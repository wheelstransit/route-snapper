//! A Rust library to accurately project an ordered list of stops onto a route line,
//! intelligently handling cases where the route overlaps itself.
//!
//! This library uses a high-performance greedy algorithm accelerated by a spatial
//! index (R-tree), with the assumption that the first stop is at the start of the
//! route line and the last stop is at the end.
//!
//! # Algorithm Overview
//!
//! 1.  **Anchoring:** The first stop is fixed to the start of the route (`distance = 0`),
//!     and the last stop is fixed to the end (`distance = total_route_length`).
//!
//! 2.  **Build Spatial Index:** An R-tree is built from the line segments of the
//!     route for near-instantaneous querying of geographically nearby segments.
//!
//! 3.  **Adaptive Sequential Projection:** Intermediate stops (between the first and
//!     last) are processed in order. For each stop, an adaptive search finds the
//!     most plausible forward projection by choosing the geographically closest
//!     candidate from a pool of valid forward options.
//!
//! # Example Usage
//!
//! ```rust
//! use route_snapper::{project_stops, Stop};
//! use geo_types::{Coord, LineString};
//!
//! fn main() {
//!     let route_line = LineString::from(vec![
//!         Coord { x: 0.0, y: 0.0 },   // Start (matches Stop "A")
//!         Coord { x: 100.0, y: 0.0 },
//!         Coord { x: 200.0, y: 0.0 }, // End of cul-de-sac
//!         Coord { x: 100.0, y: 0.0 },
//!         Coord { x: 0.0, y: 0.0 },   // End (matches Stop "D")
//!     ]);
//!
//!     let stops = vec![
//!         Stop { id: "A".to_string(), location: Coord { x: 0.0, y: 1.0 } },   // First stop
//!         Stop { id: "B".to_string(), location: Coord { x: 99.0, y: 2.0 } },  // Intermediate
//!         Stop { id: "C".to_string(), location: Coord { x: 101.0, y: -2.0 } }, // Intermediate (on overlap)
//!         Stop { id: "D".to_string(), location: Coord { x: 0.0, y: -1.0 } },  // Last stop
//!     ];
//!
//!     let results = project_stops(&route_line, &stops, None).unwrap();
//!
//!     assert!((results[0].distance_along_route - 0.0).abs() < 1e-9);    // First stop anchored
//!     assert!((results[1].distance_along_route - 99.0).abs() < 1e-9);   // Snapped correctly
//!     assert!((results[2].distance_along_route - 299.0).abs() < 1e-9);  // Snapped correctly on return trip
//!     assert!((results[3].distance_along_route - 400.0).abs() < 1e-9);  // Last stop anchored
//! }
//! ```

use geo::prelude::*;
use geo_types::{Coord, Line, LineString, Point};
use rstar::{RTree, AABB};

/// Represents a single bus stop with a unique identifier and location.
#[derive(Debug, Clone)]
pub struct Stop {
    pub id: String,
    pub location: Coord<f64>,
}

/// The output struct representing a stop's successful projection onto the route.
#[derive(Debug, PartialEq)]
pub struct ProjectedStop {
    /// The original stop's ID.
    pub id: String,
    /// The original stop's real-world location.
    pub original_location: Coord<f64>,
    /// The coordinate of the projection on the route line.
    pub projected_location: Coord<f64>,
    /// The cumulative distance from the start of the route line to the projected point.
    pub distance_along_route: f64,
    /// The geographical distance between the original location and the projected location (projection error).
    pub projection_error: f64,
}

/// A configuration struct for the projection algorithm.
#[derive(Debug, Clone, Copy)]
pub struct ProjectionConfig {}

impl Default for ProjectionConfig {
    fn default() -> Self {
        Self {}
    }
}

/// Custom error types for the library.
#[derive(Debug, PartialEq)]
pub enum ProjectionError {
    /// The provided route `LineString` has fewer than two points.
    RouteIsEmpty,
    /// The provided slice of `Stop`s is empty.
    NoStopsProvided,
    /// The algorithm failed to find a valid forward projection for an intermediate stop.
    NoProjectionFound,
}

/// Internal struct to hold data for each route segment in the R-tree.
#[derive(Debug, Clone, Copy)]
struct RouteSegment {
    line: Line<f64>,
    cumulative_distance: f64,
}

/// The number of valid forward candidates to find before stopping the search.
const CANDIDATE_POOL_SIZE: usize = 5;
/// A safety limit to prevent infinite loops on malformed data.
const MAX_SEGMENTS_TO_CHECK: usize = 1000;

/// The main function of the library.
pub fn project_stops(
    route_line: &LineString<f64>,
    stops: &[Stop],
    _config: Option<ProjectionConfig>,
) -> Result<Vec<ProjectedStop>, ProjectionError> {
    if route_line.0.len() < 2 {
        return Err(ProjectionError::RouteIsEmpty);
    }
    if stops.is_empty() {
        return Err(ProjectionError::NoStopsProvided);
    }

    let mut projected_stops = Vec::with_capacity(stops.len());

    // Handle the single-stop case (anchored to start).
    if stops.len() == 1 {
        let stop = &stops[0];
        let start_coord = route_line.0[0];
        projected_stops.push(ProjectedStop {
            id: stop.id.clone(),
            original_location: stop.location,
            projected_location: start_coord,
            distance_along_route: 0.0,
            projection_error: Point(stop.location).euclidean_distance(&Point(start_coord)),
        });
        return Ok(projected_stops);
    }

    // Pre-calculate cumulative distances and total length
    let cumulative_distances: Vec<f64> = std::iter::once(0.0)
        .chain(route_line.lines().scan(0.0, |state, line| {
            *state += Point(line.start).euclidean_distance(&Point(line.end));
            Some(*state)
        }))
        .collect();
    let total_route_length = *cumulative_distances.last().unwrap();

    // 1. Anchor the first stop
    let first_stop = &stops[0];
    let start_coord = route_line.0[0];
    projected_stops.push(ProjectedStop {
        id: first_stop.id.clone(),
        original_location: first_stop.location,
        projected_location: start_coord,
        distance_along_route: 0.0,
        projection_error: Point(first_stop.location).euclidean_distance(&Point(start_coord)),
    });
    let mut last_distance_along_route = 0.0;

    // 2. Process intermediate stops, if any
    if stops.len() > 2 {
        let rtree = build_rtree(route_line, &cumulative_distances);
        for stop in &stops[1..stops.len() - 1] {
            let stop_point = Point::from(stop.location);
            let search_point = [stop.location.x, stop.location.y];

            let mut valid_candidates = Vec::new();
            let mut segments_checked = 0;

            for segment in rtree.nearest_neighbor_iter(&search_point) {
                segments_checked += 1;

                let closest_pt = segment.line.closest_point(&stop_point);
                let (projected_point, projected_location) = match closest_pt {
                    geo::Closest::Intersection(p) | geo::Closest::SinglePoint(p) => (p, p.into()),
                    geo::Closest::Indeterminate => continue,
                };

                let dist_on_segment =
                    Point(segment.line.start).euclidean_distance(&projected_point);
                let distance_along_route = segment.cumulative_distance + dist_on_segment;

                if distance_along_route >= last_distance_along_route {
                    let projection_error = stop_point.euclidean_distance(&projected_point);
                    valid_candidates.push(ProjectedStop {
                        id: stop.id.clone(),
                        original_location: stop.location,
                        projected_location,
                        distance_along_route,
                        projection_error,
                    });
                }

                if valid_candidates.len() >= CANDIDATE_POOL_SIZE
                    || segments_checked >= MAX_SEGMENTS_TO_CHECK
                {
                    break;
                }
            }

            let best_candidate = valid_candidates.into_iter().min_by(|a, b| {
                a.projection_error.partial_cmp(&b.projection_error).unwrap()
            });

            if let Some(winner) = best_candidate {
                last_distance_along_route = winner.distance_along_route;
                projected_stops.push(winner);
            } else {
                return Err(ProjectionError::NoProjectionFound);
            }
        }
    }

    // 3. Anchor the last stop
    let last_stop = stops.last().unwrap();
    let end_coord = *route_line.0.last().unwrap();
    projected_stops.push(ProjectedStop {
        id: last_stop.id.clone(),
        original_location: last_stop.location,
        projected_location: end_coord,
        distance_along_route: total_route_length,
        projection_error: Point(last_stop.location).euclidean_distance(&Point(end_coord)),
    });

    Ok(projected_stops)
}

/// Builds the R-tree from pre-calculated cumulative distances.
fn build_rtree(route_line: &LineString<f64>, cumulative_distances: &[f64]) -> RTree<RouteSegment> {
    let segments: Vec<RouteSegment> = route_line
        .lines()
        .enumerate()
        .map(|(i, line)| RouteSegment {
            line,
            cumulative_distance: cumulative_distances[i],
        })
        .collect();

    RTree::bulk_load(segments)
}

impl rstar::RTreeObject for RouteSegment {
    type Envelope = AABB<[f64; 2]>;

    fn envelope(&self) -> Self::Envelope {
        let rect = self.line.bounding_rect();
        AABB::from_corners([rect.min().x, rect.min().y], [rect.max().x, rect.max().y])
    }
}

impl rstar::PointDistance for RouteSegment {
    fn distance_2(&self, point: &[f64; 2]) -> f64 {
        let p = Point::new(point[0], point[1]);
        let closest_point = self.line.closest_point(&p);
        let distance_2 = match closest_point {
            geo::Closest::Intersection(p_on_line) | geo::Closest::SinglePoint(p_on_line) => {
                p.euclidean_distance(&p_on_line)
            }
            geo::Closest::Indeterminate => 0.0,
        };
        distance_2 * distance_2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create a stop.
    fn make_stop(id: &str, x: f64, y: f64) -> Stop {
        Stop {
            id: id.to_string(),
            location: Coord { x, y },
        }
    }

    #[test]
    fn test_anchoring_with_two_stops() {
        let route_line =
            LineString::from(vec![Coord { x: 0.0, y: 0.0 }, Coord { x: 100.0, y: 0.0 }]);
        let stops = vec![make_stop("1", 1.0, 1.0), make_stop("2", 99.0, -1.0)];

        let results = project_stops(&route_line, &stops, None).unwrap();
        assert_eq!(results.len(), 2);
        // First stop is anchored to start
        assert_eq!(results[0].distance_along_route, 0.0);
        assert_eq!(results[0].projected_location, (Coord { x: 0.0, y: 0.0 }));
        // Last stop is anchored to end
        assert_eq!(results[1].distance_along_route, 100.0);
        assert_eq!(results[1].projected_location, (Coord { x: 100.0, y: 0.0 }));
    }

    #[test]
    fn test_simple_linear_route_with_intermediate() {
        let route_line =
            LineString::from(vec![Coord { x: 0.0, y: 0.0 }, Coord { x: 100.0, y: 0.0 }]);
        let stops = vec![
            make_stop("start", 0.0, 1.0),
            make_stop("mid", 50.0, -2.0),
            make_stop("end", 100.0, 1.0),
        ];

        let results = project_stops(&route_line, &stops, None).unwrap();
        assert_eq!(results.len(), 3);
        assert!((results[0].distance_along_route - 0.0).abs() < 1e-9);
        assert!((results[1].distance_along_route - 50.0).abs() < 1e-9);
        assert!((results[2].distance_along_route - 100.0).abs() < 1e-9);
        assert!((results[1].projection_error - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_critical_overlap_case() {
        let route_line = LineString::from(vec![
            Coord { x: 0.0, y: 0.0 },
            Coord { x: 100.0, y: 0.0 },
            Coord { x: 200.0, y: 0.0 },
            Coord { x: 100.0, y: 0.0 },
            Coord { x: 0.0, y: 0.0 },
        ]);

        let stops = vec![
            make_stop("start", 0.0, 1.0),
            make_stop("outbound", 99.0, 2.0),
            make_stop("turnaround", 201.0, -1.0),
            make_stop("inbound", 101.0, -2.0),
            make_stop("end", 0.0, -1.0),
        ];

        let results = project_stops(&route_line, &stops, None).unwrap();

        assert_eq!(results.len(), 5);
        assert!((results[0].distance_along_route - 0.0).abs() < 1e-9); // Anchored start
        assert!((results[1].distance_along_route - 99.0).abs() < 1e-9); // Intermediate outbound
        assert!((results[2].distance_along_route - 200.0).abs() < 1e-9); // Intermediate turnaround
        assert!((results[3].distance_along_route - 299.0).abs() < 1e-9); // Intermediate inbound
        assert!((results[4].distance_along_route - 400.0).abs() < 1e-9); // Anchored end
    }

    #[test]
    fn test_error_conditions() {
        let empty_route = LineString::from(vec![Coord { x: 0.0, y: 0.0 }]);
        let single_stop = vec![make_stop("1", 1.0, 1.0)];
        assert_eq!(
            project_stops(&empty_route, &single_stop, None),
            Err(ProjectionError::RouteIsEmpty)
        );

        let valid_route =
            LineString::from(vec![Coord { x: 0.0, y: 0.0 }, Coord { x: 1.0, y: 1.0 }]);
        let no_stops = vec![];
        assert_eq!(
            project_stops(&valid_route, &no_stops, None),
            Err(ProjectionError::NoStopsProvided)
        );
    }

    #[test]
    fn test_sharp_turn() {
        let route_line = LineString::from(vec![
            Coord { x: 0.0, y: 0.0 },
            Coord { x: 100.0, y: 0.0 },
            Coord { x: 100.0, y: 100.0 },
        ]);
        let stops = vec![
            make_stop("A", 0.0, 1.0),
            make_stop("B", 101.0, 50.0),
            make_stop("C", 100.0, 99.0),
        ];

        let results = project_stops(&route_line, &stops, None).unwrap();
        assert_eq!(results.len(), 3);
        assert!((results[0].distance_along_route - 0.0).abs() < 1e-9);
        assert!((results[1].distance_along_route - 150.0).abs() < 1e-9);
        assert!((results[2].distance_along_route - 200.0).abs() < 1e-9);
    }
}