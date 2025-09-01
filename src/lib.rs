//! A Rust library to accurately project an ordered list of stops onto a route line,
//! intelligently handling cases where the route overlaps itself.
//!
//! This library uses a high-performance greedy algorithm accelerated by a spatial
//! index (R-tree). This approach is significantly faster than dynamic programming for
//! large routes, while still correctly handling complex overlap scenarios.
//!
//! # Algorithm Overview
//!
//! 1.  **Build Spatial Index:** An R-tree is built from the bounding boxes of all
//!     line segments of the route. This allows for near-instantaneous querying of
//!     geographically nearby segments. This is a one-time setup cost.
//!
//! 2.  **Sequential Greedy Projection:** Stops are processed in their given order.
//!     For each stop:
//!     a. **Query Index:** The R-tree is queried to find a small set of the
//!        **nearest candidate segments** to the stop's location.
//!     b. **Local Projection:** The stop is projected onto only these candidate
//!        segments to find precise candidate points on the line.
//!     c. **Disambiguation:** From all candidates that are *forward* of the previous
//!        stop, the one with the **smallest projection error** (most geographically
//!        plausible) is chosen.
//!
//! # Example Usage
//!
//! ```rust
//! use route_snapper::{project_stops, Stop};
//! use geo_types::{Coord, LineString};
//!
//! fn main() {
//!     // A route that goes out and comes back on the same path (overlap)
//!     let route_line = LineString::from(vec![
//!         Coord { x: 0.0, y: 0.0 },   // Start
//!         Coord { x: 100.0, y: 0.0 }, // Point A
//!         Coord { x: 200.0, y: 0.0 }, // End of cul-de-sac
//!         Coord { x: 100.0, y: 0.0 }, // Back at Point A
//!         Coord { x: 0.0, y: 0.0 },   // Back at Start
//!     ]);
//!
//!     // Stops are ordered by travel sequence. Stop 3 is geographically near Stop 1.
//!     let stops = vec![
//!         Stop { id: "1".to_string(), location: Coord { x: 99.0, y: 2.0 } },  // Outbound
//!         Stop { id: "2".to_string(), location: Coord { x: 201.0, y: -1.0 } }, // At the end
//!         Stop { id: "3".to_string(), location: Coord { x: 101.0, y: -2.0 } }, // Inbound
//!     ];
//!
//!     let results = project_stops(&route_line, &stops, None).unwrap();
//!
//!     // The distance for Stop 3 should be greater than Stop 2's distance,
//!     // demonstrating the overlap was handled correctly.
//!     assert!((results[0].distance_along_route - 99.0).abs() < 1e-9);
//!     assert!((results[1].distance_along_route - 200.0).abs() < 1e-9);
//!     assert!((results[2].distance_along_route - 299.0).abs() < 1e-9);
//!     
//!     println!("Successfully projected stops:");
//!     for stop in results {
//!         println!(
//!             "Stop '{}' -> distance_along_route: {:.2}",
//!             stop.id, stop.distance_along_route
//!         );
//!     }
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
/// Note: With the new algorithm, this struct is currently unused but kept for API stability.
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
    /// The algorithm failed to find a valid forward projection for a stop.
    /// This may indicate out-of-order stops or a large gap between the stop and the route.
    NoProjectionFound,
}

/// Internal struct to hold data for each route segment in the R-tree.
#[derive(Debug, Clone, Copy)]
struct RouteSegment {
    line: Line<f64>,
    cumulative_distance: f64,
}

const K_NEAREST_NEIGHBORS: usize = 10;

/// The main function of the library.
///
/// # Arguments
/// * `route_line`: A `LineString` representing the full, sequential path of the bus.
/// * `stops`: An ordered slice of `Stop`s to project onto the line.
/// * `_config`: Optional configuration (currently unused).
///
/// # Returns
/// A `Result` containing either the ordered `Vec<ProjectedStop>` or a `ProjectionError`.
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

    // 1. Pre-computation: Build R-tree from route segments
    let rtree = build_rtree(route_line);

    // 2. Sequential Greedy Projection
    let mut projected_stops = Vec::with_capacity(stops.len());
    let mut last_distance_along_route = 0.0;

    for stop in stops {
        let stop_point = Point::from(stop.location);

        let search_point = [stop.location.x, stop.location.y];
        let candidate_segments = rtree.nearest_neighbor_iter(&search_point).take(K_NEAREST_NEIGHBORS);

        let mut best_candidate: Option<ProjectedStop> = None;

        for segment in candidate_segments {
            // Perform precise projection onto the candidate segment
            let closest_pt = segment.line.closest_point(&stop_point);
            let (projected_point, projected_location) = match closest_pt {
                geo::Closest::Intersection(p) | geo::Closest::SinglePoint(p) => (p, p.into()),
                geo::Closest::Indeterminate => continue,
            };

            let dist_on_segment = Point(segment.line.start).euclidean_distance(&projected_point);
            let distance_along_route = segment.cumulative_distance + dist_on_segment;

            // **Disambiguation:** Check if this projection is a valid forward move
            if distance_along_route >= last_distance_along_route {
                let projection_error = stop_point.euclidean_distance(&projected_point);

                let current_candidate = ProjectedStop {
                    id: stop.id.clone(),
                    original_location: stop.location,
                    projected_location,
                    distance_along_route,
                    projection_error,
                };

                // **THE FIX IS HERE**
                // From the valid forward candidates, choose the one with the smallest
                // projection error (the one that is geographically closest).
                if let Some(best) = &best_candidate {
                    if current_candidate.projection_error < best.projection_error {
                        best_candidate = Some(current_candidate);
                    }
                } else {
                    best_candidate = Some(current_candidate);
                }
            }
        }

        if let Some(winner) = best_candidate {
            last_distance_along_route = winner.distance_along_route;
            projected_stops.push(winner);
        } else {
            // If no valid forward projection was found, it's an error.
            return Err(ProjectionError::NoProjectionFound);
        }
    }

    Ok(projected_stops)
}

/// Builds the R-tree and calculates cumulative distances.
fn build_rtree(route_line: &LineString<f64>) -> RTree<RouteSegment> {
    let cumulative_distances: Vec<f64> = std::iter::once(0.0)
        .chain(route_line.lines().scan(0.0, |state, line| {
            *state += Point(line.start).euclidean_distance(&Point(line.end));
            Some(*state)
        }))
        .collect();

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

// Implement the `RTreeObject` trait for `RouteSegment`
impl rstar::RTreeObject for RouteSegment {
    type Envelope = AABB<[f64; 2]>;

    fn envelope(&self) -> Self::Envelope {
        let rect = self.line.bounding_rect();
        AABB::from_corners(
            [rect.min().x, rect.min().y],
            [rect.max().x, rect.max().y],
        )
    }
}

// Implement `PointDistance` to allow `nearest_neighbor_iter`.
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
        // R-star uses squared distance for performance.
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
    fn test_simple_linear_route() {
        let route_line = LineString::from(vec![Coord { x: 0.0, y: 0.0 }, Coord { x: 100.0, y: 0.0 }]);
        let stops = vec![
            make_stop("1", 25.0, 1.0),
            make_stop("2", 75.0, -1.0),
        ];

        let results = project_stops(&route_line, &stops, None).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "1");
        assert_eq!(results[1].id, "2");

        assert!((results[0].distance_along_route - 25.0).abs() < 1e-9);
        assert!((results[1].distance_along_route - 75.0).abs() < 1e-9);
        assert!((results[0].projection_error - 1.0).abs() < 1e-9);
        assert!((results[1].projection_error - 1.0).abs() < 1e-9);
        assert!(results[1].distance_along_route > results[0].distance_along_route);
    }

    #[test]
    fn test_critical_overlap_case() {
        // A route that goes out 200m and comes back on the same path.
        let route_line = LineString::from(vec![
            Coord { x: 0.0, y: 0.0 },   // Start
            Coord { x: 100.0, y: 0.0 }, // Point A
            Coord { x: 200.0, y: 0.0 }, // End of cul-de-sac
            Coord { x: 100.0, y: 0.0 }, // Back at Point A
            Coord { x: 0.0, y: 0.0 },   // Back at Start
        ]);
        // Total length = 200 (out) + 200 (back) = 400

        // Stop 1 is outbound near x=100.
        // Stop 2 is at the end of the line.
        // Stop 3 is inbound, also near x=100, but should be on the return path.
        let stops = vec![
            make_stop("1", 99.0, 2.0),
            make_stop("2", 201.0, -1.0),
            make_stop("3", 101.0, -2.0),
        ];

        let results = project_stops(&route_line, &stops, None).unwrap();

        assert_eq!(results.len(), 3);
        let stop1_res = &results[0];
        let stop2_res = &results[1];
        let stop3_res = &results[2];

        // Stop 1 should be on the outbound segment, around distance 99.
        assert!((stop1_res.distance_along_route - 99.0).abs() < 1e-9);
        assert_eq!(stop1_res.projected_location.x, 99.0);

        // Stop 2 should be at the turnaround point, distance 200.
        assert!((stop2_res.distance_along_route - 200.0).abs() < 1e-9);
        assert_eq!(stop2_res.projected_location.x, 200.0);

        // Stop 3 should be on the inbound segment.
        // The projection point is x=101, which is on the 3rd segment (200->100).
        // The distance along this segment is 200.0 - 101.0 = 99.0
        // The 3rd segment starts at distance 200. So, 200 + 99 = 299
        assert!(stop3_res.distance_along_route > stop2_res.distance_along_route);
        assert!((stop3_res.distance_along_route - 299.0).abs() < 1e-9);
        assert_eq!(stop3_res.projected_location.x, 101.0);
    }

    #[test]
    fn test_error_conditions() {
        let empty_route = LineString::from(vec![Coord { x: 0.0, y: 0.0 }]);
        let single_stop = vec![make_stop("1", 1.0, 1.0)];
        assert_eq!(
            project_stops(&empty_route, &single_stop, None),
            Err(ProjectionError::RouteIsEmpty)
        );

        let valid_route = LineString::from(vec![Coord { x: 0.0, y: 0.0 }, Coord { x: 1.0, y: 1.0 }]);
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
            make_stop("A", 50.0, -1.0), // On first segment
            make_stop("B", 101.0, 50.0), // On second segment
        ];

        let results = project_stops(&route_line, &stops, None).unwrap();
        assert_eq!(results.len(), 2);

        // Stop A should have distance ~50
        assert!((results[0].distance_along_route - 50.0).abs() < 1e-9);
        // Stop B should have distance 100 (from first segment) + 50 (on second) = ~150
        assert!((results[1].distance_along_route - 150.0).abs() < 1e-9);
    }
}