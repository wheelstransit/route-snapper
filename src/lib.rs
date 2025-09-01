//! A Rust library to accurately project an ordered list of stops onto a route line,
//! intelligently handling cases where the route overlaps itself.
//!
//! This library uses a high-performance greedy algorithm accelerated by a spatial
//! index (R-tree). This approach is significantly faster than dynamic programming for
//! large routes, while still correctly handling complex overlap scenarios.
//!
//! # Algorithm Overview
//!
//! 1.  **Build Spatial Index:** An R-tree is built from the line segments of the
//!     route for near-instantaneous querying of geographically nearby segments.
//!
//! 2.  **Adaptive Sequential Projection:** Stops are processed in order. For each stop:
//!     a. **Iterative Search:** An iterator of the nearest route segments is created.
//!        The algorithm pulls from this iterator one segment at a time.
//!     b. **Filtering:** Each segment is tested. If a projection onto it is a
//!        valid *forward* move from the previous stop, it's added to a pool of
//!        candidates.
//!     c. **Termination:** The search stops once a small number of valid forward
//!        candidates have been found, or a safety limit is reached. This ensures
//!        the search is fast for simple cases but robust for complex loops where
//!        the correct segment may not be the absolute closest.
//!     d. **Selection:** The best candidate from the pool is chosen based on the
//!        smallest projection error (best geographic fit).
//!
//! # Example Usage
//!
//! ```rust
//! // ... (Example usage is the same and still valid)
//! ```

use geo::prelude::*;
use geo_types::{Coord, Line, LineString, Point};
use rstar::{RTree, AABB};

// ... (Structs `Stop`, `ProjectedStop`, `ProjectionConfig`, `ProjectionError` are unchanged) ...
// --- PASTE THE STRUCT AND ERROR DEFINITIONS FROM THE PREVIOUS VERSION HERE ---

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
    RouteIsEmpty,
    NoStopsProvided,
    NoProjectionFound,
}


/// Internal struct to hold data for each route segment in the R-tree.
#[derive(Debug, Clone, Copy)]
struct RouteSegment {
    line: Line<f64>,
    cumulative_distance: f64,
}

/// The number of valid forward candidates to find before stopping the search.
/// A small number is sufficient to ensure we have a good selection to choose the best fit from.
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

    let rtree = build_rtree(route_line);
    let mut projected_stops = Vec::with_capacity(stops.len());
    let mut last_distance_along_route = 0.0;

    for stop in stops {
        let stop_point = Point::from(stop.location);
        let search_point = [stop.location.x, stop.location.y];

        let mut valid_candidates = Vec::new();
        let mut segments_checked = 0;

        // Adaptively search nearest segments until we find a pool of valid forward candidates.
        for segment in rtree.nearest_neighbor_iter(&search_point) {
            segments_checked += 1;

            let closest_pt = segment.line.closest_point(&stop_point);
            let (projected_point, projected_location) = match closest_pt {
                geo::Closest::Intersection(p) | geo::Closest::SinglePoint(p) => (p, p.into()),
                geo::Closest::Indeterminate => continue,
            };

            let dist_on_segment = Point(segment.line.start).euclidean_distance(&projected_point);
            let distance_along_route = segment.cumulative_distance + dist_on_segment;

            // If it's a valid forward projection, add it to our pool.
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

            // Stop searching once we have a decent pool of candidates or we hit the safety limit.
            if valid_candidates.len() >= CANDIDATE_POOL_SIZE || segments_checked >= MAX_SEGMENTS_TO_CHECK {
                break;
            }
        }

        // From the pool of valid forward candidates, pick the one with the best geometric fit.
        let best_candidate = valid_candidates
            .into_iter()
            .min_by(|a, b| a.projection_error.partial_cmp(&b.projection_error).unwrap());

        if let Some(winner) = best_candidate {
            last_distance_along_route = winner.distance_along_route;
            projected_stops.push(winner);
        } else {
            // If we checked many segments and still found nothing, the data is likely invalid.
            return Err(ProjectionError::NoProjectionFound);
        }
    }

    Ok(projected_stops)
}

// ... (The rest of the file: `build_rtree`, trait impls, and tests are unchanged) ...
// --- PASTE THE `build_rtree` FUNCTION, TRAIT IMPLS, AND TESTS FROM THE PREVIOUS VERSION HERE ---

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
    fn test_simple_linear_route() {
        let route_line = LineString::from(vec![Coord { x: 0.0, y: 0.0 }, Coord { x: 100.0, y: 0.0 }]);
        let stops = vec![
            make_stop("1", 25.0, 1.0),
            make_stop("2", 75.0, -1.0),
        ];

        let results = project_stops(&route_line, &stops, None).unwrap();
        assert_eq!(results.len(), 2);
        assert!((results[0].distance_along_route - 25.0).abs() < 1e-9);
        assert!((results[1].distance_along_route - 75.0).abs() < 1e-9);
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
            make_stop("1", 99.0, 2.0),
            make_stop("2", 201.0, -1.0),
            make_stop("3", 101.0, -2.0),
        ];

        let results = project_stops(&route_line, &stops, None).unwrap();
        assert_eq!(results.len(), 3);
        assert!((results[0].distance_along_route - 99.0).abs() < 1e-9);
        assert!((results[1].distance_along_route - 200.0).abs() < 1e-9);
        assert!((results[2].distance_along_route - 299.0).abs() < 1e-9);
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
            make_stop("A", 50.0, -1.0),
            make_stop("B", 101.0, 50.0),
        ];

        let results = project_stops(&route_line, &stops, None).unwrap();
        assert_eq!(results.len(), 2);
        assert!((results[0].distance_along_route - 50.0).abs() < 1e-9);
        assert!((results[1].distance_along_route - 150.0).abs() < 1e-9);
    }
}