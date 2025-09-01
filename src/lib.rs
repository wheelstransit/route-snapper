//! A Rust library to accurately project an ordered list of stops onto a route line.
//! This version uses a high-performance greedy algorithm with single-level backtracking
//! to robustly handle complex route overlaps.
//!
//! # Algorithm Overview
//!
//! 1.  **Anchoring & Spatial Index:** The first stop is fixed to the start of the route
//!     (`distance = 0`), and the last stop is fixed to the end. An R-tree is built
//!     from the route's line segments for fast geometric lookups.
//!
//! 2.  **Greedy Forward Search:** For each intermediate stop, an adaptive search
//!     finds the most plausible forward projection. It does this by creating a small
//!     pool of valid forward candidates from the nearest segments and then selecting
//!     the one with the best geographic fit (lowest projection error).
//!
//! 3.  **Backtracking on Failure:** If projecting `Stop N` fails (no valid forward
//!     candidates are found), the algorithm assumes the greedy choice for `Stop N-1`
//!     was incorrect (e.g., snapped to an early part of a large loop). It then:
//!     a. Backtracks to `Stop N-1`.
//!     b. Re-runs the search for `Stop N-1`, explicitly excluding the previous projection.
//!     c. With a new, corrected projection for `Stop N-1`, it retries projecting `Stop N`.
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
#[derive(Debug, Clone, PartialEq)]
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
    /// The algorithm failed to find a valid forward projection for a stop, even after backtracking.
    /// This likely indicates a data quality issue (e.g., out-of-order stops).
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
    if stops.len() <= 2 {
        return project_anchored_only(route_line, stops);
    }

    let cumulative_distances: Vec<f64> = std::iter::once(0.0)
        .chain(route_line.lines().scan(0.0, |state, line| {
            *state += Point(line.start).euclidean_distance(&Point(line.end));
            Some(*state)
        }))
        .collect();

    let rtree = build_rtree(route_line, &cumulative_distances);
    let mut projected_stops = Vec::with_capacity(stops.len());

    // 1. Anchor the first stop
    projected_stops.push(anchor_stop(&stops[0], &route_line.0[0], 0.0));

    // 2. Process intermediate stops with backtracking logic
    let mut i = 1;
    while i < stops.len() - 1 {
        let stop_to_project = &stops[i];
        let previous_projection = projected_stops.last().unwrap();

        let mut best_candidate = find_best_candidate(
            stop_to_project,
            &rtree,
            previous_projection.distance_along_route,
            None, // No exclusions on the first try
        );

        // BACKTRACKING LOGIC
        if best_candidate.is_none() && i > 0 {
            let bad_projection = projected_stops.pop().unwrap();
            let last_good_projection_dist =
                projected_stops.last().map_or(0.0, |p| p.distance_along_route);

            let stop_to_fix = &stops[i - 1];
            if let Some(fixed_projection) = find_best_candidate(
                stop_to_fix,
                &rtree,
                last_good_projection_dist,
                Some(bad_projection.distance_along_route),
            ) {
                projected_stops.push(fixed_projection);
                best_candidate = find_best_candidate(
                    stop_to_project,
                    &rtree,
                    projected_stops.last().unwrap().distance_along_route,
                    None,
                );
            } else {
                projected_stops.push(bad_projection); // Put it back if we couldn't find an alternative
            }
        }

        if let Some(winner) = best_candidate {
            projected_stops.push(winner);
            i += 1;
        } else {
            return Err(ProjectionError::NoProjectionFound);
        }
    }

    // 3. Anchor the last stop
    let total_route_length = *cumulative_distances.last().unwrap();
    projected_stops.push(anchor_stop(
        stops.last().unwrap(),
        route_line.0.last().unwrap(),
        total_route_length,
    ));

    Ok(projected_stops)
}

/// Helper for cases with only 0, 1, or 2 stops, which are purely anchored.
fn project_anchored_only(
    route_line: &LineString<f64>,
    stops: &[Stop],
) -> Result<Vec<ProjectedStop>, ProjectionError> {
    let mut projected_stops = Vec::new();
    if stops.is_empty() {
        return Ok(projected_stops);
    }

    projected_stops.push(anchor_stop(&stops[0], &route_line.0[0], 0.0));

    if stops.len() > 1 {
        let total_route_length = route_line.lines().map(|l| l.euclidean_length()).sum();
        projected_stops.push(anchor_stop(
            stops.last().unwrap(),
            route_line.0.last().unwrap(),
            total_route_length,
        ));
    }
    Ok(projected_stops)
}

/// Creates a `ProjectedStop` anchored to a specific point and distance.
fn anchor_stop(stop: &Stop, location: &Coord<f64>, distance: f64) -> ProjectedStop {
    ProjectedStop {
        id: stop.id.clone(),
        original_location: stop.location,
        projected_location: *location,
        distance_along_route: distance,
        projection_error: Point(stop.location).euclidean_distance(&Point(*location)),
    }
}

/// The core adaptive search function. Now accepts an optional distance to exclude.
fn find_best_candidate(
    stop: &Stop,
    rtree: &RTree<RouteSegment>,
    last_distance_along_route: f64,
    exclude_distance: Option<f64>,
) -> Option<ProjectedStop> {
    let stop_point = Point::from(stop.location);
    let search_point = [stop.location.x, stop.location.y];
    let mut valid_candidates = Vec::new();
    let mut segments_checked = 0;
    let tolerance = 1e-9;

    for segment in rtree.nearest_neighbor_iter(&search_point) {
        segments_checked += 1;
        let closest_pt = segment.line.closest_point(&stop_point);
        let (projected_point, projected_location) = match closest_pt {
            geo::Closest::Intersection(p) | geo::Closest::SinglePoint(p) => (p, p.into()),
            geo::Closest::Indeterminate => continue,
        };

        let dist_on_segment = Point(segment.line.start).euclidean_distance(&projected_point);
        let distance_along_route = segment.cumulative_distance + dist_on_segment;

        if let Some(exclude_dist) = exclude_distance {
            if (distance_along_route - exclude_dist).abs() < tolerance {
                continue;
            }
        }

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

    valid_candidates
        .into_iter()
        .min_by(|a, b| a.projection_error.partial_cmp(&b.projection_error).unwrap())
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
        let d2 = match closest_point {
            geo::Closest::Intersection(p_on_line) | geo::Closest::SinglePoint(p_on_line) => {
                p.euclidean_distance(&p_on_line)
            }
            geo::Closest::Indeterminate => 0.0,
        };
        d2 * d2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(results[0].distance_along_route, 0.0);
        assert_eq!(results[1].distance_along_route, 100.0);
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
        assert!((results[1].distance_along_route - 50.0).abs() < 1e-9);
    }

    #[test]
    fn test_critical_overlap_case() {
        let route_line = LineString::from(vec![
            Coord { x: 0.0, y: 0.0 }, Coord { x: 100.0, y: 0.0 }, Coord { x: 200.0, y: 0.0 },
            Coord { x: 100.0, y: 0.0 }, Coord { x: 0.0, y: 0.0 },
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
        assert!((results[0].distance_along_route - 0.0).abs() < 1e-9);
        assert!((results[1].distance_along_route - 99.0).abs() < 1e-9);
        assert!((results[2].distance_along_route - 200.0).abs() < 1e-9);
        assert!((results[3].distance_along_route - 299.0).abs() < 1e-9);
        assert!((results[4].distance_along_route - 400.0).abs() < 1e-9);
    }

    #[test]
    fn test_error_conditions() {
        let empty_route = LineString::from(vec![Coord { x: 0.0, y: 0.0 }]);
        assert_eq!(
            project_stops(&empty_route, &[], None),
            Err(ProjectionError::RouteIsEmpty)
        );
        let valid_route =
            LineString::from(vec![Coord { x: 0.0, y: 0.0 }, Coord { x: 1.0, y: 1.0 }]);
        assert_eq!(
            project_stops(&valid_route, &[], None),
            Err(ProjectionError::NoStopsProvided)
        );
    }
    
    #[test]
    fn test_backtracking_on_hairpin_turn() {
        // This route creates a "hairpin" where a later part of the route is
        // geographically closer to an earlier stop than its correct segment.
        let route_line = LineString::from(vec![
            Coord { x: 0.0, y: 0.0 },    // 0m
            Coord { x: 100.0, y: 0.0 },   // 100m
            Coord { x: 100.0, y: 10.0 },  // 110m (start of hairpin)
            Coord { x: 0.0, y: 10.0 },    // 210m (end of hairpin)
            Coord { x: 0.0, y: 20.0 },    // 220m
        ]);

        let stops = vec![
            make_stop("A", 0.0, 1.0),     // Anchor Start
            // Stop B is at x=90. The greedy choice will be the first segment (dist 90).
            make_stop("B", 90.0, -1.0),
            // Stop C is at x=10. The greedy choice from B's dist=90 would be the
            // hairpin return segment (dist 200), because the first segment (dist 10) is behind.
            // BUT, the geographically closest segment to B is actually the hairpin return (dist ~200).
            // A simple greedy algorithm would snap B to dist ~200, which would then fail for C.
            // Our backtracking should fix this.
            make_stop("C", 10.0, 11.0),
            make_stop("D", 0.0, 19.0),     // Anchor End
        ];
        
        let results = project_stops(&route_line, &stops, None).unwrap();

        assert_eq!(results.len(), 4);
        // A is anchored
        assert!((results[0].distance_along_route - 0.0).abs() < 1e-9);
        // B should be correctly placed on the first segment
        assert!((results[1].distance_along_route - 90.0).abs() < 1e-9);
        // C should be correctly placed on the hairpin return
        assert!((results[2].distance_along_route - 200.0).abs() < 1e-9);
        // D is anchored
        assert!((results[3].distance_along_route - 220.0).abs() < 1e-9);
    }
}