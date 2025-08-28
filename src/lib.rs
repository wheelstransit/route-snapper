//! A Rust library to accurately project an ordered list of stops onto a route line,
//! intelligently handling cases where the route overlaps itself.
//!
//! The core of this library is a dynamic programming algorithm that finds the
//! globally optimal snapping for a sequence of stops, avoiding the pitfalls of a
//! simple "find closest point" approach.
//!
//! # Algorithm Overview
//!
//! 1.  **Generate Candidate Projections:** For each stop, find all plausible projection
//!     points on the route line by identifying local minima in the distance function.
//!     This correctly finds candidates on both outbound and inbound segments in
//!     overlapping sections.
//!
//! 2.  **Define a Cost Function:** Calculate the "cost" of traveling between a
//!     candidate projection for `Stop i` and a candidate for `Stop i+1`. This cost
//!     balances projection accuracy (distance from stop to line) and path plausibility
//!     (detour factor).
//!
//! 3.  **Find Minimum Cost Path:** Use dynamic programming to find the sequence of
//!     candidates that minimizes the total accumulated cost from the first to the
//!     last stop.
//!
//! 4.  **Backtrack and Return:** Reconstruct the optimal path to produce the final
//!     ordered list of `ProjectedStop`s.
//!
//! # Example Usage
//!
//! ```rust,ignore
//! fn main() {
//!     use route_snapper::{project_stops, Stop, ProjectionConfig};
//!     use geo_types::{Coord, LineString};
//!
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
//!     assert!(results[0].distance_along_route < 100.0); // Around 99.0
//!     assert!(results[1].distance_along_route > 199.0); // Around 200.0
//!     assert!(results[2].distance_along_route > results[1].distance_along_route); // Should be around 300.0
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
use geo_types::{Coord, LineString, Point};

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
pub struct ProjectionConfig {
    /// A weight to balance the importance of projection error vs. detour cost.
    /// A higher value prioritizes paths with a more direct travel distance between stops.
    pub detour_cost_weight: f64,
}

impl Default for ProjectionConfig {
    /// Creates a default configuration with a `detour_cost_weight` of 1.0.
    fn default() -> Self {
        Self {
            detour_cost_weight: 1.0,
        }
    }
}

/// Custom error types for the library.
#[derive(Debug, PartialEq)]
pub enum ProjectionError {
    /// The provided route `LineString` has fewer than two points.
    RouteIsEmpty,
    /// The provided slice of `Stop`s is empty.
    NoStopsProvided,
    /// The algorithm failed to find a valid projection for the stops.
    NoPathFound,
}

/// A candidate projection for a single stop onto the route line.
#[derive(Debug, Clone, Copy)]
struct Candidate {
    /// The coordinate of the projection on the route line.
    projected_location: Coord<f64>,
    /// The cumulative distance from the start of the route line to the projected point.
    distance_along_route: f64,
    /// The geographical distance between the original location and the projected location.
    projection_error: f64,
}

/// The main function of the library.
///
/// # Arguments
/// * `route_line`: A `LineString` representing the full, sequential path of the bus.
/// * `stops`: An ordered slice of `Stop`s to project onto the line.
/// * `config`: Optional configuration for the projection algorithm.
///
/// # Returns
/// A `Result` containing either the ordered `Vec<ProjectedStop>` or a `ProjectionError`.
pub fn project_stops(
    route_line: &LineString<f64>,
    stops: &[Stop],
    config: Option<ProjectionConfig>,
) -> Result<Vec<ProjectedStop>, ProjectionError> {
    if route_line.0.len() < 2 {
        return Err(ProjectionError::RouteIsEmpty);
    }
    if stops.is_empty() {
        return Err(ProjectionError::NoStopsProvided);
    }

    let config = config.unwrap_or_default();

    // Pre-calculate cumulative distances along the route
    let cumulative_distances: Vec<f64> = std::iter::once(0.0)
        .chain(route_line.lines().scan(0.0, |state, line| {
            *state += Point(line.start).euclidean_distance(&Point(line.end));
            Some(*state)
        }))
        .collect();

    // 1. Generate candidate projections for each stop
    let all_candidates: Vec<Vec<Candidate>> = stops
        .iter()
        .map(|stop| find_candidate_projections(stop, route_line, &cumulative_distances))
        .collect();

    // 2. Find the minimum cost path using dynamic programming
    let (dp_table, final_stop_idx) =
        build_dp_table(&all_candidates, stops, &config)?;

    // 3. Backtrack to find the winning path
    let winning_path = backtrack(&dp_table, final_stop_idx);

    // 4. Construct the final result
    let result = winning_path
        .iter()
        .zip(stops)
        .zip(&all_candidates)
        .map(
            |(((candidate_idx, _cost), stop), candidates)| {
                let best_candidate = candidates[*candidate_idx];
                ProjectedStop {
                    id: stop.id.clone(),
                    original_location: stop.location,
                    projected_location: best_candidate.projected_location,
                    distance_along_route: best_candidate.distance_along_route,
                    projection_error: best_candidate.projection_error,
                }
            },
        )
        .collect();

    Ok(result)
}

/// Finds all plausible candidate projections for a stop on the route line.
fn find_candidate_projections(
    stop: &Stop,
    route_line: &LineString<f64>,
    cumulative_distances: &[f64],
) -> Vec<Candidate> {
    let mut candidates = Vec::new();
    for (i, line) in route_line.lines().enumerate() {
        let closest_pt = line.closest_point(&Point::from(stop.location));
        let (projected_point, projected_location) = match closest_pt {
            geo::Closest::Intersection(p) | geo::Closest::SinglePoint(p) => (p, p.into()),
            geo::Closest::Indeterminate => continue, // Should not happen on a line
        };

        let dist_on_segment = Point(line.start).euclidean_distance(&projected_point);
        let distance_along_route = cumulative_distances[i] + dist_on_segment;
        let projection_error = Point(stop.location).euclidean_distance(&projected_point);

        candidates.push(Candidate {
            projected_location,
            distance_along_route,
            projection_error,
        });
    }
    // A simple way to filter for "local minima" is to sort by projection error
    // and take the best few. For this problem, we can be more generous and keep
    // all candidates, letting the DP algorithm decide.
    candidates
}

/// Builds the dynamic programming table to find the minimum cost path.
fn build_dp_table(
    all_candidates: &[Vec<Candidate>],
    stops: &[Stop],
    config: &ProjectionConfig,
) -> Result<(Vec<Vec<(usize, f64)>>, usize), ProjectionError> {
    let mut dp_table: Vec<Vec<(usize, f64)>> = Vec::with_capacity(stops.len());

    // Initialize DP table for the first stop
    let first_stop_costs: Vec<(usize, f64)> = all_candidates[0]
        .iter()
        .map(|c| (usize::MAX, c.projection_error)) // (backpointer, cost)
        .collect();
    dp_table.push(first_stop_costs);

    // Fill the rest of the DP table
    for i in 1..stops.len() {
        let mut current_stop_costs = Vec::new();
        for (_j, current_cand) in all_candidates[i].iter().enumerate() {
            let mut min_cost = f64::INFINITY;
            let mut backpointer = 0;

            for (k, prev_cand) in all_candidates[i - 1].iter().enumerate() {
                let prev_total_cost = dp_table[i - 1][k].1;
                let transition_cost = calculate_transition_cost(
                    prev_cand,
                    current_cand,
                    &stops[i-1],
                    &stops[i],
                    config,
                );

                if prev_total_cost + transition_cost < min_cost {
                    min_cost = prev_total_cost + transition_cost;
                    backpointer = k;
                }
            }
            current_stop_costs.push((backpointer, min_cost));
        }
        dp_table.push(current_stop_costs);
    }

    // Find the best path at the very end
    let final_stop_idx = dp_table
        .last()
        .unwrap()
        .iter()
        .enumerate()
        .min_by(|a, b| a.1 .1.partial_cmp(&b.1 .1).unwrap())
        .map(|(idx, _)| idx)
        .ok_or(ProjectionError::NoPathFound)?;

    Ok((dp_table, final_stop_idx))
}

/// Calculates the cost of moving from one candidate to another.
fn calculate_transition_cost(
    prev_cand: &Candidate,
    current_cand: &Candidate,
    _prev_stop: &Stop,
    _current_stop: &Stop,
    config: &ProjectionConfig,
) -> f64 {
    let distance_on_route = current_cand.distance_along_route - prev_cand.distance_along_route;

    // **Constraint:** Disallow backward travel
    if distance_on_route <= 0.0 {
        return f64::INFINITY;
    }

    let as_crow_flies = Point(prev_cand.projected_location)
        .euclidean_distance(&Point(current_cand.projected_location));

    // **Detour Cost:** Penalize nonsensical paths
    let detour_cost = if as_crow_flies > 0.0 {
        (distance_on_route / as_crow_flies) - 1.0
    } else {
        0.0 // Avoid division by zero if points are identical
    };

    // **Total Cost:** Projection error + weighted detour cost
    current_cand.projection_error + config.detour_cost_weight * detour_cost
}

/// Backtracks through the DP table to reconstruct the optimal path.
fn backtrack(dp_table: &[Vec<(usize, f64)>], final_stop_idx: usize) -> Vec<(usize, f64)> {
    let mut path = Vec::with_capacity(dp_table.len());
    let mut current_cand_idx = final_stop_idx;
    let final_cost = dp_table.last().unwrap()[current_cand_idx].1;
    path.push((current_cand_idx, final_cost));

    for i in (1..dp_table.len()).rev() {
        let (backpointer, _cost) = dp_table[i][current_cand_idx];
        current_cand_idx = backpointer;
        let cost = dp_table[i - 1][current_cand_idx].1;
        path.push((current_cand_idx, cost));
    }

    path.reverse();
    path
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