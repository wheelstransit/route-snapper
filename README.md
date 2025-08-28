# route-snapper

A Rust library to figure out the exact location of bus stops along a bus route.

## The Problem

Imagine a bus route that travels down a street, enters a cul-de-sac (a dead-end circle), and then comes back out along the same street. 

If you just try to find the closest point on the route for a bus stop, you can get it wrong. A stop on the "return" part of the journey might be geographically very close to the "outgoing" part of the route. This library is smart enough to figure out the correct sequence.

## How It Works

Instead of just finding the single closest point on the route for each stop, the library does the following:

1.  **Finds all possibilities:** For each stop, it finds every plausible place it could be on the route line. For a stop in an overlapping section, it will find a candidate on both the outbound and inbound parts of the route.

2.  **Calculates a "cost":** It then calculates a cost to travel from each possibility for one stop to each possibility for the next. This cost is a mix of:
    *   How far the real stop is from the projected point on the line (the error).
    *   How much of a detour the bus would have to make to get between the two projected points (the detour cost).

3.  **Finds the cheapest path:** Using a method called dynamic programming, it finds the sequence of projections that has the lowest total cost, ensuring the bus is always moving forward.

## Usage

Here is a simple example of how to use the library:

```rust
use route_snapper::{project_stops, Stop};
use geo_types::{Coord, LineString};

// A route that goes out and comes back on the same path (overlap)
let route_line = LineString::from(vec![
    Coord { x: 0.0, y: 0.0 },   // Start
    Coord { x: 100.0, y: 0.0 }, // Point A
    Coord { x: 200.0, y: 0.0 }, // End of cul-de-sac
    Coord { x: 100.0, y: 0.0 }, // Back at Point A
    Coord { x: 0.0, y: 0.0 },   // Back at Start
]);

// Stops are ordered by travel sequence. Stop 3 is geographically near Stop 1.
let stops = vec![
    Stop { id: "1".to_string(), location: Coord { x: 99.0, y: 2.0 } },  // Outbound
    Stop { id: "2".to_string(), location: Coord { x: 201.0, y: -1.0 } }, // At the end
    Stop { id: "3".to_string(), location: Coord { x: 101.0, y: -2.0 } }, // Inbound
];

let results = project_stops(&route_line, &stops, None).unwrap();

// The distance for Stop 3 will be greater than Stop 2's distance,
// showing the overlap was handled correctly.
assert!(results[2].distance_along_route > results[1].distance_along_route);
```

## Building the Project

To build the library and run the tests, you can use the following commands:

```bash
# Build the library
cargo build

# Run the tests
cargo test
```
