//! Per-player road network tracking with a cached longest road.
//!
//! **Data structures.** Each player's [`RoadNetwork`] stores a 72-bit `roads`
//! bitmask (one bit per board edge), a `reachable` frontier of empty edges
//! adjacent to the network, and a cached `longest_len`.
//!
//! **`>= 5` threshold.** The Catan longest-road award requires at least 5
//! roads, so DFS is skipped entirely when `roads.count_ones() < 5`.
//!
//! **Recompute on each road (`add_road`).** Every road placed with five or
//! more down searches the whole road graph again. Extending the cache from
//! the new road's tip alone missed a longer path elsewhere whenever the
//! cache was already short: the fifth road branching off the middle of four
//! cached 3 for a road of 4, and later tips only added to that.
//!
//! **Transit-node check (`on_opponent_build`).** An opponent settlement can
//! only shorten our longest road if it sits on a transit node—one with >= 2
//! incident roads of ours. Nodes with 0 or 1 incident roads are off-network
//! or leaves and can never split a path, so DFS is skipped.
//!
//! **DFS algorithm.** `compute_longest_road` performs exhaustive simple-path
//! search. It starts DFS only from vertices with degree != 2 (leaves or
//! branch points) or under an opponent building, since an optimal path's
//! endpoints must be one of those. A pure-cycle fallback handles the all-degree-2 case.
//! `dfs_road` walks edges via a `visited` bitmask, stopping at opponent
//! buildings.

use super::board::{AdjacencyBitboards, EdgeId, NodeId};

const EDGE_MASK: u128 = (1u128 << 72) - 1;

#[derive(Clone, Copy, Debug, Default)]
pub struct RoadNetwork {
    pub roads: u128, // bits 0..72 — player's placed roads
    reachable: u128, // bits 0..72 — empty edges adjacent to network
    longest_len: u8, // cached longest simple path length
}

impl RoadNetwork {
    #[cfg(test)]
    pub fn new() -> Self {
        Self::default()
    }

    /// Player placed a settlement/city — expand frontier from the new building.
    pub fn add_building(&mut self, nid: NodeId, adj: &AdjacencyBitboards, opp_roads: u128) {
        self.reachable |= adj.node_adj_edges[nid.0 as usize];
        self.reachable &= !(self.roads | opp_roads) & EDGE_MASK;
    }

    /// Player placed a road. Updates roads, expands frontier, recomputes longest.
    /// `update_longest`: false during setup (longest road award not active).
    pub fn add_road(
        &mut self,
        eid: EdgeId,
        adj: &AdjacencyBitboards,
        opp_buildings: u64,
        opp_roads: u128,
        update_longest: bool,
    ) {
        let endpoints = adj.edge_endpoints[eid.0 as usize];
        self.roads |= 1u128 << eid.0;
        self.reachable &= !(1u128 << eid.0);

        // Expand frontier from each endpoint not blocked by opponent building
        let mut ep = endpoints;
        while ep != 0 {
            let node = ep.trailing_zeros() as usize;
            ep &= ep - 1;
            if opp_buildings & (1u64 << node) == 0 {
                self.reachable |= adj.node_adj_edges[node];
            }
        }
        self.reachable &= !(self.roads | opp_roads) & EDGE_MASK;

        if update_longest && self.roads.count_ones() >= 5 {
            self.longest_len = compute_longest_road(self.roads, adj, opp_buildings);
        }
    }

    /// Opponent placed a building — may shrink frontier, recomputes longest (might split).
    pub fn on_opponent_build(
        &mut self,
        nid: NodeId,
        adj: &AdjacencyBitboards,
        my_buildings: u64,
        opp_buildings: u64,
    ) {
        let affected = adj.node_adj_edges[nid.0 as usize] & self.reachable;
        self.reachable &= !affected;

        // Re-add any edge still reachable via its OTHER endpoint
        let mut rem = affected;
        while rem != 0 {
            let eid = rem.trailing_zeros() as usize;
            rem &= rem - 1;
            let other = (adj.edge_endpoints[eid] & !(1u64 << nid.0)).trailing_zeros() as usize;
            let mine = my_buildings & (1u64 << other) != 0;
            let open_road_end =
                opp_buildings & (1u64 << other) == 0 && adj.node_adj_edges[other] & self.roads != 0;
            if mine || open_road_end {
                self.reachable |= 1u128 << eid;
            }
        }

        let incident_roads = adj.node_adj_edges[nid.0 as usize] & self.roads;
        if self.roads.count_ones() >= 5 && incident_roads.count_ones() >= 2 {
            self.longest_len = compute_longest_road(self.roads, adj, opp_buildings);
        }
    }

    /// Remove an edge from this player's frontier (opponent built a road there).
    pub fn remove_edge(&mut self, eid: EdgeId) {
        self.reachable &= !(1u128 << eid.0);
    }

    /// Bitmask of empty edges connected to the player's network.
    pub fn reachable_edges(&self) -> u128 {
        self.reachable
    }

    /// Cached longest simple path length.
    pub fn longest_road(&self) -> u8 {
        self.longest_len
    }

    /// Longest simple path if the nodes in `opp_buildings` held the
    /// opponent's buildings, which break a road through them: to see what
    /// a building somewhere would cut.
    pub fn longest_road_against(&self, adj: &AdjacencyBitboards, opp_buildings: u64) -> u8 {
        compute_longest_road(self.roads, adj, opp_buildings)
    }

    /// Bitmask of nodes that lie on the current longest road path.
    pub fn longest_road_nodes(&self, adj: &AdjacencyBitboards, opp_buildings: u64) -> u64 {
        if self.roads.count_ones() < 5 {
            return 0;
        }
        compute_longest_road_nodes(self.roads, adj, opp_buildings)
    }
}

/// Compute longest road length AND return the node bitmask of the best path.
fn compute_longest_road_nodes(
    player_roads: u128,
    adj: &AdjacencyBitboards,
    opp_buildings: u64,
) -> u64 {
    if player_roads == 0 {
        return 0;
    }

    let degree = road_degrees(player_roads, adj);
    let mut best_len = 0u8;
    let mut best_nodes = 0u64;
    // Roads in a component with a start below; the rest are loops.
    let mut covered = 0u128;

    for node in 0..54usize {
        if degree[node] != 0 && (degree[node] != 2 || opp_buildings & (1u64 << node) != 0) {
            if adj.node_adj_edges[node] & player_roads & !covered != 0 {
                covered |= component_roads(node, player_roads, adj);
            }
            let incident = adj.node_adj_edges[node] & player_roads;
            let mut rem = incident;
            while rem != 0 {
                let eid = rem.trailing_zeros() as usize;
                rem &= rem - 1;
                let other = (adj.edge_endpoints[eid] & !(1u64 << node)).trailing_zeros() as usize;
                let mut path_nodes = 0u64;
                let len = dfs_road_nodes(
                    other,
                    1u128 << eid,
                    player_roads,
                    adj,
                    opp_buildings,
                    &mut path_nodes,
                ) + 1;
                // The starting node is always part of the path
                path_nodes |= 1u64 << node;
                if len > best_len {
                    best_len = len;
                    best_nodes = path_nodes;
                }
            }
        }
    }

    // A component whose every node has two roads (a loop) has no start
    // above, even when another component does: walk it from any node.
    let mut loops = player_roads & !covered;
    while loops != 0 {
        let start = adj.edge_endpoints[loops.trailing_zeros() as usize].trailing_zeros() as usize;
        let mut rem = adj.node_adj_edges[start] & player_roads;
        while rem != 0 {
            let eid = rem.trailing_zeros() as usize;
            rem &= rem - 1;
            let other = (adj.edge_endpoints[eid] & !(1u64 << start)).trailing_zeros() as usize;
            let mut path_nodes = 0u64;
            let len = dfs_road_nodes(
                other,
                1u128 << eid,
                player_roads,
                adj,
                opp_buildings,
                &mut path_nodes,
            ) + 1;
            path_nodes |= 1u64 << start;
            if len > best_len {
                best_len = len;
                best_nodes = path_nodes;
            }
        }
        loops &= !component_roads(start, player_roads, adj);
    }

    best_nodes
}

/// DFS that also tracks the best path's node set.
/// Returns the length, and writes the best path's node bitmask into `best_path_nodes`.
fn dfs_road_nodes(
    node: usize,
    visited: u128,
    player_roads: u128,
    adj: &AdjacencyBitboards,
    opp_buildings: u64,
    best_path_nodes: &mut u64,
) -> u8 {
    if opp_buildings & (1u64 << node) != 0 {
        return 0;
    }
    let reachable = adj.node_adj_edges[node] & player_roads & !visited;
    if reachable == 0 {
        // Leaf: this node is on the path
        *best_path_nodes = 1u64 << node;
        return 0;
    }
    let mut best = 0u8;
    *best_path_nodes = 1u64 << node;
    let mut rem = reachable;
    while rem != 0 {
        let eid = rem.trailing_zeros() as usize;
        rem &= rem - 1;
        let next = (adj.edge_endpoints[eid] & !(1u64 << node)).trailing_zeros() as usize;
        let mut child_nodes = 0u64;
        let len = dfs_road_nodes(
            next,
            visited | (1u128 << eid),
            player_roads,
            adj,
            opp_buildings,
            &mut child_nodes,
        ) + 1;
        if len > best {
            best = len;
            *best_path_nodes = (1u64 << node) | child_nodes;
        }
    }
    best
}

fn compute_longest_road(player_roads: u128, adj: &AdjacencyBitboards, opp_buildings: u64) -> u8 {
    if player_roads == 0 {
        return 0;
    }

    // Compute degree of each vertex in the road subgraph.
    // The longest trail's endpoints must have degree != 2 (leaves or branch
    // points) or sit on an opponent building (which splits the network), so
    // we only need to start DFS from those vertices.
    let degree = road_degrees(player_roads, adj);
    let mut best = 0u8;
    // Roads in a component with a start below; the rest are loops.
    let mut covered = 0u128;

    for node in 0..54usize {
        if degree[node] != 0 && (degree[node] != 2 || opp_buildings & (1u64 << node) != 0) {
            if adj.node_adj_edges[node] & player_roads & !covered != 0 {
                covered |= component_roads(node, player_roads, adj);
            }
            let incident = adj.node_adj_edges[node] & player_roads;
            let mut rem = incident;
            while rem != 0 {
                let eid = rem.trailing_zeros() as usize;
                rem &= rem - 1;
                let other = (adj.edge_endpoints[eid] & !(1u64 << node)).trailing_zeros() as usize;
                best =
                    best.max(dfs_road(other, 1u128 << eid, player_roads, adj, opp_buildings) + 1);
            }
        }
    }

    // A component whose every node has two roads (a loop) has no start
    // above, even when another component does: walk it from any node.
    let mut loops = player_roads & !covered;
    while loops != 0 {
        let start = adj.edge_endpoints[loops.trailing_zeros() as usize].trailing_zeros() as usize;
        let mut rem = adj.node_adj_edges[start] & player_roads;
        while rem != 0 {
            let eid = rem.trailing_zeros() as usize;
            rem &= rem - 1;
            let other = (adj.edge_endpoints[eid] & !(1u64 << start)).trailing_zeros() as usize;
            best = best.max(dfs_road(other, 1u128 << eid, player_roads, adj, opp_buildings) + 1);
        }
        loops &= !component_roads(start, player_roads, adj);
    }

    best
}

/// Each node's number of roads in `player_roads`.
fn road_degrees(player_roads: u128, adj: &AdjacencyBitboards) -> [u8; 54] {
    let mut degree = [0u8; 54];
    let mut remaining = player_roads;
    while remaining != 0 {
        let eid = remaining.trailing_zeros() as usize;
        remaining &= remaining - 1;
        let mut ep = adj.edge_endpoints[eid];
        while ep != 0 {
            degree[ep.trailing_zeros() as usize] += 1;
            ep &= ep - 1;
        }
    }
    degree
}

/// The roads of `player_roads` connected to `node`, buildings ignored.
fn component_roads(node: usize, player_roads: u128, adj: &AdjacencyBitboards) -> u128 {
    let mut roads = 0u128;
    let mut seen = 1u64 << node;
    let mut frontier = seen;
    while frontier != 0 {
        let n = frontier.trailing_zeros() as usize;
        frontier &= frontier - 1;
        let mut new = adj.node_adj_edges[n] & player_roads & !roads;
        roads |= new;
        while new != 0 {
            let ends = adj.edge_endpoints[new.trailing_zeros() as usize] & !seen;
            new &= new - 1;
            seen |= ends;
            frontier |= ends;
        }
    }
    roads
}

fn dfs_road(
    node: usize,
    visited: u128,
    player_roads: u128,
    adj: &AdjacencyBitboards,
    opp_buildings: u64,
) -> u8 {
    if opp_buildings & (1u64 << node) != 0 {
        return 0;
    }
    let reachable = adj.node_adj_edges[node] & player_roads & !visited;
    let mut best = 0u8;
    let mut rem = reachable;
    while rem != 0 {
        let eid = rem.trailing_zeros() as usize;
        rem &= rem - 1;
        let next = (adj.edge_endpoints[eid] & !(1u64 << node)).trailing_zeros() as usize;
        best = best.max(
            dfs_road(
                next,
                visited | (1u128 << eid),
                player_roads,
                adj,
                opp_buildings,
            ) + 1,
        );
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::topology::Topology;

    fn make_topo() -> Topology {
        Topology::from_seed(42)
    }

    #[test]
    fn add_road_expands_frontier() {
        let topo = make_topo();
        let adj = &topo.adj;
        let mut net = RoadNetwork::new();

        // Place a building at node 0 to start the network
        net.add_building(NodeId(0), adj, 0);
        let initial = net.reachable_edges();
        assert!(initial != 0, "building should create frontier");

        // Pick the first reachable edge and add it as a road
        let eid = initial.trailing_zeros() as u8;
        net.add_road(EdgeId(eid), adj, 0, 0, false);

        // Frontier should now include edges at the new endpoint
        let after = net.reachable_edges();
        // The placed road's edge should be gone from frontier
        assert_eq!(after & (1u128 << eid), 0);
        // But we should have MORE reachable edges (expanded from far endpoint)
        assert!(
            after.count_ones() >= initial.count_ones(),
            "frontier should grow or stay same after road: before={}, after={}",
            initial.count_ones(),
            after.count_ones()
        );
    }

    #[test]
    fn opponent_building_shrinks_frontier() {
        let topo = make_topo();
        let adj = &topo.adj;
        let mut net = RoadNetwork::new();

        // Place building and road to create a frontier
        net.add_building(NodeId(0), adj, 0);
        let eid = net.reachable_edges().trailing_zeros() as u8;
        net.add_road(EdgeId(eid), adj, 0, 0, false);

        // Find the far endpoint of the road
        let far = (adj.edge_endpoints[eid as usize] & !(1u64 << 0)).trailing_zeros() as u8;

        let before = net.reachable_edges();
        // Edges at the far endpoint that are in our frontier
        let edges_at_far = adj.node_adj_edges[far as usize] & before;
        assert!(edges_at_far != 0, "should have frontier edges at far node");

        // Opponent builds at far endpoint — blocks transit through that node
        let my_buildings = 1u64 << 0; // our building at node 0
        let opp_buildings = 1u64 << far;
        net.on_opponent_build(NodeId(far), adj, my_buildings, opp_buildings);

        let after = net.reachable_edges();
        // Edges at far that went to nodes NOT connected to our network should be gone
        // The edge connecting to node 0 (our building) should remain
        assert!(
            after.count_ones() <= before.count_ones(),
            "frontier should shrink or stay same: before={}, after={}",
            before.count_ones(),
            after.count_ones()
        );
    }

    #[test]
    fn longest_road_cached_correctly() {
        let topo = make_topo();
        let adj = &topo.adj;
        let mut net = RoadNetwork::new();

        // Build a network from node 0
        net.add_building(NodeId(0), adj, 0);

        // Build a chain of 5 roads
        let mut current_node = 0usize;
        let mut visited_nodes = 1u64 << 0;
        for i in 0..5 {
            let candidates = adj.node_adj_edges[current_node] & net.reachable_edges();
            assert!(candidates != 0, "should have reachable edge for road {i}");
            let eid = candidates.trailing_zeros() as u8;
            let far = (adj.edge_endpoints[eid as usize] & !visited_nodes).trailing_zeros() as usize;
            net.add_road(EdgeId(eid), adj, 0, 0, true);
            visited_nodes |= 1u64 << far;
            current_node = far;
        }

        assert!(
            net.longest_road() >= 5,
            "5 roads in a chain should give longest >= 5, got {}",
            net.longest_road()
        );
    }

    #[test]
    fn add_road_skips_dfs_under_five() {
        let topo = make_topo();
        let adj = &topo.adj;
        let mut net = RoadNetwork::new();

        net.add_building(NodeId(0), adj, 0);

        // Build 4 roads with update_longest: true
        let mut current_node = 0usize;
        let mut visited_nodes = 1u64 << 0;
        for _ in 0..4 {
            let candidates = adj.node_adj_edges[current_node] & net.reachable_edges();
            let eid = candidates.trailing_zeros() as u8;
            let far = (adj.edge_endpoints[eid as usize] & !visited_nodes).trailing_zeros() as usize;
            net.add_road(EdgeId(eid), adj, 0, 0, true);
            visited_nodes |= 1u64 << far;
            current_node = far;
        }

        // With < 5 roads, DFS should not have run, longest stays 0
        assert_eq!(net.longest_road(), 0, "longest should be 0 with < 5 roads");
    }

    #[test]
    fn bridge_two_components_recomputes() {
        let topo = make_topo();
        let adj = &topo.adj;
        let mut net = RoadNetwork::new();

        // Build first component: 3 roads from node 0
        net.add_building(NodeId(0), adj, 0);
        let mut current = 0usize;
        let mut visited = 1u64 << 0;
        for _ in 0..3 {
            let candidates = adj.node_adj_edges[current] & net.reachable_edges();
            let eid = candidates.trailing_zeros() as u8;
            let far = (adj.edge_endpoints[eid as usize] & !visited).trailing_zeros() as usize;
            net.add_road(EdgeId(eid), adj, 0, 0, true);
            visited |= 1u64 << far;
            current = far;
        }
        let component1_end = current;

        // Build second component: place building at a disconnected node and build 3 roads
        // Find a node not adjacent to our network
        let mut start2 = None;
        for n in 0..54u8 {
            if visited & (1u64 << n) == 0 && adj.node_adj_nodes[n as usize] & visited == 0 {
                start2 = Some(n as usize);
                break;
            }
        }
        let start2 = start2.expect("should find disconnected node");
        net.add_building(NodeId(start2 as u8), adj, 0);
        current = start2;
        visited |= 1u64 << start2;
        for _ in 0..3 {
            let candidates = adj.node_adj_edges[current] & net.reachable_edges();
            if candidates == 0 {
                break;
            }
            let eid = candidates.trailing_zeros() as u8;
            let far_mask = adj.edge_endpoints[eid as usize] & !visited;
            if far_mask == 0 {
                break;
            }
            let far = far_mask.trailing_zeros() as usize;
            net.add_road(EdgeId(eid), adj, 0, 0, true);
            visited |= 1u64 << far;
            current = far;
        }

        let before = net.longest_road();

        // Try to find a bridging edge between component1_end and current network node
        // that would connect the two components
        let edges_at_end = adj.node_adj_edges[component1_end];
        let mut bridge_eid = None;
        let mut rem = edges_at_end & net.reachable_edges();
        while rem != 0 {
            let eid = rem.trailing_zeros() as u8;
            rem &= rem - 1;
            let far = (adj.edge_endpoints[eid as usize] & !(1u64 << component1_end))
                .trailing_zeros() as usize;
            // Check if far endpoint connects to the second component
            if adj.node_adj_edges[far] & net.roads != 0 {
                bridge_eid = Some(eid);
                break;
            }
        }

        if let Some(eid) = bridge_eid {
            net.add_road(EdgeId(eid), adj, 0, 0, true);
            // After bridging, longest should be >= before
            assert!(
                net.longest_road() >= before,
                "bridging should maintain or increase longest: before={}, after={}",
                before,
                net.longest_road()
            );
        }
        // If no bridge found, the test still validates the two-component case
    }

    #[test]
    fn leaf_extension_onto_opponent_building_counts() {
        let topo = make_topo();
        let adj = &topo.adj;
        let mut net = RoadNetwork::new();

        net.add_building(NodeId(0), adj, 0);

        // A chain of 5 roads whose last road ends on an opponent building.
        let mut current = 0usize;
        let mut visited = 1u64 << 0;
        let mut opp_buildings = 0u64;
        for i in 0..5 {
            let candidates = adj.node_adj_edges[current] & net.reachable_edges();
            let eid = candidates.trailing_zeros() as u8;
            let far = (adj.edge_endpoints[eid as usize] & !visited).trailing_zeros() as usize;
            if i == 4 {
                opp_buildings = 1u64 << far;
            }
            net.add_road(EdgeId(eid), adj, opp_buildings, 0, true);
            visited |= 1u64 << far;
            current = far;
        }

        assert_eq!(net.longest_road(), 5);
    }

    /// Four roads in a line and a fifth off a middle node: the line is the
    /// longest road, not the new road's tip plus one side.
    #[test]
    fn fifth_road_off_the_middle_keeps_the_line() {
        let topo = make_topo();
        let adj = &topo.adj;
        // A middle node with a third edge, and two roads out each way.
        let (middle, spur) = (0..54usize)
            .find_map(|n| {
                let edges = adj.node_adj_edges[n];
                (edges.count_ones() == 3).then_some((n, edges))
            })
            .expect("a node with three edges");
        let mut line = 0u128;
        let mut ends = Vec::new();
        let mut rem = spur;
        for _ in 0..2 {
            let eid = rem.trailing_zeros() as usize;
            rem &= rem - 1;
            line |= 1u128 << eid;
            ends.push((adj.edge_endpoints[eid] & !(1u64 << middle)).trailing_zeros() as usize);
        }
        let spur = rem.trailing_zeros() as usize;
        let mut seen = (1u64 << middle) | (1u64 << ends[0]) | (1u64 << ends[1]);
        for &end in &ends {
            let eid = (0..72usize)
                .find(|&e| {
                    let ep = adj.edge_endpoints[e];
                    ep & (1u64 << end) != 0 && ep & !seen != 0 && line & (1u128 << e) == 0
                })
                .expect("an edge out of each end");
            line |= 1u128 << eid;
            seen |= adj.edge_endpoints[eid];
        }
        assert_eq!(compute_longest_road(line, adj, 0), 4);
        // The spur's far end is new, so the old cache extended from its tip.
        let tip = adj.edge_endpoints[spur] & !(1u64 << middle);
        assert_eq!(tip & seen, 0, "the spur leads off the line");

        let mut net = RoadNetwork::new();
        let mut rem = line;
        while rem != 0 {
            net.add_road(EdgeId(rem.trailing_zeros() as u8), adj, 0, 0, true);
            rem &= rem - 1;
        }
        net.add_road(EdgeId(spur as u8), adj, 0, 0, true);
        assert_eq!(net.longest_road(), 4);
    }

    #[test]
    fn cycle_split_by_opponent_building_counts_whole_loop() {
        let topo = make_topo();
        let adj = &topo.adj;

        // The six roads around one hex, with an opponent building on a corner.
        let corners = adj.tile_nodes[9];
        let mut roads = 0u128;
        for eid in 0..72 {
            if adj.edge_endpoints[eid] & !corners == 0 {
                roads |= 1u128 << eid;
            }
        }
        assert_eq!(roads.count_ones(), 6);

        // Opposite the corner the loop walk starts from, so that start is
        // not on the opponent building.
        let opp_buildings = 1u64 << corners.trailing_zeros();
        assert_eq!(compute_longest_road(roads, adj, opp_buildings), 6);
        assert_eq!(
            compute_longest_road_nodes(roads, adj, opp_buildings),
            corners
        );
    }

    /// The longest trail by trying every start and every way on, for
    /// checking the pruned search.
    fn brute_longest(roads: u128, adj: &AdjacencyBitboards, opp_buildings: u64) -> u8 {
        fn walk(node: usize, used: u128, roads: u128, adj: &AdjacencyBitboards, opp: u64) -> u8 {
            let mut best = 0;
            let mut rem = adj.node_adj_edges[node] & roads & !used;
            while rem != 0 {
                let eid = rem.trailing_zeros() as usize;
                rem &= rem - 1;
                let next = (adj.edge_endpoints[eid] & !(1u64 << node)).trailing_zeros() as usize;
                let on = if opp & (1u64 << next) != 0 {
                    0
                } else {
                    walk(next, used | 1u128 << eid, roads, adj, opp)
                };
                best = best.max(1 + on);
            }
            best
        }
        (0..54)
            .map(|n| walk(n, 0, roads, adj, opp_buildings))
            .max()
            .unwrap_or(0)
    }

    #[test]
    fn a_loop_beside_another_component_counts() {
        // From a warm-up game where his engine saw this road win the game:
        // ten roads closing a loop, one apart, and the opponent's buildings
        // elsewhere. Starts came only from the lone road, so the loop was
        // never walked and the longest road read 1.
        let topo = make_topo();
        let adj = &topo.adj;
        let roads = [34, 36, 37, 38, 39, 41, 47, 56, 67, 69, 70]
            .iter()
            .fold(0u128, |m, &e| m | 1u128 << e);
        let opp = [7, 9, 13, 21].iter().fold(0u64, |m, &n| m | 1u64 << n);
        assert_eq!(compute_longest_road(roads, adj, opp), 10);
        assert_eq!(compute_longest_road_nodes(roads, adj, opp).count_ones(), 10);
    }

    #[test]
    fn pruned_search_matches_brute_force() {
        let topo = make_topo();
        let adj = &topo.adj;
        let mut rng = fastrand::Rng::with_seed(11);
        for _ in 0..20_000 {
            // A few random walks, so networks have branches, loops and
            // several components, plus scattered opponent buildings.
            let mut roads = 0u128;
            for _ in 0..rng.usize(1..4) {
                let mut node = rng.usize(..54);
                for _ in 0..rng.usize(1..9) {
                    let edges = adj.node_adj_edges[node];
                    let pick = rng.usize(..edges.count_ones() as usize);
                    let mut rem = edges;
                    for _ in 0..pick {
                        rem &= rem - 1;
                    }
                    let eid = rem.trailing_zeros() as usize;
                    roads |= 1u128 << eid;
                    node = (adj.edge_endpoints[eid] & !(1u64 << node)).trailing_zeros() as usize;
                }
            }
            let mut opp = 0u64;
            for _ in 0..rng.usize(0..4) {
                opp |= 1u64 << rng.usize(..54);
            }
            let want = brute_longest(roads, adj, opp);
            assert_eq!(
                compute_longest_road(roads, adj, opp),
                want,
                "roads {roads:#x} opp {opp:#x}"
            );
            let nodes = compute_longest_road_nodes(roads, adj, opp);
            assert!(want == 0 || nodes.count_ones() >= u32::from(want).min(2));
        }
    }
}
