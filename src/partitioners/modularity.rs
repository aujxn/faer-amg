use std::{
    collections::{HashMap, HashSet, VecDeque},
    sync::Arc,
    usize,
};

use faer::{col::AsColRef, linalg::matmul::dot::inner_prod, sparse::SparseRowMat, Col};
use petgraph::{
    algo::{bellman_ford, bridges, connected_components},
    graph::NodeIndex,
    visit::{EdgeRef, IntoNodeIdentifiers},
    Graph, Undirected,
};
use rayon::{
    iter::{
        IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
        IntoParallelRefMutIterator, ParallelIterator,
    },
    slice::ParallelSliceMut,
};

use crate::partitioners::{AggregateGraph, PartitionerCallback};

use super::{AdjacencyList, Partition};
use log::{info, trace, warn};

// TODO: probably should split greedy merge aggregation and 'improve' aggregation into two objects
#[derive(Clone)]
pub struct ModularityPartitioner {
    pub(crate) base_strength: AdjacencyList,
    pub(crate) strength: AdjacencyList,
    pub(crate) base_row_sums: Col<f64>,
    pub(crate) row_sums: Col<f64>,
    pub(crate) inverse_total: f64,
    pub(crate) coarsening_factor: f64,
    pub(crate) agg_size_penalty: f64,
    pub(crate) dist_penalty: f64,
    pub(crate) partition: Partition,
    pub(crate) agg_sizes: Vec<usize>,
    pub(crate) node_weights: Vec<usize>,
    pub(crate) callback: Option<PartitionerCallback>,
    pub(crate) distances: Option<Vec<f64>>,
    pub(crate) centers: Option<Vec<usize>>,
    pub(crate) bridges: HashSet<usize>,
    pub(crate) endpoints: HashSet<usize>,
}

impl ModularityPartitioner {
    pub(crate) fn new(
        strength: AdjacencyList,
        coarsening_factor: f64,
        agg_size_penalty: f64,
        dist_penalty: f64,
        callback: Option<PartitionerCallback>,
    ) -> Self {
        let n = strength.nodes.len();
        // TODO: this is lazy
        let ones = Col::ones(n);
        let row_sums: Vec<f64> = strength
            .nodes
            .par_iter()
            .enumerate()
            .map(|(i, neighbors)| {
                let mut sum = 0.0;
                for (j, w) in neighbors.iter() {
                    assert_ne!(*j, i);
                    sum += w;
                }
                sum
            })
            .collect();
        let mut row_sums = Col::from_iter(row_sums);

        let mut counter = 0;
        let mut total_bad = 0.0;
        let mut min = 0.0;

        for sum in row_sums.iter_mut() {
            if *sum < 0.0 {
                counter += 1;
                if *sum < min {
                    min = *sum;
                }
                total_bad += *sum;
                *sum = 0.0;
            }
        }
        if counter > 0 {
            warn!(
            "{} of {} rows had negative rowsums. Average negative: {:.1e}, worst negative: {:.1e}",
            counter,
            row_sums.nrows(),
            total_bad / (counter as f64),
            min
        );
        }

        let total = inner_prod(
            row_sums.transpose(),
            faer::Conj::No,
            ones.as_col_ref(),
            faer::Conj::No,
        );

        let inverse_total = total.recip();

        Self {
            base_strength: strength.clone(),
            strength,
            base_row_sums: row_sums.clone(),
            row_sums,
            inverse_total,
            coarsening_factor,
            agg_size_penalty,
            dist_penalty,
            partition: Partition::singleton(n),
            agg_sizes: vec![1; n],
            node_weights: vec![1; n],
            callback,
            distances: None,
            centers: None,
            bridges: HashSet::new(),
            endpoints: HashSet::new(),
        }
    }

    pub fn get_partition(&self) -> &Partition {
        &self.partition
    }

    pub fn max_and_min_weighted_aggs(&self) -> ((usize, usize), (usize, usize)) {
        let mut max_agg = 0;
        let mut min_agg = 0;
        let mut max_size = 0;
        let mut min_size = 0;

        for (agg_idx, agg) in self.partition.aggregates().iter().enumerate() {
            let agg_size = agg
                .iter()
                .copied()
                .map(|node_idx| self.node_weights[node_idx])
                .sum();
            if agg_size > max_size {
                max_size = agg_size;
                max_agg = agg_idx;
            }
            if agg_size < max_size {
                min_size = agg_size;
                min_agg = agg_idx;
            }
        }
        ((max_agg, max_size), (min_agg, min_size))
    }

    pub fn partition(&mut self, step_cf: f64) {
        while self.partition.cf() < step_cf {
            let (pairs, unmatched) = self.greedy_matching(step_cf);
            if pairs.len() == 0 {
                warn!("Greedy partitioner terminated because no more matches are possible. target cf: {:.2} achieved: {:.2}", self.coarsening_factor, self.partition.cf());
                break;
            }
            self.strength.pairwise_merge(&pairs, &unmatched);
            self.partition.pairwise_merge(&pairs, &unmatched);
            self.pairwise_merge_rowsums(&pairs, &unmatched);
            self.update_agg_sizes();
        }
    }

    pub fn update_agg_sizes(&mut self) {
        self.agg_sizes = self
            .partition
            .agg_to_node
            .par_iter()
            .map(|agg| {
                agg.iter()
                    .copied()
                    .map(|node_id| self.node_weights[node_id])
                    .sum()
            })
            .collect();
    }

    pub fn aggregate(&mut self) {
        self.partition.validate();
        self.base_strength.aggregate(&self.partition);
        self.strength = self.base_strength.clone();

        let new_nodes = self.partition.naggs();
        let mut new_node_weights = vec![0; new_nodes];
        for (agg_id, agg) in self.partition.aggregates().iter().enumerate() {
            for node_id in agg.iter().copied() {
                new_node_weights[agg_id] += self.node_weights[node_id];
            }
        }
        self.base_row_sums = self
            .base_strength
            .nodes
            .iter()
            .map(|neighborhood| neighborhood.iter().map(|(_, a_ij)| a_ij).sum::<f64>())
            .collect();
        self.partition = Partition::singleton(new_nodes);
        self.agg_sizes = new_node_weights.clone();
        self.node_weights = new_node_weights;
    }

    pub fn modularity(&self) -> f64 {
        self.partition
            .node_to_agg
            .par_iter()
            .copied()
            .enumerate()
            .map(|(node_i, agg_i)| {
                let mut agg_sum = 0.0;
                for (node_j, a_ij) in self.base_strength.nodes[node_i].iter().copied() {
                    let agg_j = self.partition.node_to_agg[node_j];
                    if agg_i == agg_j {
                        agg_sum += a_ij
                            - self.base_row_sums[node_i]
                                * self.base_row_sums[node_j]
                                * self.inverse_total;
                    }
                }
                agg_sum
            })
            .sum::<f64>()
            * self.inverse_total
    }

    pub fn total_agg_size_cost(&self) -> f64 {
        self.agg_sizes
            .par_iter()
            .copied()
            .map(|agg_size| {
                let diff = self.coarsening_factor - agg_size as f64;
                diff.powf(2.0) * self.agg_size_penalty
            })
            .sum::<f64>()
    }

    pub fn total_dist_cost(&self) -> f64 {
        //self.distances.iter().sum::<f64>() * self.dist_penalty
        if let Some(distances) = self.distances.as_deref() {
            distances.iter().map(|dist| dist.powf(2.)).sum::<f64>() * self.dist_penalty
        } else {
            let agg_graphs = self.agg_graphs();
            let center_data = self.centers_and_distances(&agg_graphs);
            center_data
                .iter()
                .map(|(_center, _endpoints, distances)| {
                    distances
                        .iter()
                        .map(|(_node_id, dist)| dist.powf(2.))
                        .sum::<f64>()
                })
                .sum::<f64>()
                * self.dist_penalty
        }
    }

    pub fn total_edge_cost(&self) -> f64 {
        let mut cost = 0.0;
        for (node_i, neighborhood) in self.strength.nodes.iter().enumerate() {
            let agg_i = self.partition.node_assignments()[node_i];
            for (node_j, a_ij) in neighborhood
                .iter()
                .filter(|node_j| node_j.0 > node_i)
                .copied()
            {
                if self.partition.node_assignments()[node_j] != agg_i {
                    cost += a_ij;
                    //a_ij - self.row_sums[node_i] * self.row_sums[node_j] * self.inverse_total;
                }
            }
        }
        cost
    }

    pub fn into_partition(self) -> Partition {
        self.partition
    }

    fn pairwise_merge_rowsums(&mut self, pairs: &Vec<(usize, usize)>, unmatched: &Vec<usize>) {
        // NOTE: could be par w/ unsafe
        let mut new_row_sums = Col::zeros(self.partition.naggs());
        for (new_idx, pair) in pairs.iter().copied().enumerate() {
            new_row_sums[new_idx] = self.row_sums[pair.0] + self.row_sums[pair.1];
        }
        for (new_idx, old_idx) in unmatched.iter().copied().enumerate() {
            new_row_sums[new_idx] = self.row_sums[old_idx];
        }
        self.row_sums = new_row_sums;
    }

    fn generate_modularity_triplets(&self) -> Vec<(usize, usize, f64)> {
        let iters: Vec<_> = self
            .strength
            .nodes
            .iter()
            .enumerate()
            .map(|(node_i, neighborhood)| {
                neighborhood
                    .iter()
                    .copied()
                    .filter_map(move |(node_j, strength_weight)| {
                        if node_i > node_j && strength_weight > 0.0 {
                            let mut modularity_weight = strength_weight
                                - self.inverse_total
                                    * self.row_sums[node_i]
                                    * self.row_sums[node_j];
                            let total_node_weight =
                                (self.agg_sizes[node_i] + self.agg_sizes[node_j]) as f64;
                            if total_node_weight > self.coarsening_factor {
                                modularity_weight -= self.agg_size_penalty
                                    * (total_node_weight - self.coarsening_factor).powf(2.0);
                            } else if total_node_weight < self.coarsening_factor {
                                modularity_weight += self.agg_size_penalty
                                    * (total_node_weight - self.coarsening_factor).powf(2.0);
                            }
                            Some((node_i, node_j, modularity_weight))
                            /*
                            if modularity_weight > 0.0 {
                                Some((node_i, node_j, modularity_weight))
                            } else {
                                None
                            }
                            */
                        } else {
                            None
                        }
                    })
            })
            .collect();
        iters.into_par_iter().flatten_iter().collect()
    }

    fn greedy_matching(&self, step_cf: f64) -> (Vec<(usize, usize)>, Vec<usize>) {
        let vertex_count = self.row_sums.nrows();
        let target_matches = ((vertex_count as f64 - (self.partition.nnodes() as f64 / step_cf))
            .ceil()) as usize
            + 1;

        let mut pairs: Vec<(usize, usize)> =
            Vec::with_capacity(vertex_count.min(target_matches) / 2);
        let mut unmatched: Vec<usize> = Vec::new();

        let mut wants_to_merge = self.generate_modularity_triplets();
        if wants_to_merge.is_empty() {
            return (pairs, unmatched);
        }
        wants_to_merge.par_sort_unstable_by(|(_, _, w1), (_, _, w2)| w1.partial_cmp(w2).unwrap());

        let mut alive = vec![true; vertex_count];

        // NOTE: is serial... could be par with change of algorithm to Luby's style local max.
        // but complexity is linear in `nnz` so not priority
        loop {
            match wants_to_merge.pop() {
                None => break,
                Some((i, j, _w)) => {
                    if alive[i] && alive[j] {
                        alive[i] = false;
                        alive[j] = false;
                        pairs.push((i, j));
                    }
                }
            }
            if pairs.len() > target_matches {
                break;
            }
        }

        unmatched = alive
            .into_par_iter()
            .enumerate()
            .filter(|(_, flag)| *flag)
            .map(|(i, _)| i)
            .collect();

        (pairs, unmatched)
    }

    fn delta_q(&self, node_i: usize, source_agg: Option<usize>, dest_agg: usize) -> f64 {
        let mut in_degree = 0.0;
        let mut out_degree = 0.0;
        let old_dst_size = self.agg_sizes[dest_agg] as f64;
        let new_dst_size = (self.agg_sizes[dest_agg] + self.node_weights[node_i]) as f64;
        let cf = self.coarsening_factor;
        let old_size_cost;
        let new_size_cost;
        let row_sums = &self.base_row_sums;
        let mut old_dist_cost = 0.0;
        let mut new_dist_cost = f64::MAX;

        match source_agg {
            Some(source_agg) => {
                assert_ne!(source_agg, dest_agg);
                for (node_j, strength_ij) in self.base_strength.nodes[node_i].iter().copied() {
                    let agg_j = self.partition.node_assignments()[node_j];
                    let degree = strength_ij;
                    //strength_ij - row_sums[node_i] * row_sums[node_j] * self.inverse_total;
                    if agg_j == source_agg {
                        in_degree += degree;
                    } else if agg_j == dest_agg {
                        out_degree += degree;
                        if let Some(distances) = self.distances.as_deref() {
                            let candidate_dist =
                                (distances[node_j] + strength_ij.recip()).powf(2.0);
                            if candidate_dist < new_dist_cost {
                                new_dist_cost = candidate_dist;
                            }
                        }
                    }
                }
                let old_src_size = self.agg_sizes[source_agg] as f64;
                let new_src_size = (self.agg_sizes[source_agg] - self.node_weights[node_i]) as f64;
                old_size_cost = (old_dst_size - cf).powf(2.0) + (old_src_size - cf).powf(2.0);
                new_size_cost = (new_dst_size - cf).powf(2.0) + (new_src_size - cf).powf(2.0);
                if let Some(distances) = self.distances.as_deref() {
                    old_dist_cost = distances[node_i].powf(2.0);
                }
            }
            None => {
                for (node_j, strength_ij) in self.base_strength.nodes[node_i].iter().copied() {
                    let agg_j = self.partition.node_assignments()[node_j];
                    if agg_j == dest_agg {
                        let degree = strength_ij;
                        //strength_ij - row_sums[node_i] * row_sums[node_j] * self.inverse_total;
                        out_degree += degree;

                        if let Some(distances) = self.distances.as_deref() {
                            let candidate_dist =
                                (distances[node_j] + strength_ij.recip()).powf(2.0);
                            if candidate_dist < new_dist_cost {
                                new_dist_cost = candidate_dist;
                            }
                        }
                    }
                }
                old_size_cost = (old_dst_size - cf).powf(2.0);
                new_size_cost = (new_dst_size - cf).powf(2.0);
            }
        }
        if self.distances.is_none() {
            new_dist_cost = 0.0;
        }
        let delta_degree = out_degree - in_degree;
        let delta_size = old_size_cost - new_size_cost;
        let delta_dist = old_dist_cost - new_dist_cost;
        delta_degree + self.agg_size_penalty * delta_size + self.dist_penalty * delta_dist
    }

    pub fn improve(&mut self, max_iter: usize) {
        //let mut update_center = false;
        for pass in 0..max_iter {
            let agg_graphs = self.agg_graphs();
            self.update_centers_and_distances(&agg_graphs);
            /*
            if update_center {
                self.update_centers_and_distances(&agg_graphs);
                //update_center = false;
            } else if self.centers.is_some() {
                self.update_distances(&agg_graphs);
            } else {
                self.update_bridges(&agg_graphs);
            }
            */

            /*
            let mut swaps: Vec<(usize, usize, f64)> = self
                .partition
                .node_to_agg
                .copied()
                .enumerate()
                .filter_map(|(node_i, agg_i)| {
            */
            let mut swaps: Vec<(usize, usize, f64)> = self
                .endpoints
                .par_iter()
                .copied()
                .filter_map(|node_i| {
                    let agg_i = self.partition.node_assignments()[node_i];
                    if self.agg_sizes[agg_i] == self.node_weights[node_i]
                    //|| self.bridges.contains(&node_i)
                    {
                        // If we are only member of aggregate we can't leave, we want number of aggs
                        // to stay the same... There are probably better approaches than this.
                        return None;
                    }
                    if let Some(centers) = self.centers.as_deref() {
                        if centers[agg_i] == node_i {
                            return None;
                        }
                    }
                    let neighborhood = &self.base_strength.nodes[node_i];
                    let connected_aggs: HashSet<usize> = neighborhood
                        .iter()
                        .copied()
                        .map(|(node_j, _)| self.partition.node_assignments()[node_j])
                        .filter(|agg_j| *agg_j != agg_i)
                        .collect();

                    connected_aggs
                        .iter()
                        .copied()
                        .map(|agg_j| (agg_j, self.delta_q(node_i, Some(agg_i), agg_j)))
                        .filter(|(_dst_agg, delta_q)| *delta_q > 0.0)
                        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                        .map(|(best_agg, delta_q)| (node_i, best_agg, delta_q))
                })
                .collect();

            if swaps.is_empty() && self.centers.is_none() {
                //update_center = true;
            } else if swaps.is_empty() && self.centers.is_some() {
                info!("Pass {} is local minimum with no swaps.", pass);
                break;
            }

            let mut alive_nodes = vec![true; self.partition.nnodes()];
            let mut alive_aggs = vec![true; self.partition.naggs()];
            swaps.par_sort_unstable_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

            let mut true_swaps = 0;
            for (node_id, new_agg, _) in swaps {
                let old_agg = self.partition.node_to_agg[node_id];
                if alive_nodes[node_id] && alive_aggs[new_agg] && alive_aggs[old_agg] {
                    //if alive_nodes[node_id] {
                    assert_ne!(new_agg, old_agg);
                    self.partition.node_to_agg[node_id] = new_agg;
                    assert!(self.agg_sizes[old_agg] > self.node_weights[node_id]);
                    self.agg_sizes[old_agg] -= self.node_weights[node_id];
                    self.agg_sizes[new_agg] += self.node_weights[node_id];
                    let result = self.partition.agg_to_node[old_agg].remove(&node_id);
                    assert!(result);
                    self.partition.agg_to_node[new_agg].insert(node_id);
                    true_swaps += 1;

                    alive_aggs[new_agg] = false;
                    alive_aggs[old_agg] = false;
                    alive_nodes[node_id] = false;
                    for (neighbor, _) in self.base_strength.nodes[node_id].iter().copied() {
                        alive_nodes[neighbor] = false;
                        alive_aggs[self.partition.node_assignments()[neighbor]] = false;
                    }
                }
            }
            // invalidate the centers and distances
            /*
            if true_swaps > 1 {
                self.centers = None;
                self.distances = None;
            }
            */
            if let Some(callback) = self.callback.as_ref() {
                callback(pass, &self);
            }
            info!("swaps: {true_swaps}");
        }
    }

    /*
    pub fn fix_disconnected(&mut self) {
        let strength = &self.base_strength;

        let mut isolated: HashSet<usize> = self
            .partition
            .aggregates()
            .par_iter()
            .enumerate()
            .zip(self.agg_sizes.par_iter_mut())
            .map(|((agg_id, agg), agg_size)| {
                let mut local_graph: Graph<usize, f64, Undirected> = Graph::new_undirected();
                let mut node_indices: HashMap<usize, usize> = HashMap::new();
                for node_id in agg.iter().copied() {
                    let idx = local_graph.add_node(1).index();
                    node_indices.insert(node_id, idx);
                }
                for node_i in agg.iter().copied() {
                    let idx_i = *node_indices.get(&node_i).unwrap();
                    for (node_j, a_ij) in strength.nodes[node_i].iter().copied() {
                        if node_i > node_j {
                            if agg_id == self.partition.node_assignments()[node_j] {
                                let idx_j = *node_indices.get(&node_j).unwrap();
                                local_graph.add_edge(
                                    NodeIndex::new(idx_i),
                                    NodeIndex::new(idx_j),
                                    a_ij,
                                );
                            }
                        }
                    }
                }

                if connected_components(&local_graph) > 1 {
                    let largest_subgraph = find_largest_connected_component(&local_graph);
                    let local_isolated: Vec<usize> = node_indices
                        .iter()
                        .filter(|(_node_id, local_id)| !largest_subgraph.contains(&local_id))
                        .map(|(node_id, _local_id)| *node_id)
                        .collect();
                    for node_id in local_isolated.iter().copied() {
                        let node_weight = self.node_weights[node_id];
                        assert!(node_weight < *agg_size);
                        *agg_size -= node_weight;
                    }
                    local_isolated
                } else {
                    Vec::new()
                }
            })
            .flatten()
            .collect();

        for node_idx in isolated.iter() {
            let agg = self.partition.node_assignments()[*node_idx];
            let result = self.partition.agg_to_node[agg].remove(node_idx);
            assert!(result);
        }

        if isolated.len() > 0 {
            warn!(
                "disconnected aggregate detected! isolated {} nodes.",
                isolated.len()
            );
        }

        let mut i = 0;
        let mut queue: VecDeque<usize> = isolated.iter().copied().collect();
        loop {
            i += 1;
            let next = queue.pop_front();
            match next {
                None => break,
                Some(node_i) => {
                    if let Some((new_agg, strength, node_j)) =
                        self.get_max_out_agg(node_i, &isolated)
                    {
                        self.partition.node_to_agg[node_i] = new_agg;
                        self.partition.agg_to_node[new_agg].insert(node_i);
                        self.agg_sizes[new_agg] += self.node_weights[node_i];
                        self.distances[node_i] = self.distances[node_j] + strength.recip();
                        isolated.remove(&node_i);
                    } else {
                        queue.push_back(node_i);
                    }
                }
            }
            if i > 10000 {
                info!("isolated: {:?}", isolated);
                for node_id in isolated {
                    print!("\nNode {}: ", node_id);
                    for neighbor in self.base_strength.nodes[node_id].iter() {
                        print!("({}, {:.2}), ", neighbor.0, neighbor.1);
                    }
                    // TODO: validate that strength graph is fully connected.........
                }
                panic!("something broke");
            }
        }
    }

    fn get_max_out_agg(
        &self,
        node_i: usize,
        isolated: &HashSet<usize>,
    ) -> Option<(usize, f64, usize)> {
        let neighborhood = &self.base_strength.nodes[node_i];

        neighborhood
            .iter()
            .filter(|(node_j, _)| !isolated.contains(node_j))
            .copied()
            .map(|(node_j, strength)| {
                let agg_j = self.partition.node_to_agg[node_j];
                (agg_j, self.delta_q(node_i, None, agg_j), strength, node_j)
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(agg_j, _, strength, node_j)| (agg_j, strength, node_j))
    }
    */

    fn agg_graphs(&self) -> Vec<AggregateGraph> {
        self.partition
            .aggregates()
            .par_iter()
            .map(|agg| self.base_strength.subgraph(agg))
            .collect()
    }

    fn update_bridges(&mut self, agg_graphs: &Vec<AggregateGraph>) {
        self.bridges = agg_graphs
            .par_iter()
            .zip(self.partition.aggregates().par_iter())
            .map(|(agg_graph, agg)| {
                let local_to_global_node_idx: Vec<usize> = agg.iter().copied().collect();
                let bridges = bridges(&agg_graph.graph);
                let mut bridges_set = HashSet::new();
                for bridge in bridges {
                    let (local_i, local_j) = agg_graph.graph.edge_endpoints(bridge.id()).unwrap();
                    if agg_graph.graph.neighbors(local_i).count() > 1 {
                        bridges_set.insert(local_to_global_node_idx[local_i.index()]);
                    }
                    if agg_graph.graph.neighbors(local_j).count() > 1 {
                        bridges_set.insert(local_to_global_node_idx[local_j.index()]);
                    }
                }
                bridges_set
            })
            .reduce(|| HashSet::new(), |a, b| a.union(&b).copied().collect());
    }

    fn centers_and_distances(
        &self,
        agg_graphs: &Vec<AggregateGraph>,
    ) -> Vec<(usize, HashSet<usize>, Vec<(usize, f64)>)> {
        agg_graphs
            .par_iter()
            .zip(self.partition.aggregates().par_iter())
            .map(|(agg_graph, agg)| {
                let mut min_ecc = f64::MAX;
                let mut center_id = *agg.first().expect("empty aggregate has no center");
                let mut best_paths = None;
                for (node_id, local_idx) in agg_graph.index_map.iter() {
                    let paths = bellman_ford(&agg_graph.graph, *local_idx).expect("negative cycle");
                    let eccentricity = *paths
                        .distances
                        .iter()
                        .max_by(|a, b| a.partial_cmp(b).expect("compared nan"))
                        .unwrap();
                    if eccentricity < min_ecc {
                        center_id = *node_id;
                        min_ecc = eccentricity;
                        best_paths = Some(paths);
                    }
                }
                let paths = best_paths.unwrap();

                let local_to_global: Vec<usize> = agg.iter().copied().collect();
                let mut endpoints: HashSet<usize> = agg.iter().copied().collect();
                for node_idx in paths.predecessors.iter().filter_map(|maybe_local_idx| {
                    maybe_local_idx.map(|local_idx| local_to_global[local_idx.index()])
                }) {
                    endpoints.remove(&node_idx);
                }

                (
                    center_id,
                    endpoints,
                    agg.iter()
                        .cloned()
                        .zip(paths.distances.into_iter())
                        .collect::<Vec<(usize, f64)>>(),
                )
            })
            .collect()
    }

    fn update_centers_and_distances(&mut self, agg_graphs: &Vec<AggregateGraph>) {
        let center_data: Vec<(usize, HashSet<usize>, Vec<(usize, f64)>)> =
            self.centers_and_distances(agg_graphs);

        let mut all_centers = vec![];
        let mut all_distances = vec![0.0; self.base_strength.nodes.len()];
        let mut all_endpoints: HashSet<usize> = HashSet::new();
        for (center_id, endpoints, distances) in center_data {
            all_centers.push(center_id);
            for (node_id, dist) in distances {
                all_distances[node_id] = dist;
            }
            all_endpoints.extend(endpoints.iter());
        }
        self.centers = Some(all_centers);
        self.distances = Some(all_distances);
        self.endpoints = all_endpoints;
    }

    fn update_distances(&mut self, agg_graphs: &Vec<AggregateGraph>) {
        assert!(self.centers.is_some());
        let distances: Vec<(usize, f64)> = agg_graphs
            .par_iter()
            .zip(self.partition.aggregates().par_iter())
            .zip(self.centers.as_deref().unwrap().par_iter())
            .map(|((agg_graph, agg), center)| {
                let local_center = agg_graph.index_map.get(center).unwrap();
                let distances = bellman_ford(&agg_graph.graph, *local_center)
                    .expect("negative cycle")
                    .distances;
                agg.iter().cloned().zip(distances.into_iter())
            })
            .flatten_iter()
            .collect();
        let mut all_distances = vec![0.0; self.base_strength.nodes.len()];
        for (node_id, dist) in distances {
            all_distances[node_id] = dist;
        }
        self.distances = Some(all_distances);
    }
}

/*
fn find_largest_connected_component<N, E>(graph: &Graph<N, E, Undirected>) -> HashSet<usize> {
    let mut visited = HashSet::new();
    let mut largest_component: Option<HashSet<usize>> = None;
    let mut max_size = 0;

    for node_idx in graph.node_identifiers() {
        if !visited.contains(&node_idx) {
            let mut current_component = HashSet::new();
            let mut queue = VecDeque::new();

            queue.push_back(node_idx);
            visited.insert(node_idx);

            while let Some(current_node) = queue.pop_front() {
                current_component.insert(current_node.index());
                for neighbor in graph.neighbors(current_node) {
                    if !visited.contains(&neighbor) {
                        visited.insert(neighbor);
                        queue.push_back(neighbor);
                    }
                }
            }

            if current_component.len() > max_size {
                max_size = current_component.len();
                largest_component = Some(current_component);
            }
        }
    }
    largest_component.expect("empty graph provided to find largest component")
}
*/
