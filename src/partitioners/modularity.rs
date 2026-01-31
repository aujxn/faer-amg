use super::{AdjacencyList, Partition};
use crate::partitioners::PartitionerConfig;
use faer::{col::AsColRef, linalg::matmul::dot::inner_prod, Col};
use log::{trace, warn};
use rayon::{
    iter::{
        IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
    },
    slice::ParallelSliceMut,
};
use std::{collections::HashSet, usize};

// TODO: probably should split greedy merge aggregation and 'improve' aggregation into two objects
#[derive(Clone)]
pub struct Partitioner {
    config: PartitionerConfig,
    base_strength: AdjacencyList,
    strength: AdjacencyList,
    base_row_sums: Col<f64>,
    row_sums: Col<f64>,
    inverse_total: f64,
    partition: Partition,
    agg_sizes: Vec<usize>,
    node_weights: Vec<usize>,
}

impl Partitioner {
    pub(super) fn new(
        strength: AdjacencyList,
        starting_partition: Option<Partition>,
        node_weights: Option<Vec<usize>>,
        config: PartitionerConfig,
    ) -> Self {
        let n = strength.nodes.len();
        // TODO: this is lazy
        let ones = Col::ones(n);
        let base_row_sums: Vec<f64> = strength
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
        let mut base_row_sums = Col::from_iter(base_row_sums);

        let mut counter = 0;
        let mut total_bad = 0.0;
        let mut min = 0.0;

        for sum in base_row_sums.iter_mut() {
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
            base_row_sums.nrows(),
            total_bad / (counter as f64),
            min
        );
        }

        let total = inner_prod(
            base_row_sums.transpose(),
            faer::Conj::No,
            ones.as_col_ref(),
            faer::Conj::No,
        );

        let inverse_total = total.recip();
        let node_weights = node_weights.unwrap_or(vec![1; n]);

        let partition;
        let agg_sizes;
        let row_sums;

        match starting_partition {
            Some(starting_partition) => {
                partition = starting_partition;
                agg_sizes = partition
                    .agg_to_node
                    .par_iter()
                    .map(|agg| {
                        agg.iter()
                            .copied()
                            .map(|node_id| node_weights[node_id])
                            .sum()
                    })
                    .collect();

                let row_sums_vec: Vec<f64> = partition
                    .agg_to_node
                    .par_iter()
                    .map(|agg| {
                        agg.iter()
                            .copied()
                            .map(|node_id| base_row_sums[node_id])
                            .sum::<f64>()
                    })
                    .collect();
                row_sums = Col::from_iter(row_sums_vec);
            }
            None => {
                partition = Partition::singleton(n);
                agg_sizes = node_weights.clone();
                row_sums = base_row_sums.clone();
            }
        }

        Self {
            config,
            base_strength: strength.clone(),
            strength,
            base_row_sums: row_sums.clone(),
            row_sums,
            inverse_total,
            partition,
            agg_sizes,
            node_weights,
        }
    }

    pub fn rebase(&mut self, base_partition: Partition) {
        self.partition = base_partition;
        self.aggregate();
        self.initialize_partition();
        self.improve_partition();
    }

    pub(crate) fn get_base_strength(&self) -> &AdjacencyList {
        &self.base_strength
    }

    pub(crate) fn get_agg_sizes(&self) -> &Vec<usize> {
        &self.agg_sizes
    }

    pub fn get_partition(&self) -> &Partition {
        &self.partition
    }

    pub fn max_and_min_weighted_aggs(&self) -> ((usize, usize), (usize, usize)) {
        let mut max_agg = 0;
        let mut min_agg = 0;
        let mut max_size = 0;
        let mut min_size = usize::MAX;

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
            if agg_size < min_size {
                min_size = agg_size;
                min_agg = agg_idx;
            }
        }
        ((max_agg, max_size), (min_agg, min_size))
    }

    pub fn initialize_partition(&mut self) {
        let cf = self.config.coarsening_factor;
        while self.partition.cf() < cf {
            let (pairs, unmatched) = self.greedy_matching(cf);
            if pairs.len() == 0 {
                warn!("Greedy partitioner terminated because no more matches are possible. target cf: {:.2} achieved: {:.2}", cf, self.partition.cf());
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

    fn aggregate(&mut self) {
        self.partition.validate();
        self.base_strength.aggregate(&self.partition);
        self.strength = self.base_strength.clone();

        let new_nodes = self.partition.naggs();
        /*
        let mut new_node_weights = vec![0; new_nodes];
        for (agg_id, agg) in self.partition.aggregates().iter().enumerate() {
            for node_id in agg.iter().copied() {
                new_node_weights[agg_id] += self.node_weights[node_id];
            }
        }
        */
        let new_node_weights = vec![1; new_nodes];
        self.base_row_sums = self
            .base_strength
            .nodes
            .iter()
            .map(|neighborhood| neighborhood.iter().map(|(_, a_ij)| a_ij).sum::<f64>())
            .collect();
        self.row_sums = self.base_row_sums.clone();
        self.partition = Partition::singleton(new_nodes);
        self.agg_sizes = new_node_weights.clone();
        self.node_weights = new_node_weights;
        self.inverse_total = self.base_row_sums.sum().recip();
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
            .map(|agg_size| self.size_cost(agg_size))
            .sum::<f64>()
    }

    pub fn total_edge_cost(&self) -> f64 {
        let mut cost = 0.0;
        for (node_i, neighborhood) in self.base_strength.nodes.iter().enumerate() {
            let agg_i = self.partition.node_assignments()[node_i];
            //print!("node {} in agg {}: ", node_i, agg_i);
            for (node_j, a_ij) in neighborhood
                .iter()
                .filter(|node_j| node_j.0 > node_i)
                .copied()
            {
                let agg_j = self.partition.node_assignments()[node_j];
                //print!("({}, {}, {:.2e})  ", node_j, agg_j, a_ij);
                if agg_j != agg_i {
                    cost += a_ij;
                    //a_ij - self.row_sums[node_i] * self.row_sums[node_j] * self.inverse_total;
                }
            }
            //println!();
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
        let cf = self.config.coarsening_factor;
        let agg_pen = self.config.agg_size_penalty;
        let agg_sizes = &self.agg_sizes;
        let row_sums = &self.row_sums;
        let inv_t = self.inverse_total;

        let iters: Vec<_> = self
            .strength
            .nodes
            .iter()
            .enumerate()
            .map(|(node_i, neighborhood)| {
                neighborhood
                    .iter()
                    .filter(move |(node_j, _strength)| node_i > *node_j)
                    .copied()
                    .map(move |(node_j, strength)| {
                        let expected = inv_t * row_sums[node_i] * row_sums[node_j];
                        let mut modularity_weight = strength - expected;
                        let new_node_weight = (agg_sizes[node_i] + agg_sizes[node_j]) as f64;
                        let square_diff = (new_node_weight - cf).powf(2.0);
                        if new_node_weight > cf {
                            modularity_weight -= agg_pen * square_diff;
                        } else {
                            modularity_weight += agg_pen * square_diff;
                        }
                        (node_i, node_j, modularity_weight)
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

    fn size_cost(&self, size: usize) -> f64 {
        let cf = self.config.coarsening_factor;
        let relative_diff = (size as f64 - cf).abs() / cf;
        (4.0 * relative_diff).powf(4.) * self.config.agg_size_penalty
    }

    fn delta_q(&self, node_i: usize, source_agg: Option<usize>, dest_agg: usize) -> f64 {
        let mut in_degree = 0.0;
        let mut out_degree = 0.0;
        let old_dst_size = self.agg_sizes[dest_agg];
        let new_dst_size = self.agg_sizes[dest_agg] + self.node_weights[node_i];
        let old_size_cost;
        let new_size_cost;

        match source_agg {
            Some(source_agg) => {
                assert_ne!(source_agg, dest_agg);
                for (node_j, strength_ij) in self.base_strength.nodes[node_i].iter().copied() {
                    let agg_j = self.partition.node_assignments()[node_j];
                    let degree = strength_ij;
                    if agg_j == source_agg {
                        in_degree += degree;
                    } else if agg_j == dest_agg {
                        out_degree += degree;
                    }
                }
                let old_src_size = self.agg_sizes[source_agg];
                let new_src_size = self.agg_sizes[source_agg] - self.node_weights[node_i];
                old_size_cost = self.size_cost(old_dst_size) + self.size_cost(old_src_size);
                new_size_cost = self.size_cost(new_dst_size) + self.size_cost(new_src_size);
            }
            None => {
                for (node_j, strength_ij) in self.base_strength.nodes[node_i].iter().copied() {
                    let agg_j = self.partition.node_assignments()[node_j];
                    if agg_j == dest_agg {
                        let degree = strength_ij;
                        //strength_ij - row_sums[node_i] * row_sums[node_j] * self.inverse_total;
                        out_degree += degree;
                    }
                }
                old_size_cost = self.size_cost(old_dst_size);
                new_size_cost = self.size_cost(new_dst_size);
            }
        }
        let delta_degree = out_degree - in_degree;
        let delta_size = old_size_cost - new_size_cost;

        let agg_pen = self.config.agg_size_penalty;

        delta_degree + agg_pen * delta_size
    }

    pub fn improve_partition(&mut self) {
        let max_iter = self.config.max_improvement_iters;
        for pass in 0..max_iter {
            // NOTE: This can break aggregates into disconnected components... not great
            let mut swaps: Vec<(usize, usize, f64)> = self
                .partition
                .node_to_agg
                .par_iter()
                .copied()
                .enumerate()
                .filter_map(|(node_i, agg_i)| {
                    if self.agg_sizes[agg_i] == self.node_weights[node_i] {
                        // If we are only member of aggregate we can't leave, we want number of aggs
                        // to stay the same... There are probably better approaches than this.
                        return None;
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
            if swaps.is_empty() {
                if let Some(callback) = self.config.callback.as_ref() {
                    callback.call(pass, &self);
                }
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
            if let Some(callback) = self.config.callback.as_ref() {
                callback.call(pass, &self);
            }
            trace!("swaps: {true_swaps}");
        }
    }
}
