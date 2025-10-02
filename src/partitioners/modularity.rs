use std::{collections::HashMap, sync::Arc};

use faer::{col::AsColRef, linalg::matmul::dot::inner_prod, sparse::SparseRowMat, Col};
use rayon::{
    iter::{
        IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
    },
    slice::ParallelSliceMut,
};

use super::{AdjacencyList, Partition};
use log::{info, trace, warn};

enum AggSize {
    TooBig,
    TooSmall,
}

pub struct ModularityPartitioner {
    pub base_strength: AdjacencyList,
    strength: AdjacencyList,
    pub base_row_sums: Col<f64>,
    row_sums: Col<f64>,
    pub inverse_total: f64,
    pub coarsening_factor: f64,
    pub agg_size_penalty: f64,
    pub partition: Partition,
}

impl ModularityPartitioner {
    pub fn new(
        mat: Arc<SparseRowMat<usize, f64>>,
        strength: AdjacencyList,
        coarsening_factor: f64,
        agg_size_penalty: f64,
    ) -> Self {
        // TODO: this is lazy
        let ones = Col::ones(mat.nrows());
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
            partition: Partition::singleton(mat.clone()),
        }
    }

    pub fn partition(&mut self) {
        while self.partition.cf() < self.coarsening_factor {
            let (pairs, unmatched) = self.greedy_matching();
            if pairs.len() == 0 {
                warn!("Greedy partitioner terminated because no more matches are possible. target cf: {:.2} achieved: {:.2}", self.coarsening_factor, self.partition.cf());
                break;
            }
            self.strength.pairwise_merge(&pairs, &unmatched);
            self.partition.pairwise_merge(&pairs, &unmatched);
            self.pairwise_merge_rowsums(&pairs, &unmatched);
        }
    }

    pub fn modularity(&self) -> f64 {
        self.partition
            .agg_to_node
            .par_iter()
            .map(|agg| {
                let mut agg_sum = 0.0;
                for i in agg.iter().copied() {
                    for j in agg.iter().copied() {
                        // NOTE: could optimize on assumption of symmetry and no diagonal
                        agg_sum += self.base_strength.get(i, j)
                            - self.base_row_sums[i] * self.base_row_sums[j] * self.inverse_total;
                    }
                }
                agg_sum
            })
            .sum::<f64>()
            * self.inverse_total
    }

    pub fn total_agg_size_cost(&self) -> f64 {
        (0..self.partition.agg_to_node.len())
            .into_par_iter()
            .map(|agg_id| self.agg_size_cost(agg_id).0)
            .sum::<f64>()
    }

    fn agg_size_cost(&self, agg_id: usize) -> (f64, AggSize) {
        let agg_size = self.partition.agg_to_node[agg_id].len() as f64;
        let diff = self.coarsening_factor - agg_size;
        let cost = diff.powf(2.0) * self.agg_size_penalty;
        //let cost = diff.abs() * self.agg_size_penalty;
        if agg_size > self.coarsening_factor {
            (cost, AggSize::TooBig)
        } else {
            (cost, AggSize::TooSmall)
        }
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
                            let modularity_weight = strength_weight
                                - self.inverse_total
                                    * self.row_sums[node_i]
                                    * self.row_sums[node_j];
                            if modularity_weight > 0.0 {
                                Some((node_i, node_j, modularity_weight))
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    })
            })
            .collect();
        iters.into_par_iter().flatten_iter().collect()
    }

    fn greedy_matching(&self) -> (Vec<(usize, usize)>, Vec<usize>) {
        let vertex_count = self.row_sums.nrows();
        let target_matches = ((vertex_count as f64
            - (self.partition.nnodes() as f64 / self.coarsening_factor))
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

    pub fn improve(&mut self, max_iter: usize) {
        let strength = &self.base_strength;
        let row_sums = &self.base_row_sums;
        let inverse_total = self.inverse_total;
        //let dbg_iters = (max_iter / 20).max(1);
        let dbg_iters = 1;
        for pass in 0..max_iter {
            let mut swaps: Vec<(usize, usize, f64)> = self
                .partition
                .node_to_agg
                .par_iter()
                .copied()
                .enumerate()
                .filter_map(|(node_i, agg_id)| {
                    let mut out_deg: HashMap<usize, f64> = HashMap::new();
                    let agg = &self.partition.agg_to_node[agg_id];
                    let neighborhood = &strength.nodes[node_i];

                    let (cost, size) = self.agg_size_cost(agg_id);
                    let in_size_obj = match size {
                        AggSize::TooBig => -cost,
                        AggSize::TooSmall => cost
                    };
                    let in_deg: f64 = agg
                        .iter()
                        .copied()
                        .map(|node_j| {
                            let a_ij = strength.get(node_i, node_j);
                            a_ij - row_sums[node_i] * row_sums[node_j] * inverse_total
                        })
                        .sum::<f64>()
                        + in_size_obj;

                    for node_j in neighborhood.iter().map(|(j, _)| *j) {
                        if !agg.contains(&node_j) {
                            let agg_j = self.partition.node_to_agg[node_j];
                            assert_ne!(
                                agg_j, agg_id,
                                "Node {} from agg {} is connected to node {} from agg {}.",
                                node_i, agg_id, node_j, agg_j
                            );
                            if out_deg.get(&agg_j).is_none() {

                                let (cost, size) = self.agg_size_cost(agg_j);
                                let out_size_obj = match size {
                                    AggSize::TooBig => -cost,
                                    AggSize::TooSmall => cost
                                };
                                let deg: f64 = self.partition.agg_to_node[agg_j]
                                    .iter()
                                    .copied()
                                    .map(|node_j| {
                                        let a_ij = strength.get(node_i, node_j);
                                        a_ij - row_sums[node_i] * row_sums[node_j] * inverse_total
                                    })
                                    .sum::<f64>()
                                    + out_size_obj;
                                out_deg.insert(agg_j, deg);
                            }
                        }
                    }

                    let max_out = out_deg
                        .into_iter()
                        .max_by(|a, b| {
                            a.1.partial_cmp(&b.1)
                                .expect(&format!("can't compare {} and {}", a.1, b.1))
                        })
                        .map(|(new_agg, deg)| (new_agg, deg));

                    match max_out {
                        Some((new_agg, max_deg)) => trace!("node {} is part of agg {} with {} nodes and {:.2} in-degree and {:.2} max out-degree to join agg {} with {} nodes", node_i, agg_id, agg.len(), in_deg, max_deg, new_agg, self.partition.node_assignments()[new_agg]),
                        None =>trace!("node {} is part of agg {} with {} nodes and {:.2} in-degree and no options to leave!", node_i, agg_id, agg.len(), in_deg),
                    }
                    max_out.map(|(new_agg, max_deg)| (node_i, new_agg, max_deg - in_deg))
                })
                .collect();

            let mut alive_nodes = vec![true; self.partition.nnodes()];
            let mut alive_aggs = vec![true; self.partition.naggs()];
            swaps.par_sort_unstable_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

            if swaps.is_empty() {
                trace!("Pass {} is local minimum with no swaps.", pass);
                break;
            }
            let mut true_swaps = 0;
            for (node_id, new_agg, _) in swaps {
                let old_agg = self.partition.node_to_agg[node_id];
                if alive_nodes[node_id] && alive_aggs[new_agg] && alive_aggs[old_agg] {
                    //if alive_nodes[node_id] {
                    assert_ne!(new_agg, old_agg);
                    self.partition.node_to_agg[node_id] = new_agg;
                    let result = self.partition.agg_to_node[old_agg].remove(&node_id);
                    assert!(result);
                    self.partition.agg_to_node[new_agg].insert(node_id);
                    true_swaps += 1;

                    alive_aggs[new_agg] = false;
                    alive_nodes[node_id] = false;
                    for (neighbor, _) in strength.nodes[node_id].iter().copied() {
                        alive_nodes[neighbor] = false;
                        alive_aggs[self.partition.node_assignments()[neighbor]] = false;
                    }
                }
            }
            let old_agg_count = self.partition.agg_to_node.len();
            self.partition.agg_to_node.retain(|agg| !agg.is_empty());
            let new_agg_count = self.partition.agg_to_node.len();

            if old_agg_count > new_agg_count {
                for (agg_id, agg) in self.partition.agg_to_node.iter().enumerate() {
                    for node_id in agg.iter().copied() {
                        self.partition.node_to_agg[node_id] = agg_id;
                    }
                }
            }
            if pass % dbg_iters == 0 {
                info!(
                    "Pass {}:\n\t{} swaps\n\t{:.4} modularity\n\t{:.4} size penalty",
                    pass,
                    true_swaps,
                    self.modularity(),
                    self.total_agg_size_cost()
                );
                self.partition.info();
            }
            #[cfg(debug_assertions)]
            self.partition.validate();
        }
    }
}
