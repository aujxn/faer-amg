use std::{collections::HashMap, sync::Arc, usize};

use faer::{col::AsColRef, linalg::matmul::dot::inner_prod, sparse::SparseRowMat, Col};
use rayon::{
    iter::{
        IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
    },
    slice::ParallelSliceMut,
};

use super::{AdjacencyList, Partition};
use log::{info, trace, warn};

pub struct ModularityPartitioner {
    pub(crate) base_strength: AdjacencyList,
    strength: AdjacencyList,
    pub base_row_sums: Col<f64>,
    row_sums: Col<f64>,
    pub inverse_total: f64,
    pub coarsening_factor: f64,
    pub agg_size_penalty: f64,
    pub partition: Partition,
    pub agg_sizes: Vec<usize>,
}

impl ModularityPartitioner {
    pub(crate) fn new(
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
            agg_sizes: vec![1; mat.nrows()],
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
        self.agg_sizes = self.partition.agg_to_node.iter().map(|agg| agg.len()).collect();
    }

    pub fn update_agg_sizes(&mut self) {

        self.agg_sizes = vec![0; self.partition.aggregates().len()];
        for (_, agg_id) in self.partition.node_to_agg.iter().copied().enumerate() {
            self.agg_sizes[agg_id] += 1;
        }
        /*
        self.agg_sizes = self
            .partition
            .aggregates()
            .iter()
            .map(|agg| agg.len())
            .collect();
        */
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
                    let agg_j =  self.partition.node_to_agg[node_j];
                    if agg_i == agg_j {
                        agg_sum += a_ij - self.base_row_sums[node_i] * self.base_row_sums[node_j] * self.inverse_total;
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

    fn move_size_penalty(&self, src_node: usize, dst_node: usize) -> f64 {
        let src_agg = self.partition.node_to_agg[src_node];
        let dst_agg = self.partition.node_to_agg[dst_node];
        let delta: f64 = self.agg_sizes[src_agg] as f64 - self.agg_sizes[dst_agg] as f64;
        if delta > 0.0 {
            delta.powf(2.0) * self.agg_size_penalty
        } else {
            -delta.powf(2.0) * self.agg_size_penalty
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
        /* Pairwise merge results in valid partition object so don't need this unless
        * that algorithm is simplified
        }
        */
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
                    if self.agg_sizes[agg_id] == 1 {
                        // If we are only member of aggregate we can't leave, we want number of aggs
                        // to stay the same... There are probably better approaches than this.
                        return None;
                    }
                    let mut out_deg: HashMap<usize, f64> = HashMap::new();
                    let neighborhood = &strength.nodes[node_i];

                    let mut in_deg: f64 = 0.0;

                    for (node_j, a_ij) in neighborhood
                        .iter()
                        .copied() 
                    {
                        assert_ne!(node_i, node_j, "Diagonal entries in strength matrix...");
                        if self.partition.node_assignments()[node_j] == agg_id {
                            in_deg += a_ij - row_sums[node_i] * row_sums[node_j] * inverse_total;
                        } else {
                            let agg_j = self.partition.node_to_agg[node_j];

                            match out_deg.get_mut(&agg_j) {
                                None => {
                                    let size_penalty = self.move_size_penalty(node_i, node_j);
                                    let deg = size_penalty + a_ij - row_sums[node_i] * row_sums[node_j] * inverse_total;
                                    out_deg.insert(agg_j, deg);
                                },
                                Some(out_deg) => {
                                    *out_deg += a_ij - row_sums[node_i] * row_sums[node_j] * inverse_total;
                                }
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
                        Some((new_agg, max_deg)) => {
                            trace!("node {} is part of agg {} with {} nodes and {:.2} in-degree and {:.2} max out-degree to join agg {} with {} nodes", node_i, agg_id, self.agg_sizes[agg_id], in_deg, max_deg, new_agg, self.agg_sizes[new_agg]);

                            if max_deg > in_deg {
                                Some((node_i, new_agg, max_deg - in_deg))
                            } else { None }
                        },
                        None => {
                                trace!("node {} is part of agg {} with {} nodes and {:.2} in-degree and no options to leave!", node_i, agg_id, self.agg_sizes[node_i], in_deg);
                            None
                        },
                    }
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
                    self.agg_sizes[old_agg] -= 1;
                    self.agg_sizes[new_agg] += 1;
                    //let result = self.partition.agg_to_node[old_agg].remove(&node_id);
                    //assert!(result);
                    //self.partition.agg_to_node[new_agg].insert(node_id);
                    true_swaps += 1;

                    alive_aggs[new_agg] = false;
                    alive_aggs[old_agg] = false;
                    alive_nodes[node_id] = false;
                    for (neighbor, _) in strength.nodes[node_id].iter().copied() {
                        alive_nodes[neighbor] = false;
                        alive_aggs[self.partition.node_assignments()[neighbor]] = false;
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
                //self.partition.info();
                let n_aggs = self.agg_sizes.len();
                let mut min_agg = usize::MAX;
                let mut max_agg = 0;
                for agg_size in self.agg_sizes.iter().copied() {
                    if agg_size < min_agg {
                        min_agg = agg_size;
                    }
                    if agg_size > max_agg {
                        max_agg = agg_size;
                    }
                }
                info!(
                    "Partition has {} aggs ({:.2} avg size) with min size of {} and max size of {}",
                    n_aggs,
                    self.base_strength.nodes.len() as f64 / n_aggs as f64,
                    min_agg,
                    max_agg
                );
            }
            #[cfg(debug_assertions)]
            self.partition.validate();
        }
    }
}
