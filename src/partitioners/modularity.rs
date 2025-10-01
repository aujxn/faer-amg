use std::sync::Arc;

use faer::{col::AsColRef, linalg::matmul::dot::inner_prod, sparse::SparseRowMat, Col};
use rayon::{
    iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};

use super::{AdjacencyList, Partition};
use log::{info, warn};

pub struct ModularityPartitioner {
    strength: AdjacencyList,
    row_sums: Col<f64>,
    inverse_total: f64,
    coarsening_factor: f64,
    agg_size_penalty: f64,
    partition: Partition,
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
        let mut row_sums = mat.as_ref() * ones.as_col_ref();

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
            row_sums.ncols(),
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
            strength,
            row_sums,
            inverse_total,
            coarsening_factor,
            agg_size_penalty,
            partition: Partition::singleton(mat.clone()),
        }
    }

    pub fn modularity(mut self) -> Partition {
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
        let vertex_count = self.row_sums.ncols();
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

    fn refinement(&mut self, _max_iter: usize) {
        unimplemented!()
    }
}
