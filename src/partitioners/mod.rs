use std::{collections::BTreeSet, sync::Arc};

use faer::{
    sparse::{SparseRowMat, SparseRowMatRef},
    ColRef, Mat,
};

pub mod modularity;

// NOTE: for now no generics, will refactor to fully generic interfaces when API stabilizes...

#[derive(Clone)]
pub struct Partition {
    mat: Arc<SparseRowMat<usize, f64>>,
    node_to_agg: Vec<usize>,
    agg_to_node: Vec<BTreeSet<usize>>,
}

pub struct PartitionBuilder {
    pub mat: Arc<SparseRowMat<usize, f64>>,
    pub near_nullspace: Arc<Mat<f64>>,
    pub coarsening_factor: f64,
    pub max_agg_size: Option<usize>,
    pub min_agg_size: Option<usize>,
    pub max_refinement_iters: usize,
}

impl PartitionBuilder {
    pub fn new(
        mat: Arc<SparseRowMat<usize, f64>>,
        near_nullspace: Arc<Mat<f64>>,
        coarsening_factor: f64,
        max_agg_size: Option<usize>,
        min_agg_size: Option<usize>,
        max_refinement_iters: usize,
    ) -> Self {
        Self {
            mat,
            near_nullspace,
            coarsening_factor,
            max_agg_size,
            min_agg_size,
            max_refinement_iters,
        }
    }
}

impl PartitionBuilder {
    pub fn build(&self) -> Partition {
        unimplemented!()
    }
}

struct AdjacencyList {
    nodes: Vec<Vec<(usize, f64)>>,
}

impl AdjacencyList {
    pub fn new_strength_graph(mat: SparseRowMatRef<usize, f64>, near_null: ColRef<f64>) -> Self {
        let mut nodes = vec![Vec::new(); mat.ncols()];
        for triplet in mat.triplet_iter() {
            if triplet.row != triplet.col {
                let strength = -near_null[triplet.row] * triplet.val * near_null[triplet.col];
                nodes[triplet.row].push((triplet.col, strength));
            }
        }

        Self { nodes }
    }

    pub fn pairwise_merge(&mut self, pairs: &Vec<(usize, usize)>, unmatched: &Vec<usize>) {
        let fine_n = self.nodes.len();
        let pairs_n = pairs.len();
        let coarse_n = fine_n - pairs_n;
        let mut merged_nodes = Vec::with_capacity(coarse_n);

        let mut agg_ids = vec![0; fine_n];
        for (agg_id, (i, j)) in pairs.iter().enumerate() {
            agg_ids[*i] = agg_id;
            agg_ids[*j] = agg_id;
        }
        for (agg_id, i) in unmatched.iter().enumerate() {
            agg_ids[*i] = agg_id + pairs_n;
        }

        self.map_indices(&agg_ids);

        // TODO: par iter
        for (i, j) in pairs.iter() {
            merged_nodes.push(self.merge_pair(*i, *j));
        }

        for i in unmatched.iter() {
            let mut swap = Vec::new();
            std::mem::swap(&mut swap, &mut self.nodes[*i]);
            merged_nodes.push(swap);
        }

        self.nodes = merged_nodes;
    }

    pub fn aggregate(&mut self, partition: &Partition) {
        self.map_indices(&partition.node_to_agg);
        // TODO: par iter
        self.nodes = partition
            .agg_to_node
            .iter()
            .map(|agg| self.merge_agg(agg))
            .collect();
    }

    fn map_indices(&mut self, agg_ids: &Vec<usize>) {
        // TODO: par iter
        for neighbors in self.nodes.iter_mut() {
            for edge in neighbors.iter_mut() {
                edge.0 = agg_ids[edge.0];
            }
            neighbors.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        }
    }

    fn merge_pair(&self, i: usize, j: usize) -> Vec<(usize, f64)> {
        let max_len = self.nodes[i].len() + self.nodes[j].len();
        let mut merged: Vec<(usize, f64)> = Vec::with_capacity(max_len);

        let neighbors_i = &self.nodes[i];
        let len_i = neighbors_i.len();
        let mut idx_i = 0;

        let neighbors_j = &self.nodes[j];
        let len_j = neighbors_j.len();
        let mut idx_j = 0;

        while idx_i < len_i && idx_j < len_j {
            let (id_i, w_i) = neighbors_i[idx_i];
            let (id_j, w_j) = neighbors_j[idx_j];
            let to_add;
            if id_i == id_j {
                to_add = (id_i, w_i + w_j);
                idx_i += 1;
                idx_j += 1;
            } else if id_i < id_j {
                to_add = (id_i, w_i);
                idx_i += 1;
            } else {
                to_add = (id_j, w_j);
                idx_j += 1;
            }

            match merged.last_mut() {
                Some(pair) => {
                    if to_add.0 == pair.0 {
                        pair.1 += to_add.1;
                    } else {
                        merged.push(to_add);
                    }
                }
                None => merged.push(to_add),
            }
        }
        merged
    }

    fn merge_agg(&self, _agg: &BTreeSet<usize>) -> Vec<(usize, f64)> {
        unimplemented!()
    }
}
