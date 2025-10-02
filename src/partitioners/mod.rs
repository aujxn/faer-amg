use std::{collections::BTreeSet, sync::Arc};

use faer::{
    sparse::{SparseRowMat, SparseRowMatRef},
    Col, ColRef, Mat, MatRef,
};
use log::info;
use rayon::iter::{IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator};

mod modularity;

// NOTE: for now no generics, will refactor to fully generic interfaces when API stabilizes...

#[derive(Clone)]
pub struct Partition {
    mat: Arc<SparseRowMat<usize, f64>>,
    node_to_agg: Vec<usize>,
    agg_to_node: Vec<BTreeSet<usize>>,
}

impl Partition {
    pub fn naggs(&self) -> usize {
        self.agg_to_node.len()
    }

    pub fn nnodes(&self) -> usize {
        self.node_to_agg.len()
    }

    pub fn cf(&self) -> f64 {
        self.node_to_agg.len() as f64 / self.agg_to_node.len() as f64
    }

    pub fn node_assignments(&self) -> &[usize] {
        &self.node_to_agg
    }

    pub fn info(&self) {
        let mut max_agg = usize::MIN;
        let mut min_agg = usize::MAX;
        for agg in self.agg_to_node.iter() {
            if agg.len() > max_agg {
                max_agg = agg.len();
            }
            if agg.len() < min_agg {
                min_agg = agg.len();
            }
        }
        info!(
            "Partition has {} aggs ({:.2} avg size) with min size of {} and max size of {}",
            self.agg_to_node.len(),
            self.node_to_agg.len() as f64 / self.agg_to_node.len() as f64,
            min_agg,
            max_agg
        );
    }
    pub fn singleton(mat: Arc<SparseRowMat<usize, f64>>) -> Self {
        let node_to_agg = (0..mat.nrows()).collect();
        let agg_to_node = (0..mat.nrows()).map(|i| BTreeSet::from([i])).collect();
        Self {
            mat,
            node_to_agg,
            agg_to_node,
        }
    }

    pub fn update_node_to_agg(&mut self) {
        for (agg_id, agg) in self.agg_to_node.iter().enumerate() {
            for idx in agg.iter().copied() {
                self.node_to_agg[idx] = agg_id;
            }
        }
    }

    pub fn update_agg_to_node(&mut self) {
        let n_aggs: usize = self
            .node_to_agg
            .iter()
            .copied()
            .max()
            .expect("You tried to update an empty partition");
        let mut new_aggs = vec![BTreeSet::new(); n_aggs + 1];
        for (node_id, agg_id) in self.node_to_agg.iter().copied().enumerate() {
            new_aggs[agg_id].insert(node_id);
        }
        self.agg_to_node = new_aggs;
    }

    pub fn from_node_to_agg(mat: Arc<SparseRowMat<usize, f64>>, node_to_agg: Vec<usize>) -> Self {
        let mut new_part = Self {
            mat,
            agg_to_node: Vec::new(),
            node_to_agg,
        };
        new_part.update_agg_to_node();
        new_part
    }

    pub fn from_agg_to_node(
        mat: Arc<SparseRowMat<usize, f64>>,
        agg_to_node: Vec<BTreeSet<usize>>,
    ) -> Self {
        let mut new_part = Self {
            mat,
            agg_to_node,
            node_to_agg: Vec::new(),
        };
        new_part.update_node_to_agg();
        new_part
    }

    pub fn pairwise_merge(&mut self, pairs: &Vec<(usize, usize)>, unmatched: &Vec<usize>) {
        // NOTE: Could be par with unsafe
        let pair_iter = pairs.iter().copied().map(|(agg_i, agg_j)| {
            let a = &self.agg_to_node[agg_i];
            let b = &self.agg_to_node[agg_j];
            debug_assert!(a.is_disjoint(b));
            a.union(b).cloned().collect()
        });
        let singleton_iter = unmatched
            .iter()
            .copied()
            .map(|agg_i| self.agg_to_node[agg_i].clone());

        let new_aggs = pair_iter.chain(singleton_iter).collect();

        self.agg_to_node = new_aggs;
        self.update_node_to_agg();
    }

    pub fn compose(&mut self, other: &Partition) {
        #[cfg(debug_assertions)]
        self.validate();
        #[cfg(debug_assertions)]
        other.validate();
        assert_eq!(self.agg_to_node.len(), other.node_to_agg.len());
        let mut new_agg_to_node = vec![BTreeSet::new(); other.agg_to_node.len()];
        for (i, agg_id) in self.node_to_agg.iter_mut().enumerate() {
            *agg_id = other.node_to_agg[*agg_id];
            new_agg_to_node[*agg_id].insert(i);
        }
        self.agg_to_node = new_agg_to_node;
        #[cfg(debug_assertions)]
        self.validate();
    }

    pub fn validate(&self) {
        let mut visited = vec![false; self.node_to_agg.len()];
        for (agg_id, agg) in self.agg_to_node.iter().enumerate() {
            for node_id in agg.iter().copied() {
                assert_eq!(agg_id, self.node_to_agg[node_id]);
                assert!(!visited[node_id]);
                visited[node_id] = true;
            }
        }
        assert!(visited.into_iter().all(|visited| visited));
    }
}

pub struct PartitionBuilder {
    pub mat: Arc<SparseRowMat<usize, f64>>,
    pub near_null: Mat<f64>,
    pub coarsening_factor: f64,
    pub agg_size_penalty: f64,
    pub max_refinement_iters: usize,
}

impl PartitionBuilder {
    pub fn new(
        mat: Arc<SparseRowMat<usize, f64>>,
        near_null: Mat<f64>,
        coarsening_factor: f64,
        agg_size_penalty: f64,
        max_refinement_iters: usize,
    ) -> Self {
        Self {
            mat,
            near_null,
            coarsening_factor,
            agg_size_penalty,
            max_refinement_iters,
        }
    }

    pub fn build(&self) -> Partition {
        let base_strength =
            AdjacencyList::new_strength_graph(self.mat.as_ref().as_ref(), self.near_null.as_ref());
        let mut partitioner = modularity::ModularityPartitioner::new(
            self.mat.clone(),
            base_strength.clone(),
            self.coarsening_factor,
            self.agg_size_penalty,
        );

        partitioner.partition();
        partitioner.improve(self.max_refinement_iters);
        partitioner.into_partition()
    }
}

#[derive(Debug, Clone)]
struct AdjacencyList {
    nodes: Vec<Vec<(usize, f64)>>,
}

impl AdjacencyList {
    pub fn new_strength_graph(mat: SparseRowMatRef<usize, f64>, near_null: MatRef<f64>) -> Self {
        let mut nodes = vec![Vec::new(); mat.ncols()];
        for triplet in mat.triplet_iter() {
            let mut strength = 0.0;
            if triplet.row != triplet.col {
                for vec in near_null.col_iter() {
                    strength += -vec[triplet.row] * triplet.val * vec[triplet.col];
                }
                nodes[triplet.row].push((triplet.col, strength));
            }
        }

        Self { nodes }
    }

    pub fn get(&self, node_i: usize, node_j: usize) -> f64 {
        let neighborhood = &self.nodes[node_i];
        match neighborhood.binary_search_by(|probe| probe.0.cmp(&node_j)) {
            Ok(idx) => neighborhood[idx].1,
            Err(_) => 0.0,
        }
    }

    pub fn pairwise_merge(&mut self, pairs: &Vec<(usize, usize)>, unmatched: &Vec<usize>) {
        let fine_n = self.nodes.len();
        let pairs_n = pairs.len();

        let mut agg_ids = vec![0; fine_n];
        // NOTE: could be par with unsafe
        for (agg_id, (i, j)) in pairs.iter().enumerate() {
            agg_ids[*i] = agg_id;
            agg_ids[*j] = agg_id;
        }
        for (agg_id, i) in unmatched.iter().enumerate() {
            agg_ids[*i] = agg_id + pairs_n;
        }

        self.map_indices(&agg_ids);

        // NOTE: could avoid clone and re-use old memory with some opt here
        self.nodes = pairs
            .par_iter()
            .copied()
            .map(|(i, j)| self.merge_pair(i, j))
            .chain(unmatched.par_iter().copied().map(|i| self.nodes[i].clone()))
            .collect();
    }

    pub fn aggregate(&mut self, partition: &Partition) {
        self.map_indices(&partition.node_to_agg);
        self.nodes = partition
            .agg_to_node
            .par_iter()
            .map(|agg| self.merge_agg(agg))
            .collect();
    }

    fn map_indices(&mut self, agg_ids: &Vec<usize>) {
        self.nodes.par_iter_mut().for_each(|neighbors| {
            for edge in neighbors.iter_mut() {
                edge.0 = agg_ids[edge.0];
            }
            neighbors.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        });
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
