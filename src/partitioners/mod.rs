use faer::{
    prelude::Reborrow,
    sparse::{SparseRowMatRef, SymbolicSparseRowMatRef},
    Col, Mat, MatRef,
};
use rayon::{
    iter::{
        IntoParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
    },
    slice::ParallelSliceMut,
};
use std::{
    collections::{BTreeSet, BinaryHeap, VecDeque},
    fmt::{self, Debug, Formatter},
    sync::Arc,
};

pub mod modularity;
pub mod multilevel;

use crate::{core::SparseMatOp, partitioners::modularity::Partitioner};

#[derive(Clone)]
pub struct Partition {
    node_to_agg: Vec<usize>,
    agg_to_node: Vec<BTreeSet<usize>>,
}

#[derive(Debug, Clone, Copy)]
pub struct PartitionStats {
    pub aggs: usize,
    pub nodes: usize,
    pub cf: f64,
    pub agg_size_min: usize,
    pub agg_size_max: usize,
    pub agg_size_avg: f64,
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

    pub fn aggregates(&self) -> &Vec<BTreeSet<usize>> {
        &self.agg_to_node
    }

    pub fn singleton(n_nodes: usize) -> Self {
        let node_to_agg = (0..n_nodes).collect();
        let agg_to_node = (0..n_nodes).map(|i| BTreeSet::from([i])).collect();
        Self {
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

    pub fn from_node_to_agg(node_to_agg: Vec<usize>) -> Self {
        let mut new_part = Self {
            agg_to_node: Vec::new(),
            node_to_agg,
        };
        new_part.update_agg_to_node();
        new_part
    }

    pub fn from_agg_to_node(agg_to_node: Vec<BTreeSet<usize>>) -> Self {
        let mut new_part = Self {
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

    pub fn aggregate_sizes(&self) -> Vec<usize> {
        self.agg_to_node.iter().map(|agg| agg.len()).collect()
    }

    pub fn info(&self) -> PartitionStats {
        let aggs = self.naggs();
        let nodes = self.nnodes();
        if aggs == 0 {
            return PartitionStats {
                aggs,
                nodes,
                cf: 0.0,
                agg_size_min: 0,
                agg_size_max: 0,
                agg_size_avg: 0.0,
            };
        }
        let mut min_size = usize::MAX;
        let mut max_size = 0usize;
        let mut sum = 0usize;
        for agg in &self.agg_to_node {
            let size = agg.len();
            min_size = min_size.min(size);
            max_size = max_size.max(size);
            sum += size;
        }
        if min_size == usize::MAX {
            min_size = 0;
        }
        let avg = if aggs > 0 {
            sum as f64 / aggs as f64
        } else {
            0.0
        };

        PartitionStats {
            aggs,
            nodes,
            cf: self.cf(),
            agg_size_min: min_size,
            agg_size_max: max_size,
            agg_size_avg: avg,
        }
    }
}

impl fmt::Debug for Partition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let stats = self.info();
        write!(
            f,
            "Partition {{ aggs: {}, nodes: {}, cf: {:.2}, agg_size_min: {}, agg_size_max: {}, agg_size_avg: {:.2} }}",
            stats.aggs,
            stats.nodes,
            stats.cf,
            stats.agg_size_min,
            stats.agg_size_max,
            stats.agg_size_avg,
        )
    }
}

#[derive(Clone)]
pub struct PartitionerCallback {
    f: Arc<dyn Fn(usize, &Partitioner) + Send + Sync>,
    name: String,
}

impl Debug for PartitionerCallback {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("PartitionerCallback")
            .field("name", &self.name)
            .finish()
    }
}

impl PartitionerCallback {
    pub fn new(f: Arc<dyn Fn(usize, &Partitioner) + Send + Sync>) -> Self {
        Self {
            f,
            name: "Unnamed Callback".to_string(),
        }
    }

    pub fn new_named(f: Arc<dyn Fn(usize, &Partitioner) + Send + Sync>, name: String) -> Self {
        Self { f, name }
    }

    fn call(&self, iter: usize, partitioner: &Partitioner) {
        self.f.as_ref()(iter, partitioner);
    }
}

#[derive(Debug, Clone)]
pub struct PartitionerConfig {
    pub coarsening_factor: f64,
    pub agg_size_penalty: f64,
    pub max_improvement_iters: usize,
    pub callback: Option<PartitionerCallback>,
}

impl Default for PartitionerConfig {
    fn default() -> Self {
        Self {
            coarsening_factor: 8.0,
            agg_size_penalty: 1e0,
            max_improvement_iters: 100,
            callback: None,
        }
    }
}

impl PartitionerConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn build(
        &self,
        mat: SparseMatOp,
        near_null: Arc<Mat<f64>>,
        starting_partition: Option<Partition>,
        weights: Option<&Vec<f64>>,
    ) -> Partitioner {
        let mat_ref = mat.mat_ref();
        let nn_ref = near_null.as_ref().as_ref();
        let block_size = mat.block_size();
        assert_eq!(mat_ref.nrows(), mat_ref.ncols());
        assert_eq!(near_null.nrows(), mat_ref.nrows());
        if let Some(part) = starting_partition.as_ref() {
            part.validate();
            assert_eq!(part.nnodes(), mat_ref.nrows() / mat.block_size());
        }
        let mut strength = match weights {
            Some(weights) => {
                AdjacencyList::new_ls_strength_graph(mat_ref.rb(), nn_ref.rb(), weights, 3)
            }
            None => {
                unimplemented!();
            }
        };
        if block_size > 1 {
            let node_to_agg = (0..mat_ref.nrows())
                .map(|node_id| node_id / block_size)
                .collect();
            let block_reduce = Partition::from_node_to_agg(node_to_agg);
            strength.aggregate(&block_reduce);
            strength.filter_diag();
        }
        let needs_init = starting_partition.is_none();
        let mut partitioner = Partitioner::new(strength, starting_partition, None, self.clone());
        if needs_init {
            partitioner.initialize_partition();
        }
        partitioner.improve_partition();
        partitioner
    }

    fn build_from_strength(
        &self,
        strength: AdjacencyList,
        starting_partition: Option<Partition>,
        node_weights: Option<Vec<usize>>,
    ) -> Partitioner {
        Partitioner::new(strength, starting_partition, node_weights, self.clone())
    }

    pub fn build_partition(
        &self,
        mat: SparseMatOp,
        near_null: Arc<Mat<f64>>,
        weights: Option<&Vec<f64>>,
    ) -> Partition {
        let partitioner = self.build(mat, near_null, None, weights);
        partitioner.into_partition()
    }
}

#[derive(Debug, Clone)]
pub(crate) struct AdjacencyList {
    nodes: Vec<Vec<(usize, f64)>>,
}

impl AdjacencyList {
    pub fn new_ls_strength_graph(
        mat: SparseRowMatRef<usize, f64>,
        near_null: MatRef<f64>,
        weights: &Vec<f64>,
        max_depth: usize,
    ) -> Self {
        let mut nodes: Vec<Vec<(usize, f64)>> = vec![Vec::new(); mat.ncols()];
        let w = Col::from_iter(weights.iter().copied().take(near_null.ncols())).into_diagonal();

        let theta: f64 = 0.5;
        let eps: f64 = 1e-30;

        for i in 0..mat.nrows() {
            let local_nodes = extract_local_subgraph(mat.symbolic(), i, max_depth);
            let vi = near_null.row(i);
            let vi_norm = (vi.rb() * w.rb() * vi.transpose()).max(eps);
            for &j in local_nodes.iter().filter(|j| **j > i) {
                let vj = near_null.row(j);
                let vj_norm = (vj.rb() * w.rb() * vj.transpose()).max(eps);
                let vi_w_vjt = vi.rb() * w.rb() * vj.transpose();
                let rho2 = (vi_w_vjt * vi_w_vjt) / (vi_norm * vj_norm);
                let symm_relative = 2.0 * (1.0 - rho2).max(0.0).sqrt();
                nodes[i].push((j, symm_relative));
                nodes[j].push((i, symm_relative));
            }
        }

        let eps = 1e-12;
        let alpha = 4.0; // tune: bigger => more contrast (weaker edges closer to 0)

        for neighborhood in nodes.iter_mut() {
            // neighborhood: Vec<(j, d_ij)> where d smaller = stronger
            neighborhood.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            let keep_f = (neighborhood.len() as f64 * theta).floor() as usize;
            let keep = keep_f.max(1);
            neighborhood.truncate(keep);

            let d_min = neighborhood.first().expect("graph is disconnected").1;
            let d_max = neighborhood.last().expect("unreachable").1;

            if (d_max - d_min).abs() < eps {
                for (_j, w) in neighborhood.iter_mut() {
                    *w = 1.0;
                }
            } else {
                for (_j, d) in neighborhood.iter_mut() {
                    let t = (d_max - *d) / (d_max - d_min + eps); // 1 at best, 0 at worst
                    *d = t.powf(alpha);
                }
            }

            neighborhood.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        }

        Self { nodes }
    }

    pub fn maximal_independent_set(&self, f_points: &mut [bool]) -> Vec<usize> {
        let mut new_c_points = Vec::new();
        let mut degrees: Vec<(usize, f64)> = self
            .nodes
            .iter()
            .enumerate()
            .filter(|(i, _)| f_points[*i])
            .map(|(i, neighborhood)| {
                let strength_degree: f64 = neighborhood
                    .iter()
                    .filter(|(j, _)| f_points[*j])
                    .map(|(_, w)| *w)
                    .sum();
                (i, strength_degree)
            })
            .collect();
        degrees.par_sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).expect("bad float"));
        for (i, _) in degrees {
            if f_points[i] {
                f_points[i] = false;
                new_c_points.push(i);
                for (j, _) in self.nodes[i].iter() {
                    f_points[*j] = false;
                }
            }
        }
        assert!(f_points.iter().all(|b| !b));
        new_c_points
    }

    /*
    pub fn nodes(&self) -> &Vec<Vec<(usize, f64)>> {
        &self.nodes
    }

    pub fn get(&self, node_i: usize, node_j: usize) -> f64 {
        let neighborhood = &self.nodes[node_i];
        match neighborhood.binary_search_by(|probe| probe.0.cmp(&node_j)) {
            Ok(idx) => neighborhood[idx].1,
            Err(_) => 0.0,
        }
    }
    */

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
        let neighborhoods: Vec<(Vec<(usize, f64)>, f64)> = partition
            .agg_to_node
            .par_iter()
            .map(|agg| self.merge_agg(agg))
            .collect();
        // NOTE: this is normalizing while including self loops. (and in the
        // PartitionerConfig::build case these self loops are immediatelly thrown away.
        // this might be bad
        //
        // Also I believe this is bugged...
        let max: f64 = neighborhoods
            .iter()
            .map(|(_, local_max_strength)| *local_max_strength)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();

        self.nodes = neighborhoods
            .into_par_iter()
            .map(|(mut neighborhood, _)| {
                for (_, strength) in neighborhood.iter_mut() {
                    *strength /= max;
                }
                neighborhood
            })
            .collect();
    }

    pub fn filter_diag(&mut self) {
        for (node_i, row) in self.nodes.iter_mut().enumerate() {
            row.retain(|(node_j, _)| *node_j != node_i);
        }
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

        while idx_i < len_i {
            let to_add = neighbors_i[idx_i];
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
            idx_i += 1;
        }

        while idx_j < len_j {
            let to_add = neighbors_j[idx_j];
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
            idx_j += 1;
        }
        merged
    }

    fn merge_agg(&self, agg: &BTreeSet<usize>) -> (Vec<(usize, f64)>, f64) {
        #[derive(Clone)]
        struct HeapEntry {
            local_id: usize,
            neighbor: usize,
            weight: f64,
        }

        impl PartialOrd for HeapEntry {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(other.neighbor.cmp(&self.neighbor))
            }
        }
        impl Ord for HeapEntry {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                other.neighbor.cmp(&self.neighbor)
            }
        }
        impl PartialEq for HeapEntry {
            fn eq(&self, other: &Self) -> bool {
                self.neighbor == other.neighbor
            }
        }
        impl Eq for HeapEntry {}

        let mut merge_queue: Vec<VecDeque<(usize, f64)>> = agg
            .iter()
            .copied()
            .map(|node_id| self.nodes[node_id].iter().copied().collect())
            .collect();
        let mut min_heap: BinaryHeap<HeapEntry> = BinaryHeap::new();
        for (local_id, neighborhood) in merge_queue.iter_mut().enumerate() {
            let (neighbor, weight) = neighborhood
                .pop_front()
                .expect("empty neighborhood means graph is disconnected...");
            let entry = HeapEntry {
                local_id,
                neighbor,
                weight,
            };
            min_heap.push(entry);
        }

        let mut combined: Vec<(usize, f64)> = Vec::new();
        loop {
            match min_heap.pop() {
                Some(entry) => {
                    if let Some(last) = combined.last_mut() {
                        if last.0 == entry.neighbor {
                            last.1 += entry.weight;
                        } else {
                            combined.push((entry.neighbor, entry.weight));
                        }
                    } else {
                        combined.push((entry.neighbor, entry.weight));
                    }
                    if let Some((neighbor, weight)) = merge_queue[entry.local_id].pop_front() {
                        let next_entry = HeapEntry {
                            local_id: entry.local_id,
                            neighbor,
                            weight,
                        };
                        min_heap.push(next_entry);
                    }
                }
                None => break,
            }
        }

        let local_max = combined
            .iter()
            .map(|(_, w)| *w)
            .max_by(|a, b| a.partial_cmp(b).expect("bad float"))
            .expect("empty neighborhood bad");

        (combined, local_max)
    }

    /*
    fn subgraph(&self, aggregate: &BTreeSet<usize>) -> AggregateGraph {
        let mut agg_graph = Graph::new_undirected();
        let mut index_map = HashMap::new();
        for node_i in aggregate.iter().copied() {
            let node_idx = agg_graph.add_node(());
            index_map.insert(node_i, node_idx);
        }

        for node_i in aggregate.iter().copied() {
            let neighborhood = &self.nodes[node_i];
            let node_idx_i = *index_map.get(&node_i).expect("invalid partition or graph");
            for (node_j, weight) in neighborhood
                .iter()
                .copied()
                .filter(|(node_j, _)| *node_j > node_i)
            {
                if let Some(node_idx_j) = index_map.get(&node_j) {
                    agg_graph.add_edge(node_idx_i, *node_idx_j, weight.recip());
                }
            }
        }
        AggregateGraph {
            graph: agg_graph,
            index_map,
        }
    }
    */
}

/*
struct AggregateGraph {
    graph: Graph<(), f64, Undirected>,
    index_map: HashMap<usize, NodeIndex>,
}
*/

pub fn extract_local_subgraph(
    adjacency: SymbolicSparseRowMatRef<usize>,
    center: usize,
    max_depth: usize,
) -> BTreeSet<usize> {
    let mut to_visit: VecDeque<(usize, usize)> = VecDeque::new();
    to_visit.push_back((center, 0));
    let mut visited: BTreeSet<usize> = BTreeSet::new();
    visited.insert(center);
    loop {
        match to_visit.pop_front() {
            Some((j, depth)) => {
                if depth < max_depth {
                    for neighbor in adjacency.col_idx_of_row(j) {
                        if visited.insert(neighbor) {
                            to_visit.push_back((neighbor, depth + 1));
                        }
                    }
                }
            }
            None => break visited,
        }
    }
}
