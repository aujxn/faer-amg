use super::Partition;
use crate::{
    core::SparseMatOp,
    partitioners::{Partitioner, PartitionerConfig},
};
use faer::Mat;
use std::sync::Arc;

pub struct MultilevelPartitionerConfig {
    pub partitioner_configs: Vec<PartitionerConfig>,
}

impl MultilevelPartitionerConfig {
    pub fn build(&self, mat: SparseMatOp, near_null: Arc<Mat<f64>>) -> MultilevelPartitioner {
        assert!(self.partitioner_configs.len() > 1);
        let mut partitioners = Vec::new();

        let partitioner = self.partitioner_configs[0].build(mat, near_null, None, None);
        partitioners.push(partitioner);

        for config in self.partitioner_configs.iter().skip(1) {
            let partitioner = partitioners.last().unwrap();
            let partition = partitioner.get_partition();
            let mut strength = partitioner.get_base_strength().clone();
            let new_node_weights = Some(partitioner.get_agg_sizes().clone());
            strength.aggregate(partition);
            strength.filter_diag();

            let mut new_partitioner = config.build_from_strength(strength, None, new_node_weights);
            new_partitioner.initialize_partition();
            new_partitioner.improve_partition();
            partitioners.push(new_partitioner);
        }
        MultilevelPartitioner { partitioners }
    }

    pub fn build_hierarchy(&self, mat: SparseMatOp, near_null: Arc<Mat<f64>>) -> Vec<Partition> {
        let ml_partitioner = self.build(mat, near_null);
        ml_partitioner.into_partitions()
    }
}

pub struct MultilevelPartitioner {
    partitioners: Vec<Partitioner>,
}

impl MultilevelPartitioner {
    pub fn into_partitions(self) -> Vec<Partition> {
        self.partitioners
            .into_iter()
            .map(|p| p.into_partition())
            .collect()
    }
    /*
    pub fn partition(&mut self) -> Vec<Partition> {
        let mut partition_builder = PartitionBuilder::new(
            self.mat.clone(),
            self.block_size,
            self.near_null.rb(),
            self.step_cf,
            self.agg_size_penalty,
            self.dist_penalty,
            self.max_improvement_iters,
        );
        partition_builder.callback = self.callback.clone();

        let mut fine_partitioner = partition_builder.create_partitioner();
        fine_partitioner.partition(self.step_cf);
        fine_partitioner.improve(self.max_improvement_iters);

        let mut last = fine_partitioner.clone();
        let mut partitioners: Vec<Partitioner> = vec![fine_partitioner];

        for _level in 1..self.levels {
            last.aggregate();
            last.coarsening_factor *= self.step_cf;
            last.partition(self.step_cf);

            last.improve(self.max_improvement_iters);

            last.partition.update_agg_to_node();
            partitioners.push(last);
            last = partitioners.last().unwrap().clone();
        }

        partitioners.into_iter().map(|p| p.partition).collect()
    }
    */
}
