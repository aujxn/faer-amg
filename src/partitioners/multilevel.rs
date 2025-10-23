use std::sync::Arc;

use crate::partitioners::{ModularityPartitioner, PartitionBuilder, PartitionerCallback};

use super::{AdjacencyList, Partition};
use faer::{prelude::Reborrow, sparse::SparseRowMat, MatRef};
use log::{info, trace, warn};

const STEP_CF: f64 = 4.0;

pub struct MultilevelPartitionerConfig<'a> {
    mat: Arc<SparseRowMat<usize, f64>>,
    pub block_size: usize,
    near_null: MatRef<'a, f64>,
    final_coarsening_factor: f64,
    levels: usize,
    pub agg_size_penalty: f64,
    pub dist_penalty: f64,
    pub max_improvement_iters: usize,
    pub callback: Option<PartitionerCallback>,
    step_cf: f64,
}

impl<'a> MultilevelPartitionerConfig<'a> {
    pub fn new(
        mat: Arc<SparseRowMat<usize, f64>>,
        near_null: MatRef<'a, f64>,
        final_coarsening_factor: f64,
        agg_size_penalty: f64,
        dist_penalty: f64,
    ) -> Self {
        if final_coarsening_factor < STEP_CF {
            panic!("Multilevel partitioners should target a coarsening level greater than {:.0}, use standard single level partitioner", STEP_CF.powf(2.0));
        }

        let levels = final_coarsening_factor.powf(STEP_CF.recip()).floor() as usize;
        let levels = levels.max(2);
        let step_cf = final_coarsening_factor.powf((levels as f64).recip());
        info!("ml partitioner will have {levels} levels with {step_cf:.2} reduction per level for final reduction of {final_coarsening_factor:.2}");

        Self {
            mat,
            block_size: 1,
            near_null,
            final_coarsening_factor,
            levels,
            agg_size_penalty,
            dist_penalty,
            max_improvement_iters: 50,
            callback: None,
            step_cf,
        }
    }

    pub fn build(&self) -> MultilevelPartitioner<'_> {
        MultilevelPartitioner {
            mat: self.mat.clone(),
            block_size: self.block_size,
            near_null: self.near_null,
            final_coarsening_factor: self.final_coarsening_factor,
            levels: self.levels,
            agg_size_penalty: self.agg_size_penalty,
            dist_penalty: self.dist_penalty,
            max_improvement_iters: self.max_improvement_iters,
            callback: self.callback.clone(),
            step_cf: self.step_cf,
        }
    }
}

pub struct MultilevelPartitioner<'a> {
    mat: Arc<SparseRowMat<usize, f64>>,
    block_size: usize,
    near_null: MatRef<'a, f64>,
    final_coarsening_factor: f64,
    levels: usize,
    agg_size_penalty: f64,
    dist_penalty: f64,
    max_improvement_iters: usize,
    pub callback: Option<PartitionerCallback>,
    step_cf: f64,
}

impl<'a> MultilevelPartitioner<'a> {
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
        let mut partitioners: Vec<ModularityPartitioner> = vec![fine_partitioner];

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
}
