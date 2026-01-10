use faer::{
    dyn_stack::{MemBuffer, MemStack},
    get_global_parallelism,
    mat::AsMatMut,
    matrix_free::{LinOp, Precond},
    sparse::{SparseRowMat, SparseRowMatRef},
    Mat,
};
use log::info;
use std::{fmt, sync::Arc};

use crate::{
    adaptivity::{find_near_null, smooth_vector_rand_svd},
    core::SparseMatOp,
    interpolation::smoothed_aggregation,
    par_spmm::ParSpmmOp,
    partitioners::{
        multilevel::MultilevelPartitionerConfig, Partition, PartitionStats, PartitionerConfig,
    },
    preconditioners::smoothers::{new_l1, StationaryIteration},
    utils::{matrix_stats, write_matrix_stats_table, MatrixStats, NdofsFormat},
};

#[derive(Debug, Clone)]
pub struct HierarchyConfig {
    pub coarsest_dim: usize,
    pub partitioner_config: PartitionerConfig,
    pub interp_candidate_dim: usize,
}

impl Default for HierarchyConfig {
    fn default() -> Self {
        Self {
            coarsest_dim: 1000,
            partitioner_config: PartitionerConfig::default(),
            interp_candidate_dim: 4,
        }
    }
}

impl HierarchyConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn build(&self, base_matrix: SparseMatOp, near_null: Arc<Mat<f64>>) -> Hierarchy {
        let mut hierarchy = Hierarchy::new(base_matrix, near_null, self.clone());
        hierarchy.coarsen();
        hierarchy
    }
}

#[derive(Clone)]
pub struct Hierarchy {
    operators: Vec<SparseMatOp>,
    restrictions: Vec<Arc<SparseRowMat<usize, f64>>>,
    interpolations: Vec<Arc<SparseRowMat<usize, f64>>>,
    partitions: Vec<Arc<Partition>>,
    near_nulls: Vec<Arc<Mat<f64>>>,
    candidates: Vec<Arc<Mat<f64>>>,
    config: HierarchyConfig,
}

fn write_partition_stats_table(
    f: &mut fmt::Formatter<'_>,
    title: &str,
    stats: &[PartitionStats],
) -> fmt::Result {
    if stats.is_empty() {
        writeln!(f, "{title}: <empty>")?;
        return Ok(());
    }

    writeln!(f, "{title}")?;
    writeln!(
        f,
        "{:>4}  {:>8}  {:>8}  {:>8}  {:>26}",
        "lev", "aggs", "nodes", "cf", "agg size"
    )?;
    writeln!(
        f,
        "{:-<4}  {:-<8}  {:-<8}  {:-<8}  {:-<26}",
        "", "", "", "", ""
    )?;
    writeln!(
        f,
        "{:>4}  {:>8}  {:>8}  {:>8}  {:>8} {:>8} {:>8}",
        "", "", "", "", "min", "max", "avg"
    )?;
    writeln!(
        f,
        "{:-<4}  {:-<8}  {:-<8}  {:-<8}  {:-<8} {:-<8} {:-<8}",
        "", "", "", "", "", "", ""
    )?;

    for (level, stat) in stats.iter().enumerate() {
        writeln!(
            f,
            "{:>4}  {:>8}  {:>8}  {:>8.2}  {:>8} {:>8} {:>8.2}",
            level,
            stat.aggs,
            stat.nodes,
            stat.cf,
            stat.agg_size_min,
            stat.agg_size_max,
            stat.agg_size_avg
        )?;
    }

    Ok(())
}

impl fmt::Debug for Hierarchy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let levels = self.levels();
        if levels == 0 {
            writeln!(f, "Hierarchy: <empty>")?;
            return Ok(());
        }

        let operator_stats = self
            .operators
            .iter()
            .map(|op| matrix_stats(op.mat_ref()))
            .collect::<Vec<_>>();

        write_matrix_stats_table(
            f,
            &format!("Hierarchy summary ({} levels)", levels),
            &operator_stats,
            NdofsFormat::Auto,
        )?;
        //writeln!(f, "Block sizes: {:?}", self.block_sizes)?;

        if !self.interpolations.is_empty() {
            writeln!(f)?;
            let interpolation_stats: Vec<_> = self
                .interpolations
                .iter()
                .map(|p| matrix_stats(p.as_ref().as_ref()))
                .collect();
            write_matrix_stats_table(
                f,
                "Interpolation summary",
                &interpolation_stats,
                NdofsFormat::Rectangular,
            )?;
        }

        if !self.partitions.is_empty() {
            writeln!(f)?;
            let partition_stats: Vec<_> = self.partitions.iter().map(|p| p.info()).collect();
            write_partition_stats_table(f, "Partition summary", &partition_stats)?;
        }

        writeln!(f, "Operator complexity: {:.3}", self.op_complexity())?;
        Ok(())
    }
}

impl Hierarchy {
    fn new(fine_op: SparseMatOp, fine_near_null: Arc<Mat<f64>>, config: HierarchyConfig) -> Self {
        let n_candidates = config.interp_candidate_dim;
        let candidates = fine_near_null.subcols(0, n_candidates).to_owned();
        Self {
            operators: vec![fine_op],
            restrictions: Vec::new(),
            interpolations: Vec::new(),
            partitions: Vec::new(),
            near_nulls: vec![fine_near_null],
            candidates: vec![Arc::new(candidates)],
            config,
        }
    }

    /*
    fn coarsen2(&mut self) {
        let candidate_dim = self.config.interp_candidate_dim as f64;
        let coarsest_dim = self.config.coarsest_dim;
        let cf = self.config.partitioner_config.coarsening_factor;

        let mut level = 0;

        let n_candidates = self.config.interp_candidate_dim;
        loop {
            let current_op = self.get_op(level);
            if current_op.mat_ref().nrows() < coarsest_dim {
                break;
            }
            let (coarsen_near_null, partitioner_conf) = if level == 0 {
                let coarsen_near_null = self.get_near_null(level);
                let fine_block_size = current_op.block_size() as f64;
                let ratio = candidate_dim / fine_block_size;
                let first_cf = cf * ratio;
                let mut first_partitioner_conf = self.config.partitioner_config.clone();
                first_partitioner_conf.coarsening_factor = first_cf;
                (coarsen_near_null, first_partitioner_conf)
            } else {
                let coarsen_near_null = smooth_vector_rand_svd(current_op.clone(), 100, 128);
                let coarsen_near_null = Arc::new(coarsen_near_null);
                self.near_nulls.push(coarsen_near_null.clone());
                (coarsen_near_null, self.config.partitioner_config.clone())
            };
            let interp_near_null = self.candidates[level].clone();
            let partition =
                partitioner_conf.build_partition(current_op.clone(), coarsen_near_null.clone());
            partition.validate();

            let par = get_global_parallelism();

            let (mut coarse_near_null, restriction_mat, interpolation_mat, coarse_mat) =
                smoothed_aggregation(
                    current_op.mat_ref(),
                    &partition,
                    current_op.block_size(),
                    interp_near_null.as_ref().as_ref(),
                );

            let coarse_op = SparseMatOp::new(coarse_mat, n_candidates, par);
            let l1_diag = Arc::new(new_l1(coarse_op.mat_ref()));
            let dyn_op: Arc<dyn LinOp<f64> + Send> = coarse_op
                .par_op()
                .map(|op| op as Arc<dyn LinOp<f64> + Send>)
                .unwrap_or(coarse_op.arc_mat());
            let stationary = StationaryIteration::new(dyn_op, l1_diag, 3);
            let stack_req = stationary
                .apply_in_place_scratch(coarse_near_null.ncols(), get_global_parallelism());
            let mut buf = MemBuffer::new(stack_req);
            let stack = MemStack::new(&mut buf);
            stationary.apply_in_place(
                coarse_near_null.as_mat_mut(),
                get_global_parallelism(),
                stack,
            );

            let interp_near_null = Arc::new(coarse_near_null);
            let p = Arc::new(interpolation_mat);
            let r = Arc::new(restriction_mat);
            let partition = Arc::new(partition);

            self.add_level(coarse_op, partition, interp_near_null, p, r);
            level += 1;
        }
    }
    */

    fn coarsen(&mut self) {
        let candidate_dim = self.config.interp_candidate_dim as f64;
        let coarsest_dim = self.config.coarsest_dim as f64;
        let cf = self.config.partitioner_config.coarsening_factor;

        let near_null = self.get_near_null(0);
        let op = self.get_op(0);
        let fine_block_size = op.block_size() as f64;
        let ratio = candidate_dim / fine_block_size;
        let first_cf = cf * ratio;
        let mut first_partitioner_conf = self.config.partitioner_config.clone();
        first_partitioner_conf.coarsening_factor = first_cf;

        let mut partitioner_configs: Vec<PartitionerConfig> = vec![first_partitioner_conf];
        let mut size = self.get_mat_ref(0).nrows() as f64 / cf;

        while size > coarsest_dim {
            partitioner_configs.push(self.config.partitioner_config.clone());
            size /= cf;
        }

        let ml_partitioner_config = MultilevelPartitionerConfig {
            partitioner_configs,
        };

        let partitions = ml_partitioner_config.build_hierarchy(op, near_null.clone());

        let n_candidates = self.config.interp_candidate_dim;
        let par = get_global_parallelism();

        for (level, partition) in partitions.into_iter().enumerate() {
            let fine_mat = self.current_op();
            let candidates = self.candidates.last().unwrap().clone();
            let (mut coarse_near_null, restriction_mat, interpolation_mat, coarse_mat) =
                smoothed_aggregation(
                    fine_mat.mat_ref(),
                    &partition,
                    fine_mat.block_size(),
                    candidates.as_ref().as_ref(),
                );

            let coarse_op = SparseMatOp::new(coarse_mat, n_candidates, par);
            let l1_diag = Arc::new(new_l1(coarse_op.mat_ref()));
            let dyn_op = coarse_op.dyn_op();
            let stationary = StationaryIteration::new(dyn_op, l1_diag, 3);
            let stack_req = stationary
                .apply_in_place_scratch(coarse_near_null.ncols(), get_global_parallelism());
            let mut buf = MemBuffer::new(stack_req);
            let stack = MemStack::new(&mut buf);
            stationary.apply_in_place(
                coarse_near_null.as_mat_mut(),
                get_global_parallelism(),
                stack,
            );

            let coarse_candidates = Arc::new(coarse_near_null);
            let p = Arc::new(interpolation_mat);
            let r = Arc::new(restriction_mat);
            let partition = Arc::new(partition);
            //let near_null = smooth_vector_rand_svd(coarse_op.clone(), 15, 128);
            let max_dim = coarse_op.mat_ref().nrows().max(1);
            let near_null_dim = 128.min(max_dim);
            let smoother_block = 64.min(max_dim);
            let near_null = find_near_null(coarse_op.clone(), 50, near_null_dim, smoother_block);
            self.near_nulls.push(Arc::new(near_null));
            self.add_level(coarse_op, partition, coarse_candidates, p, r);
            info!(
                "Created coarse op at level {}. Hierarchy:\n{:?}",
                level + 1,
                &self
            );
        }

        /*
        let fine_mat = self.current_mat();

        let fine_block_size = *self
            .block_sizes
            .last()
            .ok_or(HierarchyError::MissingBlockSize)?;

        /*
        if params.interp_cols > fine_near_null.ncols() {
            return Err(HierarchyError::InvalidInterpolationDim);
        }
        */

        let partition = match params.partition {
            Some(_partition) => unimplemented!(),
            None => {
                let strategy = self
                    .partition_strategy
                    .as_ref()
                    .ok_or(HierarchyError::MissingPartitionStrategy)?;
                let mut partition = if fine_block_size == 1 {
                    Partition::singleton(fine_mat.nrows())
                } else {
                    let node_to_agg: Vec<usize> = (0..fine_mat.nrows())
                        .map(|node_id| node_id / fine_block_size)
                        .collect();
                    Partition::from_node_to_agg(node_to_agg)
                };

                //let coarsening_near_null = self.near_nulls.last().unwrap();
                let coarsening_near_null = smooth_vector_rand_svd(
                    fine_mat.clone(),
                    strategy.smoothing_iters,
                    strategy.near_null_dim,
                );

                let scalar_partition = strategy.build_partition(
                    fine_mat.clone(),
                    fine_block_size,
                    coarsening_near_null.as_ref(),
                    params.coarsening_factor,
                );
                partition.compose(&scalar_partition);
                partition
            }
        };
        let partition = Arc::new(partition);

        //let interpolation_near_null = fine_near_null.subcols(0, params.interp_cols);

        //let ones: Mat<f64> = Mat::ones(fine_near_null.ncols(), 1);
        //let interpolation_near_null = fine_near_null.as_ref() * ones;

        //let interpolation_near_null = smooth_vector(fine_mat.clone(), 4, 1);
        /*
        for v in interpolation_near_null.col(0).iter() {
            println!("{:.2e}", v);
        }
        */
        let interpolation_near_null = self.near_nulls().last().unwrap();
        let (mut coarse_near_null, restriction_mat, interpolation_mat, coarse_mat) =
            smoothed_aggregation(
                fine_mat.as_ref().as_ref(),
                partition.as_ref(),
                fine_block_size,
                interpolation_near_null.as_ref(),
                //ones.as_ref(),
            );

        let par_op = Arc::new(ParSpmmOp::new(coarse_mat.as_ref(), 16));
        let l1_diag = Arc::new(new_l1(coarse_mat.as_ref().as_ref()));
        let stationary = StationaryIteration::new(par_op, l1_diag, 3);
        let stack_req =
            stationary.apply_in_place_scratch(coarse_near_null.ncols(), get_global_parallelism());
        let mut buf = MemBuffer::new(stack_req);
        let stack = MemStack::new(&mut buf);
        stationary.apply_in_place(
            coarse_near_null.as_mat_mut(),
            get_global_parallelism(),
            stack,
        );
        /*
        for triplet in interpolation_mat.triplet_iter() {
            println!("{}, {} : {:.1}", triplet.row, triplet.col, triplet.val);
        }
        */
        if coarse_mat.nrows() == fine_mat.nrows() {
            return Err(HierarchyError::CoarseningDidNotReduce);
        }

        info!(
            "Added level: {} -> {} DOFs ({} aggregates)",
            fine_mat.nrows(),
            coarse_mat.nrows(),
            partition.naggs()
        );

        self.coarse_ops.push(Arc::new(coarse_mat));
        self.restrictions.push(Arc::new(restriction_mat));
        self.interpolations.push(Arc::new(interpolation_mat));
        self.partitions.push(partition);
        self.near_nulls.push(coarse_near_null);
        Ok(self.current_mat().nrows())
        */
    }

    pub fn add_level(
        &mut self,
        coarse_op: SparseMatOp,
        partition: Arc<Partition>,
        near_null: Arc<Mat<f64>>,
        interpolation: Arc<SparseRowMat<usize, f64>>,
        restriction: Arc<SparseRowMat<usize, f64>>,
    ) {
        // interpolation and restriction should match previous operator
        assert_eq!(interpolation.nrows(), restriction.ncols());
        assert_eq!(interpolation.nrows(), self.current_mat_ref().nrows());

        // interpolation and restriction should match new operator
        assert_eq!(interpolation.ncols(), restriction.nrows());
        assert_eq!(interpolation.ncols(), coarse_op.mat_ref().ncols());

        self.operators.push(coarse_op);
        self.partitions.push(partition);
        self.restrictions.push(restriction);
        self.interpolations.push(interpolation);
        self.candidates.push(near_null);
    }

    pub fn get_config(&self) -> HierarchyConfig {
        self.config.clone()
    }

    pub fn levels(&self) -> usize {
        self.operators.len()
    }

    pub fn operators(&self) -> &[SparseMatOp] {
        &self.operators
    }

    pub fn get_op(&self, level: usize) -> SparseMatOp {
        self.operators[level].clone()
    }

    pub fn get_arc_mat(&self, level: usize) -> Arc<SparseRowMat<usize, f64>> {
        self.operators[level].arc_mat()
    }

    pub fn get_mat_ref(&self, level: usize) -> SparseRowMatRef<usize, f64> {
        self.operators[level].mat_ref()
    }

    pub fn current_op(&self) -> SparseMatOp {
        self.operators.last().unwrap().clone()
    }

    pub fn current_arc_mat(&self) -> Arc<SparseRowMat<usize, f64>> {
        self.operators.last().unwrap().arc_mat()
    }

    pub fn current_mat_ref(&self) -> SparseRowMatRef<usize, f64> {
        self.operators.last().unwrap().mat_ref()
    }

    pub fn partitions(&self) -> &[Arc<Partition>] {
        &self.partitions
    }

    pub fn get_partition(&self, level: usize) -> Arc<Partition> {
        self.partitions[level].clone()
    }

    pub fn restrictions(&self) -> &[Arc<SparseRowMat<usize, f64>>] {
        &self.restrictions
    }

    pub fn get_restriction(&self, level: usize) -> Arc<SparseRowMat<usize, f64>> {
        self.restrictions[level].clone()
    }

    pub fn interpolations(&self) -> &[Arc<SparseRowMat<usize, f64>>] {
        &self.interpolations
    }

    pub fn get_interpolation(&self, level: usize) -> Arc<SparseRowMat<usize, f64>> {
        self.interpolations[level].clone()
    }

    pub fn near_nulls(&self) -> &[Arc<Mat<f64>>] {
        &self.near_nulls
    }

    pub fn get_near_null(&self, level: usize) -> Arc<Mat<f64>> {
        self.near_nulls[level].clone()
    }

    pub fn op_complexity(&self) -> f64 {
        let total: usize = self
            .operators
            .iter()
            .map(|op| op.mat_ref().compute_nnz())
            .sum();
        let fine_nnz = self.operators[0].mat_ref().compute_nnz() as f64;
        total as f64 / fine_nnz
    }
}

/*
#[derive(Debug)]
pub enum HierarchyError {
    MissingPartitionStrategy,
    MissingNearNull,
    MissingBlockSize,
    InvalidCoarseBlockSize,
    InvalidInterpolationDim,
    CoarseningDidNotReduce,
}

impl fmt::Display for HierarchyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HierarchyError::MissingPartitionStrategy => {
                write!(f, "no partition strategy configured for hierarchy")
            }
            HierarchyError::MissingNearNull => {
                write!(f, "missing near-null space for current level")
            }
            HierarchyError::MissingBlockSize => write!(f, "missing block size for current level"),
            HierarchyError::InvalidCoarseBlockSize => {
                write!(f, "coarse block size must be positive")
            }
            HierarchyError::InvalidInterpolationDim => {
                write!(
                    f,
                    "interpolation dimension must be in 1..=near-null columns for current level"
                )
            }
            HierarchyError::CoarseningDidNotReduce => {
                write!(f, "coarsening did not reduce system dimension")
            }
        }
    }
}

impl Error for HierarchyError {}
*/
