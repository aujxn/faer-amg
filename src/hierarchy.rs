use faer::{
    dyn_stack::{MemBuffer, MemStack},
    get_global_parallelism,
    mat::{AsMatMut, AsMatRef},
    matrix_free::{LinOp, Precond},
    sparse::{SparseRowMat, SparseRowMatRef},
    Mat,
};
use log::info;
use std::{fmt, sync::Arc};

use crate::{
    core::SparseMatOp,
    interpolation::{CoarseFineSplit, InterpolationConfig},
    partitioners::{Partition, PartitionStats},
    preconditioners::smoothers::{new_l1, StationaryIteration},
    utils::{matrix_stats, write_matrix_stats_table, NdofsFormat},
};

// TODO: add interpolation config once implemented
#[derive(Debug, Clone)]
pub struct HierarchyConfig {
    pub coarsest_dim: usize,
    pub interpolation_config: InterpolationConfig,
    pub max_levels: Option<usize>,
}

impl Default for HierarchyConfig {
    fn default() -> Self {
        Self {
            coarsest_dim: 1000,
            interpolation_config: InterpolationConfig::default(),
            max_levels: None,
        }
    }
}

impl HierarchyConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn build(
        &self,
        base_matrix: SparseMatOp,
        near_null: Arc<Mat<f64>>,
        nn_weights: Arc<Vec<f64>>,
    ) -> Hierarchy {
        let mut hierarchy = Hierarchy::new(base_matrix, near_null, nn_weights, self.clone());
        hierarchy.coarsen();
        hierarchy
    }
}

#[derive(Debug, Clone)]
pub enum PartitionType {
    Aggregation(Arc<Partition>),
    Classical(Arc<CoarseFineSplit>),
}

#[derive(Clone)]
pub struct Hierarchy {
    operators: Vec<SparseMatOp>,
    restrictions: Vec<Arc<SparseRowMat<usize, f64>>>,
    interpolations: Vec<Arc<SparseRowMat<usize, f64>>>,
    partitions: Vec<PartitionType>,
    near_nulls: Vec<Arc<Mat<f64>>>,
    nn_weights: Vec<Arc<Vec<f64>>>,
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

        /* TODO: handle both variants of `PartitionType`, this only does `partitioners::Partition`
        if !self.partitions.is_empty() {
            writeln!(f)?;
            let partition_stats: Vec<_> = self.partitions.iter().map(|p| p.info()).collect();
            write_partition_stats_table(f, "Partition summary", &partition_stats)?;
        }
        */

        writeln!(f, "Operator complexity: {:.3}", self.op_complexity())?;
        writeln!(f, "Grid complexity: {:.3}", self.grid_complexity())?;
        Ok(())
    }
}

impl Hierarchy {
    fn new(
        fine_op: SparseMatOp,
        fine_near_null: Arc<Mat<f64>>,
        nn_weights: Arc<Vec<f64>>,
        config: HierarchyConfig,
    ) -> Self {
        Self {
            operators: vec![fine_op],
            restrictions: Vec::new(),
            interpolations: Vec::new(),
            partitions: Vec::new(),
            near_nulls: vec![fine_near_null],
            nn_weights: vec![nn_weights],
            config,
        }
    }

    fn coarsen(&mut self) {
        let coarsest_dim = self.config.coarsest_dim;
        let max_levels = self.config.max_levels.unwrap_or(usize::MAX);

        let par = get_global_parallelism();

        let mut level = 1;
        let mut coarse_dim = usize::MAX;

        while coarse_dim > coarsest_dim && level < max_levels {
            let fine_mat = self.current_op();
            let near_null = self.near_nulls().last().unwrap().clone();
            let nn_weights = self.nn_weights.last().unwrap().clone();

            let coarse_galerkin = self.config.interpolation_config.build(
                fine_mat,
                near_null.as_ref().as_mat_ref(),
                nn_weights.as_ref(),
            );

            let block_size = match &self.config.interpolation_config {
                InterpolationConfig::Aggregation(agg_conf) => agg_conf.candidate_dimension,
                InterpolationConfig::Classical(_) => 1,
            };

            let coarse_op = SparseMatOp::new(coarse_galerkin.coarse_mat, block_size, par);
            coarse_dim = coarse_op.mat_ref().nrows();
            let mut coarse_nn = coarse_galerkin.coarse_nn;

            let l1_diag = Arc::new(new_l1(coarse_op.mat_ref()));
            let dyn_op = coarse_op.dyn_op();
            let stationary = StationaryIteration::new(dyn_op, l1_diag, 3);
            let stack_req =
                stationary.apply_in_place_scratch(coarse_nn.ncols(), get_global_parallelism());
            let mut buf = MemBuffer::new(stack_req);
            let stack = MemStack::new(&mut buf);
            stationary.apply_in_place(coarse_nn.as_mat_mut(), get_global_parallelism(), stack);

            coarse_nn = coarse_nn.qr().compute_thin_Q();

            let coarse_nn = Arc::new(coarse_nn);
            let p = Arc::new(coarse_galerkin.interpolation);
            let r = Arc::new(coarse_galerkin.restriction);

            /* TODO: two level only for now
            let max_dim = coarse_op.mat_ref().nrows();
            let near_null_dim = 128.min(max_dim);
            let smoother_block = 64.min(max_dim);
            let near_null = find_near_null(coarse_op.clone(), 100, near_null_dim, smoother_block);
            self.near_nulls.push(Arc::new(near_null));
            */
            self.add_level(coarse_op, coarse_galerkin.partition, coarse_nn, p, r);
            info!(
                "Created coarse op at level {}. Hierarchy:\n{:?}",
                level, &self
            );
            level += 1;
        }
    }

    pub fn add_level(
        &mut self,
        coarse_op: SparseMatOp,
        partition: PartitionType,
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
        self.near_nulls.push(near_null);
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

    pub fn get_mat_ref(&self, level: usize) -> SparseRowMatRef<'_, usize, f64> {
        self.operators[level].mat_ref()
    }

    pub fn current_op(&self) -> SparseMatOp {
        self.operators.last().unwrap().clone()
    }

    pub fn current_arc_mat(&self) -> Arc<SparseRowMat<usize, f64>> {
        self.operators.last().unwrap().arc_mat()
    }

    pub fn current_mat_ref(&self) -> SparseRowMatRef<'_, usize, f64> {
        self.operators.last().unwrap().mat_ref()
    }

    pub fn partitions(&self) -> &[PartitionType] {
        &self.partitions
    }

    pub fn get_partition(&self, level: usize) -> PartitionType {
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

    // TODO: make multilevel
    pub fn get_nn_weights(&self, level: usize) -> &Vec<f64> {
        self.nn_weights[level].as_ref()
    }

    pub fn grid_complexity(&self) -> f64 {
        let total: usize = self.operators.iter().map(|op| op.mat_ref().nrows()).sum();
        let fine_rows = self.operators[0].mat_ref().nrows() as f64;
        total as f64 / fine_rows
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
