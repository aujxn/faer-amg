use std::{error::Error, fmt, sync::Arc};

use faer::{
    linalg::solvers::LltError as DenseLltError,
    matrix_free::{BiPrecond, LinOp},
    sparse::SparseRowMat,
    Mat, MatRef,
};
use log::info;

use crate::{
    interpolation::smoothed_aggregation,
    partitioners::{Partition, PartitionBuilder, PartitionStats, PartitionerCallback},
    preconditioners::{
        block_smoothers::{BlockSmoother, BlockSmootherType},
        coarse_solvers::DenseCholeskySolve,
        multigrid::MultiGrid,
    },
    utils::{matrix_stats, MatrixStats},
};

#[derive(Clone)]
pub struct Hierarchy {
    fine_op: Arc<SparseRowMat<usize, f64>>,
    coarse_ops: Vec<Arc<SparseRowMat<usize, f64>>>,
    restrictions: Vec<Arc<SparseRowMat<usize, f64>>>,
    interpolations: Vec<Arc<SparseRowMat<usize, f64>>>,
    partitions: Vec<Arc<Partition>>,
    near_nulls: Vec<Mat<f64>>,
    block_sizes: Vec<usize>,
    partition_strategy: Option<PartitionStrategy>,
}

enum NdofsFormat {
    Auto,
    Rectangular,
}

fn write_matrix_stats_table(
    f: &mut fmt::Formatter<'_>,
    title: &str,
    stats: &[MatrixStats],
    ndofs_format: NdofsFormat,
) -> fmt::Result {
    if stats.is_empty() {
        writeln!(f, "{}: <empty>", title)?;
        return Ok(());
    }

    writeln!(f, "{title}")?;
    writeln!(
        f,
        "{:>4}  {:>12}  {:>10}  {:>26}  {:>26}  {:>26}",
        "lev", "ndofs", "sparsity", "entries / row", "weights", "rowsums"
    )?;
    writeln!(
        f,
        "{:-<4}  {:-<12}  {:-<10}  {:-<26}  {:-<26}  {:-<26}",
        "", "", "", "", "", ""
    )?;
    writeln!(
        f,
        "{:>4}  {:>12}  {:>10}  {:>8} {:>8} {:>8}  {:>8} {:>8} {:>8}  {:>8} {:>8} {:>8}",
        "", "", "", "min", "max", "avg", "min", "max", "avg", "min", "max", "avg"
    )?;
    writeln!(
        f,
        "{:-<4}  {:-<12}  {:-<10}  {:-<8} {:-<8} {:-<8}  {:-<8} {:-<8} {:-<8}  {:-<8} {:-<8} {:-<8}",
        "", "", "", "", "", "", "", "", "", "", "", ""
    )?;

    for (level, stat) in stats.iter().enumerate() {
        let ndofs = match ndofs_format {
            NdofsFormat::Rectangular => format!("{}x{}", stat.rows, stat.cols),
            NdofsFormat::Auto => {
                if stat.rows == stat.cols {
                    format!("{}", stat.rows)
                } else {
                    format!("{}x{}", stat.rows, stat.cols)
                }
            }
        };

        writeln!(
            f,
            "{:>4}  {:>12}  {:>10.2}  {:>8.2} {:>8.2} {:>8.2}  {:>8.2} {:>8.2} {:>8.2}  {:>8.2} {:>8.2} {:>8.2}",
            level,
            ndofs,
            stat.sparsity,
            stat.entries_min,
            stat.entries_max,
            stat.entries_avg,
            stat.weight_min,
            stat.weight_max,
            stat.weight_avg,
            stat.rowsum_min,
            stat.rowsum_max,
            stat.rowsum_avg
        )?;
    }

    Ok(())
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

        let mut operator_stats = Vec::with_capacity(levels);
        operator_stats.push(matrix_stats(self.fine_op.as_ref().as_ref()));
        operator_stats.extend(
            self.coarse_ops
                .iter()
                .map(|mat| matrix_stats(mat.as_ref().as_ref())),
        );
        write_matrix_stats_table(
            f,
            &format!("Hierarchy summary ({} levels)", levels),
            &operator_stats,
            NdofsFormat::Auto,
        )?;
        writeln!(f, "Block sizes: {:?}", self.block_sizes)?;

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
    pub fn new(
        fine_op: Arc<SparseRowMat<usize, f64>>,
        fine_near_null: Mat<f64>,
        fine_block_size: usize,
    ) -> Self {
        Self {
            fine_op,
            coarse_ops: Vec::new(),
            restrictions: Vec::new(),
            interpolations: Vec::new(),
            partitions: Vec::new(),
            near_nulls: vec![fine_near_null],
            block_sizes: vec![fine_block_size],
            partition_strategy: None,
        }
    }

    pub fn set_partition_strategy(&mut self, strategy: PartitionStrategy) {
        self.partition_strategy = Some(strategy);
    }

    pub fn partition_strategy(&self) -> Option<&PartitionStrategy> {
        self.partition_strategy.as_ref()
    }

    pub fn partition_strategy_mut(&mut self) -> Option<&mut PartitionStrategy> {
        self.partition_strategy.as_mut()
    }

    pub fn add_level(&mut self, params: AddLevelParams) -> Result<usize, HierarchyError> {
        if params.coarse_block_size == 0 {
            return Err(HierarchyError::InvalidCoarseBlockSize);
        }

        let fine_mat = self.current_mat();
        let fine_near_null = self
            .near_nulls
            .last()
            .ok_or(HierarchyError::MissingNearNull)?
            .as_ref();
        let fine_block_size = *self
            .block_sizes
            .last()
            .ok_or(HierarchyError::MissingBlockSize)?;

        let partition = match params.partition {
            Some(partition) => partition,
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
                let scalar_partition = strategy.build_partition(
                    fine_mat.clone(),
                    fine_block_size,
                    fine_near_null,
                    params.coarsening_factor,
                );
                partition.compose(&scalar_partition);
                partition
            }
        };
        let partition = Arc::new(partition);

        let (coarse_near_null, restriction_mat, interpolation_mat, coarse_mat) =
            smoothed_aggregation(
                fine_mat.as_ref().as_ref(),
                &partition,
                fine_block_size,
                fine_near_null,
            );

        if coarse_mat.nrows() == fine_mat.nrows() {
            return Err(HierarchyError::CoarseningDidNotReduce);
        }

        let coarse_near_null = coarse_near_null.qr().compute_thin_Q();

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
        self.block_sizes.push(params.coarse_block_size);
        Ok(self.current_mat().nrows())
    }

    pub fn levels(&self) -> usize {
        self.coarse_ops.len() + 1
    }

    pub fn coarse_level_count(&self) -> usize {
        self.coarse_ops.len()
    }

    pub fn is_empty(&self) -> bool {
        self.coarse_ops.is_empty()
    }

    pub fn get_mat(&self, level: usize) -> Arc<SparseRowMat<usize, f64>> {
        if level == 0 {
            self.fine_op.clone()
        } else {
            self.coarse_ops[level - 1].clone()
        }
    }

    pub fn current_mat(&self) -> Arc<SparseRowMat<usize, f64>> {
        if let Some(mat) = self.coarse_ops.last() {
            mat.clone()
        } else {
            self.fine_op.clone()
        }
    }

    pub fn fine_mat(&self) -> &Arc<SparseRowMat<usize, f64>> {
        &self.fine_op
    }

    pub fn block_size(&self, level: usize) -> usize {
        self.block_sizes[level]
    }

    pub fn current_block_size(&self) -> usize {
        *self
            .block_sizes
            .last()
            .expect("Hierarchy must track at least one block size")
    }

    pub fn partitions(&self) -> &[Arc<Partition>] {
        &self.partitions
    }

    pub fn get_partition(&self, level: usize) -> &Arc<Partition> {
        &self.partitions[level]
    }

    pub fn restrictions(&self) -> &[Arc<SparseRowMat<usize, f64>>] {
        &self.restrictions
    }

    pub fn interpolations(&self) -> &[Arc<SparseRowMat<usize, f64>>] {
        &self.interpolations
    }

    pub fn coarse_ops(&self) -> &[Arc<SparseRowMat<usize, f64>>] {
        &self.coarse_ops
    }

    pub fn near_nulls(&self) -> &[Mat<f64>] {
        &self.near_nulls
    }

    pub fn op_complexity(&self) -> f64 {
        let nnz: Vec<usize> = (0..self.levels())
            .map(|lvl| self.get_mat(lvl).compute_nnz())
            .collect();
        let fine_nnz = nnz
            .first()
            .copied()
            .unwrap_or_else(|| self.fine_op.compute_nnz());
        let total_nnz = nnz.into_iter().sum::<usize>() as f64;
        total_nnz / fine_nnz.max(1) as f64
    }
}

#[derive(Clone)]
pub struct PartitionStrategy {
    agg_size_penalty: f64,
    dist_penalty: f64,
    max_improvement_iters: usize,
    callback: Option<PartitionerCallback>,
}

impl PartitionStrategy {
    pub fn new(agg_size_penalty: f64, dist_penalty: f64, max_improvement_iters: usize) -> Self {
        Self {
            agg_size_penalty,
            dist_penalty,
            max_improvement_iters,
            callback: None,
        }
    }

    pub fn with_callback(mut self, callback: PartitionerCallback) -> Self {
        self.callback = Some(callback);
        self
    }

    pub fn set_callback(&mut self, callback: Option<PartitionerCallback>) {
        self.callback = callback;
    }

    pub fn build_partition(
        &self,
        mat: Arc<SparseRowMat<usize, f64>>,
        block_size: usize,
        near_null: MatRef<'_, f64>,
        coarsening_factor: f64,
    ) -> Partition {
        let mut builder = PartitionBuilder::new(
            mat,
            block_size,
            near_null,
            coarsening_factor,
            self.agg_size_penalty,
            self.dist_penalty,
            self.max_improvement_iters,
        );
        builder.callback = self.callback.clone();
        builder.build()
    }
}

#[derive(Clone, Debug)]
pub struct AddLevelParams {
    pub coarsening_factor: f64,
    pub coarse_block_size: usize,
    pub partition: Option<Partition>,
}

#[derive(Debug)]
pub enum HierarchyError {
    MissingPartitionStrategy,
    MissingNearNull,
    MissingBlockSize,
    InvalidCoarseBlockSize,
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
            HierarchyError::CoarseningDidNotReduce => {
                write!(f, "coarsening did not reduce system dimension")
            }
        }
    }
}

impl Error for HierarchyError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoarseSolverKind {
    DenseCholesky,
}

#[derive(Clone, Copy, Debug)]
pub struct MultigridBuilder {
    block_smoother: BlockSmootherType,
    coarse_solver: CoarseSolverKind,
    cycle_type: usize,
}

impl Default for MultigridBuilder {
    fn default() -> Self {
        Self {
            block_smoother: BlockSmootherType::SparseCholesky,
            coarse_solver: CoarseSolverKind::DenseCholesky,
            cycle_type: 1,
        }
    }
}

impl MultigridBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_block_smoother(mut self, smoother: BlockSmootherType) -> Self {
        self.block_smoother = smoother;
        self
    }

    pub fn with_coarse_solver(mut self, coarse_solver: CoarseSolverKind) -> Self {
        self.coarse_solver = coarse_solver;
        self
    }

    pub fn with_cycle_type(mut self, cycle_type: usize) -> Self {
        self.cycle_type = cycle_type.max(1);
        self
    }

    pub fn build(&self, hierarchy: &Hierarchy) -> Result<MultiGrid, MultigridBuildError> {
        if hierarchy.levels() == 0 {
            return Err(MultigridBuildError::EmptyHierarchy);
        }

        let level_count = hierarchy.levels();
        let mut smoothers: Vec<Arc<dyn BiPrecond<f64> + Send>> = Vec::with_capacity(level_count);

        for level in 0..level_count {
            if level + 1 == level_count {
                let op = hierarchy.get_mat(level);
                let solver = match self.coarse_solver {
                    CoarseSolverKind::DenseCholesky => {
                        DenseCholeskySolve::from_sparse(op.as_ref().as_ref())
                            .map_err(MultigridBuildError::CoarseSolveFailed)?
                    }
                };
                smoothers.push(Arc::new(solver));
            } else {
                let op = hierarchy.get_mat(level);
                let partition = Arc::clone(hierarchy.get_partition(level));
                let smoother = BlockSmoother::new(
                    op.as_ref().as_ref(),
                    partition,
                    self.block_smoother,
                    hierarchy.block_size(level),
                );
                smoothers.push(Arc::new(smoother));
            }
        }

        let finest_op = hierarchy.get_mat(0) as Arc<dyn LinOp<f64> + Send>;
        let mut multigrid = MultiGrid::new(finest_op, smoothers[0].clone());

        for level in 0..hierarchy.coarse_level_count() {
            let op = hierarchy.get_mat(level + 1) as Arc<dyn LinOp<f64> + Send>;
            let smoother = smoothers[level + 1].clone();
            let r = Arc::clone(&hierarchy.restrictions[level]) as Arc<dyn LinOp<f64> + Send>;
            let p = Arc::clone(&hierarchy.interpolations[level]) as Arc<dyn LinOp<f64> + Send>;
            multigrid.add_level(op, smoother, r, p);
        }

        if self.cycle_type != 1 {
            multigrid = multigrid.with_cycle_type(self.cycle_type);
        }

        Ok(multigrid)
    }
}

#[derive(Debug)]
pub enum MultigridBuildError {
    EmptyHierarchy,
    CoarseSolveFailed(DenseLltError),
}

impl fmt::Display for MultigridBuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MultigridBuildError::EmptyHierarchy => write!(f, "hierarchy has no levels"),
            MultigridBuildError::CoarseSolveFailed(_) => {
                write!(f, "failed to factorize coarsest grid for multigrid")
            }
        }
    }
}

impl Error for MultigridBuildError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            MultigridBuildError::CoarseSolveFailed(err) => Some(err),
            _ => None,
        }
    }
}
