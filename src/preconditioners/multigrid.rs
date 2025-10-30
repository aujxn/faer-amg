use crate::{
    hierarchy::Hierarchy,
    partitioners::{Partition, PartitionerConfig},
    preconditioners::{block_smoothers::BlockSmootherConfig, coarse_solvers::CoarseSolverKind},
};
use faer::{
    dyn_stack::{MemStack, StackReq},
    matrix_free::{BiLinOp, BiPrecond, LinOp, Precond},
    reborrow::*,
    Mat, MatMut, MatRef, Par,
};
use std::sync::Arc;

/// This builder helps construct multigrid preconditioners for basic configurations. The number of
/// smoothing steps and the smoother configuration is the same for all levels in the hierarchy.
///
/// If `coarse_solver` is `None` then the same smoothing is applied on coarsest level instead of
/// solving exactly. Default is Cholesky decomposition.
///
/// For more custom / complex configurations the `Multigrid::add_level` API can be used directly.
#[derive(Debug, Clone)]
pub struct MultigridConfig {
    pub mu: usize,
    pub smoother_config: BlockSmootherConfig,
    pub smoothing_steps: usize,
    pub coarse_solver: Option<CoarseSolverKind>,
}

impl Default for MultigridConfig {
    fn default() -> Self {
        Self {
            mu: 1,
            smoother_config: BlockSmootherConfig::default(),
            smoothing_steps: 1,
            coarse_solver: Some(CoarseSolverKind::Cholesky),
        }
    }
}

impl MultigridConfig {
    pub fn new() -> Self {
        Self::default()
    }

    /// Constructs a `Multigrid` operator from this config.
    pub fn build(&self, hierarchy: Hierarchy) -> Multigrid {
        /*
        pub fn build(&self, hierarchy: Hierarchy) -> Result<Multigrid, MultigridBuildError> {
            if hierarchy.levels() == 0 {
                return Err(MultigridBuildError::EmptyHierarchy);
            }
        */

        let level_count = hierarchy.levels();
        let mut smoothers: Vec<Arc<dyn BiPrecond<f64> + Send>> = Vec::with_capacity(level_count);

        /*
        let fine_near_null = hierarchy.get_near_null(0);
        let fine_op = hierarchy.get_op(0);
        let mut partition_config = hierarchy.get_config().partitioner_config.clone();
        partition_config.coarsening_factor = 128.0;
        partition_config.max_improvement_iters = 1000;
        //partition_config.agg_size_penalty = 0.0;
        let mut partitioner = partition_config.build(fine_op, fine_near_null, None);
        let mut smoother_partitions = vec![Arc::new(partitioner.get_partition().clone())];
        for part in hierarchy.partitions() {
            partitioner.rebase(part.as_ref().clone());
            smoother_partitions.push(Arc::new(partitioner.get_partition().clone()));
        }
        */
        let mut partition_config = hierarchy.get_config().partitioner_config.clone();
        partition_config.coarsening_factor = 128.0;
        partition_config.max_improvement_iters = 300;
        let mut smoother_partitions = vec![];
        for level in 0..(level_count - 1) {
            let op = hierarchy.get_op(level);
            let near_null = hierarchy.get_near_null(level);
            let partition = partition_config.build_partition(op, near_null);
            smoother_partitions.push(Arc::new(partition));
        }

        for level in 0..level_count {
            let smoother: Arc<dyn BiPrecond<f64> + Send> =
                if level + 1 == level_count && self.coarse_solver.is_some() {
                    let op = hierarchy.get_mat_ref(level);
                    self.coarse_solver.unwrap().build_from_sparse(op)
                } else {
                    assert_ne!(level, level_count - 1);
                    let op = hierarchy.get_op(level);
                    Arc::new(
                        self.smoother_config
                            .build(op, smoother_partitions[level].clone()),
                    )
                };
            smoothers.push(smoother);
        }

        let finest_op = hierarchy.get_op(0);
        let spmm_op = match finest_op.par_op() {
            Some(par_op) => par_op as Arc<dyn LinOp<f64> + Send>,
            None => finest_op.arc_mat(),
        };
        let mut multigrid = Multigrid::new(spmm_op, smoothers[0].clone());

        for level in 1..level_count {
            let op = hierarchy.get_op(level);
            let spmm_op = match op.par_op() {
                Some(par_op) => par_op as Arc<dyn LinOp<f64> + Send>,
                None => finest_op.arc_mat(),
            };
            let smoother = smoothers[level].clone();
            let r = hierarchy.get_restriction(level - 1);
            let p = hierarchy.get_interpolation(level - 1);
            multigrid.add_level(spmm_op, smoother, r, p);
        }
        multigrid.with_cycle_type(self.mu)
    }
}

/// An abstract multigrid preconditioner. See `add_level` for manual construction details or use
/// `MultigridConfig` as a builder for a standard AMG setup. Example `simple_geometric` shows the
/// construction for a 1d finite difference for the negative second derivative and example `amg`
/// shows usage for an SPD matrix system loaded from disk.
#[derive(Debug, Clone)]
pub struct Multigrid {
    operators: Vec<Arc<dyn LinOp<f64> + Send>>,
    smoothers: Vec<Arc<dyn BiPrecond<f64> + Send>>,
    interpolations: Vec<Arc<dyn LinOp<f64> + Send>>,
    restrictions: Vec<Arc<dyn LinOp<f64> + Send>>,
    cycle_type: usize,
}

const DEBUG: bool = false;

impl Multigrid {
    /// Initializes a `Multigrid` with only a single level. `smoother` should (quickly) approximate
    /// the inverse of `op`. When using this as a preconditioner for a solver such as conjugate
    /// gradient, `op` here is the same operator as the one used in the system solver.
    ///
    /// Currently only symmetric multigrid is supported, so calling `apply_transpose` will simply
    /// call `apply` but this is subject to change.
    pub fn new(op: Arc<dyn LinOp<f64> + Send>, smoother: Arc<dyn BiPrecond<f64> + Send>) -> Self {
        Self {
            operators: vec![op],
            smoothers: vec![smoother],
            interpolations: Vec::new(),
            restrictions: Vec::new(),
            cycle_type: 1,
        }
    }

    /// Calls the next level `mu` times at each level in the hierarchy. In the literature, `mu=1`
    /// is called a V-cycle, `mu=2` a W-cycle, and `mu > 2` a mu-cycle. Algorithmic complexity
    /// scales $O(\ell^\mu)$ where the grid has $\ell$ levels.
    pub fn with_cycle_type(mut self, mu: usize) -> Self {
        self.cycle_type = mu;
        self
    }

    /// Adds a level to the hierarchy. Currently only symmetric multigrid is supported, so `r` and
    /// `p` should be transpose operations of each other but this is subject to change.
    /// Additionally, `r.ncols()` and `p.nrows()` must match the size of the last operator added
    /// and `r.nrows()` and `p.ncols()` must match the size of the current `op` argument.
    /// Traditionally these operators satisfy the 'Galerkin assembly': `op = r * previous_op * p`,
    /// but this is not a strict requirement.
    ///
    /// `smoother` should quickly approximate the inverse of `op`. Although only symmetric
    /// multigrid is supported does *NOT* mean that `smoother` must be symmetric. For example, the
    /// `GaussSeidel` smoother is not a symmetric operator but the resulting multigrid is since
    /// `smoother.apply` is called for forward smoothing (down) and `smoother.apply_transpose` is
    /// called for backward smoothing (up).
    pub fn add_level(
        &mut self,
        op: Arc<dyn LinOp<f64> + Send>,
        smoother: Arc<dyn BiPrecond<f64> + Send>,
        r: Arc<dyn LinOp<f64> + Send>,
        p: Arc<dyn LinOp<f64> + Send>,
    ) {
        self.operators.push(op);
        self.smoothers.push(smoother);
        self.interpolations.push(p);
        self.restrictions.push(r);
    }

    /// How many levels in the underlying grid / hierarchy
    pub fn levels(&self) -> usize {
        self.operators.len()
    }

    /// See `with_cycle_type`
    pub fn cycle_type(&self) -> usize {
        self.cycle_type
    }

    fn init_cycle(
        &self,
        mut out: MatMut<'_, f64>,
        rhs: MatRef<'_, f64>,
        par: Par,
        stack: &mut MemStack,
    ) {
        let mut v = Mat::zeros(out.nrows(), out.ncols());
        self.cycle(v.as_mut(), rhs, 0, par, stack);
        out += v;
    }

    fn cycle(
        &self,
        mut v: MatMut<'_, f64>,
        f: MatRef<'_, f64>,
        level: usize,
        par: Par,
        stack: &mut MemStack,
    ) {
        let smoother = &self.smoothers[level];
        if level == self.operators.len() - 1 {
            smoother.apply(v, f, par, stack);
            return;
        }
        let mut work = Mat::zeros(v.nrows(), v.ncols());
        let op = &self.operators[level];

        forward_smooth(v.rb_mut(), f, op.clone(), smoother.clone(), par, stack, 1);
        if DEBUG {
            op.apply(work.rb_mut(), v.as_ref(), par, stack);
            work = f - work.rb();
            let norm = work.norm_l2();
            for _ in 0..level + 1 {
                print!("\t");
            }
            println!("{:?}", norm);
        }

        if level < self.operators.len() - 1 {
            let restrict = &self.restrictions[level];
            let interp = &self.interpolations[level];
            let mut v_coarse = Mat::zeros(restrict.nrows(), v.ncols());
            let mut f_coarse = Mat::zeros(restrict.nrows(), v.ncols());

            // compute fine residual and restrict to coarse residual
            op.apply(work.rb_mut(), v.as_ref(), par, stack);
            work = f - work.rb();
            restrict.apply(f_coarse.as_mut(), work.as_ref(), par, stack);

            for _ in 0..self.cycle_type {
                self.cycle(v_coarse.as_mut(), f_coarse.as_ref(), level + 1, par, stack);
            }

            interp.apply(work.rb_mut(), v_coarse.as_ref(), par, stack);
            v += &work;
            backward_smooth(v.rb_mut(), f, op.clone(), smoother.clone(), par, stack, 1);
            if DEBUG {
                op.apply(work.rb_mut(), v.as_ref(), par, stack);
                work = f - work.rb();
                let norm = work.norm_l2();
                for _ in 0..level + 1 {
                    print!("\t");
                }
                println!("{:?}", norm);
            }
        }
    }
}

fn forward_smooth(
    x: MatMut<'_, f64>,
    b: MatRef<'_, f64>,
    op: Arc<dyn LinOp<f64>>,
    pc: Arc<dyn BiPrecond<f64>>,
    par: Par,
    stack: &mut MemStack,
    max_iter: usize,
) {
    let mut work = Mat::zeros(x.nrows(), x.ncols());
    let mut x = x;
    // first iteration of forward `x` is 0 so residual is `b`
    pc.apply(x.rb_mut(), b, par, stack);
    for _ in 1..max_iter {
        op.apply(work.rb_mut(), x.rb(), par, stack);
        let mut r = b - &work;
        pc.apply_in_place(r.rb_mut(), par, stack);
        x += r;
    }
}

fn backward_smooth(
    x: MatMut<'_, f64>,
    b: MatRef<'_, f64>,
    op: Arc<dyn LinOp<f64>>,
    pc: Arc<dyn BiPrecond<f64>>,
    par: Par,
    stack: &mut MemStack,
    max_iter: usize,
) {
    let mut work = Mat::zeros(x.nrows(), x.ncols());
    let mut x = x;
    for _ in 0..max_iter {
        op.apply(work.rb_mut(), x.rb(), par, stack);
        let mut r = b - &work;
        pc.transpose_apply_in_place(r.rb_mut(), par, stack);
        x += r;
    }
}

impl LinOp<f64> for Multigrid {
    // TODO: low level API
    fn apply_scratch(&self, rhs_ncols: usize, par: Par) -> StackReq {
        let _rhs_ncols = rhs_ncols;
        let _par = par;
        StackReq::EMPTY
    }

    fn nrows(&self) -> usize {
        if self.operators.is_empty() {
            unreachable!("Cannot determine dimension of partially uninitialized Multigrid.");
        }
        self.operators[0].nrows()
    }

    fn ncols(&self) -> usize {
        if self.operators.is_empty() {
            unreachable!("Cannot determine dimension of partially uninitialized Multigrid.");
        }
        self.operators[0].ncols()
    }

    fn apply(&self, out: MatMut<'_, f64>, rhs: MatRef<'_, f64>, par: Par, stack: &mut MemStack) {
        let mut out = out;
        out.fill(0.0);
        self.init_cycle(out, rhs, par, stack);
    }

    fn conj_apply(
        &self,
        out: MatMut<'_, f64>,
        rhs: MatRef<'_, f64>,
        par: Par,
        stack: &mut MemStack,
    ) {
        // TODO: only real multigrid for now
        self.apply(out, rhs, par, stack);
    }
}

impl BiLinOp<f64> for Multigrid {
    fn transpose_apply_scratch(&self, rhs_ncols: usize, par: Par) -> StackReq {
        // TODO : only symmetric multigrid
        self.apply_scratch(rhs_ncols, par)
    }

    fn transpose_apply(
        &self,
        out: MatMut<'_, f64>,
        rhs: MatRef<'_, f64>,
        par: Par,
        stack: &mut MemStack,
    ) {
        // TODO : only symmetric multigrid
        self.apply(out, rhs, par, stack);
    }

    fn adjoint_apply(
        &self,
        out: MatMut<'_, f64>,
        rhs: MatRef<'_, f64>,
        par: Par,
        stack: &mut MemStack,
    ) {
        // TODO : only symmetric multigrid
        self.conj_apply(out, rhs, par, stack);
    }
}

// TODO: auto impl are fine for now but custom would be better
impl Precond<f64> for Multigrid {}
impl BiPrecond<f64> for Multigrid {}

/*
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
*/
