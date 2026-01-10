use std::sync::Arc;
use std::time::Instant;

use faer::diag::Diag;
use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::linalg::matmul::dot::inner_prod;
use faer::linalg::temp_mat_scratch;
use faer::matrix_free::{BiLinOp, BiPrecond, LinOp, Precond};
use faer::prelude::ReborrowMut;
use faer::stats::prelude::StandardNormal;
use faer::stats::{CwiseMatDistribution, DistributionExt};
use faer::{get_global_parallelism, Col, ColRef, Mat, MatMut, MatRef, Par};
use log::info;
use rand::rng;

use crate::core::SparseMatOp;
use crate::decompositions::rand_svd::rand_svd;
use crate::hierarchy::HierarchyConfig;
use crate::partitioners::PartitionerConfig;
use crate::preconditioners::block_smoothers::BlockSmootherConfig;
use crate::preconditioners::composite::Composite;
use crate::preconditioners::multigrid::MultigridConfig;
use crate::preconditioners::smoothers::{new_l1, StationaryIteration};

#[derive(Clone, Debug)]
pub struct AdaptiveConfig {
    pub hierarchy_config: HierarchyConfig,
    pub multigrid_config: MultigridConfig,
    pub target_convergence: Option<f64>,
    pub max_components: usize,
    pub test_iters: usize,
    pub coarsening_near_null_dim: usize,
    pub include_constant_first_near_null: bool,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            hierarchy_config: HierarchyConfig::default(),
            multigrid_config: MultigridConfig::default(),
            target_convergence: None,
            max_components: 5,
            test_iters: 50,
            coarsening_near_null_dim: 32,
            include_constant_first_near_null: true,
        }
    }
}

impl AdaptiveConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn build(&self, mat: SparseMatOp) -> Composite {
        let l1_diag = Arc::new(new_l1(mat.mat_ref()));
        let oversample = 5;
        let par = get_global_parallelism();

        let mut error_prop = ErrorPropogator {
            op: mat.dyn_op(),
            pc: l1_diag,
        };
        let (_u, s, smoothed) = rand_svd(
            &error_prop,
            self.coarsening_near_null_dim - 1,
            oversample,
            self.test_iters,
        );
        report_convergence(s.column_vector());

        let mut with_constant = Mat::ones(smoothed.nrows(), self.coarsening_near_null_dim);
        with_constant
            .subcols_mut(1, self.coarsening_near_null_dim - 1)
            .copy_from(smoothed);
        let basis = with_constant.qr().compute_thin_Q();

        let initial_near_null = Arc::new(basis);
        let initial_hierarchy = self.hierarchy_config.build(mat.clone(), initial_near_null);
        info!("Hierarchy 1 info:\n{:?}", initial_hierarchy);
        let first_component = self.multigrid_config.build(initial_hierarchy);
        let mut composite = Composite::new(mat.dyn_op(), Arc::new(first_component));

        for n_components in 1..self.max_components {
            info!(
                "Constructed {} components. Testing convergence...",
                n_components
            );

            error_prop.pc = Arc::new(composite.clone());
            let (u, s, smoothed) = rand_svd(
                &error_prop,
                self.coarsening_near_null_dim,
                oversample,
                self.test_iters,
            );
            report_convergence(s.column_vector());
            let mut mem = MemBuffer::new(StackReq::any_of(&[
                error_prop.apply_scratch(self.coarsening_near_null_dim, par),
                error_prop.transpose_apply_scratch(self.coarsening_near_null_dim, par),
            ]));
            let stack = MemStack::new(&mut mem);
            let mut maybe_u = Mat::zeros(u.nrows(), u.ncols());
            error_prop.apply(maybe_u.as_mut(), smoothed.as_ref(), par, stack);
            let maybe_s = Col::from_fn(u.ncols(), |i| maybe_u.col(i).norm_l2());
            report_convergence(maybe_s.as_ref());
            let maybe_s = Col::from_fn(u.ncols(), |i| {
                inner_prod(
                    u.col(i).adjoint(),
                    faer::Conj::No,
                    maybe_u.col(i),
                    faer::Conj::No,
                )
            });
            report_convergence(maybe_s.as_ref());

            let near_null = Arc::new(smoothed);
            let hierarchy = self.hierarchy_config.build(mat.clone(), near_null);
            info!("Hierarchy {} info:\n{:?}", n_components + 1, hierarchy);
            let component = self.multigrid_config.build(hierarchy);
            composite.push(Arc::new(component));
        }

        composite
    }
}

#[derive(Clone, Debug)]
pub struct ErrorPropogator {
    pub op: Arc<dyn LinOp<f64> + Send>,
    pub pc: Arc<dyn LinOp<f64> + Send>,
}

impl LinOp<f64> for ErrorPropogator {
    fn apply_scratch(&self, rhs_ncols: usize, par: Par) -> StackReq {
        let req = StackReq::all_of(&[temp_mat_scratch::<f64>(self.nrows(), rhs_ncols); 3]);
        req.and(StackReq::any_of(&[
            self.op.apply_scratch(rhs_ncols, par),
            self.pc.apply_scratch(rhs_ncols, par),
        ]))
    }

    fn nrows(&self) -> usize {
        self.op.nrows()
    }

    fn ncols(&self) -> usize {
        self.pc.ncols()
    }

    fn apply(&self, out: MatMut<'_, f64>, rhs: MatRef<'_, f64>, par: Par, stack: &mut MemStack) {
        let mut work = Mat::zeros(out.nrows(), out.ncols());
        let mut out = out;
        self.op.apply(out.rb_mut(), rhs.as_ref(), par, stack);
        self.pc.apply(work.rb_mut(), out.as_ref(), par, stack);
        out.copy_from(rhs);
        out -= work;
    }

    fn conj_apply(
        &self,
        out: MatMut<'_, f64>,
        rhs: MatRef<'_, f64>,
        par: Par,
        stack: &mut MemStack,
    ) {
        self.apply(out, rhs, par, stack);
    }
}

impl BiLinOp<f64> for ErrorPropogator {
    fn transpose_apply_scratch(&self, rhs_ncols: usize, par: Par) -> StackReq {
        self.apply_scratch(rhs_ncols, par)
    }

    fn transpose_apply(
        &self,
        out: MatMut<'_, f64>,
        rhs: MatRef<'_, f64>,
        par: Par,
        stack: &mut MemStack,
    ) {
        //self.apply(out, rhs, par, stack);
        let mut work = Mat::zeros(out.nrows(), out.ncols());
        let mut out = out;
        self.pc.apply(out.rb_mut(), rhs.as_ref(), par, stack);
        self.op.apply(work.rb_mut(), out.as_ref(), par, stack);
        out.copy_from(rhs);
        out -= work;
    }

    fn adjoint_apply(
        &self,
        out: MatMut<'_, f64>,
        rhs: MatRef<'_, f64>,
        par: Par,
        stack: &mut MemStack,
    ) {
        self.transpose_apply(out, rhs, par, stack);
    }
}

impl Precond<f64> for ErrorPropogator {}
impl BiPrecond<f64> for ErrorPropogator {}

fn report_convergence(s: ColRef<f64>) {
    let convergence_factors: String = s
        .iter()
        .map(|singular_value| format!("{:.3} ", singular_value))
        .collect();
    info!("sampled convergence factors: {}", convergence_factors);
}

/// Builds an approximate near-null space by repeatedly smoothing random vectors and
/// extracting the dominant right singular vectors via randomized SVD.
pub fn smooth_vector_rand_svd(
    op: SparseMatOp,
    iterations: usize,
    near_null_dim: usize,
) -> Mat<f64> {
    let l1_diag = Arc::new(new_l1(op.mat_ref()));

    let error_p = ErrorPropogator {
        op: op.dyn_op(),
        pc: l1_diag,
    };
    let (_u, s, v) = rand_svd(error_p, near_null_dim, 10, iterations);
    report_convergence(s.column_vector());

    v
}

pub fn find_near_null(
    mat: SparseMatOp,
    iterations: usize,
    near_null_dim: usize,
    smoothing_block_size: usize,
) -> Mat<f64> {
    let simple_pc = new_l1(mat.mat_ref());
    let (smooth_basis, _singular_values) =
        smooth_vector(mat.clone(), Arc::new(simple_pc), iterations, near_null_dim);

    let partitioner_config = PartitionerConfig {
        coarsening_factor: smoothing_block_size as f64,
        max_improvement_iters: 50,
        ..Default::default()
    };
    let block_smoother_config = BlockSmootherConfig {
        partitioner_config,
        ..Default::default()
    };
    let block_pc = block_smoother_config.build(mat.clone(), Arc::new(smooth_basis));
    let (smooth_basis, _singular_values) =
        smooth_vector(mat.clone(), Arc::new(block_pc), iterations, near_null_dim);
    smooth_basis
}

fn smooth_vector(
    mat: SparseMatOp,
    pc: Arc<dyn LinOp<f64> + Send>,
    iterations: usize,
    near_null_dim: usize,
) -> (Mat<f64>, Diag<f64>) {
    let iteration = ErrorPropogator {
        op: mat.dyn_op(),
        pc,
    };

    let n = mat.mat_ref().nrows();

    let rng = &mut rng();
    let mut x = CwiseMatDistribution {
        nrows: n,
        ncols: near_null_dim * 2,
        dist: StandardNormal,
    }
    .rand::<Mat<f64>>(rng);
    x = x.qr().compute_thin_Q();

    let par = get_global_parallelism();
    let stack_req = iteration.apply_in_place_scratch(x.ncols(), par);
    let mut buf = MemBuffer::new(stack_req);
    let stack = MemStack::new(&mut buf);
    let start = Instant::now();
    info!("Starting search for smooth vectors");
    let pre_smooth_iters = iterations.saturating_sub(5);
    for _ in 0..pre_smooth_iters {
        iteration.apply_in_place(x.as_mut(), par, stack);
    }
    let duration = Instant::now() - start;
    info!(
        "{:.2e} secs per error prop iter",
        duration.as_secs_f64() / pre_smooth_iters as f64
    );
    x = x.qr().compute_thin_Q();

    for _ in 0..std::cmp::min(iterations, 5) {
        iteration.apply_in_place(x.as_mut(), par, stack);
    }
    let svd = x.thin_svd().unwrap();

    let u = svd.U().subcols(0, near_null_dim).to_owned();
    let s = svd
        .S()
        .column_vector()
        .subrows(0, near_null_dim)
        .to_owned()
        .into_diagonal();

    let duration = Instant::now() - start;
    info!(
        "Finished smooth vec search in {} seconds",
        duration.as_secs()
    );
    (u, s)
}
