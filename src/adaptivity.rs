use std::sync::Arc;

use faer::stats::prelude::StandardNormal;
use faer::stats::DistributionExt;
use faer::Mat;
use faer::{sparse::SparseRowMat, stats::CwiseMatDistribution};
use log::info;
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::core::SparseMatOp;
use crate::decompositions::rand_svd::rand_svd;
use crate::hierarchy::HierarchyConfig;
use crate::par_spmv::ParSpmmOp;
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
    pub interp_candidate_dim: usize,
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
            interp_candidate_dim: 4,
            coarsening_near_null_dim: 32,
            include_constant_first_near_null: true,
        }
    }
}

impl AdaptiveConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn build(&self) -> Composite {
        unimplemented!()
    }
}

/// Builds an approximate near-null space by repeatedly smoothing random vectors and
/// extracting the dominant right singular vectors via randomized SVD.
pub fn smooth_vector_rand_svd(
    op: SparseMatOp,
    iterations: usize,
    near_null_dim: usize,
) -> Mat<f64> {
    let l1_diag = Arc::new(new_l1(op.mat_ref()));

    let par_op = op.par_op().unwrap();
    let stationary = StationaryIteration::new(par_op, l1_diag, iterations);
    //let stationary = StationaryIteration::new(mat, l1_diag, iterations);
    let (_u, s, v) = rand_svd(stationary, near_null_dim);

    let convergence_factors: String = s
        .into_column_vector()
        .iter()
        .map(|singular_value| format!("{:.2} ", singular_value.powf((iterations as f64).recip())))
        .collect();
    info!(
        "near-null smoothing convergence factors: {}",
        convergence_factors
    );

    v
}

pub fn smooth_vector(
    mat: Arc<SparseRowMat<usize, f64>>,
    iterations: usize,
    near_null_dim: usize,
) -> Mat<f64> {
    let mat_ref = mat.as_ref();
    let l1_diag = new_l1(mat_ref.as_ref().as_ref());

    // could just reuse this...
    //let stationary = StationaryIteration::new(mat, l1_diag, iterations);

    let n = mat.nrows();

    let rng = &mut StdRng::seed_from_u64(42);
    let mut x = CwiseMatDistribution {
        nrows: n,
        ncols: near_null_dim,
        dist: StandardNormal,
    }
    .rand::<Mat<f64>>(rng);
    let mut r: Mat<f64>;

    for _ in 0..=iterations {
        x = x.qr().compute_thin_Q();
        r = mat.as_ref().as_ref() * &x;
        r = &l1_diag * r;
        x -= r;
    }

    let convergence_factors: String = x
        .col_iter()
        .map(|col| format!("{:.2} ", col.norm_l2()))
        .collect();
    info!(
        "converge factor after {} iterations: {}",
        iterations, convergence_factors
    );

    x
}
