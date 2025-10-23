use std::sync::Arc;

use faer::sparse::SparseRowMat;
use faer::Mat;
use log::info;

use crate::{
    decompositions::rand_svd::rand_svd,
    preconditioners::smoothers::{new_l1, StationaryIteration},
};

/// Builds an approximate near-null space by repeatedly smoothing random vectors and
/// extracting the dominant right singular vectors via randomized SVD.
pub fn smooth_vector_rand_svd(
    mat: Arc<SparseRowMat<usize, f64>>,
    iterations: usize,
    near_null_dim: usize,
) -> Mat<f64> {
    let l1_diag = Arc::new(new_l1(mat.as_ref().as_ref()));

    let stationary = StationaryIteration::new(mat, l1_diag, iterations);
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
