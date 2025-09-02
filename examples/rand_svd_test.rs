use faer::{
    dyn_stack::{MemBuffer, MemStack},
    mat::Mat,
    stats::prelude::*,
    Col, Par,
};
use faer_amg::decompositions::rand_svd::rand_svd;

/// Generate test matrix A = U * Σ * V^T from random unitary matrices and exponentially decaying
/// singular values.
fn generate_test_matrix(
    m: usize,
    n: usize,
    decay_rate: f64,
) -> (Mat<f64>, Mat<f64>, Mat<f64>, Col<f64>) {
    let rng = &mut StdRng::seed_from_u64(42);

    // Generate random unitary matrices U (m x m) and V (n x n)
    let u_full = UnitaryMat {
        dim: m,
        standard_normal: StandardNormal,
    }
    .rand::<Mat<f64>>(rng);
    let v_full = UnitaryMat {
        dim: n,
        standard_normal: StandardNormal,
    }
    .rand::<Mat<f64>>(rng);

    // Take first `rank` columns of U and V
    let rank = m.min(n);
    let u = u_full.as_ref().subcols(0, rank).to_owned();
    let v = v_full.as_ref().subcols(0, rank).to_owned();

    // Generate exponentially decaying singular values
    let singular_values = Col::<f64>::from_iter((0..rank).map(|i| (-decay_rate * i as f64).exp()));

    let a = u.as_ref() * singular_values.as_diagonal() * v.transpose();

    (a, u, v, singular_values)
}

/// Calculate accuracy score based on recovered singular values and subspace alignment
fn calculate_accuracy_score(
    q: &Mat<f64>,
    svd_result: &faer::linalg::solvers::Svd<f64>,
    u_ref: &Mat<f64>,
    v_ref: &Mat<f64>,
    sigma_ref: &Col<f64>,
    k: usize,
) -> f64 {
    // Get truncated approximation: A_k ≈ Q * U_tilde * Σ * V^T
    let u_tilde = svd_result.U();
    let s_recovered = svd_result.S();
    let v_recovered = svd_result.V();

    // Construct the recovered left singular vectors: U_recovered = Q * U_tilde
    let u_recovered = q.as_ref() * u_tilde;

    // Calculate subspace alignment for U: ||U_ref^T * U_recovered||_F^2 / rank
    let u_projection = u_ref.as_ref().subcols(0, k).transpose() * u_recovered.as_ref();
    let u_alignment = u_projection.squared_norm_l2() / k as f64;

    // Calculate subspace alignment for V: ||V_ref^T * V_recovered||_F^2 / rank
    let v_projection = v_ref.as_ref().subcols(0, k).transpose() * v_recovered;
    let v_alignment = v_projection.squared_norm_l2() / k as f64;

    // Calculate singular value recovery ratio
    let recovered_sum: f64 = s_recovered.column_vector().iter().take(k).sum();
    let reference_sum: f64 = sigma_ref.as_ref().iter().take(k).sum();
    let sigma_ratio = recovered_sum / reference_sum;

    println!("U subspace alignment: {:.6}", u_alignment);
    println!("V subspace alignment: {:.6}", v_alignment);
    println!("Singular value recovery ratio: {:.6}", sigma_ratio);
    let ref_vals: Vec<f64> = sigma_ref.as_ref().iter().take(k.min(5)).cloned().collect();
    println!(
        "Reference singular values (first {}): {:?}",
        k.min(5),
        ref_vals
    );
    let recovered_vals: Vec<f64> = s_recovered
        .column_vector()
        .iter()
        .take(k.min(5))
        .cloned()
        .collect();
    println!(
        "Recovered singular values (first {}): {:?}",
        k.min(5),
        recovered_vals
    );

    // Combined score: geometric mean of alignments weighted by singular value recovery
    let subspace_score = (u_alignment * v_alignment).sqrt();
    subspace_score * sigma_ratio
}

fn main() {
    println!("=== Randomized SVD Test with Manufactured Solution ===\n");

    let m = 200; // Number of rows
    let n = 150; // Number of columns
    let k = 50; // Number of singular vectors to recover
    let decay_rate = 0.1; // Exponential decay rate for singular values

    println!("Matrix dimensions: {}x{}", m, n);
    println!("Target rank (k): {}", k);
    println!("Singular value decay rate: {}\n", decay_rate);

    // Generates test matrix with a manufactured solution
    let (a, u_ref, v_ref, sigma_ref) = generate_test_matrix(m, n, decay_rate);

    // Compute randomized SVD using the matrix directly (Mat<f64> implements BiLinOp)
    let par = Par::Seq;
    let stack_req = faer::dyn_stack::StackReq::new::<f64>(m * n + k * (m + n));
    let mut buf = MemBuffer::new(stack_req);
    let mut stack = MemStack::new(&mut buf);

    println!("Computing randomized SVD...");
    let result = rand_svd(&a, k, par, &mut stack);

    match result {
        Ok((q, svd_result)) => {
            let accuracy = calculate_accuracy_score(&q, &svd_result, &u_ref, &v_ref, &sigma_ref, k);
            println!("Accuracy Score: {:.4} / 1.0", accuracy);
        }
        Err(e) => {
            println!("Randomized SVD failed: {:?}", e);
        }
    }
}
