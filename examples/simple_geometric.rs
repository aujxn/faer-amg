use core::f64;
use std::sync::Arc;

use faer::{
    dyn_stack::{MemBuffer, MemStack},
    mat::generic::Mat,
    matrix_free::{
        conjugate_gradient::{
            conjugate_gradient, conjugate_gradient_scratch, CgError, CgInfo, CgParams,
        },
        BiPrecond, LinOp, Precond,
    },
    prelude::{Reborrow, ReborrowMut},
    sparse::{SparseRowMat, Triplet},
    Col, MatMut, MatRef, Par,
};
use faer_amg::preconditioners::{
    multigrid::MultiGrid,
    smoothers::{new_jacobi, CholeskySolve},
};

/// # 1D Geometric Multigrid Example
///
/// This example demonstrates a geometric multigrid preconditioner for solving the
/// one-dimensional Poisson equation:
///
/// ```text
/// -u'' = 1   on the interval (0,1)
/// u(0) = u(1) = 0   (homogeneous Dirichlet boundary conditions)
/// ```
///
/// The problem is discretized using a finite difference method on uniformly spaced grids,
/// creating a symmetric positive definite tridiagonal linear system. The multigrid method
/// uses:
///
/// - **Prolongation (Interpolation)**: Linear interpolation that averages neighboring
///   coarse grid values to estimate fine grid values
/// - **Restriction**: Full weighting that satisfies the variational property, making it
///   suitable for use with conjugate gradient methods
/// - **Smoothers**: Weighted Jacobi iterations on intermediate levels and direct Cholesky
///   solve on the coarsest level
/// - **Cycle**: V-cycle multigrid iterations
///
/// The example compares four solution approaches:
/// 1. PCG with simple weighted Jacobi preconditioning
/// 2. Stationary iteration with weighted Jacobi
/// 3. PCG with multigrid preconditioning  
/// 4. Stationary iteration with multigrid
///
/// The multigrid method should demonstrate mesh-independent convergence, solving the
/// system in a small number of iterations regardless of the fine grid resolution.

/* TODO:
* - Use analytic solution for error computation
*/

const LENGTH: f64 = 1.0;

/// Creates sparse matrix prolongation operator defined by linear interpolation stencil.
/// - new grid points are added between each pair of coarse points
/// - the new grid point values are the average of the existing neighbors
fn make_interpolation(n_coarse: usize) -> SparseRowMat<usize, f64> {
    let n_fine = 2 * n_coarse + 1;
    let mut triplets: Vec<Triplet<_, _, _>> = Vec::new();
    let scale = 1.0 / 2.0_f64;

    for col in 0..n_coarse {
        let row_start = 2 * col;
        triplets.push(Triplet::new(row_start, col, 1.0 * scale));
        triplets.push(Triplet::new(row_start + 1, col, 2.0 * scale));
        triplets.push(Triplet::new(row_start + 2, col, 1.0 * scale));
    }

    SparseRowMat::try_new_from_triplets(n_fine, n_coarse, &triplets).unwrap()
}

/// Creates sparse matrix restriction operator (full weighting).
/// - satisfies variational property with linear interpolation operator (off by a scalar multiple
/// of its transpose, needed for pcg)
fn make_restriction(n_coarse: usize) -> SparseRowMat<usize, f64> {
    let n_fine = 2 * n_coarse + 1;
    let mut triplets: Vec<Triplet<_, _, _>> = Vec::new();
    let scale = 1.0 / 4.0_f64;

    for row in 0..n_coarse {
        let col_start = 2 * row;
        triplets.push(Triplet::new(row, col_start, 1.0 * scale));
        triplets.push(Triplet::new(row, col_start + 1, 2.0 * scale));
        triplets.push(Triplet::new(row, col_start + 2, 1.0 * scale));
    }

    SparseRowMat::try_new_from_triplets(n_coarse, n_fine, &triplets).unwrap()
}

/// Creates the finite difference matrix for computing the negative second derivative.
fn make_finite_difference(n_elements: usize) -> SparseRowMat<usize, f64> {
    let mut triplets: Vec<Triplet<_, _, _>> = Vec::new();
    let h = LENGTH / n_elements as f64;
    let diag_val = 2.0 / (h * h); // Second derivative finite difference
    let off_diag_val = -1.0 / (h * h);

    let n_dofs = n_elements - 1; // Interior points only (homogeneous Dirichlet BC)
    for i in 0..n_dofs {
        triplets.push(Triplet::new(i, i, diag_val));
        if i > 0 {
            triplets.push(Triplet::new(i, i - 1, off_diag_val));
        }
        if i + 1 < n_dofs {
            triplets.push(Triplet::new(i, i + 1, off_diag_val));
        }
    }
    SparseRowMat::try_new_from_triplets(n_dofs, n_dofs, &triplets).unwrap()
}

/// An iterative solver based on stationary linear iterations.
/// Returns the number of iterations taken.
fn stationary_solver(
    x: MatMut<f64>,
    b: MatRef<f64>,
    op: Arc<dyn LinOp<f64>>,
    pc: impl Precond<f64>,
    par: Par,
    stack: &mut MemStack,
    max_iter: usize,
    rel_tolerance: f64,
    print_progress: bool,
) -> usize {
    let mut work = Mat::zeros(x.nrows(), x.ncols());
    let mut iter = 0;
    let mut x = x;
    let b_norm = b.norm_l2();
    let print_iters = if print_progress {
        (max_iter / 10).max(1)
    } else {
        max_iter + 1
    };

    loop {
        op.apply(work.rb_mut(), x.rb(), par, stack);
        let mut r = b - &work;
        let r_norm = r.norm_l2();
        let rel_residual = r_norm / b_norm;

        iter += 1;
        if print_progress && iter % print_iters == 1 {
            println!("iter {}: {:.2e}", iter, rel_residual);
        }

        if rel_residual < rel_tolerance || iter >= max_iter {
            break;
        }

        pc.apply_in_place(r.rb_mut(), par, stack);
        x += r;
    }

    iter
}

fn report_cg_result(result: Result<CgInfo<f64>, CgError<f64>>) -> usize {
    match result {
        Ok(info) => {
            println!(
                "Solved in {} iters with (abs, rel) residual norms: ({:.2e}, {:.2e})",
                info.iter_count, info.abs_residual, info.rel_residual,
            );
            info.iter_count
        }
        Err(e) => {
            println!("CG failed: {:?}", e);
            0
        }
    }
}

fn main() {
    let base_elements: usize = 10;
    let min_refinements: usize = 2;
    let max_refinements: usize = 10;
    let max_iters = 6000;
    let rel_tolerance = 1e-8;
    let par = Par::Seq;

    let mut results: Vec<(usize, usize, usize, usize)> = Vec::new();

    println!("=== 1D Geometric Multigrid Refinement Study ===");
    println!("Target relative residual: {:.0e}", rel_tolerance);
    println!();

    for refinement in min_refinements..=max_refinements {
        let n_elements = base_elements * 2usize.pow(refinement as u32);
        let n_dofs = n_elements - 1;

        println!(
            "Refinement {}: {} elements, {} DOFs",
            refinement, n_elements, n_dofs
        );
        println!("Grid spacing h = {:.2e}", 1.0 / n_elements as f64);

        let fine_mat = make_finite_difference(n_elements);
        let arc_mat = Arc::new(fine_mat);
        let simple_pc = new_jacobi(&arc_mat.as_ref().as_ref(), 0.66);

        let fine_smoother = Arc::new(new_jacobi(&arc_mat.as_ref().as_ref(), 0.66));
        let mut mg_pc = MultiGrid::new(arc_mat.clone(), fine_smoother.clone());

        for level in 1..=refinement {
            let coarse_elements = base_elements * 2usize.pow((refinement - level) as u32);
            let coarse_dofs = coarse_elements - 1;

            let r = Arc::new(make_restriction(coarse_dofs));
            let p = Arc::new(make_interpolation(coarse_dofs));
            let mat = Arc::new(make_finite_difference(coarse_elements));
            let smoother: Arc<dyn BiPrecond<f64> + Send>;

            if level == refinement {
                let cholesky = CholeskySolve::new(mat.rb());
                smoother = Arc::new(cholesky);
            } else {
                smoother = Arc::new(new_jacobi(&mat.as_ref().as_ref(), 0.66));
            }

            mg_pc.add_level(mat, smoother, r, p);
        }

        let rhs: Col<f64, _> = Col::full(n_dofs, 1.0);
        let mut dst: Col<f64, _> = Col::zeros(n_dofs);

        let params = CgParams {
            abs_tolerance: f64::EPSILON,
            rel_tolerance,
            max_iters,
            ..Default::default()
        };

        let stack_req = conjugate_gradient_scratch(&mg_pc, arc_mat.as_ref(), 1, par);
        let mut buf = MemBuffer::new(stack_req);
        let stack = MemStack::new(&mut buf);

        // 1. PCG with Jacobi
        println!("  PCG + Jacobi:");
        dst.fill(0.0);
        let result = conjugate_gradient(
            dst.as_mat_mut(),
            &simple_pc,
            arc_mat.as_ref(),
            rhs.as_mat(),
            params,
            |_| {},
            par,
            stack,
        );
        let pcg_jacobi_iters = report_cg_result(result);

        // 2. PCG + Multigrid
        println!("  PCG + Multigrid:");
        dst.fill(0.0);
        let result = conjugate_gradient(
            dst.as_mat_mut(),
            &mg_pc,
            arc_mat.as_ref(),
            rhs.as_mat(),
            params,
            |_| {},
            par,
            stack,
        );
        let pcg_mg_iters = report_cg_result(result);

        // 3. Stationary + Multigrid
        println!("  Stationary + Multigrid:");
        dst.fill(0.0);
        let stat_mg_iters = stationary_solver(
            dst.as_mat_mut(),
            rhs.as_mat(),
            arc_mat.clone(),
            &mg_pc,
            par,
            stack,
            max_iters,
            rel_tolerance,
            true,
        );

        results.push((n_dofs, pcg_jacobi_iters, pcg_mg_iters, stat_mg_iters));
        println!();
    }

    println!("=== RESULTS SUMMARY ===");
    println!(
        "{:>8} {:>12} {:>12} {:>12}",
        "DOFs", "PCG+Jacobi", "PCG+MG", "Stat+MG"
    );
    println!("{:-<64}", "");
    for (n_dofs, pcg_jac, pcg_mg, stat_mg) in results {
        println!(
            "{:>8} {:>12} {:>12} {:>12}",
            n_dofs, pcg_jac, pcg_mg, stat_mg
        );
    }
}
