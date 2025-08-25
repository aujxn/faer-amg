use core::f64;
use std::sync::Arc;

use faer::{
    dyn_stack::{MemBuffer, MemStack},
    mat::generic::Mat,
    matrix_free::{
        conjugate_gradient::{
            conjugate_gradient, conjugate_gradient_scratch, CgError, CgInfo, CgParams,
        },
        LinOp, Precond,
    },
    prelude::{Reborrow, ReborrowMut},
    sparse::{SparseRowMat, Triplet},
    Col, MatMut, MatRef, Par,
};
use faer_amg::preconditioners::{multigrid::MultiGrid, smoothers::new_jacobi};

/* TODO:
* - Solve coarsest exactly
* - Refinement study for pcg and stationary
* - Full description of what this example does
* - Use analytic solution for error computation
*/

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
    let h = 1.0 / n_elements as f64;
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
fn stationary_solver(
    x: MatMut<f64>,
    b: MatRef<f64>,
    op: Arc<dyn LinOp<f64>>,
    pc: impl Precond<f64>,
    par: Par,
    stack: &mut MemStack,
    max_iter: usize,
) {
    let mut work = Mat::zeros(x.nrows(), x.ncols());
    let mut iter = 0;
    let mut x = x;
    let print_iters = (max_iter / 10).max(1);
    loop {
        op.apply(work.rb_mut(), x.rb(), par, stack);
        let mut r = b - &work;
        iter += 1;
        if iter % print_iters == 1 {
            let r_norm = r.norm_l2();
            println!("iter {}: {:.2e}", iter, r_norm);
        }
        pc.apply_in_place(r.rb_mut(), par, stack);
        x += r;
        if iter == max_iter {
            break;
        }
    }
}

fn report_cg_result(result: Result<CgInfo<f64>, CgError<f64>>) {
    match result {
        Ok(info) => {
            println!(
                "Solved in {} iters with (abs, rel) residual norms: ({:.2e}, {:.2e})",
                info.iter_count, info.abs_residual, info.rel_residual,
            );
        }
        Err(e) => {
            println!("Multigrid failed: {:?}", e);
        }
    }
}

fn main() {
    let coarse_elements: usize = 10;
    let n_levels: usize = 7;

    /*
    let print_errs = |x: MatRef<f64>| {
        let err_norm = (solution.as_mat() - x).norm_l2();
        println!("{}", err_norm);
    };
    */
    let fine_n_elements: usize = coarse_elements * 2u32.pow(n_levels as u32) as usize;
    println!(
        "Fine grid elements: {}, Fine DOFs: {}",
        fine_n_elements,
        fine_n_elements - 1
    );

    let fine_mat = make_finite_difference(fine_n_elements);
    let arc_mat = Arc::new(fine_mat);
    let fine_smoother = Arc::new(new_jacobi(&arc_mat.as_ref().as_ref(), 0.66));
    let mut pc = MultiGrid::new(arc_mat.clone(), fine_smoother.clone());

    for level in 1..=n_levels {
        let n_elements: usize = coarse_elements * 2u32.pow((n_levels - level) as u32) as usize;
        let n_dofs = n_elements - 1;

        let r = Arc::new(make_restriction(n_dofs));
        let p = Arc::new(make_interpolation(n_dofs));
        let mat = Arc::new(make_finite_difference(n_elements));
        let smoother = Arc::new(new_jacobi(&mat.as_ref().as_ref(), 0.66));

        println!(
            "Adding level {}: coarse DOFs = {}, fine DOFs = {}",
            level,
            p.ncols(),
            p.nrows()
        );
        pc.add_level(mat, smoother, r, p);
    }

    let max_iters = 1000;
    let params = CgParams {
        abs_tolerance: f64::EPSILON,
        rel_tolerance: 1e-12,
        max_iters,
        ..Default::default()
    };
    let par = Par::Seq;
    let stack_req = conjugate_gradient_scratch(&pc, arc_mat.as_ref(), 1, par);
    let mut buf = MemBuffer::new(stack_req);
    let stack = MemStack::new(&mut buf);

    let simple_pc = new_jacobi(&arc_mat.as_ref().as_ref(), 0.66);
    let n_freedofs = fine_n_elements - 1;
    let rhs: Col<f64, _> = Col::ones(n_freedofs);
    let mut dst: Col<f64, _> = Col::zeros(n_freedofs);

    println!("\nRunning PCG with weighted Jacobi preconditioner...");
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

    report_cg_result(result);

    println!("\nRunning stationary iteration with weighted Jacobi...");
    dst.fill(0.0);
    stationary_solver(
        dst.as_mat_mut(),
        rhs.as_mat(),
        arc_mat.clone(),
        &simple_pc,
        par,
        stack,
        max_iters,
    );

    println!("\nRunning PCG with multigrid preconditioner...");
    dst.fill(0.0);
    let result = conjugate_gradient(
        dst.as_mat_mut(),
        &pc,
        arc_mat.as_ref(),
        rhs.as_mat(),
        params,
        |_| {},
        par,
        stack,
    );

    report_cg_result(result);

    println!("\nRunning stationary iteration with multigrid...");
    dst.fill(0.0);
    stationary_solver(
        dst.as_mat_mut(),
        rhs.as_mat(),
        arc_mat.clone(),
        &pc,
        par,
        stack,
        max_iters,
    );
}
