use faer::diag::Diag;
use faer::dyn_stack::MemBuffer;
use faer::dyn_stack::MemStack;
use faer::dyn_stack::StackReq;
use faer::get_global_parallelism;
use faer::linalg::matmul::dot::inner_prod;
use faer::mat::AsMatMut;
use faer::matrix_free::BiLinOp;
use faer::prelude::*;
use faer::reborrow::ReborrowMut;
use faer::stats::prelude::*;
use faer::stats::CwiseMatDistribution;
use log::info;
use rand::rng;
// TODO: Low level implementation later...
pub fn rand_svd_scratch(
    _op: impl BiLinOp<f64>,
    _truncation_dim: usize,
    _oversample: usize,
) -> StackReq {
    unimplemented!()
}

/// Returns an orthonormal matrix, `Q`, which approximates the range of `op`, $A$,
/// and the intermediate `svd`, $\widetilde U \Sigma V^\star$. The randomized low
/// rank truncated approximation can be obtained by $A \approx Q \widetilde U \Sigma V^\star$.
pub fn rand_svd(
    op: impl BiLinOp<f64>,
    l: usize,
    oversample: usize,
    subspace_iter: usize,
    //stack: &mut MemStack,
    //params: Spec<SvdParams, T>,
) -> (Mat<f64>, Diag<f64>, Mat<f64>) {
    let rng = &mut rng();
    let m = op.nrows();
    let n = op.ncols();

    // could just default to normal svd if this happens and not crash
    assert!(m > l + oversample);
    assert!(n > l + oversample);

    let par = get_global_parallelism();
    let mut mem = MemBuffer::new(StackReq::any_of(&[
        op.apply_scratch(l + oversample, par),
        op.transpose_apply_scratch(l + oversample, par),
    ]));
    let stack = MemStack::new(&mut mem);

    let mut range = Mat::zeros(m, l + oversample);
    let mut b = Mat::zeros(n, l + oversample);
    let gaussian = CwiseMatDistribution {
        nrows: n,
        ncols: l + oversample,
        dist: StandardNormal,
    }
    .rand::<Mat<f64>>(rng);
    /*
    let mut stack = stack;
    let (mut range, mut stack) = unsafe { temp_mat_uninit::<f64, _, _>(m, l, stack.rb_mut()) };
    let mut range = range.as_mat_mut();
    let (mut b, stack) = unsafe { temp_mat_uninit::<f64, _, _>(n, l, stack.rb_mut()) };
    let mut b = b.as_mat_mut();
    */
    op.apply(range.rb_mut(), gaussian.as_ref(), par, stack);

    if subspace_iter > 0 {
        let mut work = Mat::zeros(n, l + oversample);
        for _ in 0..subspace_iter {
            op.transpose_apply(work.as_mat_mut(), range.as_ref(), par, stack);
            op.apply(range.as_mat_mut(), work.as_ref(), par, stack);
        }
    }

    let qr = range.qr();
    let q = qr.compute_thin_Q();
    op.transpose_apply(b.rb_mut(), q.as_ref(), par, stack);

    let svd = b.transpose().thin_svd().unwrap();
    let u_tilde = svd.U().subcols(0, l);
    let s = svd
        .S()
        .column_vector()
        .subrows(0, l)
        .to_owned()
        .into_diagonal();
    let v = svd.V().subcols(0, l).to_owned();

    let u = q * u_tilde;

    /*
    let mut maybe_u = Mat::zeros(u.nrows(), u.ncols());
    op.apply(maybe_u.as_mut(), v.as_ref(), par, stack);
    let maybe_s = Col::from_iter(maybe_u.col_iter().map(|col| col.norm_l2())).into_diagonal();
    let convergence_factors: String = maybe_s
        .into_column_vector()
        .iter()
        .map(|norm| format!("{:.2} ", norm))
        .collect();
    info!("Column L2 norms of A v: {}", convergence_factors);
    */

    (u, s, v)
}
