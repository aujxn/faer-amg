use faer::dyn_stack::MemStack;
use faer::linalg::solvers::Svd;
use faer::linalg::svd::SvdError;
use faer::linalg::temp_mat_uninit;
use faer::mat::AsMatMut;
use faer::matrix_free::BiLinOp;
use faer::prelude::*;
use faer::reborrow::ReborrowMut;
use faer::stats::prelude::*;
use faer::stats::CwiseMatDistribution;

/// Returns an orthonormal matrix, `Q`, which approximates the range of `op`, $A$,
/// and the intermediate `svd`, $\widetilde U \Sigma V^\star$. The randomized low
/// rank truncated approximation can be obtained by $A \approx Q \widetilde U \Sigma V^\star$.
pub fn rand_svd(
    op: impl BiLinOp<f64>,
    n_vecs: usize,
    par: Par,
    stack: &mut MemStack,
) -> Result<(Mat<f64>, Svd<f64>), SvdError> {
    let rng = &mut StdRng::seed_from_u64(0);
    let m = op.nrows();
    let n = op.ncols();
    let l = n_vecs;
    let gaussian = CwiseMatDistribution {
        nrows: n,
        ncols: l,
        dist: StandardNormal,
    }
    .rand::<Mat<f64>>(rng);

    let mut stack = stack;
    let (mut range, mut stack) = unsafe { temp_mat_uninit::<f64, _, _>(m, l, stack.rb_mut()) };
    let mut range = range.as_mat_mut();
    let (mut b, stack) = unsafe { temp_mat_uninit::<f64, _, _>(n, l, stack.rb_mut()) };
    let mut b = b.as_mat_mut();

    op.apply(range.rb_mut(), gaussian.as_ref(), par, stack);
    let qr = range.qr();
    let q = qr.compute_thin_Q();
    op.transpose_apply(b.rb_mut(), q.as_ref(), par, stack);

    b.transpose().thin_svd().map(|svd| (q, svd))
}
