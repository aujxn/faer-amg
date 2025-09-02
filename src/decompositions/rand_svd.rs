use faer::diag::Diag;
use faer::dyn_stack::MemBuffer;
use faer::dyn_stack::MemStack;
use faer::dyn_stack::StackReq;
use faer::get_global_parallelism;
use faer::matrix_free::BiLinOp;
use faer::prelude::*;
use faer::reborrow::ReborrowMut;
use faer::stats::prelude::*;
use faer::stats::CwiseMatDistribution;
use faer_traits::ComplexField;

// TODO: Low level and generic implementations later...
pub fn rand_svd_scratch<T: ComplexField>(
    _op: impl BiLinOp<T>,
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
    l: usize, // TODO: Low level impl with stack and generics
              //op: impl BiLinOp<T>,
              //oversample: usize,
              //subspace_iter: usize,
              //s: DiagMut<T>,
              //u: Option<MatMut<T>>,
              //v: Option<MatMut<T>>,
              //par: Par,
              //stack: &mut MemStack,
              //params: Spec<SvdParams, T>,
) -> (Mat<f64>, Diag<f64>, Mat<f64>) {
    let rng = &mut StdRng::seed_from_u64(0);
    let m = op.nrows();
    let n = op.ncols();

    let par = get_global_parallelism();
    let mut mem = MemBuffer::new(StackReq::any_of(&[
        op.apply_scratch(l, par),
        op.transpose_apply_scratch(l, par),
    ]));
    let stack = MemStack::new(&mut mem);

    let mut range = Mat::zeros(m, l);
    let mut b = Mat::zeros(n, l);
    let gaussian = CwiseMatDistribution {
        nrows: n,
        ncols: l,
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
    let qr = range.qr();
    let q = qr.compute_thin_Q();
    op.transpose_apply(b.rb_mut(), q.as_ref(), par, stack);

    let svd = b.transpose().thin_svd().unwrap();
    let u_tilde = svd.U();
    let s = svd.S();
    let v = svd.V();

    let u = q.as_ref() * u_tilde;
    (u, s.column_vector().cloned().into_diagonal(), v.cloned())
}
