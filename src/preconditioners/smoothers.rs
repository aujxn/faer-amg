use std::sync::Arc;

use faer::{
    diag::Diag,
    dyn_stack::{MemStack, StackReq},
    linalg::solvers::SolveCore,
    mat::{AsMatMut, AsMatRef},
    matrix_free::{BiLinOp, BiPrecond, LinOp, Precond},
    prelude::{Reborrow, ReborrowMut},
    sparse::{
        linalg::solvers::{Llt, SymbolicLlt},
        SparseRowMatRef,
    },
    Mat, MatMut, MatRef, Par,
};

pub type L2 = Diag<f64>;
pub type L1 = Diag<f64>;

pub fn new_l2(mat: SparseRowMatRef<usize, f64>) -> L2 {
    let nrows = mat.nrows();
    let diag_sqrt: Vec<f64> = (0..nrows)
        .map(|i| mat.get(i, i).map(|v| v.sqrt()).unwrap())
        .collect();

    let mut l2_inverse: L2 = Diag::zeros(nrows);

    for triplet in mat.triplet_iter() {
        let scale = diag_sqrt[triplet.row] / diag_sqrt[triplet.col];
        l2_inverse[triplet.row] += triplet.val.abs() * scale;
    }

    l2_inverse
        .column_vector_mut()
        .iter_mut()
        .for_each(|d| *d = 1.0 / *d);
    l2_inverse
}

pub fn new_l1(mat: SparseRowMatRef<usize, f64>) -> L1 {
    let nrows = mat.nrows();
    let mut l1_inverse: L1 = Diag::zeros(nrows);

    for triplet in mat.triplet_iter() {
        l1_inverse[triplet.row] += triplet.val.abs();
    }

    l1_inverse
        .column_vector_mut()
        .iter_mut()
        .for_each(|d| *d = 1.0 / *d);
    l1_inverse
}

pub fn new_jacobi(mat: &SparseRowMatRef<usize, f64>, omega: f64) -> L1 {
    let nrows = mat.nrows();
    let mut jacobi: L1 = Diag::zeros(nrows);

    for (i, val) in jacobi.column_vector_mut().iter_mut().enumerate() {
        *val = omega / *mat.get(i, i).unwrap();
    }
    jacobi
}

#[derive(Clone, Debug)]
pub struct StationaryIteration {
    mat: Arc<dyn LinOp<f64> + Send>,
    prec: Arc<dyn LinOp<f64> + Send>,
    iters: usize,
}

impl StationaryIteration {
    pub fn new(
        mat: Arc<dyn LinOp<f64> + Send>,
        prec: Arc<dyn LinOp<f64> + Send>,
        iters: usize,
    ) -> Self {
        Self { mat, prec, iters }
    }
}

impl LinOp<f64> for StationaryIteration {
    // TODO: low level API
    fn apply_scratch(&self, rhs_ncols: usize, par: Par) -> StackReq {
        let _rhs_ncols = rhs_ncols;
        let _par = par;
        StackReq::EMPTY
    }

    fn nrows(&self) -> usize {
        self.mat.nrows()
    }

    fn ncols(&self) -> usize {
        self.mat.ncols()
    }

    fn apply(&self, out: MatMut<'_, f64>, rhs: MatRef<'_, f64>, par: Par, stack: &mut MemStack) {
        let mut out = out;
        let mut work1 = Mat::zeros(out.nrows(), out.ncols());
        let mut work2 = Mat::zeros(out.nrows(), out.ncols());
        work1.copy_from(rhs);
        for _ in 0..self.iters {
            self.mat
                .apply(work2.rb_mut(), work1.as_mat_ref(), par, stack);
            self.prec
                .apply(out.rb_mut(), work2.as_mat_ref(), par, stack);
            work1 -= out.rb();
        }
        out.copy_from(work1);
    }

    fn conj_apply(
        &self,
        out: MatMut<'_, f64>,
        rhs: MatRef<'_, f64>,
        par: Par,
        stack: &mut MemStack,
    ) {
        // TODO: only real for now
        self.apply(out, rhs, par, stack);
    }
}

impl BiLinOp<f64> for StationaryIteration {
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
        let mut out = out;
        let mut work1 = Mat::zeros(out.nrows(), out.ncols());
        let mut work2 = Mat::zeros(out.nrows(), out.ncols());
        work1.copy_from(rhs);
        for _ in 0..self.iters {
            self.prec
                .apply(work2.rb_mut(), work1.as_mat_ref(), par, stack);
            self.mat.apply(out.rb_mut(), work2.as_mat_ref(), par, stack);
            work1 -= out.rb();
        }
        out.copy_from(work1);
    }

    fn adjoint_apply(
        &self,
        out: MatMut<'_, f64>,
        rhs: MatRef<'_, f64>,
        par: Par,
        stack: &mut MemStack,
    ) {
        // only real
        self.transpose_apply(out, rhs, par, stack);
    }
}

/// Cholesky solver abstraction that implements linear operator interface.
#[derive(Clone, Debug)]
pub struct CholeskySolve {
    decomp: Llt<usize, f64>,
    size: usize,
}

// TODO: probably use low level API?
impl CholeskySolve {
    pub fn new(sym_mat: SparseRowMatRef<usize, f64>) -> Self {
        let symb_llt =
            SymbolicLlt::try_new(sym_mat.symbolic().transpose(), faer::Side::Upper).unwrap();
        let decomp =
            Llt::try_new_with_symbolic(symb_llt, sym_mat.transpose(), faer::Side::Upper).unwrap();
        let size = sym_mat.nrows();
        Self { decomp, size }
    }
}

impl LinOp<f64> for CholeskySolve {
    // TODO: remove allocations in apply and learn how to use this stack
    fn apply_scratch(&self, rhs_ncols: usize, par: Par) -> StackReq {
        let _rhs_ncols = rhs_ncols;
        let _par = par;
        StackReq::EMPTY
    }

    fn nrows(&self) -> usize {
        self.size
    }

    fn ncols(&self) -> usize {
        self.size
    }

    fn apply(&self, out: MatMut<'_, f64>, rhs: MatRef<'_, f64>, par: Par, stack: &mut MemStack) {
        let _par = par;
        let _stack = stack;
        let mut out = out;
        out.copy_from(rhs);
        // TODO: use low level api...
        self.decomp.solve_in_place_with_conj(faer::Conj::No, out);
    }

    fn conj_apply(
        &self,
        out: MatMut<'_, f64>,
        rhs: MatRef<'_, f64>,
        par: Par,
        stack: &mut MemStack,
    ) {
        let _par = par;
        let _stack = stack;
        let mut out = out;
        out.copy_from(rhs);
        // TODO: use low level api...
        self.decomp.solve_in_place_with_conj(faer::Conj::Yes, out);
    }
}

impl BiLinOp<f64> for CholeskySolve {
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
        self.apply(out, rhs, par, stack);
    }
    fn adjoint_apply(
        &self,
        out: MatMut<'_, f64>,
        rhs: MatRef<'_, f64>,
        par: Par,
        stack: &mut MemStack,
    ) {
        self.apply(out, rhs, par, stack);
    }
}

impl Precond<f64> for CholeskySolve {
    fn apply_in_place_scratch(&self, rhs_ncols: usize, par: Par) -> StackReq {
        let _rhs_ncols = rhs_ncols;
        let _par = par;
        StackReq::EMPTY
    }
    fn apply_in_place(&self, rhs: MatMut<'_, f64>, par: Par, stack: &mut MemStack) {
        let _par = par;
        let _stack = stack;
        self.decomp.solve_in_place_with_conj(faer::Conj::No, rhs);
    }
    fn conj_apply_in_place(&self, rhs: MatMut<'_, f64>, par: Par, stack: &mut MemStack) {
        let _par = par;
        let _stack = stack;
        self.decomp.solve_in_place_with_conj(faer::Conj::Yes, rhs);
    }
}

impl BiPrecond<f64> for CholeskySolve {
    fn transpose_apply_in_place_scratch(&self, rhs_ncols: usize, par: Par) -> StackReq {
        self.apply_in_place_scratch(rhs_ncols, par)
    }
    fn transpose_apply_in_place(&self, rhs: MatMut<'_, f64>, par: Par, stack: &mut MemStack) {
        self.apply_in_place(rhs, par, stack);
    }
    fn adjoint_apply_in_place(&self, rhs: MatMut<'_, f64>, par: Par, stack: &mut MemStack) {
        self.conj_apply_in_place(rhs, par, stack);
    }
}
