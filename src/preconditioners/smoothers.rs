use std::sync::Arc;

use faer::{
    diag::Diag,
    dyn_stack::{MemStack, StackReq},
    linalg::solvers::SolveCore,
    matrix_free::{BiLinOp, BiPrecond, LinOp, Precond},
    sparse::{
        linalg::solvers::{Llt, SymbolicLlt},
        SparseRowMatRef,
    },
    Index, MatMut, MatRef, Par,
};
use faer_traits::{
    math_utils::{abs1, add, mul, recip, sqrt},
    RealField,
};

pub type L2<T> = Diag<T>;
pub type L1<T> = Diag<T>;

pub fn new_l2<I: Index, T: RealField>(mat: &SparseRowMatRef<I, T>) -> L2<T> {
    let nrows = mat.nrows();
    let diag_sqrt: Vec<T> = (0..nrows).map(|i| sqrt(mat.get(i, i).unwrap())).collect();

    let mut l2_inverse: L2<T> = Diag::zeros(nrows);

    for triplet in mat.triplet_iter() {
        let scale: T = mul(&diag_sqrt[triplet.row], &recip(&diag_sqrt[triplet.col]));
        l2_inverse[triplet.row] = add(&l2_inverse[triplet.row], &(abs1(triplet.val) * scale));
    }

    l2_inverse
        .column_vector_mut()
        .iter_mut()
        .for_each(|d| *d = recip(d));
    l2_inverse
}

pub fn new_l1<I: Index, T: RealField>(mat: &SparseRowMatRef<I, T>) -> L1<T> {
    let nrows = mat.nrows();
    let mut l1_inverse: L1<T> = Diag::zeros(nrows);

    for triplet in mat.triplet_iter() {
        l1_inverse[triplet.row] = add(&l1_inverse[triplet.row], &abs1(triplet.val));
    }

    l1_inverse
        .column_vector_mut()
        .iter_mut()
        .for_each(|d| *d = recip(d));
    l1_inverse
}

pub fn new_jacobi<I: Index, T: RealField>(mat: &SparseRowMatRef<I, T>, omega: T) -> L1<T> {
    let nrows = mat.nrows();
    let mut jacobi: L1<T> = Diag::zeros(nrows);

    for (i, val) in jacobi.column_vector_mut().iter_mut().enumerate() {
        *val = recip(mat.get(i, i).unwrap()).mul_by_ref(&omega);
    }
    jacobi
}

/// Cholesky solver abstraction that implements linear operator interface.
#[derive(Clone, Debug)]
pub struct CholeskySolve<I: Index, T: RealField> {
    decomp: Llt<I, T>,
    size: usize,
}

// TODO: probably use low level API?
impl<I: Index, T: RealField> CholeskySolve<I, T> {
    pub fn new(sym_mat: SparseRowMatRef<I, T>) -> Self {
        let symb_llt =
            SymbolicLlt::try_new(sym_mat.symbolic().transpose(), faer::Side::Upper).unwrap();
        let decomp =
            Llt::try_new_with_symbolic(symb_llt, sym_mat.transpose(), faer::Side::Upper).unwrap();
        let size = sym_mat.nrows();
        Self { decomp, size }
    }
}

impl<I: Index, T: RealField> LinOp<T> for CholeskySolve<I, T> {
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

    fn apply(&self, out: MatMut<'_, T>, rhs: MatRef<'_, T>, par: Par, stack: &mut MemStack) {
        let _par = par;
        let _stack = stack;
        let mut out = out;
        out.copy_from(rhs);
        // TODO: use low level api...
        self.decomp.solve_in_place_with_conj(faer::Conj::No, out);
    }

    fn conj_apply(&self, out: MatMut<'_, T>, rhs: MatRef<'_, T>, par: Par, stack: &mut MemStack) {
        let _par = par;
        let _stack = stack;
        let mut out = out;
        out.copy_from(rhs);
        // TODO: use low level api...
        self.decomp.solve_in_place_with_conj(faer::Conj::Yes, out);
    }
}

impl<I: Index, T: RealField> BiLinOp<T> for CholeskySolve<I, T> {
    fn transpose_apply_scratch(&self, rhs_ncols: usize, par: Par) -> StackReq {
        self.apply_scratch(rhs_ncols, par)
    }
    fn transpose_apply(
        &self,
        out: MatMut<'_, T>,
        rhs: MatRef<'_, T>,
        par: Par,
        stack: &mut MemStack,
    ) {
        self.apply(out, rhs, par, stack);
    }
    fn adjoint_apply(
        &self,
        out: MatMut<'_, T>,
        rhs: MatRef<'_, T>,
        par: Par,
        stack: &mut MemStack,
    ) {
        self.apply(out, rhs, par, stack);
    }
}

impl<I: Index, T: RealField> Precond<T> for CholeskySolve<I, T> {
    fn apply_in_place_scratch(&self, rhs_ncols: usize, par: Par) -> StackReq {
        let _rhs_ncols = rhs_ncols;
        let _par = par;
        StackReq::EMPTY
    }
    fn apply_in_place(&self, rhs: MatMut<'_, T>, par: Par, stack: &mut MemStack) {
        let _par = par;
        let _stack = stack;
        self.decomp.solve_in_place_with_conj(faer::Conj::No, rhs);
    }
    fn conj_apply_in_place(&self, rhs: MatMut<'_, T>, par: Par, stack: &mut MemStack) {
        let _par = par;
        let _stack = stack;
        self.decomp.solve_in_place_with_conj(faer::Conj::Yes, rhs);
    }
}

impl<I: Index, T: RealField> BiPrecond<T> for CholeskySolve<I, T> {
    fn transpose_apply_in_place_scratch(&self, rhs_ncols: usize, par: Par) -> StackReq {
        self.apply_in_place_scratch(rhs_ncols, par)
    }
    fn transpose_apply_in_place(&self, rhs: MatMut<'_, T>, par: Par, stack: &mut MemStack) {
        self.apply_in_place(rhs, par, stack);
    }
    fn adjoint_apply_in_place(&self, rhs: MatMut<'_, T>, par: Par, stack: &mut MemStack) {
        self.conj_apply_in_place(rhs, par, stack);
    }
}
