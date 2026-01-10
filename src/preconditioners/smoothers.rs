use std::sync::Arc;

use faer::{
    diag::Diag,
    dyn_stack::{MemStack, StackReq},
    linalg::temp_mat_scratch,
    mat::AsMatRef,
    matrix_free::{BiLinOp, BiPrecond, LinOp, Precond},
    prelude::{Reborrow, ReborrowMut},
    sparse::SparseRowMatRef,
    Mat, MatMut, MatRef, Par,
};

#[derive(Debug, Clone)]
pub enum SmootherKind {
    L1,
    L2,
    Jacobi(f64),
    GaussSeidel,
    SymGaussSeidel,
}

impl SmootherKind {
    pub fn build(&self, mat: SparseRowMatRef<usize, f64>) -> Arc<dyn BiPrecond<f64> + Send> {
        match &self {
            SmootherKind::SymGaussSeidel => unimplemented!(),
            SmootherKind::GaussSeidel => unimplemented!(),
            SmootherKind::L1 => Arc::new(new_l1(mat)),
            SmootherKind::L2 => Arc::new(new_l2(mat)),
            SmootherKind::Jacobi(omega) => Arc::new(new_jacobi(mat, *omega)),
        }
    }
}

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
        .for_each(|d| *d = d.recip());
    l1_inverse
}

pub fn new_jacobi(mat: SparseRowMatRef<usize, f64>, omega: f64) -> L1 {
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

    pub fn set_op(&mut self, op: Arc<dyn LinOp<f64> + Send>) {
        self.mat = op;
    }

    pub fn get_op(&self) -> Arc<dyn LinOp<f64> + Send> {
        self.mat.clone()
    }

    pub fn set_pc(&mut self, pc: Arc<dyn LinOp<f64> + Send>) {
        self.prec = pc;
    }

    pub fn get_pc(&self) -> Arc<dyn LinOp<f64> + Send> {
        self.prec.clone()
    }

    pub fn set_iters(&mut self, iters: usize) {
        self.iters = iters;
    }

    pub fn get_iters(&self) -> usize {
        self.iters
    }
}

impl LinOp<f64> for StationaryIteration {
    fn apply_scratch(&self, rhs_ncols: usize, par: Par) -> StackReq {
        let req = StackReq::all_of(&[temp_mat_scratch::<f64>(self.nrows(), rhs_ncols); 3]);
        req.and(StackReq::any_of(&[
            self.mat.apply_scratch(rhs_ncols, par),
            self.prec.apply_scratch(rhs_ncols, par),
        ]))
    }

    fn nrows(&self) -> usize {
        self.mat.nrows()
    }

    fn ncols(&self) -> usize {
        self.mat.ncols()
    }

    fn apply(&self, out: MatMut<'_, f64>, rhs: MatRef<'_, f64>, par: Par, stack: &mut MemStack) {
        let mut out = out;
        let mut x = Mat::zeros(out.nrows(), out.ncols());
        let mut r = Mat::zeros(out.nrows(), out.ncols());
        // initial residual is rhs
        self.prec.apply(x.rb_mut(), rhs, par, stack);
        for _ in 1..self.iters {
            self.mat.apply(r.rb_mut(), x.as_ref(), par, stack);
            r = x.as_ref() - r;
            self.prec.apply(out.rb_mut(), r.as_ref(), par, stack);
            x += out.rb();
        }
        out.copy_from(x);
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

impl Precond<f64> for StationaryIteration {}
impl BiPrecond<f64> for StationaryIteration {}
