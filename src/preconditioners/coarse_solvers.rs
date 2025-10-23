use faer::{
    dyn_stack::{MemStack, StackReq},
    linalg::solvers::{Llt as DenseLlt, LltError as DenseLltError, SolveCore},
    matrix_free::{BiLinOp, BiPrecond, LinOp, Precond},
    sparse::SparseRowMatRef,
    Conj, Mat, MatMut, MatRef, Par, Side,
};

/// Dense Cholesky preconditioner used for coarse solves where building a sparse factorization
/// would be overkill relative to the sub-problem size.
#[derive(Clone, Debug)]
pub struct DenseCholeskySolve {
    decomp: DenseLlt<f64>,
    size: usize,
}

impl DenseCholeskySolve {
    pub fn from_sparse(mat: SparseRowMatRef<usize, f64>) -> Result<Self, DenseLltError> {
        let dense = mat.to_dense();
        Self::from_dense(dense)
    }

    pub fn from_dense(matrix: Mat<f64>) -> Result<Self, DenseLltError> {
        let size = matrix.nrows();
        let decomp = matrix.llt(Side::Lower)?;
        Ok(Self { decomp, size })
    }
}

impl LinOp<f64> for DenseCholeskySolve {
    fn apply_scratch(&self, rhs_ncols: usize, par: Par) -> StackReq {
        let _ = (rhs_ncols, par);
        StackReq::EMPTY
    }

    fn nrows(&self) -> usize {
        self.size
    }

    fn ncols(&self) -> usize {
        self.size
    }

    fn apply(&self, out: MatMut<'_, f64>, rhs: MatRef<'_, f64>, par: Par, stack: &mut MemStack) {
        let _ = (par, stack);
        let mut out = out;
        out.copy_from(rhs);
        self.decomp.solve_in_place_with_conj(Conj::No, out);
    }

    fn conj_apply(
        &self,
        out: MatMut<'_, f64>,
        rhs: MatRef<'_, f64>,
        par: Par,
        stack: &mut MemStack,
    ) {
        let _ = (par, stack);
        let mut out = out;
        out.copy_from(rhs);
        self.decomp.solve_in_place_with_conj(Conj::Yes, out);
    }
}

impl BiLinOp<f64> for DenseCholeskySolve {
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

impl Precond<f64> for DenseCholeskySolve {
    fn apply_in_place_scratch(&self, rhs_ncols: usize, par: Par) -> StackReq {
        self.apply_scratch(rhs_ncols, par)
    }

    fn apply_in_place(&self, rhs: MatMut<'_, f64>, par: Par, stack: &mut MemStack) {
        let _ = (par, stack);
        self.decomp.solve_in_place_with_conj(Conj::No, rhs);
    }

    fn conj_apply_in_place(&self, rhs: MatMut<'_, f64>, par: Par, stack: &mut MemStack) {
        let _ = (par, stack);
        self.decomp.solve_in_place_with_conj(Conj::Yes, rhs);
    }
}

impl BiPrecond<f64> for DenseCholeskySolve {
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
