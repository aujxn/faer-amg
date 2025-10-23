use std::{
    collections::{BTreeSet, HashSet},
    sync::Arc,
};

use faer::{
    dyn_stack::{MemBuffer, MemStack, StackReq},
    matrix_free::{BiLinOp, BiPrecond, LinOp, Precond},
    prelude::ReborrowMut,
    sparse::{SparseRowMat, SparseRowMatRef, Triplet},
    traits::MulByRef,
    Mat, MatMut, MatRef, Par,
};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

use crate::{partitioners::Partition, preconditioners::smoothers::CholeskySolve};

#[derive(Copy, Clone, Debug)]
#[non_exhaustive]
pub enum BlockSmootherType {
    /// Performs one symmetric Gauss-Seidel sweep per block.
    GaussSeidel,
    /// Decomposes each block into LDL^T with no fill-in. Resulting decomposition has same sparsity
    /// pattern as the block matrix created by removing connections between aggregates in the
    /// partition.
    IncompleteCholesky,
    /// If the sparsity of a block is below 30% then solve with sparse Cholesky decomposition but
    /// otherwise just use dense Cholesky decompositions. Warning: large blocks can consume lots of
    /// memory even when the sparse decomposition is done.
    AutoCholesky,
    /// Solve each block by cacheing a sparse Cholesky decomposition of each block using the
    /// provided reordering. Warning: If blocks are large and the reordering approach doesn't
    /// effectively reduce fill in this can consume lots of memory.
    SparseCholesky,
    /// Solve each block by cacheing a dense Cholesky decomposition of each block. Warning: if the
    /// blocks are large this can consume all available memory very quickly.
    DenseCholesky,
    /// Solve each block to a relative accuracy provided (1e-6 is probably okay most the time but
    /// problem dependent) iteratively with conjugate gradient.
    ConjugateGradient(f64),
}

#[derive(Debug, Clone)]
pub struct BlockSmoother {
    partition: Arc<Partition>,
    blocks: Vec<Arc<dyn BiPrecond<f64> + Send + Sync>>,
    dim: usize,
}

impl BlockSmoother {
    pub fn new(
        mat: SparseRowMatRef<usize, f64>,
        partition: Arc<Partition>,
        smoother: BlockSmootherType,
        vdim: usize,
    ) -> Self {
        let blocks: Vec<Arc<dyn BiPrecond<f64> + Send + Sync>> = partition
            .aggregates()
            .par_iter()
            .map(|agg| {
                let csr = if vdim == 1 {
                    Self::diagonally_compensate(agg, mat)
                } else {
                    Self::diagonally_compensate_vector(agg, mat, vdim)
                };

                match smoother {
                    BlockSmootherType::GaussSeidel => {
                        unimplemented!()
                    }
                    BlockSmootherType::IncompleteCholesky => {
                        unimplemented!()
                    }
                    BlockSmootherType::AutoCholesky => {
                        unimplemented!()
                    }
                    BlockSmootherType::DenseCholesky => {
                        unimplemented!()
                    }
                    BlockSmootherType::SparseCholesky => {
                        let pc: Arc<dyn BiPrecond<f64> + Sync + Send> =
                            Arc::new(CholeskySolve::new(csr.as_ref()));
                        pc
                    }
                    BlockSmootherType::ConjugateGradient(_tolerance) => {
                        unimplemented!()
                    }
                }
            })
            .collect();

        Self {
            partition,
            blocks,
            dim: mat.nrows(),
        }
    }

    fn diagonally_compensate(
        agg: &BTreeSet<usize>,
        mat: SparseRowMatRef<usize, f64>,
    ) -> SparseRowMat<usize, f64> {
        let block_size = agg.len();
        let agg: Vec<usize> = agg.iter().copied().collect();
        let mut block_triplets = Vec::new();
        let symbolic = mat.symbolic();

        for (ic, i) in agg.iter().copied().enumerate() {
            let row_i_vals = mat.val_of_row(i);
            let row_i_cols = symbolic.col_idx_of_row(i);
            let a_ii = mat.get(i, i).unwrap();
            for (j, val) in row_i_cols.zip(row_i_vals.iter()) {
                match agg.binary_search(&j) {
                    Ok(jc) => {
                        block_triplets.push(Triplet::new(ic, jc, *val));
                    }
                    Err(_) => {
                        let a_jj = mat.get(j, j).unwrap();
                        block_triplets.push(Triplet::new(
                            ic,
                            ic,
                            0.5 * (a_ii / a_jj).sqrt() * val.abs(),
                        ));
                    }
                }
            }
        }

        SparseRowMat::try_new_from_triplets(block_size, block_size, &block_triplets).unwrap()
    }

    fn diagonally_compensate_vector(
        agg: &BTreeSet<usize>,
        mat: SparseRowMatRef<usize, f64>,
        vdim: usize,
    ) -> SparseRowMat<usize, f64> {
        let block_size = agg.len();
        assert_eq!(block_size % vdim, 0);
        let agg: Vec<usize> = agg.iter().copied().collect();
        let mut block_triplets = Vec::new();
        let symbolic = mat.symbolic();

        let mut to_compensate = HashSet::new();

        for (ic, i) in agg.iter().copied().enumerate() {
            let values_row_i = mat.val_of_row(i);
            for (j, val) in symbolic.col_idx_of_row(i).zip(values_row_i.iter()) {
                match agg.binary_search(&j) {
                    Ok(jc) => {
                        block_triplets.push(Triplet::new(ic, jc, *val));
                    }
                    Err(_) => {
                        let ic_start = ic - (ic % vdim);
                        let i_start = i - (i % vdim);
                        let j_start = j - (j % vdim);
                        to_compensate.insert((ic_start, (i_start, j_start)));
                    }
                }
            }
        }

        for (ic, (i, j)) in to_compensate {
            let mut block_a_ij: Mat<f64> = Mat::zeros(vdim, vdim);
            for i_off in 0..vdim {
                for j_off in 0..vdim {
                    if let Some(val) = mat.get(i + i_off, j + j_off) {
                        block_a_ij[(i_off, j_off)] -= val;
                    }
                }
            }

            let svd = block_a_ij.svd().unwrap();
            let u = svd.U();
            let s = svd.S();
            let usut: Mat<f64> = u.mul_by_ref(&s.mul_by_ref(&u.transpose()));

            for i_off in 0..vdim {
                for j_off in 0..vdim {
                    block_triplets.push(Triplet::new(
                        ic + i_off,
                        ic + j_off,
                        0.5 * usut[(i_off, j_off)],
                    ));
                }
            }
        }

        SparseRowMat::try_new_from_triplets(block_size, block_size, &block_triplets).unwrap()
    }
}

impl LinOp<f64> for BlockSmoother {
    // TODO: low level API
    fn apply_scratch(&self, rhs_ncols: usize, par: Par) -> StackReq {
        let _rhs_ncols = rhs_ncols;
        let _par = par;
        StackReq::EMPTY
    }

    fn nrows(&self) -> usize {
        self.dim
    }

    fn ncols(&self) -> usize {
        self.dim
    }

    fn apply(&self, out: MatMut<'_, f64>, rhs: MatRef<'_, f64>, par: Par, stack: &mut MemStack) {
        let _stack = stack;
        // TODO: no allocate, use `MemStack`
        let smoothed_parts: Vec<Mat<f64>> = self
            .partition
            .aggregates()
            .par_iter()
            .enumerate()
            .map(|(block_idx, agg)| {
                let mut r_part = Mat::zeros(agg.len(), rhs.ncols());
                for (i, mut part_row) in agg.iter().copied().zip(r_part.row_iter_mut()) {
                    let rhs_row = rhs.row(i);
                    part_row.copy_from(rhs_row);
                }
                // TODO: probably have to split stack
                let mut buf = MemBuffer::new(StackReq::new::<i32>(0));
                let stack = MemStack::new(&mut buf);
                self.blocks[block_idx].apply_in_place(r_part.as_mut(), par, stack);
                r_part
            })
            .collect();

        let mut out = out;
        // TODO: make this par iter
        for (smoothed_part, agg) in smoothed_parts
            .iter()
            .zip(self.partition.aggregates().iter())
        {
            for (i, r_i) in agg.iter().copied().zip(smoothed_part.row_iter()) {
                out.rb_mut().row_mut(i).copy_from(r_i);
            }
        }
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

impl BiLinOp<f64> for BlockSmoother {
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
        let _stack = stack;
        // TODO: no allocate, use `MemStack`
        let smoothed_parts: Vec<Mat<f64>> = self
            .partition
            .aggregates()
            .par_iter()
            .enumerate()
            .map(|(block_idx, agg)| {
                let mut r_part = Mat::zeros(agg.len(), rhs.ncols());
                for (i, mut part_row) in agg.iter().copied().zip(r_part.row_iter_mut()) {
                    let rhs_row = rhs.row(i);
                    part_row.copy_from(rhs_row);
                }
                // TODO: probably have to split stack
                let mut buf = MemBuffer::new(StackReq::new::<i32>(0));
                let stack = MemStack::new(&mut buf);
                self.blocks[block_idx].transpose_apply_in_place(r_part.as_mut(), par, stack);
                r_part
            })
            .collect();

        let mut out = out;
        // TODO: make this par iter
        for (smoothed_part, agg) in smoothed_parts
            .iter()
            .zip(self.partition.aggregates().iter())
        {
            for (i, r_i) in agg.iter().copied().zip(smoothed_part.row_iter()) {
                out.rb_mut().row_mut(i).copy_from(r_i);
            }
        }
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

// TODO: auto impl are fine for now but custom would be better
impl Precond<f64> for BlockSmoother {}
impl BiPrecond<f64> for BlockSmoother {}
