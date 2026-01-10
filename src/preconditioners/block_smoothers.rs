use std::{
    collections::{BTreeSet, HashSet},
    sync::Arc,
};

use faer::{
    dyn_stack::{MemBuffer, MemStack, StackReq},
    matrix_free::{BiLinOp, BiPrecond, LinOp, Precond},
    prelude::{Reborrow, ReborrowMut},
    sparse::{SparseRowMat, SparseRowMatRef, Triplet},
    Mat, MatMut, MatRef, Par,
};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

use crate::{
    core::SparseMatOp,
    partitioners::{Partition, PartitionerConfig},
    preconditioners::coarse_solvers::CoarseSolverKind,
};

const STEP_CF: f64 = 4.0;

#[derive(Copy, Clone, Debug)]
#[non_exhaustive]
pub enum BlockSmootherKind {
    /// Performs one symmetric Gauss-Seidel sweep per block.
    GaussSeidel,
    /// Decomposes each block into LDL^T with no fill-in. Resulting decomposition has same sparsity
    /// pattern as the block matrix created by removing connections between aggregates in the
    /// partition.
    IncompleteCholesky,
    /// Solves each block with a specified solver.
    BlockSolver(CoarseSolverKind),
}

#[derive(Debug, Clone)]
pub struct BlockSmootherConfig {
    pub block_smoother_kind: BlockSmootherKind,
    pub partitioner_config: PartitionerConfig,
}

impl Default for BlockSmootherConfig {
    fn default() -> Self {
        Self {
            block_smoother_kind: BlockSmootherKind::BlockSolver(CoarseSolverKind::Cholesky),
            partitioner_config: PartitionerConfig::default(),
        }
    }
}

impl BlockSmootherConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn build(&self, base_matrix: SparseMatOp, near_null: Arc<Mat<f64>>) -> BlockSmoother {
        /*
        let mut partition_config = self.partitioner_config.clone();
        let n_levels = 4;
        let cf: f64 = 128.;
        partition_config.coarsening_factor = cf.powf(1. / n_levels as f64);
        partition_config.max_improvement_iters = 100;
        let ml_partitioner_config = MultilevelPartitionerConfig {
            partitioner_configs: vec![partition_config; n_levels],
        };
        */
        let partition = self
            .partitioner_config
            .build_partition(base_matrix.clone(), near_null);
        BlockSmoother::new(base_matrix, Arc::new(partition), self.block_smoother_kind)
    }

    pub fn build_from_partition(
        &self,
        base_matrix: SparseMatOp,
        partition: Arc<Partition>,
    ) -> BlockSmoother {
        BlockSmoother::new(base_matrix, partition, self.block_smoother_kind)
    }
}

#[derive(Debug, Clone)]
pub struct BlockSmoother {
    partition: Arc<Partition>,
    blocks: Vec<Arc<dyn BiPrecond<f64> + Send>>,
    dim: usize,
    vdim: usize,
}

impl BlockSmoother {
    fn new(op: SparseMatOp, partition: Arc<Partition>, smoother: BlockSmootherKind) -> Self {
        let vdim = op.block_size();
        let mat = op.mat_ref();
        assert_eq!(mat.nrows(), partition.nnodes() * vdim);
        let blocks: Vec<Arc<dyn BiPrecond<f64> + Send>> = partition
            .aggregates()
            .par_iter()
            .map(|agg| {
                let csr = if vdim == 1 {
                    diagonally_compensate(agg, mat)
                } else {
                    /*
                    let agg = agg
                        .iter()
                        .map(|block_i| {
                            (0..vdim)
                                .map(|offset_i| block_i * vdim + offset_i)
                                .collect::<Vec<_>>()
                        })
                        .flatten()
                        .collect();
                    diagonally_compensate(&agg, mat)
                    */
                    diagonally_compensate_vector(agg, mat, vdim)
                };
                /*
                let mut test = csr.clone().into_transpose();
                sub_assign(test.as_dyn_mut(), csr.transpose());
                for v in test.val().iter() {
                    if v.abs() > 1e-10 {
                        panic!("not symmetric submatrix");
                    }
                }
                */

                match smoother {
                    BlockSmootherKind::GaussSeidel => {
                        unimplemented!()
                    }
                    BlockSmootherKind::IncompleteCholesky => {
                        unimplemented!()
                    }
                    BlockSmootherKind::BlockSolver(coarse_solver_kind) => {
                        coarse_solver_kind.build_from_sparse(csr.as_ref())
                    }
                }
            })
            .collect();

        Self {
            partition,
            blocks,
            dim: mat.nrows(),
            vdim,
        }
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
                let mut r_part = Mat::zeros(agg.len() * self.vdim, rhs.ncols());
                for (i, mut part_row) in agg
                    .iter()
                    .map(|i| {
                        let i = i * self.vdim;
                        i..(i + self.vdim)
                    })
                    .flatten()
                    .zip(r_part.row_iter_mut())
                {
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

        smoothed_parts
            .par_iter()
            .zip(self.partition.aggregates().par_iter())
            .for_each(|(smoothed_part, agg)| {
                // SAFETY: as long as partition is valid and disjoint then no data race on indexing
                // rows
                let mut out = unsafe { out.rb().const_cast() };
                for (i, r_i) in agg
                    .iter()
                    .map(|i| {
                        let i = i * self.vdim;
                        i..(i + self.vdim)
                    })
                    .flatten()
                    .zip(smoothed_part.row_iter())
                {
                    out.rb_mut().row_mut(i).copy_from(r_i);
                }
            });
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
        self.apply(out, rhs, par, stack);
        /*
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
        */
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

pub fn diagonally_compensate(
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

pub fn diagonally_compensate_vector(
    agg: &BTreeSet<usize>,
    mat: SparseRowMatRef<usize, f64>,
    vdim: usize,
) -> SparseRowMat<usize, f64> {
    let block_size = agg.len() * vdim;
    let agg: Vec<usize> = agg.iter().copied().collect();
    let mut block_triplets = Vec::new();
    let symbolic = mat.symbolic();

    let mut to_compensate = HashSet::new();
    let mut diag: Vec<Mat<f64>> = vec![Mat::zeros(vdim, vdim); agg.len()];

    for (block_ic, block_i) in agg.iter().copied().enumerate() {
        for offset_i in 0..vdim {
            let i = block_i * vdim + offset_i;
            let ic = block_ic * vdim + offset_i;
            let values_row_i = mat.val_of_row(i);
            for (j, val) in symbolic.col_idx_of_row(i).zip(values_row_i.iter()) {
                let block_j = j / vdim;
                let offset_j = j % vdim;

                if block_j == block_i {
                    assert_eq!(diag[block_ic][(offset_i, offset_j)], 0.0);
                    diag[block_ic][(offset_i, offset_j)] = *val;
                } else {
                    match agg.binary_search(&block_j) {
                        Ok(block_jc) => {
                            let jc = block_jc * vdim + offset_j;
                            block_triplets.push(Triplet::new(ic, jc, *val));
                        }
                        Err(_) => {
                            let i_start = i - offset_i;
                            let j_start = j - offset_j;
                            to_compensate.insert((block_ic, (i_start, j_start)));
                        }
                    }
                }
            }
        }
    }

    for (block_ic, (i, j)) in to_compensate {
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
        let usut: Mat<f64> = u * (s * u.transpose());

        diag[block_ic] += 0.5 * usut;
    }

    for (block_ic, diag_block) in diag.iter().enumerate() {
        let ic = block_ic * vdim;
        for i_off in 0..vdim {
            for j_off in 0..vdim {
                block_triplets.push(Triplet::new(
                    ic + i_off,
                    ic + j_off,
                    diag_block[(i_off, j_off)],
                ));
            }
        }
    }

    SparseRowMat::try_new_from_triplets(block_size, block_size, &block_triplets).unwrap()
}
