use std::{collections::BTreeMap, time::Instant};

use faer::{
    dyn_stack::{MemStack, StackReq},
    matrix_free::LinOp,
    prelude::{Reborrow, ReborrowMut},
    sparse::{linalg::matmul::sparse_dense_matmul, SparseColMat, SparseRowMatRef, Triplet},
    MatMut, MatRef, Par,
};
use log::info;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};

pub const PAR_BLOCK_SIZE: usize = 8192;

#[derive(Debug)]
pub struct ParSpmmOp {
    nrows: usize,
    ncols: usize,
    block_rows: Vec<BlockRow>,
}

#[derive(Debug)]
pub struct BlockRow {
    block_cols: Vec<usize>,
    blocks: Vec<SparseColMat<usize, f64>>,
}

impl ParSpmmOp {
    pub fn new(mat: SparseRowMatRef<usize, f64>, par: Par) -> Self {
        let start = Instant::now();
        let n_threads = par.degree();
        if n_threads == 1 {
            panic!("not a parallel operator with a single thread");
        }
        let block_nrows = (mat.nrows() + PAR_BLOCK_SIZE - 1) / PAR_BLOCK_SIZE;
        //let block_ncols = (mat.ncols() + PAR_BLOCK_SIZE - 1) / PAR_BLOCK_SIZE;

        let block_rows: Vec<BlockRow> = (0..block_nrows)
            .into_par_iter()
            .map(|block_i| {
                let mut nonzero_blocks: BTreeMap<usize, Vec<Triplet<usize, usize, f64>>> =
                    BTreeMap::new();
                let start_row = block_i * PAR_BLOCK_SIZE;
                let end_row = std::cmp::min(start_row + PAR_BLOCK_SIZE, mat.ncols());

                for (local_i, i) in (start_row..end_row).enumerate() {
                    let row_vals = mat.val_of_row(i);
                    let cols = mat.col_idx_of_row(i);
                    for (j, val) in cols.zip(row_vals.iter()) {
                        let block_j = j / PAR_BLOCK_SIZE;
                        let local_j = j % PAR_BLOCK_SIZE;
                        let triplet = Triplet {
                            row: local_i,
                            col: local_j,
                            val: *val,
                        };
                        if let Some(triplets) = nonzero_blocks.get_mut(&block_j) {
                            triplets.push(triplet);
                        } else {
                            let triplets = vec![triplet];
                            nonzero_blocks.insert(block_j, triplets);
                        }
                    }
                }

                let local_n_rows = end_row - start_row;
                let mut blocks: Vec<SparseColMat<usize, f64>> =
                    Vec::with_capacity(nonzero_blocks.len());
                let mut block_cols: Vec<usize> = Vec::with_capacity(nonzero_blocks.len());
                for (block_j, triplets) in nonzero_blocks {
                    let end_col = std::cmp::min(mat.ncols(), (block_j + 1) * PAR_BLOCK_SIZE);
                    let start_col = block_j * PAR_BLOCK_SIZE;
                    let local_n_cols = end_col - start_col;
                    let block =
                        SparseColMat::try_new_from_triplets(local_n_rows, local_n_cols, &triplets)
                            .unwrap();
                    blocks.push(block);
                    block_cols.push(block_j);
                }
                BlockRow { block_cols, blocks }
            })
            .collect();

        let duration = Instant::now() - start;
        info!(
            "Finished par op construction in {} seconds",
            duration.as_secs()
        );
        ParSpmmOp {
            nrows: mat.nrows(),
            ncols: mat.ncols(),
            block_rows,
        }
    }

    pub fn implementation(&self, out: MatMut<f64>, rhs: MatRef<f64>, stack: &mut MemStack) {
        let _ = stack;
        self.block_rows
            .par_iter()
            .enumerate()
            .for_each(|(block_i, block_row)| {
                let start = block_i * PAR_BLOCK_SIZE;
                let end = std::cmp::min(start + PAR_BLOCK_SIZE, self.nrows);
                let nrows = end - start;
                // SAFETY: thread non-overlapping partition of rows
                let out = unsafe { out.rb().subrows(start, nrows).const_cast() };
                block_row.spmm(out, rhs.rb());
            });
    }
}

impl BlockRow {
    fn spmm(&self, out: MatMut<f64>, rhs: MatRef<f64>) {
        let mut out = out;
        out.fill(0.0);

        for (block_j, mat) in self.block_cols.iter().zip(self.blocks.iter()) {
            let start = block_j * PAR_BLOCK_SIZE;
            let n_cols = mat.ncols();
            let rhs_slice = rhs.subrows(start, n_cols);
            sparse_dense_matmul(
                out.rb_mut(),
                faer::Accum::Add,
                mat.as_ref(),
                rhs_slice,
                1.0,
                Par::Seq,
            );
        }
    }
}

impl LinOp<f64> for ParSpmmOp {
    fn nrows(&self) -> usize {
        self.nrows
    }

    fn ncols(&self) -> usize {
        self.ncols
    }

    fn apply_scratch(&self, rhs_ncols: usize, par: faer::Par) -> StackReq {
        let _ = par;
        let _ = rhs_ncols;
        //temp_mat_scratch::<f64>(self.ws_rows, rhs_ncols)
        StackReq::empty()
    }

    fn apply(&self, out: MatMut<f64>, rhs: MatRef<f64>, par: Par, stack: &mut MemStack) {
        let _par = par;
        self.implementation(out, rhs, stack);
    }

    fn conj_apply(&self, out: MatMut<f64>, rhs: MatRef<f64>, par: Par, stack: &mut MemStack) {
        self.apply(out, rhs, par, stack);
    }
}
