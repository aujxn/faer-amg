use std::{
    collections::{HashMap, HashSet},
    thread,
};

use faer::{
    dyn_stack::{MemStack, StackReq},
    matrix_free::LinOp,
    prelude::{Reborrow, ReborrowMut},
    sparse::{linalg::matmul::dense_sparse_matmul, SparseRowMat, SparseRowMatRef, Triplet},
    Mat, MatMut, MatRef, Par,
};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

#[derive(Debug)]
pub struct ParSpmmOp {
    nrows: usize,
    ncols: usize,
    partition: Vec<usize>,
    work: Vec<ThreadJob>,
}

#[derive(Debug)]
struct ThreadJob {
    diag: SparseRowMat<usize, f64>,
    off_diag: SparseRowMat<usize, f64>,
    local_to_global_map: Vec<usize>,
}

impl ParSpmmOp {
    pub fn new(mat: SparseRowMatRef<usize, f64>, par: Par) -> Self {
        let n_threads = par.degree();
        if n_threads == 1 {
            panic!("not a parallel operator with a single thread");
        }
        let nnz = mat.compute_nnz();
        let per_thread = nnz / n_threads;
        let row_ptrs = mat.row_ptr();
        let nrows = mat.nrows();
        let ncols = mat.ncols();
        let mut partition = vec![0; n_threads + 1];
        partition[n_threads] = nrows;

        match mat.row_nnz() {
            None => {
                let mut nnz_counter = 0;
                let mut thread_id = 1;
                for (row, (start, end)) in row_ptrs.iter().zip(row_ptrs.iter().skip(1)).enumerate()
                {
                    let row_nnz = end - start;
                    nnz_counter += row_nnz;

                    if nnz_counter > per_thread {
                        if nnz_counter > 2 * per_thread {
                            panic!(
                                "This algorithm only makes sense when nnz per row is somewhat balanced..."
                            );
                        }

                        partition[thread_id] = row + 1;
                        nnz_counter = 0;
                        thread_id += 1;
                        if thread_id == n_threads {
                            break;
                        }
                    }
                }
                if thread_id != n_threads {
                    panic!("Matrix nnz per row is too unbalanced for this... Does the matrix have almost all nonzero values in a few rows?");
                }
            }
            Some(_nnz_per_col) => {
                unimplemented!();
            }
        }
        let work: Vec<ThreadJob> = (0..n_threads)
            .into_par_iter()
            .map(|thread_id| {
                ThreadJob::new(mat.as_ref(), partition[thread_id], partition[thread_id + 1])
            })
            .collect();

        /*
        info!("{:?}", partition);
        for (tid, job) in work.iter().enumerate() {
            info!(
                "tid: {}, diag shape / nnz: {}x{} / {}, off diag: {}x{} / {}",
                tid,
                job.diag.nrows(),
                job.diag.ncols(),
                job.diag.compute_nnz(),
                job.off_diag.nrows(),
                job.off_diag.ncols(),
                job.off_diag.compute_nnz(),
            );
        }

        let ws_rows: usize = work.iter().map(|work| work.local_to_global_map.len()).sum();
        */

        ParSpmmOp {
            nrows,
            ncols,
            partition,
            work,
        }
    }

    pub fn implementation(&self, out: MatMut<f64>, rhs: MatRef<f64>, stack: &mut MemStack) {
        let _ = stack;
        thread::scope(|s| {
            for (thread_id, job) in self.work.iter().enumerate() {
                //let (mut ws, _) = temp_mat_zeroed::<f64, _, _>(self.ws_rows, rhs.ncols(), stack);
                //let mut ws = ws.as_mat_mut();
                let start = self.partition[thread_id];
                let end = self.partition[thread_id + 1];
                let nrows = end - start;
                // SAFETY: thread non-overlapping partition of rows
                let mut out = unsafe { out.rb().subrows(start, nrows).const_cast() };
                let rhs_diag = rhs.subrows(start, nrows);
                let _handle = s.spawn(move || {
                    let mut ws: Mat<f64> = Mat::zeros(job.off_diag.ncols(), rhs.ncols());
                    for (ws_row, rhs_row_idx) in ws
                        .row_iter_mut()
                        .zip(job.local_to_global_map.iter().copied())
                    {
                        for (dst, src) in
                            ws_row.iter_mut().zip(rhs.row(rhs_row_idx).iter().copied())
                        {
                            *dst = src;
                        }
                    }
                    job.spmm(out.rb_mut(), rhs_diag, ws.as_ref());
                });
            }
        });
    }
}

impl ThreadJob {
    fn new(mat: SparseRowMatRef<usize, f64>, start: usize, end: usize) -> Self {
        assert!(end > start);
        let nrows = end - start;
        let symbolic = mat.symbolic();

        let mut unique_cols: HashSet<usize> = HashSet::new();
        for row_cols in (start..end).map(|i| symbolic.col_idx_of_row(i)) {
            unique_cols.extend(row_cols.filter(|j| *j < start || *j >= end));
        }
        let mut local_to_global_map: Vec<usize> = unique_cols.into_iter().collect();
        local_to_global_map.sort_unstable();
        let global_map: HashMap<usize, usize> = HashMap::from_iter(
            local_to_global_map
                .iter()
                .copied()
                .enumerate()
                .map(|(a, b)| (b, a)),
        );

        let mut diag: Vec<Triplet<usize, usize, f64>> = Vec::new();
        let mut off_diag: Vec<Triplet<usize, usize, f64>> = Vec::new();

        for row in start..end {
            for (col, val) in symbolic
                .col_idx_of_row(row)
                .zip(mat.val_of_row(row).iter().copied())
            {
                if col < start || col >= end {
                    off_diag.push(Triplet {
                        row: row - start,
                        col: *global_map.get(&col).unwrap(),
                        val,
                    });
                } else {
                    diag.push(Triplet {
                        row: row - start,
                        col: col - start,
                        val,
                    });
                }
            }
        }

        let diag = SparseRowMat::try_new_from_triplets(nrows, nrows, &diag).unwrap();
        let off_diag =
            SparseRowMat::try_new_from_triplets(nrows, local_to_global_map.len(), &off_diag)
                .unwrap();

        ThreadJob {
            diag,
            off_diag,
            local_to_global_map,
        }
    }

    fn spmm(&self, out: MatMut<f64>, rhs_diag: MatRef<f64>, rhs_off_diag: MatRef<f64>) {
        let mut out = out;
        dense_sparse_matmul(
            out.rb_mut().transpose_mut(),
            faer::Accum::Replace,
            rhs_diag.transpose(),
            self.diag.transpose(),
            1.0,
            Par::Seq,
        );

        dense_sparse_matmul(
            out.transpose_mut(),
            faer::Accum::Add,
            rhs_off_diag.transpose(),
            self.off_diag.transpose(),
            1.0,
            Par::Seq,
        );
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
