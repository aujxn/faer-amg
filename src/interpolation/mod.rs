use faer::{
    diag::DiagRef,
    dyn_stack::{MemBuffer, MemStack},
    get_global_parallelism,
    matrix_free::{LinOp, Precond},
    prelude::{Reborrow, Solve},
    sparse::{ops::add_assign, SparseRowMat, SparseRowMatRef, Triplet},
    Col, ColRef, Mat, MatRef, Row, RowRef,
};
use itertools::Itertools;
use log::{info, trace};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::{collections::BTreeSet, f64, sync::Arc};

// TODO: split this up into seperate modules for aggregation and ruge-stuben like interpolation schemes
//pub mod classical;
//pub mod aggregation;

use crate::{
    adaptivity::ErrorPropogator,
    core::SparseMatOp,
    hierarchy::PartitionType,
    partitioners::{extract_local_subgraph, AdjacencyList, Partition},
    preconditioners::block_smoothers::BlockSmootherConfig,
    utils::{matrix_stats, MatrixStats},
};

#[derive(Clone, Debug)]
pub struct InterpolationInfo {
    pub stats: MatrixStats,
}

impl InterpolationInfo {
    pub fn new(p: &SparseRowMat<usize, f64>) -> Self {
        Self {
            stats: matrix_stats(p.as_ref()),
        }
    }

    pub fn stats(&self) -> &MatrixStats {
        &self.stats
    }
}

#[derive(Clone, Copy, Debug)]
pub enum InterpMode {
    Constrained,
    Regularized,
}

#[derive(Clone, Debug)]
struct LsInterpResult {
    /// Active weights, aligned with `set` (same length).
    weights: Row<f64>,
    /// Local indices into the candidate `vc` rows.
    set: Vec<usize>,
    /// Weighted squared error.
    err: f64,
}

impl LsInterpResult {
    fn empty(err: f64) -> Self {
        Self {
            weights: Row::<f64>::zeros(0),
            set: Vec::new(),
            err,
        }
    }

    fn size(&self) -> usize {
        self.set.len()
    }
}

// negative weights allowed for regularized LS approach
fn validate_weights(p: RowRef<f64>, min_abs: f64, min_rel: f64) -> bool {
    let mut max_w: f64 = 0.0;
    for &x in p.iter() {
        if !x.is_finite() || x.abs() < min_abs {
            return false;
        }
        max_w = max_w.max(x.abs());
    }

    let rel_thr = min_rel * max_w;
    for &x in p.iter() {
        if x.abs() < rel_thr {
            return false;
        }
    }
    true
}

// only positive weights allowed for constrained LS approach
fn validate_weights_constrained(p: RowRef<f64>, min_abs: f64, min_rel: f64, feas: f64) -> bool {
    let mut max_w: f64 = 0.0;
    let mut sum: f64 = 0.0;
    for &x in p.iter() {
        if !x.is_finite() || x < min_abs {
            return false;
        }
        max_w = max_w.max(x);
        sum += x;
    }

    if sum > 1.0 + feas {
        return false;
    }

    let rel_thr = min_rel * max_w;

    for &x in p.iter() {
        if x < rel_thr {
            return false;
        }
    }

    true
}

fn extract_gram_rhs(gram: MatRef<f64>, g: ColRef<f64>, idx: &[usize]) -> (Mat<f64>, Col<f64>) {
    let r = idx.len();
    let mut gram_ff = Mat::zeros(r, r);
    let mut gf = Col::zeros(r);
    for (ii, &i) in idx.iter().enumerate() {
        gf[ii] = g[i];
        for (jj, &j) in idx.iter().enumerate() {
            gram_ff[(ii, jj)] = gram[(i, j)];
        }
    }
    (gram_ff, gf)
}

fn eval_err(gram_ff: MatRef<f64>, p_row: RowRef<f64>, gf: ColRef<f64>, btb: f64) -> f64 {
    let p_col = p_row.transpose();
    let quad = p_row.rb() * gram_ff * p_col.rb();
    let lin = gf.transpose() * p_col.rb();
    btb + quad - 2.0 * lin
}

fn weighted_least_squares(
    gram_ff: MatRef<f64>,
    gf: ColRef<f64>,
    btb: f64,
) -> Option<(Row<f64>, f64)> {
    let min_abs = 1e-10;
    let min_rel = 1e-2;
    let eta = 1e-2;
    let lambda = eta
        * gram_ff
            .self_adjoint_eigenvalues(faer::Side::Upper)
            .unwrap()
            .last()
            .unwrap();

    let k = gram_ff.nrows();
    let gram_reg = Mat::<f64>::identity(k, k) * lambda + gram_ff;
    let eig_decomp = gram_reg.self_adjoint_eigen(faer::Side::Upper).unwrap();

    let pt = eig_decomp.pseudoinverse() * gf;
    let p = pt.transpose();

    if validate_weights(p.rb(), min_abs, min_rel) {
        Some((p.to_owned(), eval_err(gram_ff.rb(), p.rb(), gf, btb)))
    } else {
        None
    }
}

fn constrained_subset_qp(
    gram_ff: MatRef<f64>,
    gf: ColRef<f64>,
    btb: f64,
) -> Option<(Row<f64>, f64)> {
    let r = gram_ff.nrows();

    let feas_tol = 1e-12;
    let min_abs = 1e-10;
    let min_rel = 1e-2;

    // ---------- Candidate A (sum constraint inactive)
    if let Ok(eig) = gram_ff.self_adjoint_eigen(faer::Side::Upper) {
        let pt = eig.pseudoinverse() * gf; // r x 1
        let p = pt.transpose(); // 1 x r

        if validate_weights_constrained(p, min_abs, min_rel, feas_tol) {
            return Some((p.to_owned(), eval_err(gram_ff.rb(), p.rb(), gf, btb)));
        }
    }

    // ---------- Candidate B (sum constraint active: sum(p)=1)
    let n = r + 1;

    // KKT = [[G, 1], [1^T, 0]]
    let mut kkt = Mat::<f64>::ones(n, n);
    kkt.submatrix_mut(0, 0, r, r).copy_from(gram_ff);
    kkt[(r, r)] = 0.0;

    let mut rhs = Col::<f64>::ones(n);
    rhs.subrows_mut(0, r).copy_from(gf);

    let decomp = faer::linalg::solvers::Lblt::new(kkt.as_ref(), faer::Side::Upper);
    let mut sol = rhs;
    decomp.solve_in_place(sol.as_mut());

    let pt = sol.subrows(0, r).to_owned();
    let p = pt.transpose();

    if validate_weights_constrained(p, min_abs, min_rel, feas_tol) {
        return Some((p.to_owned(), eval_err(gram_ff.rb(), p.rb(), gf, btb)));
    }

    None
}

fn ls_interp_weights(
    vf: RowRef<f64>,
    vc: MatRef<f64>, // L x k (rows are candidate C-points)
    d: DiagRef<f64>, // k x k (Q)
    max_interp: usize,
    gamma: Option<f64>,
    mode: InterpMode,
) -> LsInterpResult {
    let k = vf.ncols();
    let l = vc.nrows();

    assert_eq!(k, vc.ncols(), "vf and vc k mismatch");
    assert_eq!(k, d.nrows(), "weight matrix wrong size");
    assert_eq!(k, d.ncols(), "weight matrix wrong size");

    let local_max = l.min(max_interp);

    // gram = V Q V^T, g = V Q vf^T, btb = vf Q vf^T
    let vc_d = vc.rb() * d.rb(); // l x k
    let gram = vc_d.rb() * vc.transpose(); // l x l
    let g = vc_d.rb() * vf.transpose(); // l x 1
    let btb = vf.rb() * d.rb() * vf.transpose();

    // best accepted interpolation (set / weights / error)
    let mut accepted = LsInterpResult::empty(btb);

    for r in 1..=local_max {
        // best interpolation of size `r`
        let mut best_r: Option<LsInterpResult> = None;

        for idx in (0..l).combinations(r) {
            let (gram_ff, gf) = extract_gram_rhs(gram.as_ref(), g.rb(), &idx);

            let cand = match mode {
                InterpMode::Constrained => {
                    constrained_subset_qp(gram_ff.as_ref(), gf.as_ref(), btb)
                }
                InterpMode::Regularized => {
                    weighted_least_squares(gram_ff.as_ref(), gf.as_ref(), btb)
                }
            }
            .map(|(w, err)| LsInterpResult {
                weights: w,
                set: idx.clone(),
                err,
            });

            if let Some(cand) = cand {
                match &best_r {
                    None => best_r = Some(cand),
                    Some(best) => {
                        if cand.err < best.err {
                            best_r = Some(cand);
                        }
                    }
                }
            }
        }

        // if no candidate of size r exists, continue to next size
        let Some(best_r) = best_r else { continue };

        // acceptance rule relative to currently accepted solution
        let accept = match gamma {
            None => best_r.err < accepted.err,
            Some(gam) => {
                let dr = (best_r.size() - accepted.size()) as f64;
                best_r.err < accepted.err.powf(gam * dr)
            }
        };

        if accept {
            accepted = best_r;
        }
    }

    accepted
}

#[derive(Debug, Clone)]
pub struct CoarseFineSplit {
    c_points: Vec<usize>,
}

impl CoarseFineSplit {
    pub fn new(c_points: Vec<usize>) -> Self {
        assert!(c_points.is_sorted(), "`c_points` should be sorted");
        Self { c_points }
    }
    pub fn coarse_idx(&self, fine_idx: usize) -> Option<usize> {
        self.c_points.binary_search(&fine_idx).ok()
    }

    pub fn fine_idx(&self, coarse_idx: usize) -> usize {
        self.c_points[coarse_idx]
    }

    pub fn c_points(&self) -> &Vec<usize> {
        &self.c_points
    }

    pub fn into_c_points(self) -> Vec<usize> {
        self.c_points
    }
}

/* TODO: create config / builder pattern for interpolation (least squares or aggregation based)
*
* LS config options:
* - depth / depth_ls
* - constrained / regularized
* - max_interp
* - tau
*
* - CR sub-options:
*   - sigma func?
*   - CR smoother
*   - target convergence
*   - relax steps
*/
pub fn least_squares(
    fine_mat: SparseMatOp,
    smoother_partition: Arc<Partition>,
    // TODO:? block_size: usize,
    near_null: MatRef<f64>,
    nn_weights: Vec<f64>,
) -> (
    Mat<f64>,
    SparseRowMat<usize, f64>,
    SparseRowMat<usize, f64>,
    SparseRowMat<usize, f64>,
    PartitionType,
) {
    //let _ = block_size;
    let matref = fine_mat.mat_ref();
    let n_fine = matref.nrows();
    let k = near_null.ncols();
    let d = Col::from_iter(nn_weights.iter().copied().take(near_null.ncols())).into_diagonal();
    let mut u0 = Col::zeros(n_fine);
    for col in near_null.col_iter() {
        u0 += col;
    }
    let par = get_global_parallelism();

    let depth = 3;
    let strength_graph =
        AdjacencyList::new_ls_strength_graph(matref, near_null, &nn_weights, depth);

    let mut reduction_factor = 1.0;
    let target_reduction = 0.15;
    let relax_steps = 5;
    let mut c_points = BTreeSet::new();
    #[derive(Copy, Clone, PartialEq, Eq, Debug)]
    enum PointType {
        F,
        C,
        N,
    }
    let mut partition = vec![PointType::F; n_fine];

    let mut cr_iter = 1;
    while reduction_factor > target_reduction {
        let mut f_points: Vec<bool> = partition.iter().map(|p| *p == PointType::F).collect();
        let new_c_points = strength_graph.maximal_independent_set(&mut f_points);
        for c_point in new_c_points.iter() {
            assert_eq!(partition[*c_point], PointType::F);
            partition[*c_point] = PointType::C;
        }
        c_points.extend(new_c_points);
        let identity_f = Col::from_iter(partition.iter().map(|point_type| match point_type {
            PointType::C => 0.0,
            _ => 1.0,
        }))
        .into_diagonal();
        let u0f = identity_f.rb() * u0.rb();
        let start_norm = u0f.norm_l2();
        let mut unuf = u0f.clone();
        let mut af = identity_f.rb() * matref * identity_f.rb();
        for (i, point_type) in partition.iter().copied().enumerate() {
            if point_type == PointType::C {
                *af.get_mut(i, i).unwrap() = 1.0;
            }
        }
        let af = SparseMatOp::new(af, 1, par);
        // TODO: should abstract to `fn SparseMatOp -> Precond<f64>` instead of passing in
        // partition for use of any smoother in API
        let mf = BlockSmootherConfig::default()
            .build_from_partition(af.clone(), smoother_partition.clone());

        let iteration = ErrorPropogator {
            op: af.dyn_op(),
            pc: Arc::new(mf),
        };
        let stack_req = iteration.apply_in_place_scratch(unuf.ncols(), par);
        let mut buf = MemBuffer::new(stack_req);
        let stack = MemStack::new(&mut buf);

        for _ in 0..relax_steps {
            iteration.apply_in_place(unuf.as_mat_mut(), par, stack);
        }
        let end_norm = unuf.norm_l2();
        reduction_factor = (end_norm / start_norm).powf(1. / relax_steps as f64);
        trace!(
            "CR iter {}, {} c-points, {:.2} reduction factor\n\t{:.2e} start norm and {:.2e} end norm",
            cr_iter,
            c_points.len(),
            reduction_factor,
            start_norm,
            end_norm
        );
        let tol = 1.0 - reduction_factor;
        let inf_norm = unuf.norm_max();
        for i in 0..n_fine {
            let sigma_i = unuf[i].abs() / inf_norm;
            if sigma_i > tol {
                partition[i] = PointType::F;
            } else if partition[i] == PointType::F {
                partition[i] = PointType::N;
            }
        }
        cr_iter += 1;
    }
    info!(
        "Compatible Relaxation completed in {} iters with {} c-points",
        cr_iter,
        c_points.len()
    );

    let n_coarse = c_points.len();
    let max_interp = 3;
    let mut p_triplets = Vec::new();
    let d_ls = depth + 2;
    let tau = 1.2;

    let mut coarse_near_null = Mat::zeros(c_points.len(), k);
    for ((coarse_idx, c_point), mut coarse_row) in c_points
        .iter()
        .enumerate()
        .zip(coarse_near_null.row_iter_mut())
    {
        coarse_row.copy_from(near_null.row(*c_point));
        p_triplets.push(Triplet::new(*c_point, coarse_idx, 1.0));
    }

    let interp_data: Vec<(usize, Row<f64>, Vec<usize>)> = (0..n_fine)
        .into_par_iter()
        .filter(|i| partition[*i] != PointType::C)
        .map(|i| {
            let mut local = extract_local_subgraph(matref.symbolic(), i, d_ls);
            local.retain(|j| partition[*j] == PointType::C);
            let c_point_local_to_global: Vec<usize> = local.iter().copied().collect();
            let vf = near_null.row(i);
            let l = local.len();
            let mut vc = Mat::zeros(l, k);
            for (mut row, j) in vc.row_iter_mut().zip(local.iter().copied()) {
                row.copy_from(near_null.row(j));
            }

            let res = ls_interp_weights(
                vf,
                vc.as_ref(),
                d.as_ref(),
                max_interp,
                Some(tau),
                InterpMode::Constrained,
            );

            (
                i,
                res.weights,
                res.set
                    .into_iter()
                    .map(|local_i| c_point_local_to_global[local_i])
                    .collect(),
            )
        })
        .collect();

    let partition = CoarseFineSplit::new(c_points.into_iter().collect());
    for (i, best_p, best_set) in interp_data {
        for (p_ij, fine_j) in best_p.iter().copied().zip(best_set) {
            p_triplets.push(Triplet::new(i, partition.coarse_idx(fine_j).unwrap(), p_ij));
        }
    }

    let p = SparseRowMat::try_new_from_triplets(n_fine, n_coarse, &p_triplets)
        .expect("failed to create interp sparse row matrix");
    let debug_stats = matrix_stats(p.as_ref());
    info!("{:?}", debug_stats);

    let r = p
        .transpose()
        .to_row_major()
        .expect("failed to transpose interp to form restriction");
    let mat_coarse = &r * &(matref * &p);
    (
        coarse_near_null,
        r,
        p,
        mat_coarse,
        PartitionType::Classical(Arc::new(partition)),
    )
}

pub fn smoothed_aggregation(
    fine_mat: SparseRowMatRef<usize, f64>,
    partition: Arc<Partition>,
    block_size: usize,
    near_null: MatRef<f64>,
) -> (
    Mat<f64>,
    SparseRowMat<usize, f64>,
    SparseRowMat<usize, f64>,
    SparseRowMat<usize, f64>,
    PartitionType,
) {
    let n_fine = fine_mat.nrows();
    let n_coarse = partition.aggregates().len();
    assert_eq!(n_fine % block_size, 0);
    assert_eq!(n_fine, partition.nnodes() * block_size);
    assert_eq!(n_fine, near_null.nrows());
    let k = near_null.ncols();
    let mut coarse_near_null = Mat::zeros(n_coarse * k, k);

    // n_fine by n_coarse * k
    let mut p: Vec<Triplet<usize, usize, f64>> = Vec::new();
    for (coarse_idx, agg) in partition.aggregates().iter().enumerate() {
        let local_rows = agg.len() * block_size;
        // could relax this but makes things much messier
        assert!(
            local_rows >= k,
            "Agg size of {} cannot support near-null dimension of {}",
            local_rows,
            k
        );
        let mut local = Mat::zeros(local_rows, k);
        for (local_j, j) in agg.iter().copied().enumerate() {
            local
                .subrows_mut(local_j * block_size, block_size)
                .copy_from(near_null.subrows(j * block_size, block_size));
            /*
            for offset in 0..block_size {
                for (dest, src) in local
                    .row_mut(local_j * block_size + offset)
                    .iter_mut()
                    .zip(near_null.row(j * block_size + offset).iter())
                {
                    *dest = *src;
                }
            }
            */
        }

        let svd = local.thin_svd().unwrap();
        let q = svd.U();
        let s = svd.S();
        let r = svd.V();

        /*
        let s_first = s[0];
        let s_last = s[k - 1];
        if s_last / s_first < 1e-6 {
            warn!(
                "Local basis in linearly dependent...\n\tsingular values: {:?}",
                s
            );
        }
        */
        let r = s * r.transpose();

        coarse_near_null.subrows_mut(coarse_idx * k, k).copy_from(r);

        for (local_i, fine_i) in agg.iter().copied().enumerate() {
            let col_start = coarse_idx * k;
            let sub_q = q.subrows(local_i * block_size, block_size);
            let row_start = fine_i * block_size;
            for offset_i in 0..block_size {
                for offset_j in 0..k {
                    p.push(Triplet {
                        row: row_start + offset_i,
                        col: col_start + offset_j,
                        val: sub_q[(offset_i, offset_j)],
                    });
                }
            }
        }
    }

    let mut p = SparseRowMat::try_new_from_triplets(n_fine, n_coarse * k, &p)
        .expect("failed to create SA interp sparse row matrix");
    let debug_stats = matrix_stats(p.as_ref());
    info!("{:?}", debug_stats);

    if block_size == 1 {
        p = smooth_interpolation(fine_mat, p.as_ref(), 0.66);
    } else {
        p = block_jacobi(fine_mat, block_size, p.as_ref());
    };
    /*
    let m_inv = csr_block_smoother(partition, fine_mat.rb(), block_size);
    p = smooth_p(fine_mat, m_inv.as_ref(), p.as_ref());
    */

    let r = p
        .transpose()
        .to_row_major()
        .expect("failed to transpose SA interp to form restriction");
    let mat_coarse = &r * &(fine_mat * &p);
    (
        coarse_near_null,
        r,
        p,
        mat_coarse,
        PartitionType::Aggregation(partition),
    )
}

/*
fn csr_block_smoother(
    partition: &Partition,
    mat: SparseRowMatRef<usize, f64>,
    vdim: usize,
) -> SparseRowMat<usize, f64> {
    let _n_aggs = partition.naggs();
    let n = mat.nrows();
    let permutation: Vec<usize> = partition
        .aggregates()
        .iter()
        .map(|agg| {
            agg.iter()
                .map(|block_idx| {
                    let start = block_idx * vdim;
                    let end = start + vdim;
                    (start..end).collect::<Vec<usize>>()
                })
                .flatten()
                .collect::<Vec<usize>>()
        })
        .flatten()
        .collect();

    let diag_block_inv: Vec<Mat<f64>> = partition
        .aggregates()
        .par_iter()
        .map(|agg| {
            let local = if vdim == 1 {
                diagonally_compensate(agg, mat.rb())
            } else {
                diagonally_compensate_vector(agg, mat.rb(), vdim)
            };
            local
                .to_dense()
                .self_adjoint_eigen(faer::Side::Upper)
                .unwrap()
                .pseudoinverse()
        })
        .collect();

    let mut triplets = Vec::new();
    let mut start_idx = 0;
    for block in diag_block_inv {
        let local_dim = block.nrows();
        for local_i in 0..local_dim {
            for local_j in 0..local_dim {
                let val = block[(local_i, local_j)];
                let i = start_idx + local_i;
                let j = start_idx + local_j;
                let row = permutation[i];
                let col = permutation[j];
                triplets.push(Triplet { row, col, val });
            }
        }
        start_idx += local_dim;
    }
    SparseRowMat::try_new_from_triplets(n, n, &triplets).unwrap()
}

pub fn local_diag_inv(
    agg: &BTreeSet<usize>,
    mat: SparseRowMatRef<usize, f64>,
    vdim: usize,
    rhs: MatMut<f64>,
) {
    let mut rhs = rhs;
    for (local_block_idx, _global_block_idx) in agg.iter().enumerate() {
        let mut block_inv = Mat::zeros(vdim, vdim);
        //let global_start = global_block_idx * vdim;
        let local_start = local_block_idx * vdim;
        for offset_i in 0..vdim {
            //let global_i = global_start + offset_i;
            for offset_j in 0..vdim {
                //let global_j = global_start + offset_j;
                //let mat_ij = mat.get(global_i, global_j).unwrap_or(&0.0);
                let mat_ij = mat
                    .get(local_start + offset_i, local_start + offset_j)
                    .unwrap_or(&0.0);
                block_inv[(offset_i, offset_j)] = *mat_ij;
            }
        }
        let sub_rhs = rhs.rb_mut().subrows_mut(local_start, vdim);
        block_inv.partial_piv_lu().solve_in_place(sub_rhs);
    }
    rhs *= 0.5;
}
*/

pub fn smooth_interpolation(
    mat: SparseRowMatRef<usize, f64>,
    p: SparseRowMatRef<usize, f64>,
    jacobi_weight: f64,
) -> SparseRowMat<usize, f64> {
    let diag_inv_iter = (0..mat.nrows()).map(|i| {
        let diag_val = mat.get(i, i).unwrap();
        assert!(*diag_val > 1e-6, "Diagonal nearly zero: {:.2e}", diag_val);
        jacobi_weight * diag_val.recip()
    });

    let mut smoothed = mat * p;
    for (row, scalar) in diag_inv_iter.enumerate() {
        let row_vals = smoothed.val_of_row_mut(row);
        for v in row_vals.iter_mut() {
            *v *= -scalar;
        }
    }
    add_assign(smoothed.transpose_mut(), p.transpose());
    smoothed
    /* TODO: I don't know why this is different from the above... it's concerning.
    let diag_inv =
        Col::from_iter((0..mat.nrows()).map(|i| jacobi_weight * mat.get(i, i).unwrap().recip()))
            .into_diagonal();

    let ap = mat * p;
    let smoothed = ap.transpose() * diag_inv.as_ref();
    let diff_symbolic = union_symbolic(p.transpose().symbolic(), smoothed.symbolic())
        .expect("failed to compute sparsity pattern for smoothed interpolation");
    let nnz = diff_symbolic.row_idx().len();
    let mut diff_numeric = SparseColMat::new(diff_symbolic, vec![0.0; nnz]);
    sub_into(diff_numeric.as_dyn_mut(), p.transpose(), smoothed.as_ref());
    diff_numeric.into_transpose()
    */
}

pub fn block_jacobi(
    mat: SparseRowMatRef<usize, f64>,
    block_size: usize,
    p: SparseRowMatRef<usize, f64>,
) -> SparseRowMat<usize, f64> {
    let ndofs = mat.nrows();
    let n_blocks = ndofs / block_size;
    let mut d_inv: Vec<Triplet<usize, usize, f64>> = Vec::new();

    for block_idx in 0..n_blocks {
        let start = block_idx * block_size;

        let mut block = Mat::zeros(block_size, block_size);
        for i in 0..block_size {
            for j in 0..block_size {
                //if i >= j {
                if let Some(v) = mat.get(start + i, start + j) {
                    block[(i, j)] = *v;
                }
                //}
            }
        }

        /*
        let zero_test = block.as_ref() - block.transpose();
        assert!(zero_test.norm_max() < 1e-12);
        */

        let spectral = block
            .self_adjoint_eigen(faer::Side::Lower)
            .expect("failed to decompose small local eigenproblem");
        let u = spectral.U();
        let s = spectral.S();
        let s = s
            .column_vector()
            .iter()
            .map(|v| {
                assert!(
                    *v > 1e-6,
                    "block diagonal is nearly singular with eigval of: {:.3e}",
                    v
                );
                v.recip()
            })
            .collect::<Col<f64>>()
            .into_diagonal();

        let block = u.as_ref() * s * u.transpose();
        //let block = spectral.pseudoinverse();

        for i in 0..block_size {
            for j in 0..block_size {
                d_inv.push(Triplet::new(start + i, start + j, -0.66 * block[(i, j)]));
            }
        }
    }

    let d_inv = SparseRowMat::try_new_from_triplets(ndofs, ndofs, &d_inv)
        .expect("failed to create sparse row mat from block diag inverse triplets");
    let ap = mat * p;
    let mut smoothed = d_inv * ap;
    //let d_inv_a = d_inv * mat;
    //let mut smoothed = d_inv_a * p;
    add_assign(smoothed.transpose_mut(), p.transpose());
    smoothed
}

pub fn smooth_p(
    mat: SparseRowMatRef<usize, f64>,
    m_inv: SparseRowMatRef<usize, f64>,
    p: SparseRowMatRef<usize, f64>,
) -> SparseRowMat<usize, f64> {
    let mut ap = mat * p;
    ap *= -1.0;
    let mut smoothed = m_inv * ap;
    add_assign(smoothed.transpose_mut(), p.transpose());
    smoothed
}
