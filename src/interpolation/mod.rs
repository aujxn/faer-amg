use std::{
    collections::{BTreeMap, BTreeSet},
    f64,
};

use faer::{
    diag::{self, Diag, DiagRef},
    linalg::matmul::dot::inner_prod,
    mat::{AsMatMut, AsMatRef},
    matrix_free::LinOp,
    prelude::{Reborrow, ReborrowMut, Solve, SolveLstsq},
    sparse::{ops::add_assign, SparseRowMat, SparseRowMatRef, Triplet},
    Col, ColRef, Mat, MatMut, MatRef, Row, RowRef,
};
use log::{info, warn};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::{
    partitioners::{AdjacencyList, Partition},
    preconditioners::block_smoothers::{diagonally_compensate, diagonally_compensate_vector},
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

/*
#[derive(Clone, Debug, Serialize)]
pub struct InterpViz {
    pub functions: Mat<f64>,
    pub interpolated: Mat<f64>,
    pub c_points: Vec<usize>,
}
*/

fn best_coarse(agg: &BTreeSet<usize>, near_null: MatRef<f64>, w: DiagRef<f64>) -> (usize, f64) {
    let mut best_node_id = *agg.first().unwrap();
    let mut best_error = f64::MAX;

    let mut vf = Mat::zeros(agg.len(), near_null.ncols());
    for (node_id, mut local_row) in agg.iter().zip(vf.row_iter_mut()) {
        local_row.copy_from(near_null.row(*node_id));
    }

    /*
    let trace = vf
        .col_iter()
        .zip(w.column_vector().iter())
        .map(|(vi, wi)| vi.squared_norm_l2() * wi)
        .sum::<f64>();
    */
    //println!("{:?}", vf);

    for (vc, node_id) in vf.row_iter().zip(agg.iter()) {
        //println!("------------");
        /*
        let w_vc = w.rb() * vc.transpose();
        let p = vf.rb() * w_vc.rb();
        let numerator = p.squared_norm_l2();
        let denom = vc.rb() * w_vc.rb();
        //let p = p / denom;
        let error_norm = trace - numerator / denom;
        */
        let error_norm = vf
            .row_iter()
            .map(|vf| weighted_least_squares(w, vc.as_mat(), vf.rb()).1)
            .sum::<f64>();
        //println!("{:?}", p);
        //println!("{:?}", p.rb() * vc.as_mat());
        if error_norm < best_error {
            best_error = error_norm;
            best_node_id = *node_id;
        }
    }
    //panic!();
    (best_node_id, best_error)
}

pub fn weighted_least_squares(
    d: DiagRef<f64>,
    vc: MatRef<f64>,
    vf: RowRef<f64>,
) -> (Row<f64>, f64) {
    let k = vf.ncols();
    let m = vc.nrows();
    assert_eq!(
        k,
        vc.ncols(),
        "incompatibale interpolation dimensions (columns don't match): vf: {:?} and vc: {:?}",
        vf.shape(),
        vc.shape()
    );
    assert_eq!(
        k,
        d.ncols(),
        "weight matrix wrong size: vf: {:?}, vc: {:?}, k: {}",
        vf.shape(),
        vc.shape(),
        d.nrows()
    );
    assert!(
        k > m,
        "need more test vectors than coarse points: k: {}, vc: {:?}",
        k,
        vc.shape()
    );

    let a = d.rb() * vc.transpose();
    let b = d.rb() * vf.transpose();

    let mut gram = a.transpose() * a.rb();
    let lambda = 1e-3;
    /*
    let eta = 1e-1;
    let lambda = eta
        * gram
            .self_adjoint_eigenvalues(faer::Side::Upper)
            .unwrap()
            .last()
            .unwrap()
            .sqrt();
    */
    gram.diagonal_mut().for_each_mut(|d| *d = *d + lambda);
    let llt = gram.llt(faer::Side::Upper).unwrap();

    let rhs = a.transpose() * b.rb();
    let p_t = llt.solve(rhs);
    let p = p_t.transpose();

    let reconstruction = p.rb() * vc.rb();
    let err = reconstruction.rb() - vf.rb();
    let weighted_squared_err_norm = err.rb() * d * err.transpose();

    (p.to_owned(), weighted_squared_err_norm)
}

pub fn least_squares(
    fine_mat: SparseRowMatRef<usize, f64>,
    partition: &Partition,
    block_size: usize,
    near_null: MatRef<f64>,
    nn_weights: Vec<f64>,
) -> (
    Mat<f64>,
    SparseRowMat<usize, f64>,
    SparseRowMat<usize, f64>,
    SparseRowMat<usize, f64>,
) {
    let _ = block_size;
    let n_fine = fine_mat.nrows();
    let n_coarse = partition.aggregates().len();
    let k = near_null.ncols();

    let mut w_string = String::new();

    let mut weights = Col::zeros(k);
    //for (w, nn) in weights.iter_mut().zip(near_null.col_iter()) {
    for (w, nnw) in weights.iter_mut().zip(nn_weights.iter()) {
        /*
        let rhs = fine_mat.rb() * nn;
        *w = (nn.transpose() * rhs.rb()).recip().sqrt();
        */
        *w = nnw.sqrt();
        w_string = format!("{} {:.2}", w_string, w);
    }
    info!("nn weigts: {}", w_string);
    let d = weights.into_diagonal();
    //let d = Col::from_iter(w.column_vector().iter().map(|weight| weight.sqrt())).into_diagonal();
    //let d = Diag::ones(k);
    let centers: Vec<usize> = partition
        .aggregates()
        .iter()
        //.map(|agg| best_coarse(agg, near_null, w.rb()).0)
        .map(|agg| best_coarse(agg, near_null, d.rb()).0)
        .collect();
    let mut coarse_graph = AdjacencyList::new_ls_strength_graph(fine_mat, near_null);
    coarse_graph.aggregate(partition);

    let max_interp = 3;
    let mut vc = Mat::zeros(max_interp, k);
    let mut p_triplets = Vec::new();

    for ((agg, neighborhood), (coarse_id, center)) in partition
        .aggregates()
        .iter()
        .zip(coarse_graph.nodes().iter())
        .zip(centers.iter().copied().enumerate())
    {
        /*
        let mut vf = Mat::zeros(agg.len(), k);
        for (node_id, mut local_row) in agg.iter().zip(vf.row_iter_mut()) {
            local_row.copy_from(near_null.row(*node_id));
        }
        */

        for f_point in agg.iter().copied() {
            if f_point == center {
                // we are a c point not an f point
                p_triplets.push(Triplet::new(center, coarse_id, 1.0));
                continue;
            }
            let vf = near_null.row(f_point);
            // key: coarse_id, value: row of `vc` / col of `best_weights`
            let mut interp_points: BTreeMap<usize, usize> = BTreeMap::new();
            interp_points.insert(coarse_id, 0);
            vc.row_mut(0).copy_from(near_null.row(center));
            let (local_p, error) = weighted_least_squares(d.rb(), vc.subrows(0, 1), vf.rb());
            let mut best_weights = local_p;
            let mut prev_best_err = f64::MAX;
            let mut best_err = error;

            for n_interp in 1..max_interp {
                //.min(agg.len() - 1) {
                let mut best_coarse_id = None;
                for (coarse_id, _strength) in neighborhood.iter().copied() {
                    let c_point = centers[coarse_id];
                    if interp_points.contains_key(&coarse_id) {
                        continue;
                    }
                    vc.row_mut(n_interp).copy_from(near_null.row(c_point));
                    let (local_p, error) =
                        weighted_least_squares(d.rb(), vc.subrows(0, n_interp + 1), vf.rb());
                    //assert!(error < prev_best_err, "with more interp points error should be improving... previous best: {} current: {}", prev_best_err, error);
                    if error < best_err {
                        best_coarse_id = Some(coarse_id);
                        best_err = error;
                        best_weights = local_p;
                    }
                }
                if best_err > 1.0 {
                    warn!(
                        "extremelly high local interpolation error: {:.2} with {} points",
                        best_err, n_interp
                    );
                }
                /*
                if best_err > prev_best_err.powf(1.2) {
                    break;
                }
                */
                match best_coarse_id {
                    Some(coarse_id) => {
                        let c_point = centers[coarse_id];
                        vc.row_mut(n_interp).copy_from(near_null.row(c_point));
                        interp_points.insert(coarse_id, n_interp);
                        prev_best_err = best_err;
                    }
                    None => {
                        // this could happen if there aren't many coarse neighbors, we should always at
                        // least have our center, though
                        //warn!("didn't improve at {} interpolation points...", n_interp);
                        break;
                    }
                }
            }

            if true {
                let n_points = interp_points.len();
                let p = best_weights.subcols(0, n_points);
                let vc = vc.subrows(0, n_points);
                let vf = near_null.row(f_point);
                let err = vf - p * vc;
                if p.max().unwrap().abs() > 2.0 {
                    warn!("Large interpolation weight computed in LS:\nerr:\n{:?}\nvf:\n{:?}\np:\n{:?}\nvc:\n{:?}", err, vf, p, vc);
                }
            }
            for (coarse_id, p_idx) in interp_points.iter() {
                let weight = best_weights[*p_idx];
                p_triplets.push(Triplet::new(f_point, *coarse_id, weight));
            }

            /*
            assert_eq!(interp_points.len(), best_weights.ncols());
            assert_eq!(agg.len(), best_weights.nrows());
            for (fine_point, weights) in agg.iter().zip(best_weights.row_iter()) {
                for (coarse_id, weight) in interp_points.iter().zip(weights.iter()) {
                    p_triplets.push(Triplet::new(*fine_point, *coarse_id, *weight));
                }
            }
            */
        }
    }

    let mut coarse_near_null = Mat::zeros(centers.len(), k);
    for (c_point, mut coarse_row) in centers.iter().zip(coarse_near_null.row_iter_mut()) {
        coarse_row.copy_from(near_null.row(*c_point));
    }

    let p = SparseRowMat::try_new_from_triplets(n_fine, n_coarse, &p_triplets)
        .expect("failed to create interp sparse row matrix");
    let debug_stats = matrix_stats(p.as_ref());
    info!("{:?}", debug_stats);

    let r = p
        .transpose()
        .to_row_major()
        .expect("failed to transpose interp to form restriction");
    let mat_coarse = &r * &(fine_mat * &p);
    (coarse_near_null, r, p, mat_coarse)
}

pub fn smoothed_aggregation(
    fine_mat: SparseRowMatRef<usize, f64>,
    partition: &Partition,
    block_size: usize,
    near_null: MatRef<f64>,
) -> (
    Mat<f64>,
    SparseRowMat<usize, f64>,
    SparseRowMat<usize, f64>,
    SparseRowMat<usize, f64>,
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

        /*
        let qr = local.qr();
        let q = qr.compute_thin_Q();
        let r = qr.thin_R();
        */
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
    (coarse_near_null, r, p, mat_coarse)
}

fn csr_block_smoother(
    partition: &Partition,
    mat: SparseRowMatRef<usize, f64>,
    vdim: usize,
) -> SparseRowMat<usize, f64> {
    let n_aggs = partition.naggs();
    let n = mat.nrows();
    let mut permutation: Vec<usize> = partition
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
    for (local_block_idx, global_block_idx) in agg.iter().enumerate() {
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
