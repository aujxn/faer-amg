use faer::{
    sparse::{
        ops::{add_assign, sub_into, union_symbolic},
        SparseColMat, SparseRowMat, SparseRowMatRef, Triplet,
    },
    Col, Mat, MatRef,
};

use crate::partitioners::Partition;

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
    let k = near_null.ncols();
    let mut coarse_near_null = Mat::zeros(n_coarse * k, k);

    // n_fine by n_coarse * k
    let mut p: Vec<Triplet<usize, usize, f64>> = Vec::new();
    for (coarse_idx, agg) in partition.aggregates().iter().enumerate() {
        let local_rows = agg.len();
        // could relax this but makes things much messier
        assert!(
            local_rows >= k,
            "Agg size of {} cannot support near-null dimension of {}",
            local_rows,
            k
        );
        let mut local = Mat::zeros(local_rows, k);
        for (local_j, j) in agg.iter().copied().enumerate() {
            //local.row_mut(local_j) = near_null.row(j);
            for (dest, src) in local
                .row_mut(local_j)
                .iter_mut()
                .zip(near_null.row(j).iter())
            {
                *dest = *src;
            }
        }

        let qr = local.qr();
        let q = qr.compute_thin_Q();
        let r = qr.thin_R();
        assert_eq!(q.ncols(), k);

        for (i, r_row) in r.row_iter().enumerate() {
            for (dest, src) in coarse_near_null
                .row_mut(coarse_idx * k + i)
                .iter_mut()
                .zip(r_row.iter())
            {
                *dest = *src;
            }
        }

        for (local_i, fine_i) in agg.iter().copied().enumerate() {
            let col_start = k * coarse_idx;
            for (offset_j, src) in q.row(local_i).iter().enumerate() {
                p.push(Triplet {
                    row: fine_i,
                    col: col_start + offset_j,
                    val: *src,
                });
            }
        }
    }

    let mut p = SparseRowMat::try_new_from_triplets(n_fine, n_coarse * k, &p)
        .expect("failed to create SA interp sparse row matrix");

    if block_size == 1 {
        p = smooth_interpolation(fine_mat, p.as_ref(), 0.66);
    } else {
        p = block_jacobi(fine_mat, block_size, p.as_ref());
    };

    let r = p
        .transpose()
        .to_row_major()
        .expect("failed to transpose SA interp to form restriction");
    let mat_coarse = &r * &(fine_mat * &p);
    (coarse_near_null, r, p, mat_coarse)
}

pub fn smooth_interpolation(
    mat: SparseRowMatRef<usize, f64>,
    p: SparseRowMatRef<usize, f64>,
    jacobi_weight: f64,
) -> SparseRowMat<usize, f64> {
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
                if let Some(v) = mat.get(start + i, start + j) {
                    block[(i, j)] = *v;
                }
            }
        }

        block = block
            .self_adjoint_eigen(faer::Side::Lower)
            .expect("failed to decompose small local eigenproblem")
            .pseudoinverse();

        for i in 0..block_size {
            for j in 0..block_size {
                d_inv.push(Triplet::new(start + i, start + j, block[(i, j)]));
            }
        }
    }

    let d_inv = SparseRowMat::try_new_from_triplets(ndofs, ndofs, &d_inv)
        .expect("failed to create sparse row mat from block diag inverse triplets");
    let mut d_inv_a = d_inv * mat;
    d_inv_a *= -0.66;
    let mut smoothed = d_inv_a * p;
    add_assign(smoothed.transpose_mut(), p.transpose());
    smoothed
}
