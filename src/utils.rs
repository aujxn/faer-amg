use faer::sparse::SparseRowMat;

pub fn mats_are_equal(left: &SparseRowMat<usize, f64>, right: &SparseRowMat<usize, f64>) -> bool {
    let l_shape = left.shape();
    let l_nnz = left.compute_nnz();
    let r_shape = right.shape();
    let r_nnz = right.compute_nnz();

    if l_shape != r_shape || l_nnz != r_nnz {
        println!(
            "Left: {:?}, {} nnz\nRight: {:?}, {} nnz",
            l_shape, l_nnz, r_shape, r_nnz
        );
        return false;
    }

    for (left, right) in left.triplet_iter().zip(right.triplet_iter()) {
        let absolute = (left.val - right.val).abs();
        let relative = absolute / left.val.max(*right.val);
        if left.row != right.row || left.col != right.col || absolute > 1e-12 || relative > 1e-12 {
            println!(
                "Left: {}, {} : {}\nRight: {}, {} : {}\nRelative err: {}\nAbsolute err: {}",
                left.row, left.col, left.val, right.row, right.col, right.val, relative, absolute
            );
            return false;
        }
    }
    true
}
