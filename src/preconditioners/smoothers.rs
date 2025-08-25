use faer::{diag::Diag, sparse::SparseRowMatRef, Index};
use faer_traits::{
    math_utils::{abs1, add, mul, recip, sqrt},
    RealField,
};

pub type L2<T> = Diag<T>;
pub type L1<T> = Diag<T>;

pub fn new_l2<I: Index, T: RealField>(mat: &SparseRowMatRef<I, T>) -> L2<T> {
    let nrows = mat.nrows();
    let diag_sqrt: Vec<T> = (0..nrows).map(|i| sqrt(mat.get(i, i).unwrap())).collect();

    let mut l2_inverse: L2<T> = Diag::zeros(nrows);

    for triplet in mat.triplet_iter() {
        let scale: T = mul(&diag_sqrt[triplet.row], &recip(&diag_sqrt[triplet.col]));
        l2_inverse[triplet.row] = add(&l2_inverse[triplet.row], &(abs1(triplet.val) * scale));
    }

    l2_inverse
        .column_vector_mut()
        .iter_mut()
        .for_each(|d| *d = recip(d));
    l2_inverse
}

pub fn new_l1<I: Index, T: RealField>(mat: &SparseRowMatRef<I, T>) -> L1<T> {
    let nrows = mat.nrows();
    let mut l1_inverse: L1<T> = Diag::zeros(nrows);

    for triplet in mat.triplet_iter() {
        l1_inverse[triplet.row] = add(&l1_inverse[triplet.row], &abs1(triplet.val));
    }

    l1_inverse
        .column_vector_mut()
        .iter_mut()
        .for_each(|d| *d = recip(d));
    l1_inverse
}

pub fn new_jacobi<I: Index, T: RealField>(mat: &SparseRowMatRef<I, T>, omega: T) -> L1<T> {
    let nrows = mat.nrows();
    let mut jacobi: L1<T> = Diag::zeros(nrows);

    for (i, val) in jacobi.column_vector_mut().iter_mut().enumerate() {
        *val = recip(mat.get(i, i).unwrap()).mul_by_ref(&omega);
    }
    jacobi
}
