/*
use std::sync::Arc;

use faer::{
    dyn_stack::{MemBuffer, MemStack, StackReq},
    matrix_free::{
        conjugate_gradient::{conjugate_gradient, conjugate_gradient_scratch, CgParams},
        IdentityPrecond, LinOp,
    },
    reborrow::*,
    sparse::{linalg::matmul::sparse_dense_matmul, SparseColMat, Triplet},
    Mat, MatMut, MatRef, Par,
};
use faer_traits::RealField;
*/

fn main() {
    /*
        let triplets = [
            (0usize, 0usize, 1f64),
            (1, 1, 2.0),
            (2, 2, 3.0),
            (3, 3, 3.0),
            (4, 4, 4.0),
        ];

        let triplets: Vec<Triplet<_, _, _>> = triplets
            .iter()
            .map(|t| Triplet::new(t.0, t.1, t.2))
            .collect();

        let mat = SparseColMat::try_new_from_triplets(5usize, 5usize, &triplets).unwrap();
        let ones: Col<f64, _> = Col::ones(5);
        let mut dst: Col<f64, _> = Col::zeros(5);

        sparse_dense_matmul(
            dst.as_mat_mut(),
            faer::Accum::Replace,
            mat.as_ref(),
            ones.as_mat(),
            1.0,
            Par::Seq,
        );

        /*
        let m = MultiGrid {
            operators: vec![Arc::new(mat)],
            interpolations: vec![],
            restrictions: vec![],
        };
        */
        let params = CgParams::default();
        let par = Par::Seq;
        let pc = IdentityPrecond { dim: 5 };
        let mut buf = MemBuffer::new(conjugate_gradient_scratch(pc, &mat, 1, par));
        let stack = MemStack::new(&mut buf);

        let _ = conjugate_gradient(
            dst.as_mat_mut(),
            pc,
            mat,
            ones.as_mat(),
            params,
            |_| {},
            par,
            stack,
        )
        .unwrap();

        println!("{:?}", dst);
    */
}
