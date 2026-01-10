use crate::par_spmm::ParSpmmOp;
use faer::{
    matrix_free::LinOp,
    sparse::{SparseRowMat, SparseRowMatRef},
    Par,
};
use std::sync::Arc;

/// A square sparse matrix (compressed row) with optionally specified block structure.
/// Intended for use in solvers and multigrid hierarchies.
/// Cloning this object increases hard reference counts on `Arc`s and doesn't deep copy.
#[derive(Debug, Clone)]
pub struct SparseMatOp {
    mat: Arc<SparseRowMat<usize, f64>>,
    block_size: usize,
    par_op: Option<Arc<ParSpmmOp>>,
}

impl SparseMatOp {
    /// Block size specifies the size of dense blocks in the matrix. It's not required that all
    /// values in a dense block are non-zero in the sparsity pattern of the compressed matrix but
    /// only square blocks are allowed and the matrix dimensions must be an even multiple of block
    /// size. Dense blocks are considered indivisible units in algorithms:
    /// - coarsening algorithms will only aggregate entire blocks
    /// - smoothers and matrix manipulations operate on block scale
    ///
    /// It's intended for relatively small blocks with intuition related to the underlying problem
    /// associated with the matrix operator. Examples:
    /// - matrices from a 3d PDE discretization where the solution is in the form of displacement
    /// (such as elasticity), then the natural block size is 3
    /// - for systems of PDEs with multiple unknown values the block size could be the number of
    /// unknowns
    ///
    /// NOTE that this only works if your matrix is actually ordered correctly... for the first
    /// example the matrix degrees of freedom should be ordered `x_1, y_1, z_1, ..., x_n, y_n,
    /// z_n`.
    ///
    /// The `par` argument makes an optimized operator for the parallel application of the
    /// operator. Providing more than 1 thread here will double the memory required and creates an
    /// entire copy of the matrix broken into pieces to be consumed by each thread. Dropping all
    /// copies of this object once a preconditioner is constructed should free this memory because
    /// the final preconditioner objects only maintain references to the parallel operator when
    /// provided, but this is cloned in multiple configurations and intermediate algorithm structs
    /// such as `Hierarchy` and `Partitioner` which also must be dropped if memory is limited.
    /// Currently, there is no default parallel implementation of SpMM in `faer`, partly
    /// because the performance of such an implementation and the best algorithm is highly
    /// dependent on the structure of the matrix. This implementation is under assumptions
    /// reasonable to mesh based PDE discretizations:
    /// - `nnz` per row is bounded and generally consistent
    /// - many values are clustered near the diagonal
    ///
    /// Violating these assumptions is fine but might result in suboptimal performance. If
    /// the matrix `nnz` per row is extremely unbalanced, construction of the parallel operator will
    /// panic. For this to happen most of the nonzero values have to be contained in less rows
    /// than the number of parallelism threads.
    pub fn new(mat: SparseRowMat<usize, f64>, block_size: usize, par: Par) -> Self {
        if mat.nrows() != mat.ncols() {
            panic!("SparseMatOp is only designed for square sparse matrices. Matrix dimensions are {}x{}", mat.nrows(), mat.ncols());
        }
        Self::check_block_size(mat.as_ref(), block_size);

        let mat = Arc::new(mat);
        let par_op = if par.degree() > 1 {
            Some(Arc::new(ParSpmmOp::new(mat.as_ref().as_ref(), par)))
        } else {
            None
        };

        Self {
            mat,
            block_size,
            par_op,
        }
    }

    pub fn mat_ref(&self) -> SparseRowMatRef<usize, f64> {
        self.mat.as_ref().as_ref()
    }

    pub fn arc_mat(&self) -> Arc<SparseRowMat<usize, f64>> {
        self.mat.clone()
    }

    pub fn par_op(&self) -> Option<Arc<ParSpmmOp>> {
        self.par_op.clone()
    }

    pub fn dyn_op(&self) -> Arc<dyn LinOp<f64> + Send> {
        self.par_op()
            .map(|op| op as Arc<dyn LinOp<f64> + Send>)
            .unwrap_or(self.arc_mat())
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn set_block_size(&mut self, block_size: usize) {
        Self::check_block_size(self.mat.as_ref().as_ref(), block_size);
        self.block_size = block_size;
    }

    fn check_block_size(mat: SparseRowMatRef<usize, f64>, block_size: usize) {
        if mat.nrows() % block_size != 0 {
            panic!(
                "Matrix is incompatible with provided block size. `mat.nrows() % block_size = {}` and should be 0.",
                mat.nrows() % block_size
            );
        }
    }
}
