use std::{error::Error, num::NonZeroUsize, path::PathBuf, sync::Arc};

use clap::{Parser, ValueEnum};
use env_logger;
use faer::{
    dyn_stack::{MemBuffer, MemStack, StackReq},
    matrix_free::{
        conjugate_gradient::{
            conjugate_gradient, conjugate_gradient_scratch, CgError, CgInfo, CgParams,
        },
        BiPrecond, LinOp,
    },
    sparse::SparseRowMat,
    Col, Mat,
};
use faer_amg::{
    adaptivity::smooth_vector_rand_svd,
    interpolation::smoothed_aggregation,
    partitioners::{Partition, PartitionBuilder},
    preconditioners::{
        block_smoothers::{BlockSmoother, BlockSmootherType},
        coarse_solvers::DenseCholeskySolve,
        multigrid::MultiGrid,
        smoothers::{new_l1, StationaryIteration},
    },
    utils::load_mfem_linear_system,
};
use log::{info, warn, LevelFilter};

#[derive(Parser)]
#[command(name = "amg")]
#[command(about = "Solve an MFEM system with smoothed aggregation AMG + PCG")]
struct Cli {
    /// Coefficient type to load (see aggregation example for options)
    #[arg(long, value_enum, default_value_t = CoefType::Spiral)]
    coef: CoefType,

    /// Use the blend_x variant of the coefficient data
    #[arg(long = "blend_x")]
    blend_x: bool,

    /// Number of refinements (the number after 'h' in 'h4p1')
    #[arg(long, default_value_t = 4)]
    refinements: u32,

    /// Block size of the matrix system (e.g. vector dimension per node)
    #[arg(long, default_value_t = 1)]
    block_size: usize,

    /// Near-null dimension used for smoothed aggregation (also coarse block size)
    #[arg(long, default_value_t = 4)]
    near_null_dim: usize,

    /// Smoothing iterations when building the near-null space
    #[arg(long, default_value_t = 20)]
    smoothing_iters: usize,

    /// Coarsening factor for aggregation (target fine-to-coarse ratio)
    #[arg(long, short = 'c', alias = "cf", default_value_t = 8.0)]
    coarsening_factor: f64,

    /// Maximum improvement iterations for aggregation refinement
    #[arg(long, default_value_t = 200)]
    aggregation_iters: usize,

    /// Maximum CG iterations
    #[arg(long, default_value_t = 1000)]
    max_iters: usize,

    /// Solver tolerance
    #[arg(long, default_value_t = 1e-12)]
    tolerance: f64,

    /// Target block size for diagonally compensated block smoothers
    #[arg(long, default_value_t = 128)]
    block_smoother_size: usize,

    /// Stop coarsening once the operator dimension falls at or below this size
    #[arg(long, default_value_t = 1000)]
    coarsest_dim: usize,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum CoefType {
    Constant,
    #[value(alias = "block_random")]
    BlockRandom,
    Circles,
    Radial,
    #[value(alias = "sine_tangent")]
    SineTangent,
    Spiral,
}

fn main() -> Result<(), Box<dyn Error>> {
    env_logger::builder()
        .format_timestamp(None)
        .filter_level(LevelFilter::Info)
        .try_init()
        .ok();

    let cli = Cli::parse();
    if cli.block_size == 0 {
        return Err("block_size must be positive".into());
    }
    if cli.near_null_dim == 0 {
        return Err("near_null_dim must be positive".into());
    }
    if cli.block_smoother_size == 0 {
        return Err("block_smoother_size must be positive".into());
    }
    if cli.coarsest_dim == 0 {
        return Err("coarsest_dim must be positive".into());
    }
    if cli.coarsening_factor <= 1.0 {
        warn!(
            "Coarsening factor {:.2} is <= 1.0; multigrid hierarchy may not coarsen.",
            cli.coarsening_factor
        );
    }

    faer::set_global_parallelism(faer::Par::Rayon(
        NonZeroUsize::new(16).expect("16 should be non-zero"),
    ));
    let par = faer::get_global_parallelism();

    let (data_dir, dataset_name) = system_path(&cli);
    info!(
        "Loading system {} from {}",
        dataset_name,
        data_dir.display()
    );
    let system = load_mfem_linear_system(&data_dir, &dataset_name, true)?;
    if system.rhs.ncols() == 0 {
        return Err("Loaded system has no RHS columns to solve for".into());
    }

    let matrix = Arc::new(system.matrix);
    let rhs_full = system.rhs;

    info!(
        "Matrix: {} x {} ({} nnz)",
        matrix.nrows(),
        matrix.ncols(),
        matrix.compute_nnz()
    );

    info!(
        "Constructing near-null space with {} vectors using {} smoothing iterations",
        cli.near_null_dim, cli.smoothing_iters
    );
    let fine_near_null =
        smooth_vector_rand_svd(matrix.clone(), cli.smoothing_iters, cli.near_null_dim);

    let hierarchy = build_hierarchy(&cli, matrix.clone(), fine_near_null);

    let (level_mats, restrictions, interpolations, smoothers) = hierarchy;
    info!(
        "Constructed multigrid hierarchy with {} levels",
        level_mats.len()
    );

    let mut multigrid = MultiGrid::new(
        level_mats[0].clone() as Arc<dyn LinOp<f64> + Send>,
        smoothers[0].clone(),
    );

    for level in 0..restrictions.len() {
        let op = level_mats[level + 1].clone() as Arc<dyn LinOp<f64> + Send>;
        let smoother = smoothers[level + 1].clone();
        let r = restrictions[level].clone();
        let p = interpolations[level].clone();
        multigrid.add_level(op, smoother, r, p);
    }

    info!(
        "Running PCG solve with tolerance {:.2e} and max {} iterations",
        cli.tolerance, cli.max_iters
    );
    let rhs_vec: Col<f64> = rhs_full.col(0).to_owned();
    let mut dst: Col<f64> = Col::zeros(matrix.nrows());

    let params = CgParams {
        abs_tolerance: 0.0,
        rel_tolerance: cli.tolerance,
        max_iters: cli.max_iters,
        ..Default::default()
    };

    let stack_req = conjugate_gradient_scratch(&multigrid, matrix.as_ref(), 1, par);
    let mut buf = MemBuffer::new(stack_req);
    let stack = MemStack::new(&mut buf);

    let result = conjugate_gradient(
        dst.as_mut().as_mat_mut(),
        &multigrid,
        matrix.as_ref(),
        rhs_vec.as_mat(),
        params,
        |_| {},
        par,
        stack,
    );

    report(result)?;

    let mut residual = matrix.as_ref().as_ref() * dst.as_mat();
    residual -= rhs_vec.as_mat();
    let residual_norm = residual.norm_l2();
    let rhs_norm = rhs_vec.as_mat().norm_l2();
    let rel_residual = residual_norm / rhs_norm.max(1e-32);
    info!(
        "Final relative residual {:.2e} (||Ax-b|| = {:.2e}, ||b|| = {:.2e})",
        rel_residual, residual_norm, rhs_norm
    );

    Ok(())
}

fn report(result: Result<CgInfo<f64>, CgError<f64>>) -> Result<(), Box<dyn Error>> {
    match result {
        Ok(info) => {
            info!(
                "CG converged in {} iterations (abs {:.2e}, rel {:.2e})",
                info.iter_count, info.abs_residual, info.rel_residual
            );
            Ok(())
        }
        Err(err) => Err(format!("CG failed: {err:?}").into()),
    }
}

fn system_path(cli: &Cli) -> (PathBuf, String) {
    let base_coef_dir = match cli.coef {
        CoefType::Constant => "constant",
        CoefType::BlockRandom => "block_random",
        CoefType::Circles => "circles",
        CoefType::Radial => "radial",
        CoefType::SineTangent => "sine_tangent",
        CoefType::Spiral => "spiral",
    };

    let coef_dir = if cli.blend_x {
        format!("{base_coef_dir}_blend_x")
    } else {
        base_coef_dir.to_owned()
    };

    let data_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("data")
        //.join("anisotropy")
        .join("isotropic")
        .join("2d")
        .join(coef_dir);
    let dataset_name = format!("h{}_p1", cli.refinements);
    (data_dir, dataset_name)
}

fn build_hierarchy(
    cli: &Cli,
    matrix: Arc<SparseRowMat<usize, f64>>,
    fine_near_null: Mat<f64>,
) -> (
    Vec<Arc<SparseRowMat<usize, f64>>>,
    Vec<Arc<dyn LinOp<f64> + Send>>,
    Vec<Arc<dyn LinOp<f64> + Send>>,
    Vec<Arc<dyn BiPrecond<f64> + Send>>,
) {
    let size_penalty = 1e-1;
    let dist_penalty = 1e-6;
    let block_smoother_cf = cli.block_smoother_size as f64;

    let mut level_mats = vec![matrix.clone()];
    let mut level_near_nulls = vec![fine_near_null];
    let mut level_block_sizes = vec![cli.block_size];
    let mut restrictions: Vec<Arc<dyn LinOp<f64> + Send>> = Vec::new();
    let mut interpolations: Vec<Arc<dyn LinOp<f64> + Send>> = Vec::new();

    let par = faer::get_global_parallelism();
    let mut buf = MemBuffer::new(StackReq::empty());
    let stack = MemStack::new(&mut buf);

    loop {
        let level_idx = level_mats.len() - 1;
        let mat = level_mats[level_idx].clone();
        if mat.nrows() <= cli.coarsest_dim {
            info!(
                "Stopping coarsening at level with {} unknowns (<= coarsest_dim {}).",
                mat.nrows(),
                cli.coarsest_dim
            );
            break;
        }
        let block_size = level_block_sizes[level_idx];

        let mut partition;
        if block_size == 1 {
            partition = Partition::singleton(mat.nrows());
        } else {
            let node_to_agg = (0..mat.nrows())
                .map(|node_id| node_id / block_size)
                .collect();
            partition = Partition::from_node_to_agg(node_to_agg);
        }

        let scalar_partition = {
            let near_null_ref = level_near_nulls[level_idx].as_ref();
            let block_ratio = cli.near_null_dim as f64 / block_size as f64;
            let coarsening_factor = cli.coarsening_factor * block_ratio;
            PartitionBuilder::new(
                mat.clone(),
                block_size,
                near_null_ref,
                coarsening_factor,
                size_penalty,
                dist_penalty,
                cli.aggregation_iters,
            )
            .build()
        };
        partition.compose(&scalar_partition);

        if partition.naggs() == mat.nrows() {
            info!(
                "Reached terminal level with {} DOFs; no further coarsening possible.",
                mat.nrows()
            );
            break;
        }

        let (coarse_near_null, restriction_mat, interpolation_mat, coarse_mat) = {
            let near_null_ref = level_near_nulls[level_idx].as_ref();
            smoothed_aggregation(mat.as_ref().as_ref(), &partition, block_size, near_null_ref)
        };

        info!(
            "Level {} -> {} DOFs ({} aggregates)",
            mat.nrows(),
            coarse_mat.nrows(),
            partition.naggs()
        );

        let coarse_mat = Arc::new(coarse_mat);
        let prec = Arc::new(new_l1(coarse_mat.as_ref().as_ref())) as Arc<dyn LinOp<f64> + Send>;
        let iters = 5;
        let near_null_smoother = StationaryIteration::new(coarse_mat.clone(), prec, iters);
        let mut out = Mat::zeros(coarse_near_null.nrows(), coarse_near_null.ncols());
        near_null_smoother.apply(out.as_mut(), coarse_near_null.as_ref(), par, stack);
        let coarse_near_null = out.qr().compute_thin_Q();

        let restriction_arc = Arc::new(restriction_mat);
        let interpolation_arc = Arc::new(interpolation_mat);
        restrictions.push(restriction_arc.clone() as Arc<dyn LinOp<f64> + Send>);
        interpolations.push(interpolation_arc.clone() as Arc<dyn LinOp<f64> + Send>);

        level_mats.push(coarse_mat);
        level_near_nulls.push(coarse_near_null);
        level_block_sizes.push(cli.near_null_dim);
    }

    let level_count = level_mats.len();
    let mut smoothers: Vec<Arc<dyn BiPrecond<f64> + Send>> = Vec::with_capacity(level_count);
    for idx in 0..level_count {
        let mat = level_mats[idx].clone();
        if idx + 1 == level_count {
            let solver = DenseCholeskySolve::from_sparse(mat.as_ref().as_ref())
                .expect("failed to factorize coarsest grid with dense Cholesky");
            smoothers.push(Arc::new(solver));
        } else {
            let near_null = level_near_nulls[idx].as_ref();
            let block_size = level_block_sizes[idx];
            let mut partition;
            if block_size == 1 {
                partition = Partition::singleton(mat.nrows());
            } else {
                let node_to_agg = (0..mat.nrows())
                    .map(|node_id| node_id / block_size)
                    .collect();
                partition = Partition::from_node_to_agg(node_to_agg);
            }
            let scalar_partition = PartitionBuilder::new(
                mat.clone(),
                block_size,
                near_null,
                block_smoother_cf,
                size_penalty,
                dist_penalty,
                cli.aggregation_iters,
            )
            .build();
            partition.compose(&scalar_partition);
            let partition = Arc::new(partition);
            smoothers.push(Arc::new(BlockSmoother::new(
                mat.as_ref().as_ref(),
                partition,
                BlockSmootherType::SparseCholesky,
                block_size,
            )) as Arc<dyn BiPrecond<f64> + Send>);
        }
    }

    (level_mats, restrictions, interpolations, smoothers)
}
