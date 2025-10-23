use std::{error::Error, num::NonZeroUsize, path::PathBuf, sync::Arc};

use clap::{Parser, ValueEnum};
use env_logger;
use faer::{
    dyn_stack::{MemBuffer, MemStack, StackReq},
    matrix_free::conjugate_gradient::{
        conjugate_gradient, conjugate_gradient_scratch, CgError, CgInfo, CgParams,
    },
    sparse::SparseRowMat,
    Col,
};
use faer_amg::{
    adaptivity::smooth_vector_rand_svd,
    hierarchy::{AddLevelParams, Hierarchy, HierarchyError, MultigridBuilder, PartitionStrategy},
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

    let size_penalty = 1e-1;
    let dist_penalty = 1e-6;
    let mut hierarchy = Hierarchy::new(matrix.clone(), fine_near_null, cli.block_size);
    hierarchy.set_partition_strategy(PartitionStrategy::new(
        size_penalty,
        dist_penalty,
        cli.aggregation_iters,
    ));

    loop {
        let current_mat = hierarchy.current_mat();
        if current_mat.nrows() <= cli.coarsest_dim {
            info!(
                "Stopping coarsening at level with {} unknowns (<= coarsest_dim {}).",
                current_mat.nrows(),
                cli.coarsest_dim
            );
            break;
        }

        let block_size = hierarchy.current_block_size();
        let block_ratio = cli.near_null_dim as f64 / block_size as f64;
        let coarsening_factor = cli.coarsening_factor * block_ratio;
        match hierarchy.add_level(AddLevelParams {
            coarsening_factor,
            coarse_block_size: cli.near_null_dim,
            partition: None,
        }) {
            Ok(coarse_size) => {
                if coarse_size <= cli.coarsest_dim {
                    info!(
                        "Stopping coarsening at level with {} unknowns (<= coarsest_dim {}).",
                        coarse_size, cli.coarsest_dim
                    );
                    break;
                }
            }
            Err(HierarchyError::CoarseningDidNotReduce) => {
                info!(
                    "Reached terminal level with {} DOFs; no further coarsening possible.",
                    current_mat.nrows()
                );
                break;
            }
            Err(err) => return Err(Box::new(err)),
        }
    }

    info!(
        "Constructed multigrid hierarchy with {} levels",
        hierarchy.levels()
    );
    info!("{:?}", hierarchy);

    let mut multigrid = MultigridBuilder::new()
        .build(&hierarchy)
        .map_err(|err| -> Box<dyn Error> { Box::new(err) })?;
    info!(
        "Constructed multigrid with {} operators and cycle type {}",
        multigrid.levels(),
        multigrid.cycle_type()
    );

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
