use std::{error::Error, num::NonZeroUsize, path::PathBuf, sync::Arc};

use clap::{Parser, ValueEnum};
use env_logger;
use faer::{
    dyn_stack::{MemBuffer, MemStack, StackReq},
    get_global_parallelism,
    matrix_free::{
        conjugate_gradient::{
            conjugate_gradient, conjugate_gradient_scratch, CgError, CgInfo, CgParams,
        },
        IdentityPrecond, LinOp, Precond,
    },
    prelude::ReborrowMut,
    sparse::SparseRowMat,
    stats::{
        prelude::StandardNormal, CwiseColDistribution, CwiseMatDistribution, DistributionExt,
        UnitaryMat,
    },
    Col, Mat, Par,
};
use faer_amg::{
    adaptivity::smooth_vector_rand_svd,
    core::SparseMatOp,
    decompositions::rand_svd::rand_svd,
    hierarchy::HierarchyConfig,
    partitioners::{modularity::Partitioner, PartitionerCallback, PartitionerConfig},
    preconditioners::{
        block_smoothers::BlockSmootherConfig, multigrid::MultigridConfig,
        smoothers::StationaryIteration,
    },
    utils::load_mfem_linear_system,
};
use log::{info, warn, LevelFilter};
use rand::{rngs::StdRng, SeedableRng};
//use rand::{distr::Uniform, rng, rngs::StdRng, Rng, SeedableRng};

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

    /// Near-null dimension used for partitioning/coarsening (also limits smooth vector generation)
    #[arg(long, alias = "near_null_dim", default_value_t = 64)]
    coarsening_near_null_dim: usize,

    /// Near-null dimension used when forming interpolation (must be <= coarsening_near_null_dim)
    #[arg(long, default_value_t = 4)]
    interp_near_null_dim: usize,

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
    if cli.coarsening_near_null_dim == 0 {
        return Err("coarsening_near_null_dim must be positive".into());
    }
    if cli.interp_near_null_dim == 0 {
        return Err("interp_near_null_dim must be positive".into());
    }
    if cli.interp_near_null_dim > cli.coarsening_near_null_dim {
        return Err("interp_near_null_dim must be <= coarsening_near_null_dim".into());
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

    let par = Par::Rayon(NonZeroUsize::new(16).expect("16 should be non-zero"));
    faer::set_global_parallelism(par);

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

    info!(
        "Matrix: {} x {} ({} nnz)",
        system.matrix.nrows(),
        system.matrix.ncols(),
        system.matrix.compute_nnz()
    );

    let base_mat = SparseMatOp::new(system.matrix, cli.block_size, par);
    let rhs_full = system.rhs;

    info!(
        "Constructing near-null space with {} vectors using {} smoothing iterations",
        cli.coarsening_near_null_dim, cli.smoothing_iters
    );

    let near_null = smooth_vector_rand_svd(
        base_mat.clone(),
        cli.smoothing_iters,
        cli.coarsening_near_null_dim,
    );
    let near_null = Arc::new(near_null);

    let callback = PartitionerCallback::new(Arc::new(callback));
    let partitioner_config = PartitionerConfig {
        coarsening_factor: cli.coarsening_factor,
        callback: Some(callback.clone()),
        max_improvement_iters: cli.aggregation_iters,
        ..Default::default()
    };
    let hierarch_config = HierarchyConfig {
        coarsest_dim: cli.coarsest_dim,
        partitioner_config,
        interp_candidate_dim: cli.interp_near_null_dim,
    };
    let hierarchy = hierarch_config.build(base_mat.clone(), near_null);
    info!("{:?}", hierarchy);

    let smoother_partitioner_config = PartitionerConfig {
        coarsening_factor: cli.block_smoother_size as f64,
        callback: Some(callback.clone()),
        max_improvement_iters: cli.aggregation_iters,
        ..Default::default()
    };
    let smoother_config = BlockSmootherConfig {
        partitioner_config: smoother_partitioner_config,
        ..Default::default()
    };
    let mg_config = MultigridConfig {
        smoother_config,
        ..Default::default()
    };
    let multigrid = mg_config.build(hierarchy);

    let iterations = 50;
    let par_op = base_mat.par_op().unwrap() as Arc<dyn LinOp<f64> + Send>;
    let stationary = StationaryIteration::new(par_op.clone(), Arc::new(multigrid.clone()), 1);

    let rhs_test_vecs = 5;
    let stack_req = stationary.apply_scratch(rhs_test_vecs, get_global_parallelism());
    let mut buf = MemBuffer::new(stack_req);
    let stack = MemStack::new(&mut buf);
    let rng = &mut StdRng::seed_from_u64(1337);
    let mut test = CwiseMatDistribution {
        nrows: rhs_full.nrows(),
        ncols: rhs_test_vecs,
        dist: StandardNormal,
    }
    .rand::<Mat<f64>>(rng);
    test = test.qr().compute_thin_Q();

    for i in 0..iterations {
        stationary.apply_in_place(test.rb_mut(), get_global_parallelism(), stack);
        let reductions: Vec<f64> = test.col_iter().map(|col| col.norm_l2()).collect();
        let report: String = reductions.iter().map(|rf| format!("{:.2} ", rf)).collect();
        info!("iter: {}\t {}", i, report);
        test = test.qr().compute_thin_Q();
    }

    let iterations = 5;
    let (_u, s, _v) = rand_svd(stationary, iterations);
    let convergence_factors: String = s
        .into_column_vector()
        .iter()
        .map(|singular_value| format!("{:.2} ", singular_value.powf((iterations as f64).recip())))
        .collect();
    info!(
        "near-null smoothing convergence factors: {}",
        convergence_factors
    );

    info!(
        "Running PCG solve with tolerance {:.2e} and max {} iterations",
        cli.tolerance, cli.max_iters
    );
    let rhs_vec: Col<f64> = rhs_full.col(0).to_owned();
    //let mut dst: Col<f64> = Col::zeros(matrix.nrows());

    let rng = &mut StdRng::seed_from_u64(42);
    let mut dst = CwiseColDistribution {
        nrows: rhs_full.nrows(),
        dist: StandardNormal,
    }
    .rand::<Col<f64>>(rng);

    let params = CgParams {
        abs_tolerance: 0.0,
        rel_tolerance: cli.tolerance,
        max_iters: cli.max_iters,
        initial_guess: faer::matrix_free::InitialGuessStatus::MaybeNonZero,
        ..Default::default()
    };

    let matrix = base_mat.mat_ref();
    let stack_req = conjugate_gradient_scratch(&multigrid, matrix, 1, par);
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
        .join("anisotropy")
        //.join("isotropic")
        .join("2d")
        .join(coef_dir);
    let dataset_name = format!("h{}_p1", cli.refinements);
    (data_dir, dataset_name)
}

fn callback(iter: usize, partitioner: &Partitioner) {
    if iter % 25 == 0 {
        let p = partitioner.get_partition();
        let ((_max_agg, max_size), (_min_agg, min_size)) = partitioner.max_and_min_weighted_aggs();

        let size_cost = partitioner.total_agg_size_cost() as f32;
        let dist_cost = partitioner.total_dist_cost() as f32;
        let edge_cost = partitioner.total_edge_cost() as f32;

        info!(
        "Iter: {}\n\tfine nodes: {}\n\taggregates: {}\n\tcoarsening factor: {:.2}\n\tmax / min agg size: {} / {}\n\tsize cost: {:.2e}\n\tdist cost: {:.2e}\n\tedge cost: {:.2e}",
        iter,
        p.nnodes(),
        p.naggs(),
        p.cf(),
        max_size,
        min_size,
        size_cost,
        dist_cost,
        edge_cost,
    );
    }
}
