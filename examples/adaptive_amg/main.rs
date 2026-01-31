use std::{error::Error, num::NonZeroUsize, path::PathBuf, sync::Arc};

use clap::{Parser, ValueEnum};
use env_logger;
use faer::{
    dyn_stack::{MemBuffer, MemStack, StackReq},
    get_global_parallelism,
    matrix_free::{
        conjugate_gradient::{conjugate_gradient, conjugate_gradient_scratch},
        stationary_iteration::{stationary_iteration, stationary_iteration_scratch},
        IdentityPrecond, LinOp, Precond,
    },
    prelude::ReborrowMut,
    sparse::{SparseRowMat, SparseRowMatRef},
    stats::{
        prelude::StandardNormal, CwiseColDistribution, CwiseMatDistribution, DistributionExt,
        UnitaryMat,
    },
    Col, ColRef, Mat, MatRef, Par,
};
use faer_amg::{
    adaptivity::{AdaptiveConfig, ErrorPropogator},
    core::SparseMatOp,
    decompositions::rand_svd::rand_svd,
    hierarchy::HierarchyConfig,
    par_spmm::ParSpmmOp,
    partitioners::{modularity::Partitioner, PartitionerCallback, PartitionerConfig},
    preconditioners::{
        block_smoothers::BlockSmootherConfig, composite::Composite, multigrid::MultigridConfig,
    },
    utils::{load_mfem_linear_system, test_solver},
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

    /// Maximum number of components to create for composite preconditioner
    #[arg(long, default_value_t = 3)]
    max_components: usize,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum CoefType {
    Brandt,
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

    let callback = PartitionerCallback::new(Arc::new(callback));
    let partitioner_config = PartitionerConfig {
        coarsening_factor: cli.coarsening_factor,
        callback: Some(callback.clone()),
        max_improvement_iters: cli.aggregation_iters,
        //agg_size_penalty: 1e1,
        ..Default::default()
    };
    let hierarchy_config = HierarchyConfig {
        coarsest_dim: cli.coarsest_dim,
        partitioner_config,
        interp_candidate_dim: cli.interp_near_null_dim,
    };

    let smoother_partitioner_config = PartitionerConfig {
        coarsening_factor: cli.block_smoother_size as f64,
        callback: Some(callback.clone()),
        //agg_size_penalty: 1e0,
        max_improvement_iters: cli.aggregation_iters,
        ..Default::default()
    };
    let smoother_config = BlockSmootherConfig {
        partitioner_config: smoother_partitioner_config,
        ..Default::default()
    };
    let multigrid_config = MultigridConfig {
        smoother_config,
        smoothing_steps: 3,
        ..Default::default()
    };

    let adaptive_config = AdaptiveConfig {
        hierarchy_config,
        multigrid_config,
        test_iters: cli.smoothing_iters,
        max_components: cli.max_components,
        coarsening_near_null_dim: cli.coarsening_near_null_dim,
        ..Default::default()
    };
    let mut composite = adaptive_config.build(base_mat.clone());

    let par_op = base_mat.par_op().unwrap() as Arc<dyn LinOp<f64> + Send + Sync>;

    let results = test_composite(
        &mut composite,
        par_op,
        rhs_full.as_ref(),
        cli.max_iters,
        cli.tolerance,
    );
    let pcg_table = build_results_table(&results.pcg);
    info!("Composite PCG results:\n{pcg_table}");
    let stationary_table = build_results_table(&results.stationary);
    info!("Composite stationary results:\n{stationary_table}");

    Ok(())
}

type CompositeResult = (usize, usize, usize, f64, f64, f64);

struct CompositeSolveResults {
    pcg: Vec<CompositeResult>,
    stationary: Vec<CompositeResult>,
}

fn test_composite(
    composite: &mut faer_amg::preconditioners::composite::Composite,
    matrix: Arc<dyn LinOp<f64> + Sync + Send>,
    rhs: MatRef<'_, f64>,
    max_iters: usize,
    tolerance: f64,
) -> CompositeSolveResults {
    let mut pcg_results = Vec::new();
    let mut stationary_results = Vec::new();

    let rhs_vec: ColRef<f64> = rhs.col(0);
    let rhs = Some(rhs_vec.as_mat());
    let rng = &mut StdRng::seed_from_u64(42);
    let initial_guess = CwiseColDistribution {
        nrows: rhs_vec.nrows(),
        dist: StandardNormal,
    }
    .rand::<Col<f64>>(rng);
    let initial_guess = Some(initial_guess.as_mat());

    while !composite.components().is_empty() {
        let component_count = composite.components().len();
        let arc_pc = Arc::new(composite.clone());

        let ((cg_iters, cg_residual, _dst_cg), (sli_iters, sli_residual, _dst_sli)) = test_solver(
            matrix.clone(),
            arc_pc,
            initial_guess,
            rhs,
            max_iters,
            tolerance,
        );
        pcg_results.push(build_composite_result(
            component_count,
            cg_iters,
            cg_residual,
        ));
        stationary_results.push(build_composite_result(
            component_count,
            sli_iters,
            sli_residual,
        ));

        composite.components_mut().pop();
    }

    CompositeSolveResults {
        pcg: pcg_results,
        stationary: stationary_results,
    }
}

fn build_results_table(results: &[CompositeResult]) -> String {
    let mut table = String::new();
    table.push_str("+------------+------------+------------+----------------------+----------------------+----------------------+\n");
    table.push_str("| components | iterations | v-cycles   | reduction/iter       | reduction/v-cycle    | final rel residual   |\n");
    table.push_str("+------------+------------+------------+----------------------+----------------------+----------------------+\n");
    for (
        component_count,
        iter_count,
        total_vcycles,
        reduction_per_iter,
        reduction_per_vcycle,
        rel_residual,
    ) in results
    {
        table.push_str(&format!(
            "| {:>10} | {:>10} | {:>10} | {:>20.3} | {:>20.3} | {:>20.3e} |\n",
            component_count,
            iter_count,
            total_vcycles,
            reduction_per_iter,
            reduction_per_vcycle,
            rel_residual
        ));
    }
    table.push_str("+------------+------------+------------+----------------------+----------------------+----------------------+\n");
    table
}

fn build_composite_result(
    component_count: usize,
    iter_count: usize,
    rel_residual: f64,
) -> CompositeResult {
    let vcycles_per_iter = 2 * component_count - 1;
    let total_vcycles = iter_count * vcycles_per_iter;
    let reduction_per_iter = if iter_count > 0 {
        rel_residual.powf(1.0 / iter_count as f64)
    } else {
        0.0
    };
    let reduction_per_vcycle = if total_vcycles > 0 {
        rel_residual.powf(1.0 / total_vcycles as f64)
    } else {
        0.0
    };
    (
        component_count,
        iter_count,
        total_vcycles,
        reduction_per_iter,
        reduction_per_vcycle,
        rel_residual,
    )
}

fn system_path(cli: &Cli) -> (PathBuf, String) {
    let base_coef_dir = match cli.coef {
        CoefType::Brandt => "brandt",
        CoefType::Constant => "constant",
        CoefType::BlockRandom => "block_random",
        CoefType::Circles => "circles",
        CoefType::Radial => "radial",
        CoefType::SineTangent => "sine_tangent",
        CoefType::Spiral => "spiral",
    };

    let data_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("data")
        .join("anisotropy")
        //.join("isotropic")
        .join("2d")
        .join(base_coef_dir);
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
