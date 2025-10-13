use clap::{Parser, ValueEnum};
use faer::{stats::prelude::*, ColRef, Mat};
use indicatif::ProgressBar;
use std::{
    error::Error,
    num::NonZeroUsize,
    path::PathBuf,
    sync::{Arc, Mutex},
    thread,
    time::Duration,
};

use faer_amg::{
    partitioners::PartitionBuilder,
    preconditioners::smoothers::new_l1,
    utils::{load_mfem_linear_system, MfemLinearSystem},
};
use log::{info, warn, LevelFilter};
use once_cell::sync::OnceCell;
use sci_bevy_comm::SciBeevyClient;
use tokio::runtime::Runtime;

#[derive(Parser)]
#[command(name = "aggregation")]
#[command(about = "Algebraic multigrid aggregation example")]
struct Cli {
    /// Coefficient type: constant, fan, spiral, or radial
    #[arg(long, value_enum, default_value_t = CoefType::Spiral)]
    coef: CoefType,

    /// Number of refinements (the number after 'h' in 'h4p1')
    #[arg(long, default_value_t = 4)]
    refinements: u32,

    /// Near null space dimension
    #[arg(long, default_value_t = 256)]
    near_null_dim: usize,

    /// Number of smoothing iterations
    #[arg(long, default_value_t = 200)]
    smoothing_iters: usize,

    /// Coarsening factor
    #[arg(long, short = 'c', alias = "cf", default_value_t = 32.0)]
    coarsening_factor: f64,

    /// Maximum improvement iterations
    #[arg(long, default_value_t = 1000)]
    improvement_iters: usize,

    /// Skip the smoothing visualization
    #[arg(long)]
    no_viz: bool,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum CoefType {
    Constant,
    Fan,
    Spiral,
    Radial,
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();

    env_logger::builder()
        .format_timestamp(None)
        .filter_level(LevelFilter::Info)
        .try_init()?;

    faer::set_global_parallelism(faer::Par::Rayon(NonZeroUsize::new(16).unwrap()));

    let coef_dir = match cli.coef {
        CoefType::Constant => "constant",
        CoefType::Fan => "fan",
        CoefType::Spiral => "spiral",
        CoefType::Radial => "radial",
    };

    let data_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("data")
        .join("anisotropy")
        .join("2d")
        .join(coef_dir);

    let dataset_name = format!("h{}_p1", cli.refinements);

    let system = load_mfem_linear_system(&data_dir, &dataset_name, true)?;
    if !cli.no_viz {
        initialize_visualization(&dataset_name, &system)?;
    }

    let MfemLinearSystem {
        matrix,
        rhs,
        coords,
        boundary_indices,
        ..
    } = system;

    let near_null_space =
        smooth_vector(&matrix, cli.smoothing_iters, cli.near_null_dim, cli.no_viz);

    info!(
        "Loaded system from {}\n  matrix: {} x {} ({} nnz)\n  rhs columns: {}\n  geom dim: {}\n  removed dirichlet nodes: {}",
        data_dir.join(dataset_name).display(),
        matrix.nrows(),
        matrix.ncols(),
        matrix.compute_nnz(),
        rhs.ncols(),
        coords.ncols(),
        boundary_indices.len()
    );

    let matrix = Arc::new(matrix);

    let agg_penalty = 0.0005;
    let partition_builder = PartitionBuilder::new(
        matrix.clone(),
        near_null_space,
        cli.coarsening_factor,
        agg_penalty,
        cli.improvement_iters,
    );

    let mut partitioner = partition_builder.create_partitioner();
    partitioner.partition();

    info!(
        "Generated aggregation:\n  aggregates: {}\n  fine nodes: {}\n  coarsening factor: {:.2}",
        partitioner.partition.naggs(),
        partitioner.partition.nnodes(),
        partitioner.partition.cf()
    );
    partitioner.partition.info();

    if !cli.no_viz {
        broadcast_partition(&partitioner.partition);
    }

    let n_updates = 500;
    let iters_per_update = (cli.improvement_iters / n_updates).max(1);
    for iter in 0..cli.improvement_iters {
        partitioner.improve(1);
        if !cli.no_viz && iter % iters_per_update == 0 {
            broadcast_partition(&partitioner.partition);
            //thread::sleep(Duration::from_millis(3000));
        }

        info!(
            "Improvement iteration {} -> aggregates: {}, cf: {:.2}, modularity: {:.4}, size cost: {:.4}",
            iter + 1,
            partitioner.partition.naggs(),
            partitioner.partition.cf(),
            partitioner.modularity(),
            partitioner.total_agg_size_cost()
        );
    }

    info!(
        "Final partition -> aggregates: {}, fine nodes: {}, coarsening factor: {:.2}",
        partitioner.partition.naggs(),
        partitioner.partition.nnodes(),
        partitioner.partition.cf()
    );

    Ok(())
}

struct VisualizationContext {
    runtime: Runtime,
    client: SciBeevyClient,
    mesh_vertex_count: usize,
    solution_to_mesh: Vec<usize>,
    dataset: String,
    active: bool,
}

static VIZ_CONTEXT: OnceCell<Mutex<Option<VisualizationContext>>> = OnceCell::new();

fn broadcast_partition(partition: &faer_amg::partitioners::Partition) {
    let Some(context) = VIZ_CONTEXT.get() else {
        return;
    };

    let mut guard = context
        .lock()
        .expect("visualization context mutex poisoned");

    let Some(ctx) = guard.as_mut() else {
        return;
    };

    if !ctx.active || ctx.mesh_vertex_count == 0 {
        return;
    }

    let assignments = partition.node_assignments();
    if ctx.solution_to_mesh.len() != assignments.len() {
        warn!(
            "Partition assignment length ({}) does not match mesh mapping ({}) for dataset {}; disabling visualization updates.",
            assignments.len(),
            ctx.solution_to_mesh.len(),
            ctx.dataset
        );
        ctx.active = false;
        return;
    }

    let mut mesh_assignments = vec![0u32; ctx.mesh_vertex_count];
    for (solution_idx, &agg) in assignments.iter().enumerate() {
        let mesh_idx = ctx.solution_to_mesh[solution_idx];
        let Some(slot) = mesh_assignments.get_mut(mesh_idx) else {
            warn!(
                "Mesh index {} out of bounds ({} vertices) for dataset {}; disabling visualization updates.",
                mesh_idx,
                ctx.mesh_vertex_count,
                ctx.dataset
            );
            ctx.active = false;
            return;
        };
        *slot = agg as u32;
    }

    if let Err(err) = ctx
        .runtime
        .block_on(ctx.client.update_partition_from_mapping(mesh_assignments))
    {
        warn!(
            "Failed to send partition update to visualization server for dataset {}: {err:?}. Disabling further updates.",
            ctx.dataset
        );
        ctx.active = false;
    }
}

fn initialize_visualization(
    dataset_name: &str,
    system: &MfemLinearSystem,
) -> Result<(), Box<dyn Error>> {
    let context = VIZ_CONTEXT.get_or_init(|| Mutex::new(None));
    let mut guard = context
        .lock()
        .expect("visualization context mutex poisoned");

    if guard.is_some() {
        return Ok(());
    }

    let Some(mesh_geometry) = system.mesh_geometry.clone() else {
        info!(
            "No mesh geometry found for dataset {}; smoothing visualization disabled.",
            dataset_name
        );
        return Ok(());
    };

    if mesh_geometry.vertices.is_empty() {
        warn!(
            "Mesh geometry for dataset {} has no vertices; smoothing visualization disabled.",
            dataset_name
        );
        return Ok(());
    }

    let mesh_vertex_count = mesh_geometry.vertices.len();
    if mesh_vertex_count != system.original_dimension {
        warn!(
            "Mesh vertex count ({}) does not match system dimension ({}) for dataset {}.",
            mesh_vertex_count, system.original_dimension, dataset_name
        );
    }

    let runtime = Runtime::new()?;
    let client = SciBeevyClient::connect_local();

    let healthy = match runtime.block_on(client.health_check()) {
        Ok(result) => result,
        Err(err) => {
            warn!(
                "Failed to reach visualization server health endpoint: {err:?}; smoothing visualization disabled."
            );
            false
        }
    };

    if !healthy {
        info!(
            "Visualization server not reachable; smoothing visualization disabled for dataset {}.",
            dataset_name
        );
        return Ok(());
    }

    if let Err(err) = runtime.block_on(client.upload_mesh(mesh_geometry.clone())) {
        warn!(
            "Failed to upload mesh for dataset {} to visualization server: {err:?}; smoothing visualization disabled.",
            dataset_name
        );
        return Ok(());
    }

    info!(
        "Uploaded mesh for dataset {} ({} vertices); smoothing visualization enabled.",
        dataset_name, mesh_vertex_count
    );

    *guard = Some(VisualizationContext {
        runtime,
        client,
        mesh_vertex_count,
        solution_to_mesh: system.index_mapping.solution_to_mesh.clone(),
        dataset: dataset_name.to_string(),
        active: true,
    });

    Ok(())
}

fn smooth_vector(
    mat: &faer::sparse::SparseRowMat<usize, f64>,
    iterations: usize,
    near_null_dim: usize,
    no_viz: bool,
) -> Mat<f64> {
    let mat_ref = mat.as_ref();
    let l1_diag = new_l1(&mat_ref);

    let bar = ProgressBar::new(iterations as u64);

    let rng = &mut StdRng::seed_from_u64(42);
    let n = mat.nrows();

    let mut x = CwiseMatDistribution {
        nrows: n,
        ncols: near_null_dim,
        dist: StandardNormal,
    }
    .rand::<Mat<f64>>(rng);
    let mut r;

    for iter in 0..=iterations {
        x = x.qr().compute_thin_Q();

        if iter % 5 == 0 && !no_viz {
            let to_visualize = x.col(0);
            visualize(to_visualize);
        }

        if iter == iterations {
            break;
        }

        r = mat.as_ref() * &x;
        r = &l1_diag * r;
        x -= r;
        bar.inc(1);
    }
    bar.finish();

    let convergence_factors: String = x
        .col_iter()
        .map(|col| format!("{:.2} ", col.norm_l2()))
        .collect();
    info!(
        "converge factor after {} iterations: {}",
        iterations, convergence_factors
    );

    x
}

fn visualize(vec: ColRef<f64>) {
    let Some(context) = VIZ_CONTEXT.get() else {
        return;
    };

    let mut guard = context
        .lock()
        .expect("visualization context mutex poisoned");

    let Some(ctx) = guard.as_mut() else {
        return;
    };

    if !ctx.active || ctx.mesh_vertex_count == 0 {
        return;
    }

    if ctx.solution_to_mesh.len() != vec.nrows() {
        warn!(
            "Solution to mesh mapping length ({}) does not match vector size ({}) for dataset {}; disabling visualization updates.",
            ctx.solution_to_mesh.len(),
            vec.nrows(),
            ctx.dataset
        );
        ctx.active = false;
        return;
    }

    let mut values = vec![0.0f32; ctx.mesh_vertex_count];
    for (solution_idx, value) in vec.iter().copied().enumerate() {
        let mesh_idx = ctx.solution_to_mesh[solution_idx];
        let Some(slot) = values.get_mut(mesh_idx) else {
            warn!(
                "Mesh index {} out of bounds ({} vertices) for dataset {}; disabling visualization updates.",
                mesh_idx,
                ctx.mesh_vertex_count,
                ctx.dataset
            );
            ctx.active = false;
            return;
        };
        *slot = value as f32;
    }

    if let Err(err) = ctx.runtime.block_on(ctx.client.update_function(values)) {
        warn!(
            "Failed to send smoothing update to visualization server for dataset {}: {err:?}. Disabling further updates.",
            ctx.dataset
        );
        ctx.active = false;
    }
}
