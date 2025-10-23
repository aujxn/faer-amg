use clap::{Parser, ValueEnum};
use faer::{stats::prelude::*, ColRef, Mat};
use indicatif::ProgressBar;
use std::{
    error::Error,
    num::NonZeroUsize,
    ops::DerefMut,
    path::PathBuf,
    rc::Rc,
    sync::{Arc, Mutex},
    thread,
    time::Duration,
};

use faer_amg::{
    decompositions::rand_svd::rand_svd,
    interpolation::smoothed_aggregation,
    partitioners::{
        multilevel::{self, MultilevelPartitionerConfig},
        ModularityPartitioner, Partition, PartitionBuilder,
    },
    preconditioners::smoothers::{new_l1, StationaryIteration},
    utils::{load_mfem_linear_system, MfemLinearSystem},
};
use log::{error, info, warn, LevelFilter};
use once_cell::sync::OnceCell;
use sci_bevy_comm::{PartitionMetrics, SciBeevyClient};
use tokio::runtime::Runtime;

#[derive(Parser)]
#[command(name = "aggregation")]
#[command(about = "Algebraic multigrid aggregation example")]
struct Cli {
    /// Coefficient type: constant, block_random, circles, radial, sine_tangent, or spiral
    #[arg(long, value_enum, default_value_t = CoefType::Spiral)]
    coef: CoefType,

    /// Use the blend_x variant of the coefficient data
    #[arg(long = "blend_x")]
    blend_x: bool,

    /// Apply coarsening algorithm recursively
    #[arg(long = "multilevel")]
    multilevel: bool,

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
    #[value(alias = "block_random")]
    BlockRandom,
    Circles,
    Radial,
    #[value(alias = "sine_tangent")]
    SineTangent,
    Spiral,
}

fn main() -> Result<(), Box<dyn Error>> {
    {
        let cli = Cli::parse();
        let no_viz = cli.no_viz;

        env_logger::builder()
            .format_timestamp(None)
            .filter_level(LevelFilter::Info)
            .try_init()?;

        faer::set_global_parallelism(faer::Par::Rayon(NonZeroUsize::new(16).unwrap()));

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
            .join("2d")
            .join(&coef_dir);

        let dataset_name = format!("h{}_p1", cli.refinements);

        let system = load_mfem_linear_system(&data_dir, &dataset_name, true)?;
        if !no_viz {
            initialize_visualization(&dataset_name, &system)?;
        }

        let MfemLinearSystem {
            matrix,
            rhs,
            coords,
            boundary_indices,
            ..
        } = system;

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

        //let fine_near_null_space = smooth_vector(&matrix, cli.smoothing_iters, cli.near_null_dim, no_viz);
        let matrix = Arc::new(matrix);
        let fine_near_null_space = smooth_vector_rand_svd(
            matrix.clone(),
            cli.smoothing_iters,
            cli.near_null_dim,
            no_viz,
        );

        let base_partition = Partition::singleton(matrix.nrows());
        let prev_partition = Partition::singleton(matrix.nrows());
        let level = 1;
        let safe_state: State = Arc::new(Mutex::new((level, base_partition, prev_partition)));

        let dist_penalty = 1e-6;
        let size_penalty = 1e-1;
        if cli.multilevel {
            let mut partition_builder = MultilevelPartitionerConfig::new(
                matrix.clone(),
                fine_near_null_space.as_ref(),
                cli.coarsening_factor,
                size_penalty,
                dist_penalty,
            );
            partition_builder.max_improvement_iters = cli.improvement_iters;

            partition_builder.callback = Some(Arc::new(
                move |iter: usize, partitioner: &ModularityPartitioner| {
                    callback(iter, partitioner, safe_state.clone())
                },
            ));
            let mut ml_partitioner = partition_builder.build();
            ml_partitioner.partition();
        } else {
            let mut partition_builder = PartitionBuilder::new(
                matrix.clone(),
                1,
                fine_near_null_space.as_ref(),
                cli.coarsening_factor,
                size_penalty,
                dist_penalty,
                cli.improvement_iters,
            );

            partition_builder.callback = Some(Arc::new(
                move |iter: usize, partitioner: &ModularityPartitioner| {
                    callback(iter, partitioner, safe_state.clone())
                },
            ));
            let _partition = partition_builder.build();
        }
    }
    Ok(())
}

type State = Arc<Mutex<(usize, Partition, Partition)>>;
fn callback(iter: usize, partitioner: &ModularityPartitioner, state: State) {
    let p = partitioner.get_partition();
    let ((_max_agg, max_size), (_min_agg, min_size)) = partitioner.max_and_min_weighted_aggs();

    let size_cost = partitioner.total_agg_size_cost() as f32;
    let dist_cost = partitioner.total_dist_cost() as f32;
    let edge_cost = partitioner.total_edge_cost() as f32;

    let metrics = PartitionMetrics {
        iter: iter as u32,
        size_cost,
        dist_cost,
        edge_cost,
    };

    match state.lock() {
        Ok(mut guard) => {
            let state = guard.deref_mut();
            if state.2.nnodes() != p.nnodes() {
                state.1.compose(&state.2);
                state.0 += 1;
                info!("NEW LEVEL: {}", state.0);
                broadcast_partition(state.1.node_assignments(), metrics);
            } else {
                let to_broadcast: Vec<usize> = state
                    .1
                    .node_assignments()
                    .iter()
                    .map(|old_agg| p.node_assignments()[*old_agg])
                    .collect();
                broadcast_partition(&to_broadcast, metrics);
            }
            state.2 = p.clone();
        }
        Err(err) => error!("{:?}", err),
    }

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

struct VisualizationContext {
    runtime: Runtime,
    client: SciBeevyClient,
    mesh_vertex_count: usize,
    solution_to_mesh: Vec<usize>,
    dataset: String,
    active: bool,
}

static VIZ_CONTEXT: OnceCell<Mutex<Option<VisualizationContext>>> = OnceCell::new();

fn broadcast_partition(node_assignments: &[usize], metrics: PartitionMetrics) {
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

    if ctx.solution_to_mesh.len() != node_assignments.len() {
        warn!(
            "Partition assignment length ({}) does not match mesh mapping ({}) for dataset {}; disabling visualization updates.",
            node_assignments.len(),
            ctx.solution_to_mesh.len(),
            ctx.dataset
        );
        ctx.active = false;
        return;
    }

    let mut mesh_assignments = vec![0u32; ctx.mesh_vertex_count];
    for (solution_idx, &agg) in node_assignments.iter().enumerate() {
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

    if let Err(err) = ctx.runtime.block_on(
        ctx.client
            .update_partition_from_mapping(mesh_assignments, Some(metrics)),
    ) {
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
    let l1_diag = new_l1(mat_ref);

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

fn smooth_vector_rand_svd(
    mat: Arc<faer::sparse::SparseRowMat<usize, f64>>,
    iterations: usize,
    near_null_dim: usize,
    no_viz: bool,
) -> Mat<f64> {
    let l1_diag = Arc::new(new_l1(mat.as_ref().as_ref()));

    let stationary = StationaryIteration::new(mat, l1_diag, iterations);
    let (_u, s, v) = rand_svd(stationary, near_null_dim);

    if !no_viz {
        let to_visualize = v.col(0);
        visualize(to_visualize);
    }

    let convergence_factors: String = s
        .into_column_vector()
        .iter()
        .map(|singular_value| format!("{:.2} ", singular_value.powf((iterations as f64).recip())))
        .collect();
    info!("converge factor: {}", convergence_factors);

    v
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
