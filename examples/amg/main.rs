use std::{error::Error, num::NonZeroUsize, path::PathBuf, sync::Arc};

use clap::{Parser, ValueEnum};
use env_logger;
use faer::{
    mat::AsMatRef,
    matrix_free::LinOp,
    prelude::Reborrow,
    sparse::SparseRowMatRef,
    stats::{prelude::StandardNormal, CwiseColDistribution, DistributionExt},
    Col, Mat, MatRef, Par,
};
use faer_amg::{
    adaptivity::find_near_null,
    core::SparseMatOp,
    hierarchy::{Hierarchy, HierarchyConfig, PartitionType},
    interpolation::{
        AggregationConfig, ClassicalConfig, CompatibleRelaxationConfig, InterpolationConfig,
        LeastSquaresConfig, LsSolver,
    },
    partitioners::{modularity::Partitioner, PartitionerCallback, PartitionerConfig},
    preconditioners::{block_smoothers::BlockSmootherConfig, multigrid::MultigridConfig},
    utils::{approx_convergence_factor, load_mfem_linear_system, test_solver},
};
use log::{info, warn, LevelFilter};
use rand::{rngs::StdRng, SeedableRng};
//use rand::{distr::Uniform, rng, rngs::StdRng, Rng, SeedableRng};
use serde::Serialize;
use std::fs;

#[derive(Parser)]
#[command(name = "amg")]
#[command(about = "Solve an MFEM system with smoothed aggregation AMG + PCG")]
struct Cli {
    /// Coefficient type to load (see aggregation example for options)
    #[arg(long, value_enum, default_value_t = CoefType::Spiral)]
    coef: CoefType,

    /// Use 0 vector for starting CG guess (otherwise random)
    #[arg(long, default_value_t = false)]
    zero_guess: bool,

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

#[derive(Clone, Debug, Serialize)]
pub struct InterpViz {
    pub functions: Mat<f64>,
    pub interpolation: Vec<(usize, usize, f64)>,
    pub c_points: Vec<usize>,
}

impl InterpViz {
    pub fn new(hierarchy: &Hierarchy) -> Self {
        let p = hierarchy.get_interpolation(0);
        let triplets = p
            .triplet_iter()
            .map(|triplet| (triplet.row, triplet.col, *triplet.val))
            .collect();
        let arc = hierarchy.get_near_null(0);
        let nn = arc.as_mat_ref();
        let mut functions = Mat::zeros(nn.nrows(), nn.ncols());

        for (mut out_col, in_col) in functions.col_iter_mut().zip(nn.col_iter()) {
            let max = in_col
                .iter()
                .map(|v| v.abs())
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap();
            out_col.copy_from(in_col);
            out_col /= max;
        }
        // TODO: multilevel and for SA needs to be different...
        let c_points = match hierarchy.get_partition(0) {
            PartitionType::Classical(cf_split) => cf_split.c_points().clone(),
            PartitionType::Aggregation(_partition) => Vec::new(),
        };

        Self {
            functions,
            interpolation: triplets,
            c_points,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct MeshViz {
    pub coords: Mat<f64>,
    pub interps: Vec<InterpViz>,
}

impl MeshViz {
    pub fn new(hierarchy: &Hierarchy, coords: Mat<f64>) -> Self {
        let interps = vec![InterpViz::new(hierarchy)];
        Self { coords, interps }
    }
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
    // TODO: CLI arg for aggregation (with smoothing steps) do this block instead
    let aggregation = false;
    let smoothing_steps = 1;
    let interpolation_config = if aggregation {
        let partitioner_config = PartitionerConfig {
            coarsening_factor: cli.coarsening_factor,
            callback: Some(callback.clone()),
            max_improvement_iters: cli.aggregation_iters,
            //agg_size_penalty: 1e1,
            ..Default::default()
        };
        let agg_config = AggregationConfig::new(
            smoothing_steps,
            cli.interp_near_null_dim,
            partitioner_config,
        );
        InterpolationConfig::Aggregation(agg_config)
    } else {
        let cr_options = CompatibleRelaxationConfig {
            relax_steps: 5,
            target_convergence: 0.15,
        };
        let ls_options = LeastSquaresConfig {
            search_depth: 3,
            depth_ls: 2,
            solver: LsSolver::Constrained,
            max_interp: 3,
            tau_threshold: 1.2,
        };
        let classical_config = ClassicalConfig {
            cr_options,
            ls_options,
        };
        InterpolationConfig::Classical(classical_config)
    };

    let hierarchy_config = HierarchyConfig {
        coarsest_dim: cli.coarsest_dim,
        interpolation_config,
        max_levels: Some(2),
    };
    let nn = find_near_null(
        base_mat.clone(),
        cli.smoothing_iters,
        cli.coarsening_near_null_dim - 1,
        cli.block_smoother_size as f64,
    );
    // TODO: constant should be removed from subspace during iterative orthogonalization
    let mut nn_with_constant = Mat::ones(nn.nrows(), cli.coarsening_near_null_dim);
    nn_with_constant
        .subcols_mut(1, cli.coarsening_near_null_dim - 1)
        .copy_from(nn);
    let nn_basis = nn_with_constant.qr().compute_thin_Q();

    /*
    let evd = SelfAdjointEigen::new(base_mat.mat_ref().to_dense().as_ref(), Side::Upper).unwrap();
    let mut nn_basis = evd.U().subcols(0, cli.coarsening_near_null_dim).to_owned();
    let eigvals: String = evd
        .S()
        .column_vector()
        .iter()
        .take(cli.coarsening_near_null_dim)
        .map(|eigval| format!("{:.2e} ", eigval))
        .collect();
    info!("eigvals: {}", eigvals);
    let n = nn_basis.nrows() as f64;
    let const_val = 1. / n.sqrt();
    nn_basis.col_mut(0).iter_mut().for_each(|v| *v = const_val);
    */

    let weights = create_weights(nn_basis.as_mat_ref(), base_mat.mat_ref());
    let weights = Arc::new(weights);
    let near_null = Arc::new(nn_basis);
    let hierarchy = hierarchy_config.build(base_mat.clone(), near_null, weights);
    info!("{:?}", hierarchy);

    let mesh_viz = MeshViz::new(&hierarchy, system.coords);
    let serialized = serde_json::to_string(&mesh_viz)?;
    let file_path = "./data/hierarchy_viz.json";
    fs::write(file_path, serialized)?;

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
    let multigrid_config = MultigridConfig {
        smoother_config,
        smoothing_steps: 3,
        //mu: 2,
        ..Default::default()
    };
    let multigrid = multigrid_config.build(hierarchy);

    let par_op = base_mat.par_op().unwrap() as Arc<dyn LinOp<f64> + Send>;
    let arc_pc = Arc::new(multigrid);
    /*
    println!("Preconditioner symmetry test:");
    symmetry_test(arc_pc.clone());
    let rhs_test_vecs = 10;

    let iterations = 5;
    let error_prop = ErrorPropogator {
        op: par_op.clone(),
        pc: arc_pc.clone(),
    };
    println!("Error prop symmetry test:");
    symmetry_test(Arc::new(error_prop.clone()));

    let (_u, s, v) = rand_svd(error_prop.clone(), rhs_test_vecs, 5, iterations);

    let mut mem = MemBuffer::new(StackReq::any_of(&[
        par_op.apply_scratch(rhs_test_vecs, par),
        error_prop.apply_scratch(rhs_test_vecs, par),
    ]));
    let stack = MemStack::new(&mut mem);
    let mut av = Mat::zeros(v.nrows(), v.ncols());
    let mut ev = Mat::zeros(v.nrows(), v.ncols());
    let mut a_ev = Mat::zeros(v.nrows(), v.ncols());
    par_op.apply(av.as_mut(), v.as_ref(), par, stack);
    error_prop.apply(ev.as_mut(), v.as_ref(), par, stack);
    par_op.apply(a_ev.as_mut(), ev.as_ref(), par, stack);
    let a_norms = Col::from_iter((0..rhs_test_vecs).map(|i| {
        let vtav = inner_prod(
            v.col(i).transpose(),
            faer::Conj::No,
            av.col(i),
            faer::Conj::No,
        );
        let vt_a_ev = inner_prod(
            v.col(i).transpose(),
            faer::Conj::No,
            a_ev.col(i),
            faer::Conj::No,
        );
        vt_a_ev / vtav
    }));
    let convergence_factors: String = a_norms.iter().map(|norm| format!("{:.2} ", norm)).collect();
    info!("Column A norms of E v: {}", convergence_factors);
    let convergence_factors: String = s
        .into_column_vector()
        .iter()
        .map(|singular_value| format!("{:.2} ", singular_value))
        .collect();
    info!(
        "near-null smoothing convergence factors: {}",
        convergence_factors
    );
    */

    let rhs_vec: Col<f64> = rhs_full.col(0).to_owned();
    let dst: Col<f64> = if cli.zero_guess {
        Col::zeros(base_mat.mat_ref().nrows())
    } else {
        let rng = &mut StdRng::seed_from_u64(42);
        CwiseColDistribution {
            nrows: rhs_full.nrows(),
            dist: StandardNormal,
        }
        .rand::<Col<f64>>(rng)
    };

    test_solver(
        par_op,
        arc_pc.clone(),
        Some(dst.as_mat()),
        Some(rhs_vec.as_mat()),
        cli.max_iters,
        cli.tolerance,
    );

    approx_convergence_factor(base_mat, arc_pc);

    Ok(())
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

    // TODO: add cli arg for isotropic vs anisotropic
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
        let edge_cost = partitioner.total_edge_cost() as f32;

        info!(
        "Iter: {}\n\tfine nodes: {}\n\taggregates: {}\n\tcoarsening factor: {:.2}\n\tmax / min agg size: {} / {}\n\tsize cost: {:.2e}\n\tedge cost: {:.2e}",
        iter,
        p.nnodes(),
        p.naggs(),
        p.cf(),
        max_size,
        min_size,
        size_cost,
        edge_cost,
    );
    }
}

fn create_weights(nn_basis: MatRef<f64>, mat: SparseRowMatRef<usize, f64>) -> Vec<f64> {
    nn_basis
        .col_iter()
        .map(|v| {
            let av = mat.rb() * v.rb();
            let vtav = v.transpose() * av;
            vtav.recip()
        })
        .collect()
}
