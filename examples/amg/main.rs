use std::{error::Error, num::NonZeroUsize, path::PathBuf, sync::Arc};

use clap::{Parser, ValueEnum};
use env_logger;
use faer::{
    mat::AsMatRef,
    matrix_free::LinOp,
    prelude::Reborrow,
    sparse::SparseRowMatRef,
    stats::{prelude::StandardNormal, CwiseColDistribution, DistributionExt},
    Col, ColRef, Mat, MatRef, Par,
};
use faer_amg::{
    adaptivity::find_near_null,
    adaptivity::AdaptiveConfig,
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
use log::{info, LevelFilter};
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

    /// Use smoothed aggregation interpolation with the given number of smoothing steps
    #[arg(long, value_name = "STEPS", conflicts_with = "classical_interpolation")]
    sa_interpolation: Option<usize>,

    /// Coarsening factor for smoothed aggregation (target fine-to-coarse ratio)
    #[arg(
        long,
        alias = "sa-cf",
        default_value_t = 8.0,
        requires = "sa_interpolation"
    )]
    sa_coarsening_factor: f64,

    /// Use classical interpolation (mutually exclusive with --sa-interpolation)
    #[arg(long, default_value_t = false, conflicts_with = "sa_interpolation")]
    classical_interpolation: bool,

    /// Classical interpolation options: tau=...,search=...,depth=...,max=...
    #[arg(
        long,
        value_name = "KEY=VAL,...",
        conflicts_with = "sa_interpolation",
        requires = "classical_interpolation"
    )]
    classical_opts: Option<String>,

    /// Choose between isotropic or anisotropic coefficient datasets
    #[arg(long, value_enum, default_value_t = Anisotropy::Anisotropic)]
    anisotropy: Anisotropy,

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

    /// Maximum number of multigrid levels (omit for no limit)
    #[arg(long)]
    max_levels: Option<usize>,

    /// Use a composite adaptive preconditioner with this many components
    #[arg(long, value_name = "N")]
    composite: Option<usize>,
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

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum Anisotropy {
    Isotropic,
    Anisotropic,
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
    if let Some(components) = cli.composite {
        if components == 0 {
            return Err("composite must be positive".into());
        }
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
    let use_classical = cli.classical_interpolation || cli.sa_interpolation.is_none();
    let interpolation_config = if use_classical {
        let cr_options = CompatibleRelaxationConfig {
            relax_steps: 5,
            target_convergence: 0.02,
        };
        let mut ls_options = LeastSquaresConfig {
            search_depth: 3,
            depth_ls: 2,
            solver: LsSolver::Constrained,
            max_interp: 3,
            tau_threshold: 1.2,
        };
        if let Some(opts) = cli.classical_opts.as_deref() {
            let overrides = parse_classical_opts(opts)?;
            if let Some(tau) = overrides.tau_threshold {
                ls_options.tau_threshold = tau;
            }
            if let Some(search_depth) = overrides.search_depth {
                ls_options.search_depth = search_depth;
            }
            if let Some(depth_ls) = overrides.depth_ls {
                ls_options.depth_ls = depth_ls;
            }
            if let Some(max_interp) = overrides.max_interp {
                ls_options.max_interp = max_interp;
            }
        }
        let classical_config = ClassicalConfig {
            cr_options,
            ls_options,
        };
        InterpolationConfig::Classical(classical_config)
    } else if let Some(smoothing_steps) = cli.sa_interpolation {
        let partitioner_config = PartitionerConfig {
            coarsening_factor: cli.sa_coarsening_factor,
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
        unreachable!("sa_interpolation is None and use_classical is false")
    };

    let hierarchy_config = HierarchyConfig {
        coarsest_dim: cli.coarsest_dim,
        interpolation_config,
        max_levels: cli.max_levels,
    };
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

    if let Some(components) = cli.composite {
        let adaptive_config = AdaptiveConfig {
            hierarchy_config,
            multigrid_config,
            test_iters: cli.smoothing_iters,
            max_components: components,
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
        let composite_table = build_composite_table(&results);
        info!("{composite_table}");
        return Ok(());
    }

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
    let op_complexity = hierarchy.op_complexity();

    let mesh_viz = MeshViz::new(&hierarchy, system.coords);
    let serialized = serde_json::to_string(&mesh_viz)?;
    let file_path = "./data/hierarchy_viz.json";
    fs::write(file_path, serialized)?;

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

    let ((cg_iters, _cg_residual, _dst_cg), (sli_iters, _sli_residual, _dst_sli)) = test_solver(
        par_op,
        arc_pc.clone(),
        Some(dst.as_mat()),
        Some(rhs_vec.as_mat()),
        cli.max_iters,
        cli.tolerance,
    );

    // approximates \|E\|_A where E = I - M^{-1}A is the iteration matrix with multigrid op M^{-1}
    let a_norm_of_e = approx_convergence_factor(base_mat, arc_pc);

    println!(
        "{} {} {} {}",
        cg_iters, sli_iters, a_norm_of_e, op_complexity
    );

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

    let data_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("data")
        .join(match cli.anisotropy {
            Anisotropy::Isotropic => "isotropic",
            Anisotropy::Anisotropic => "anisotropy",
        })
        .join("2d")
        .join(base_coef_dir);
    let dataset_name = format!("h{}_p1", cli.refinements);
    (data_dir, dataset_name)
}

#[derive(Default)]
struct ClassicalOverrides {
    tau_threshold: Option<f64>,
    search_depth: Option<usize>,
    depth_ls: Option<usize>,
    max_interp: Option<usize>,
}

fn parse_classical_opts(input: &str) -> Result<ClassicalOverrides, Box<dyn Error>> {
    let mut out = ClassicalOverrides::default();
    if input.trim().is_empty() {
        return Ok(out);
    }
    for item in input.split(',') {
        let item = item.trim();
        if item.is_empty() {
            continue;
        }
        let (key, value) = item
            .split_once('=')
            .ok_or_else(|| format!("Invalid classical_opts entry '{item}', expected key=value"))?;
        let key = key.trim();
        let value = value.trim();
        match key {
            "tau" | "tau_threshold" => {
                out.tau_threshold = Some(value.parse()?);
            }
            "search" | "search_depth" => {
                out.search_depth = Some(value.parse()?);
            }
            "depth" | "depth_ls" => {
                out.depth_ls = Some(value.parse()?);
            }
            "max" | "max_interp" => {
                out.max_interp = Some(value.parse()?);
            }
            _ => {
                return Err(format!(
                    "Unknown classical_opts key '{key}' (expected tau, search, depth, max)"
                )
                .into());
            }
        }
    }
    Ok(out)
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

fn build_composite_table(results: &CompositeSolveResults) -> String {
    let pcg_table = build_results_table(&results.pcg);
    let stationary_table = build_results_table(&results.stationary);
    format!(
        "Composite PCG results:\n{pcg_table}\nComposite stationary results:\n{stationary_table}"
    )
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
