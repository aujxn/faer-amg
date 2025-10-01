use faer::{stats::prelude::*, Col, Mat};
use std::{
    error::Error,
    fs::File,
    io::{BufWriter, Write},
    num::NonZeroUsize,
    path::PathBuf,
    sync::Arc,
};

use faer_amg::{
    partitioners::PartitionBuilder,
    preconditioners::smoothers::new_l1,
    utils::{load_mfem_linear_system, MfemLinearSystem},
};
use log::{info, LevelFilter};

fn main() -> Result<(), Box<dyn Error>> {
    env_logger::builder()
        .format_timestamp(None)
        .filter_level(LevelFilter::Info)
        .try_init()?;

    faer::set_global_parallelism(faer::Par::Rayon(NonZeroUsize::new(16).unwrap()));

    let data_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join("data")
        .join("anisotropy");

    let MfemLinearSystem {
        matrix,
        rhs,
        coords,
        boundary_indices,
    } = load_mfem_linear_system(&data_dir, true)?;

    let near_null_dim = 32;
    let smoothing_iters = 100;
    let near_null_space = smooth_vector(&matrix, smoothing_iters, near_null_dim);

    info!(
        "Loaded system from {}\n  matrix: {} x {} ({} nnz)\n  rhs columns: {}\n  geom dim: {}\n  removed dirichlet nodes: {}",
        data_dir.display(),
        matrix.nrows(),
        matrix.ncols(),
        matrix.compute_nnz(),
        rhs.ncols(),
        coords.ncols(),
        boundary_indices.len()
    );

    let matrix = Arc::new(matrix);

    let agg_penalty = 0.005;
    let coarsening_factor = 6.0;
    let max_refinement_iters = 3000;
    let partition = PartitionBuilder::new(
        matrix.clone(),
        near_null_space,
        coarsening_factor,
        agg_penalty,
        Some(24),
        Some(2),
        max_refinement_iters,
    )
    .build();

    info!(
        "Generated aggregation:\n  aggregates: {}\n  fine nodes: {}\n  coarsening factor: {:.2}",
        partition.naggs(),
        partition.nnodes(),
        partition.cf()
    );
    partition.info();

    let output_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join("aggregation")
        .join("anisotropy_partition.csv");
    let mut writer = BufWriter::new(File::create(&output_path)?);

    write!(writer, "node")?;
    for dim in 0..coords.ncols() {
        write!(writer, ",coord{}", dim)?;
    }
    writeln!(writer, ",aggregate")?;

    for (node_idx, &agg_id) in partition.node_assignments().iter().enumerate() {
        write!(writer, "{}", node_idx)?;
        for dim in 0..coords.ncols() {
            write!(writer, ",{}", coords[(node_idx, dim)])?;
        }
        writeln!(writer, ",{}", agg_id)?;
    }

    writer.flush()?;
    info!("Wrote partition data to {}", output_path.display());

    Ok(())
}

fn smooth_vector(
    mat: &faer::sparse::SparseRowMat<usize, f64>,
    iterations: usize,
    near_null_dim: usize,
) -> Mat<f64> {
    let mat_ref = mat.as_ref();
    let l1_diag = new_l1(&mat_ref);

    let rng = &mut StdRng::seed_from_u64(42);
    let n = mat.nrows();

    let mut x = CwiseMatDistribution {
        nrows: n,
        ncols: near_null_dim,
        dist: StandardNormal,
    }
    .rand::<Mat<f64>>(rng);
    let mut r;

    for _ in 0..iterations {
        x = x.qr().compute_thin_Q();
        r = mat.as_ref() * &x;
        r = &l1_diag * r;
        x -= r;
    }

    let convergence_factors: String = x
        .col_iter()
        .map(|col| format!("{:.2} ", col.norm_l2()))
        .collect();
    info!(
        "converge factor after {} iterations: {}",
        iterations, convergence_factors
    );

    x.qr().compute_thin_Q()
}
