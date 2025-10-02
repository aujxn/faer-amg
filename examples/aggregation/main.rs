use faer::{sparse::SparseRowMat, stats::prelude::*, Mat};
use indicatif::ProgressBar;
use std::collections::HashSet;
use std::{
    error::Error,
    fs::File,
    io::{BufWriter, Write},
    num::NonZeroUsize,
    path::{Path, PathBuf},
    process::Command,
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
        //.join("examples")
        .join("data")
        .join("anisotropy")
        .join("2d");

    let dataset_name = "h4_p1";

    let MfemLinearSystem {
        matrix,
        rhs,
        coords,
        boundary_indices,
    } = load_mfem_linear_system(&data_dir, dataset_name, true)?;

    let near_null_dim = 256;
    let smoothing_iters = 50;
    let near_null_space = smooth_vector(&matrix, smoothing_iters, near_null_dim);

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

    let agg_penalty = 0.001;
    let coarsening_factor = 16.0;
    let max_refinement_iters = 100;
    let partition = PartitionBuilder::new(
        matrix.clone(),
        near_null_space,
        coarsening_factor,
        agg_penalty,
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

    let partition_output_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join("aggregation")
        .join("anisotropy_partition.csv");
    write_partition_points(&partition, &coords, &partition_output_path)?;
    info!(
        "Wrote partition data to {}",
        partition_output_path.display()
    );

    let boundary_output_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join("aggregation")
        .join("anisotropy_boundary_segments.csv");
    let (segment_count, boundary_node_count) =
        write_boundary_segments(matrix.as_ref(), &partition, &coords, &boundary_output_path)?;
    info!(
        "Wrote {} boundary segments ({} boundary nodes) to {}",
        segment_count,
        boundary_node_count,
        boundary_output_path.display()
    );

    run_plotting_script(&partition_output_path, &boundary_output_path)?;

    Ok(())
}

fn write_partition_points(
    partition: &faer_amg::partitioners::Partition,
    coords: &Mat<f64>,
    output_path: &Path,
) -> Result<(), Box<dyn Error>> {
    let assignments = partition.node_assignments();
    assert_eq!(assignments.len(), coords.nrows());

    let mut writer = BufWriter::new(File::create(output_path)?);

    write!(writer, "node")?;
    for dim in 0..coords.ncols() {
        write!(writer, ",coord{}", dim)?;
    }
    writeln!(writer, ",aggregate")?;

    for (node_idx, &agg_id) in assignments.iter().enumerate() {
        write!(writer, "{}", node_idx)?;
        for dim in 0..coords.ncols() {
            write!(writer, ",{}", coords[(node_idx, dim)])?;
        }
        writeln!(writer, ",{}", agg_id)?;
    }

    writer.flush()?;
    Ok(())
}

fn write_boundary_segments(
    matrix: &SparseRowMat<usize, f64>,
    partition: &faer_amg::partitioners::Partition,
    coords: &Mat<f64>,
    output_path: &Path,
) -> Result<(usize, usize), Box<dyn Error>> {
    let assignments = partition.node_assignments();
    let nnodes = assignments.len();
    assert_eq!(coords.nrows(), nnodes);

    let mat_ref = matrix.as_ref();
    let symbolic = mat_ref.symbolic();
    let row_ptr = symbolic.row_ptr();
    let col_idx = symbolic.col_idx();

    let mut is_boundary = vec![false; nnodes];

    for i in 0..nnodes {
        let row_start = row_ptr[i];
        let row_end = row_ptr[i + 1];
        let agg_i = assignments[i];
        for &j in &col_idx[row_start..row_end] {
            if i == j {
                continue;
            }
            if assignments[j] != agg_i {
                is_boundary[i] = true;
                is_boundary[j] = true;
            }
        }
    }

    let boundary_node_count = is_boundary.iter().filter(|flag| **flag).count();
    let mut seen_edges = HashSet::new();
    let mut segments = Vec::new();

    for i in 0..nnodes {
        let row_start = row_ptr[i];
        let row_end = row_ptr[i + 1];
        for &j in &col_idx[row_start..row_end] {
            if i == j {
                continue;
            }

            let (a, b) = if i < j { (i, j) } else { (j, i) };
            if !seen_edges.insert((a, b)) {
                continue;
            }

            if !(is_boundary[a] && is_boundary[b]) {
                continue;
            }

            let agg = assignments[a];
            if assignments[b] != agg {
                continue;
            }

            segments.push((agg, a, b));
        }
    }

    let mut writer = BufWriter::new(File::create(output_path)?);
    write!(writer, "aggregate,start_node,end_node")?;
    for dim in 0..coords.ncols() {
        write!(writer, ",start_coord{}", dim)?;
    }
    for dim in 0..coords.ncols() {
        write!(writer, ",end_coord{}", dim)?;
    }
    writeln!(writer)?;

    for (agg, start, end) in &segments {
        write!(writer, "{},{},{}", agg, start, end)?;
        for dim in 0..coords.ncols() {
            write!(writer, ",{}", coords[(*start, dim)])?;
        }
        for dim in 0..coords.ncols() {
            write!(writer, ",{}", coords[(*end, dim)])?;
        }
        writeln!(writer)?;
    }

    writer.flush()?;

    Ok((segments.len(), boundary_node_count))
}

fn run_plotting_script(partition_csv: &Path, boundary_csv: &Path) -> Result<(), Box<dyn Error>> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let python_bin = manifest_dir.join("venv").join("bin").join("python");
    let script_path = manifest_dir
        .join("examples")
        .join("aggregation")
        .join("plot_partition.py");
    let image_dir = manifest_dir.join("target");
    std::fs::create_dir_all(&image_dir)?;
    let partition_img = image_dir.join("aggregation_partition.png");
    let boundary_img = image_dir.join("aggregation_boundary.png");

    info!("Running plotting script {}", script_path.display());
    let status = Command::new(python_bin)
        .arg(script_path)
        .arg(partition_csv)
        .arg("--boundary-csv")
        .arg(boundary_csv)
        .arg("--save")
        .arg(&partition_img)
        .arg("--boundary-save")
        .arg(&boundary_img)
        .status()?;

    if !status.success() {
        return Err(format!("Plotting script exited with non-zero status: {}", status).into());
    }

    info!(
        "Saved plots to {} and {}",
        partition_img.display(),
        boundary_img.display()
    );

    Ok(())
}

fn smooth_vector(
    mat: &faer::sparse::SparseRowMat<usize, f64>,
    iterations: usize,
    near_null_dim: usize,
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

    for _ in 0..iterations {
        x = x.qr().compute_thin_Q();
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

    x.qr().compute_thin_Q()
}
