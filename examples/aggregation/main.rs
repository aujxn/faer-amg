use faer::{sparse::SparseRowMat, stats::prelude::*, Mat};
use indicatif::ProgressBar;
use plotters::{data::Quartiles as PlottersQuartiles, drawing::DrawingAreaErrorKind, prelude::*};
use std::collections::{HashMap, HashSet};
use std::{
    error::Error,
    num::NonZeroUsize,
    path::{Path, PathBuf},
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
        .join("2d")
        .join("spiral");

    let dataset_name = "h5_p1";

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

    let agg_penalty = 0.0005;
    let coarsening_factor = 64.0;
    let max_improvement_iters = 100;
    let partition_builder = PartitionBuilder::new(
        matrix.clone(),
        near_null_space,
        coarsening_factor,
        agg_penalty,
        max_improvement_iters,
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

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let image_dir = manifest_dir.join("target");
    std::fs::create_dir_all(&image_dir)?;

    const GIF_FRAME_DELAY_MS: u32 = 250;
    const GIF_FINAL_HOLD_REPEATS: usize = 5000 / GIF_FRAME_DELAY_MS as usize; // Additional repeats for ~5s hold on last frame
    const MAX_GIF_FRAMES: usize = 60;

    let gif_path = image_dir.join("aggregation_boundary.gif");
    let gif_backend =
        BitMapBackend::gif(&gif_path, (FRAME_WIDTH, FRAME_HEIGHT), GIF_FRAME_DELAY_MS)
            .map_err(|err| Box::<dyn Error>::from(err))?;
    let gif_area = gif_backend.into_drawing_area();

    let mut frame_idx = 0usize;
    let mut last_geometry;
    let mut frame_metrics = Vec::<FrameMetrics>::new();

    let geometry = collect_aggregate_geometry(matrix.as_ref(), &partitioner.partition, &coords)?;
    frame_metrics.push(capture_frame_metrics(&partitioner, frame_idx));
    draw_full_layout(&gif_area, &geometry, &frame_metrics, frame_idx)
        .map_err(|err| Box::<dyn Error>::from(err))?;
    gif_area
        .present()
        .map_err(|err| Box::<dyn Error>::from(err))?;

    info!(
        "Rendered boundary frame {} with {} edges, {} triangles to GIF",
        frame_idx,
        geometry.edges.len(),
        geometry.triangles.len(),
    );

    last_geometry = geometry.clone();

    let frames_to_generate = MAX_GIF_FRAMES.min(max_improvement_iters.saturating_add(1));
    let iterations_to_run = frames_to_generate.saturating_sub(1);

    for iter in 0..iterations_to_run {
        partitioner.improve(1);
        frame_idx += 1;

        let geometry =
            collect_aggregate_geometry(matrix.as_ref(), &partitioner.partition, &coords)?;
        frame_metrics.push(capture_frame_metrics(&partitioner, frame_idx));
        draw_full_layout(&gif_area, &geometry, &frame_metrics, frame_idx)
            .map_err(|err| Box::<dyn Error>::from(err))?;
        gif_area
            .present()
            .map_err(|err| Box::<dyn Error>::from(err))?;

        info!(
            "Rendered boundary frame {} (iteration {}) with {} edges, {} triangles to GIF",
            frame_idx,
            iter + 1,
            geometry.edges.len(),
            geometry.triangles.len(),
        );

        last_geometry = geometry.clone();
    }

    for _ in 0..GIF_FINAL_HOLD_REPEATS {
        gif_area
            .present()
            .map_err(|err| Box::<dyn Error>::from(err))?;
    }

    let boundary_img = image_dir.join("aggregation_boundary.png");
    draw_final_frame(&last_geometry, &frame_metrics, &boundary_img)?;

    info!(
        "Saved final boundary frame {} with {} edges, {} triangles to {}",
        frame_idx,
        last_geometry.edges.len(),
        last_geometry.triangles.len(),
        boundary_img.display()
    );
    info!("GIF animation written to {}", gif_path.display());

    Ok(())
}

type EdgeSegment = (usize, (f64, f64), (f64, f64));
type TriangleFacet = (usize, [(f64, f64); 3]);

#[derive(Clone)]
struct GeometryData {
    edges: Vec<EdgeSegment>,
    triangles: Vec<TriangleFacet>,
    points: Vec<(f64, f64)>,
}

#[derive(Clone)]
struct FrameMetrics {
    iteration: usize,
    modularity: f64,
    size_cost: f64,
    quartiles: PlottersQuartiles,
}

fn collect_aggregate_geometry(
    matrix: &SparseRowMat<usize, f64>,
    partition: &faer_amg::partitioners::Partition,
    coords: &Mat<f64>,
) -> Result<GeometryData, Box<dyn Error>> {
    let assignments = partition.node_assignments();
    let nnodes = assignments.len();
    assert_eq!(coords.nrows(), nnodes);

    if coords.ncols() < 2 {
        return Err(
            "At least two coordinate dimensions are required to plot boundary segments".into(),
        );
    }

    let mat_ref = matrix.as_ref();
    let symbolic = mat_ref.symbolic();
    let row_ptr = symbolic.row_ptr();
    let col_idx = symbolic.col_idx();

    let naggs = partition.naggs();

    let mut adjacency: Vec<HashSet<usize>> = vec![HashSet::new(); nnodes];

    for i in 0..nnodes {
        let row_start = row_ptr[i];
        let row_end = row_ptr[i + 1];
        for &j in &col_idx[row_start..row_end] {
            if i == j {
                continue;
            }

            adjacency[i].insert(j);
            adjacency[j].insert(i);
        }
    }

    let mut agg_nodes: Vec<Vec<usize>> = vec![Vec::new(); naggs];
    for (node_idx, &agg) in assignments.iter().enumerate() {
        agg_nodes[agg].push(node_idx);
    }

    let mut agg_edges: Vec<HashSet<(usize, usize)>> = vec![HashSet::new(); naggs];

    for i in 0..nnodes {
        for &j in &adjacency[i] {
            if i < j && assignments[i] == assignments[j] {
                agg_edges[assignments[i]].insert((i, j));
            }
        }
    }

    let mut agg_triangles: Vec<HashSet<[usize; 3]>> = vec![HashSet::new(); naggs];

    for agg_id in 0..naggs {
        for &node in &agg_nodes[agg_id] {
            let neighbors_in_agg: Vec<usize> = adjacency[node]
                .iter()
                .copied()
                .filter(|&neighbor| assignments[neighbor] == agg_id)
                .collect();

            for a_idx in 0..neighbors_in_agg.len() {
                let a = neighbors_in_agg[a_idx];
                for b_idx in (a_idx + 1)..neighbors_in_agg.len() {
                    let b = neighbors_in_agg[b_idx];
                    if adjacency[a].contains(&b) {
                        let mut tri = [node, a, b];
                        tri.sort_unstable();
                        agg_triangles[agg_id].insert(tri);
                    }
                }
            }
        }
    }

    let mut edges = Vec::new();
    for (agg_id, edge_set) in agg_edges.into_iter().enumerate() {
        for (a, b) in edge_set {
            let start = (coords[(a, 0)], coords[(a, 1)]);
            let end = (coords[(b, 0)], coords[(b, 1)]);
            edges.push((agg_id, start, end));
        }
    }

    if edges.is_empty() {
        return Err("No edges were detected for the current partition; cannot render plot".into());
    }

    let mut triangles = Vec::new();
    for (agg_id, triangle_set) in agg_triangles.into_iter().enumerate() {
        for [a, b, c] in triangle_set {
            triangles.push((
                agg_id,
                [
                    (coords[(a, 0)], coords[(a, 1)]),
                    (coords[(b, 0)], coords[(b, 1)]),
                    (coords[(c, 0)], coords[(c, 1)]),
                ],
            ));
        }
    }

    let points = (0..nnodes)
        .map(|idx| (coords[(idx, 0)], coords[(idx, 1)]))
        .collect();

    Ok(GeometryData {
        edges,
        triangles,
        points,
    })
}

fn capture_frame_metrics(
    partitioner: &faer_amg::partitioners::ModularityPartitioner,
    iteration: usize,
) -> FrameMetrics {
    let modularity = partitioner.modularity();
    let size_cost = partitioner.total_agg_size_cost();
    let quartiles = compute_aggregate_quartiles(&partitioner.partition);
    FrameMetrics {
        iteration,
        modularity,
        size_cost,
        quartiles,
    }
}

fn compute_aggregate_quartiles(partition: &faer_amg::partitioners::Partition) -> PlottersQuartiles {
    let sizes: Vec<f64> = partition
        .aggregate_sizes()
        .into_iter()
        .map(|size| size as f64)
        .collect();
    PlottersQuartiles::new(&sizes)
}

const GEOM_WIDTH: u32 = 900;
const GEOM_HEIGHT: u32 = 900;
const SIDE_WIDTH: u32 = 900;
const METRICS_HEIGHT: u32 = 450;
const FRAME_WIDTH: u32 = GEOM_WIDTH + SIDE_WIDTH;
const FRAME_HEIGHT: u32 = GEOM_HEIGHT;
const FINAL_WIDTH: u32 = FRAME_WIDTH;
const FINAL_HEIGHT: u32 = FRAME_HEIGHT;

fn draw_final_frame(
    geometry: &GeometryData,
    metrics: &[FrameMetrics],
    output_path: &Path,
) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new(output_path, (FINAL_WIDTH, FINAL_HEIGHT)).into_drawing_area();
    let final_iter = metrics.last().map(|m| m.iteration).unwrap_or(0);
    draw_full_layout(&root, geometry, metrics, final_iter)
        .map_err(|err| Box::<dyn Error>::from(err))?;
    root.present().map_err(|err| Box::<dyn Error>::from(err))?;
    Ok(())
}

fn draw_full_layout<DB>(
    area: &DrawingArea<DB, plotters::coord::Shift>,
    geometry: &GeometryData,
    metrics: &[FrameMetrics],
    iteration: usize,
) -> Result<(), DrawingAreaErrorKind<DB::ErrorType>>
where
    DB: DrawingBackend,
{
    area.fill(&WHITE)?;
    let (geometry_area, side_area) = area.split_horizontally(GEOM_WIDTH);
    draw_geometry_on_area(&geometry_area, geometry, iteration)?;
    let (metrics_area, quartiles_area) = side_area.split_vertically(METRICS_HEIGHT);
    draw_metrics_chart(&metrics_area, metrics)?;
    draw_quartiles_chart(&quartiles_area, metrics)?;
    Ok(())
}

fn draw_geometry_on_area<DB>(
    drawing_area: &DrawingArea<DB, plotters::coord::Shift>,
    geometry: &GeometryData,
    iteration: usize,
) -> Result<(), DrawingAreaErrorKind<DB::ErrorType>>
where
    DB: DrawingBackend,
{
    let (area_width, area_height) = drawing_area.dim_in_pixel();
    let aspect = if area_height == 0 {
        GEOM_WIDTH as f64 / GEOM_HEIGHT as f64
    } else {
        area_width as f64 / area_height as f64
    };
    let (x_min, x_max, y_min, y_max) = compute_plot_bounds(geometry, aspect);
    let mut agg_colors: HashMap<usize, RGBColor> = HashMap::new();
    let mut unique_aggs: Vec<usize> = geometry
        .edges
        .iter()
        .map(|(agg, _, _)| *agg)
        .chain(geometry.triangles.iter().map(|(agg, _)| *agg))
        .collect();
    unique_aggs.sort_unstable();
    unique_aggs.dedup();
    for (idx, agg) in unique_aggs.iter().enumerate() {
        agg_colors.insert(*agg, palette_rgb(idx));
    }

    drawing_area.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(drawing_area)
        .margin(25)
        .caption(
            format!("Aggregate Graph – Iter {iteration}"),
            ("sans-serif", 45),
        )
        .set_label_area_size(LabelAreaPosition::Left, 70)
        .set_label_area_size(LabelAreaPosition::Bottom, 70)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

    chart
        .configure_mesh()
        .disable_mesh()
        .x_desc("x")
        .y_desc("y")
        .label_style(("sans-serif", 24))
        .draw()?;

    for (agg, triangle) in &geometry.triangles {
        let color = agg_colors
            .get(agg)
            .copied()
            .unwrap_or_else(|| palette_rgb(0));
        chart.draw_series(std::iter::once(Polygon::new(
            triangle.to_vec(),
            ShapeStyle::from(color.mix(0.9)).filled(),
        )))?;
    }

    for (_, start, end) in &geometry.edges {
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(start.0, start.1), (end.0, end.1)],
            ShapeStyle::from(&BLACK).stroke_width(1),
        )))?;
    }

    Ok(())
}

fn compute_plot_bounds(geometry: &GeometryData, aspect: f64) -> (f64, f64, f64, f64) {
    if geometry.points.is_empty() {
        return (0.0, 1.0, 0.0, 1.0);
    }

    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    for &(x, y) in &geometry.points {
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }

    let mut x_span = (max_x - min_x).max(f64::EPSILON);
    let mut y_span = (max_y - min_y).max(f64::EPSILON);

    let mut x_min = min_x;
    let mut x_max = max_x;
    let mut y_min = min_y;
    let mut y_max = max_y;

    if aspect.is_finite() && (x_span / y_span) > aspect {
        let desired_y_span = x_span / aspect;
        let pad = (desired_y_span - y_span) / 2.0;
        y_min -= pad;
        y_max += pad;
        y_span = desired_y_span;
    } else if aspect.is_finite() {
        let desired_x_span = y_span * aspect;
        let pad = (desired_x_span - x_span) / 2.0;
        x_min -= pad;
        x_max += pad;
        x_span = desired_x_span;
    }

    let x_pad = x_span * 0.05;
    let y_pad = y_span * 0.05;
    x_min -= x_pad;
    x_max += x_pad;
    y_min -= y_pad;
    y_max += y_pad;

    (x_min, x_max, y_min, y_max)
}

fn palette_rgb(idx: usize) -> RGBColor {
    let colors = Palette99::COLORS;
    let (r, g, b) = colors[idx % colors.len()];
    RGBColor(r, g, b)
}

fn draw_metrics_chart<DB>(
    area: &DrawingArea<DB, plotters::coord::Shift>,
    metrics: &[FrameMetrics],
) -> Result<(), DrawingAreaErrorKind<DB::ErrorType>>
where
    DB: DrawingBackend,
{
    area.fill(&WHITE)?;
    if metrics.is_empty() {
        return Ok(());
    }

    let x_min = metrics.first().unwrap().iteration as i32;
    let x_max = metrics.last().unwrap().iteration as i32;
    let x_upper = x_max + 1;
    let x_range = x_min..x_upper;

    let (mod_min, mod_max) = metrics
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |acc, m| {
            (acc.0.min(m.modularity), acc.1.max(m.modularity))
        });
    let (cost_min, cost_max) = metrics
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |acc, m| {
            (acc.0.min(m.size_cost), acc.1.max(m.size_cost))
        });

    let mod_span = if (mod_max - mod_min).abs() < 1e-9 {
        1.0
    } else {
        mod_max - mod_min
    };
    let cost_span = if (cost_max - cost_min).abs() < 1e-9 {
        1.0
    } else {
        cost_max - cost_min
    };

    let mut chart = ChartBuilder::on(area)
        .margin(15)
        .set_label_area_size(LabelAreaPosition::Left, 60)
        .set_label_area_size(LabelAreaPosition::Right, 70)
        .set_label_area_size(LabelAreaPosition::Bottom, 45)
        .caption("Modularity vs Aggregate Size Penalty", ("sans-serif", 28))
        .build_cartesian_2d(
            x_range.clone(),
            (mod_min - 0.05 * mod_span)..(mod_max + 0.05 * mod_span),
        )?
        .set_secondary_coord(
            x_range.clone(),
            (cost_min - 0.05 * cost_span)..(cost_max + 0.05 * cost_span),
        );

    chart
        .configure_mesh()
        .x_desc("Iteration")
        .y_desc("Modularity")
        .label_style(("sans-serif", 20))
        .draw()?;

    chart
        .configure_secondary_axes()
        .y_desc("Agg Size Cost")
        .label_style(("sans-serif", 20))
        .draw()?;

    chart
        .draw_series(LineSeries::new(
            metrics.iter().map(|m| (m.iteration as i32, m.modularity)),
            &BLUE,
        ))?
        .label("Modularity")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    chart
        .draw_secondary_series(LineSeries::new(
            metrics.iter().map(|m| (m.iteration as i32, m.size_cost)),
            &RED,
        ))?
        .label("Agg Size Cost")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperLeft)
        .label_font(("sans-serif", 18))
        .border_style(&BLACK)
        .draw()?;

    Ok(())
}

fn draw_quartiles_chart<DB>(
    area: &DrawingArea<DB, plotters::coord::Shift>,
    metrics: &[FrameMetrics],
) -> Result<(), DrawingAreaErrorKind<DB::ErrorType>>
where
    DB: DrawingBackend,
{
    area.fill(&WHITE)?;
    if metrics.is_empty() {
        return Ok(());
    }

    let x_min = metrics.first().unwrap().iteration as i32;
    let x_max = metrics.last().unwrap().iteration as i32;
    let x_upper = x_max + 1;
    let x_range = x_min..x_upper;

    let (min_val, max_val) = metrics
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |acc, m| {
            let values = m.quartiles.values();
            (acc.0.min(values[0]), acc.1.max(values[4]))
        });
    let span = if (max_val - min_val).abs() < 1e-6 {
        1.0
    } else {
        max_val - min_val
    };

    let mut chart = ChartBuilder::on(area)
        .margin(15)
        .set_label_area_size(LabelAreaPosition::Left, 60)
        .set_label_area_size(LabelAreaPosition::Bottom, 45)
        .caption("Aggregate Size Quartiles", ("sans-serif", 28))
        .build_cartesian_2d(
            x_range.clone(),
            (min_val - 0.1 * span)..(max_val + 0.1 * span),
        )?;

    chart
        .configure_mesh()
        .x_desc("Iteration")
        .y_desc("Aggregate Size")
        .label_style(("sans-serif", 20))
        .draw()?;

    let series_color = RGBColor(30, 144, 255);

    chart
        .draw_series(metrics.iter().map(|m| {
            Boxplot::new_vertical(m.iteration as i32, &m.quartiles)
                .width(18)
                .whisker_width(0.6)
                .style(ShapeStyle::from(series_color))
        }))?
        .label("Quartiles")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RGBColor(30, 144, 255)));

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperRight)
        .label_font(("sans-serif", 18))
        .border_style(&BLACK)
        .draw()?;

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
