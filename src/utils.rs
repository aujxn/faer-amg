use std::{
    fmt,
    fs::File,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
};

use faer::{
    sparse::{SparseRowMat, SparseRowMatRef, Triplet},
    Mat,
};
use sci_bevy_comm::{load_triangle_mesh_data, MeshGeometry};

pub fn mats_are_equal(left: &SparseRowMat<usize, f64>, right: &SparseRowMat<usize, f64>) -> bool {
    let l_shape = left.shape();
    let l_nnz = left.compute_nnz();
    let r_shape = right.shape();
    let r_nnz = right.compute_nnz();

    if l_shape != r_shape || l_nnz != r_nnz {
        println!(
            "Left: {:?}, {} nnz\nRight: {:?}, {} nnz",
            l_shape, l_nnz, r_shape, r_nnz
        );
        return false;
    }

    for (left, right) in left.triplet_iter().zip(right.triplet_iter()) {
        let absolute = (left.val - right.val).abs();
        let relative = absolute / left.val.max(*right.val);
        if left.row != right.row || left.col != right.col || absolute > 1e-12 || relative > 1e-12 {
            println!(
                "Left: {}, {} : {}\nRight: {}, {} : {}\nRelative err: {}\nAbsolute err: {}",
                left.row, left.col, left.val, right.row, right.col, right.val, relative, absolute
            );
            return false;
        }
    }
    true
}

#[derive(Clone)]
pub struct MatrixStats {
    pub rows: usize,
    pub cols: usize,
    pub nnz: usize,
    pub sparsity: f64,
    pub entries_min: f64,
    pub entries_max: f64,
    pub entries_avg: f64,
    pub weight_min: f64,
    pub weight_max: f64,
    pub weight_avg: f64,
    pub rowsum_min: f64,
    pub rowsum_max: f64,
    pub rowsum_avg: f64,
}

pub fn matrix_stats(mat: SparseRowMatRef<usize, f64>) -> MatrixStats {
    let rows = mat.nrows();
    let cols = mat.ncols();
    let nnz = mat.compute_nnz();
    let total_entries = nnz as f64;

    let mut entries_min = f64::INFINITY;
    let mut entries_max = 0.0f64;
    let mut entries_sum = 0.0f64;

    let mut weight_min = f64::INFINITY;
    let mut weight_max = f64::NEG_INFINITY;
    let mut weight_sum = 0.0f64;

    let mut rowsum_min = f64::INFINITY;
    let mut rowsum_max = f64::NEG_INFINITY;
    let mut rowsum_sum = 0.0f64;

    for row in 0..rows {
        let values = mat.val_of_row(row);
        let entries = values.len() as f64;
        entries_min = entries_min.min(entries);
        entries_max = entries_max.max(entries);
        entries_sum += entries;

        let rowsum: f64 = values.iter().copied().sum();
        rowsum_min = rowsum_min.min(rowsum);
        rowsum_max = rowsum_max.max(rowsum);
        rowsum_sum += rowsum;

        for &val in values {
            weight_min = weight_min.min(val);
            weight_max = weight_max.max(val);
            weight_sum += val;
        }
    }

    if entries_min.is_infinite() {
        entries_min = 0.0;
    }
    if weight_min.is_infinite() {
        weight_min = 0.0;
    }
    if weight_max.is_infinite() {
        weight_max = 0.0;
    }
    if rowsum_min.is_infinite() {
        rowsum_min = 0.0;
    }
    if rowsum_max.is_infinite() {
        rowsum_max = 0.0;
    }

    let entries_avg = if rows > 0 {
        entries_sum / rows as f64
    } else {
        0.0
    };

    let weight_avg = if total_entries > 0.0 {
        weight_sum / total_entries
    } else {
        0.0
    };

    let rowsum_avg = if rows > 0 {
        rowsum_sum / rows as f64
    } else {
        0.0
    };

    let sparsity = if rows == 0 || cols == 0 {
        0.0
    } else {
        nnz as f64 / (rows as f64 * cols as f64)
    };

    MatrixStats {
        rows,
        cols,
        nnz,
        sparsity,
        entries_min,
        entries_max,
        entries_avg,
        weight_min,
        weight_max,
        weight_avg,
        rowsum_min,
        rowsum_max,
        rowsum_avg,
    }
}

pub(crate) enum NdofsFormat {
    Auto,
    Rectangular,
}

pub(crate) fn write_matrix_stats_table(
    f: &mut fmt::Formatter<'_>,
    title: &str,
    stats: &[MatrixStats],
    ndofs_format: NdofsFormat,
) -> fmt::Result {
    if stats.is_empty() {
        writeln!(f, "{}: <empty>", title)?;
        return Ok(());
    }

    writeln!(f, "{title}")?;
    writeln!(
        f,
        "{:>4}  {:>12}  {:>15}  {:>10}  {:>26}  {:>26}  {:>26}",
        "lev", "ndofs", "nnz", "sparsity", "entries / row", "weights", "rowsums"
    )?;
    writeln!(
        f,
        "{:-<4}  {:-<12}  {:-<15}  {:-<10}  {:-<26}  {:-<26}  {:-<26}",
        "", "", "", "", "", "", ""
    )?;
    writeln!(
        f,
        "{:>4}  {:>12}  {:>15}  {:>10}  {:>8} {:>8} {:>8}  {:>8} {:>8} {:>8}  {:>8} {:>8} {:>8}",
        "", "", "", "", "min", "max", "avg", "min", "max", "avg", "min", "max", "avg"
    )?;
    writeln!(
        f,
        "{:-<4}  {:-<12}  {:-<15}  {:-<10}  {:-<8} {:-<8} {:-<8}  {:-<8} {:-<8} {:-<8}  {:-<8} {:-<8} {:-<8}",
        "", "", "", "", "", "", "", "", "", "", "", "", ""
    )?;

    for (level, stat) in stats.iter().enumerate() {
        let ndofs = match ndofs_format {
            NdofsFormat::Rectangular => format!("{}x{}", stat.rows, stat.cols),
            NdofsFormat::Auto => {
                if stat.rows == stat.cols {
                    format!("{}", stat.rows)
                } else {
                    format!("{}x{}", stat.rows, stat.cols)
                }
            }
        };

        writeln!(
            f,
            "{:>4}  {:>12}  {:>15}  {:>10.2}  {:>8} {:>8} {:>8.2}  {:>8.2} {:>8.2} {:>8.2}  {:>8.2} {:>8.2} {:>8.2}",
            level,
            ndofs,
            stat.nnz,
            stat.sparsity,
            stat.entries_min,
            stat.entries_max,
            stat.entries_avg,
            stat.weight_min,
            stat.weight_max,
            stat.weight_avg,
            stat.rowsum_min,
            stat.rowsum_max,
            stat.rowsum_avg
        )?;
    }

    Ok(())
}

impl fmt::Debug for MatrixStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_matrix_stats_table(f, "Matrix info:", &[self.clone()], NdofsFormat::Rectangular)?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct MfemIndexMapping {
    /// Maps original mesh vertex indices to solution vector indices (None for removed boundary nodes).
    pub mesh_to_solution: Vec<Option<usize>>,
    /// Maps solution vector indices back to the original mesh vertex indices.
    pub solution_to_mesh: Vec<usize>,
}

#[derive(Debug)]
pub struct MfemLinearSystem {
    pub matrix: SparseRowMat<usize, f64>,
    pub rhs: Mat<f64>,
    pub coords: Mat<f64>,
    pub boundary_indices: Vec<usize>,
    pub mesh_geometry: Option<MeshGeometry>,
    pub index_mapping: MfemIndexMapping,
    pub original_dimension: usize,
}

pub fn load_mfem_linear_system<P: AsRef<Path>, S: AsRef<Path>>(
    dir: P,
    name: S,
    delete_boundary: bool,
) -> Result<MfemLinearSystem, Box<dyn std::error::Error>> {
    let dir = dir.as_ref();
    let base_path = dir.join(name);

    let matrix_path = ensure_named_file_exists(&base_path, "mtx")?;
    let boundary_path = ensure_named_file_exists(&base_path, "bdy")?;
    let coords_path = ensure_named_file_exists(&base_path, "coords")?;
    let rhs_path = ensure_named_file_exists(&base_path, "rhs")?;

    let mut boundary_indices = load_boundary_indices(&boundary_path)?;
    boundary_indices.sort_unstable();
    boundary_indices.dedup();

    let (nrows, ncols, triplets) = load_matrix_triplets(&matrix_path)?;
    if nrows != ncols {
        return Err("The MFEM loader currently supports only square matrices".into());
    }

    let coords_data = load_dense_rows(&coords_path)?;
    if coords_data.len() != nrows {
        return Err(format!(
            "Coordinate rows ({}) must match matrix dimension ({})",
            coords_data.len(),
            nrows
        )
        .into());
    }

    let rhs_flat = load_rhs_values(&rhs_path)?;
    if rhs_flat.len() % nrows != 0 {
        return Err(format!(
            "RHS length ({}) must be a multiple of the matrix dimension ({})",
            rhs_flat.len(),
            nrows
        )
        .into());
    }
    let rhs_cols = rhs_flat.len() / nrows;
    let mut rhs_data = Vec::with_capacity(nrows);
    for row in 0..nrows {
        let mut row_vals = Vec::with_capacity(rhs_cols);
        for col in 0..rhs_cols {
            row_vals.push(rhs_flat[col * nrows + row]);
        }
        rhs_data.push(row_vals);
    }

    let (matrix, selection, mesh_to_solution) = if delete_boundary {
        remove_boundary_from_triplets(nrows, &boundary_indices, &triplets)?
    } else {
        let matrix = build_sparse_row_mat(nrows, ncols, &triplets)?;
        let selection = (0..nrows).collect();
        let mesh_to_solution = (0..nrows).map(Some).collect();
        (matrix, selection, mesh_to_solution)
    };

    let rhs = build_mat_from_rows(&rhs_data, &selection)?;
    let coords = build_mat_from_rows(&coords_data, &selection)?;

    let mesh_geometry = find_associated_vtk(&base_path)?
        .map(|path| load_triangle_mesh_data(&path).map(|mesh| mesh.to_mesh_geometry()))
        .transpose()?;

    Ok(MfemLinearSystem {
        matrix,
        rhs,
        coords,
        boundary_indices,
        mesh_geometry,
        index_mapping: MfemIndexMapping {
            mesh_to_solution,
            solution_to_mesh: selection.clone(),
        },
        original_dimension: nrows,
    })
}

fn ensure_named_file_exists(
    base_path: &Path,
    extension: &str,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let candidate = base_path.with_extension(extension);
    if candidate.is_file() {
        Ok(candidate)
    } else {
        Err(format!("Expected file {}", candidate.display()).into())
    }
}

fn load_boundary_indices(path: &Path) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let mut lines = BufReader::new(file).lines();

    let expected_count: usize = lines
        .next()
        .ok_or_else(|| format!("Boundary file {} is empty", path.display()))??
        .trim()
        .parse()?;

    let mut indices = Vec::with_capacity(expected_count);
    for line in lines {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        indices.push(trimmed.parse()?);
    }

    if indices.len() != expected_count {
        return Err(format!(
            "Boundary file {} expected {} entries but found {}",
            path.display(),
            expected_count,
            indices.len()
        )
        .into());
    }

    Ok(indices)
}

fn load_dense_rows(path: &Path) -> Result<Vec<Vec<f64>>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut data = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let values: Vec<f64> = line
            .split_whitespace()
            .map(str::parse)
            .collect::<Result<_, _>>()?;
        if values.is_empty() {
            continue;
        }
        data.push(values);
    }

    Ok(data)
}

fn load_rhs_values(path: &Path) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut values = Vec::new();

    for line in reader.lines() {
        let line = line?;
        for item in line.split_whitespace() {
            values.push(item.parse()?);
        }
    }

    Ok(values)
}

fn build_sparse_row_mat(
    nrows: usize,
    ncols: usize,
    triplets: &[(usize, usize, f64)],
) -> Result<SparseRowMat<usize, f64>, Box<dyn std::error::Error>> {
    let triplets: Vec<Triplet<usize, usize, f64>> = triplets
        .iter()
        .map(|&(row, col, val)| Triplet::new(row, col, val))
        .collect();
    Ok(SparseRowMat::try_new_from_triplets(
        nrows, ncols, &triplets,
    )?)
}

fn remove_boundary_from_triplets(
    nrows: usize,
    boundary_indices: &[usize],
    triplets: &[(usize, usize, f64)],
) -> Result<(SparseRowMat<usize, f64>, Vec<usize>, Vec<Option<usize>>), Box<dyn std::error::Error>>
{
    let mut is_boundary = vec![false; nrows];
    for &idx in boundary_indices {
        if idx >= nrows {
            return Err(format!(
                "Boundary index {} out of range for matrix of size {}",
                idx, nrows
            )
            .into());
        }
        is_boundary[idx] = true;
    }

    let selection: Vec<usize> = (0..nrows).filter(|&i| !is_boundary[i]).collect();
    let mut old_to_new = vec![None; nrows];
    for (new_idx, &old_idx) in selection.iter().enumerate() {
        old_to_new[old_idx] = Some(new_idx);
    }

    let mut filtered = Vec::with_capacity(triplets.len());
    for &(row, col, val) in triplets.iter() {
        if let (Some(new_row), Some(new_col)) = (old_to_new[row], old_to_new[col]) {
            filtered.push(Triplet::new(new_row, new_col, val));
        }
    }

    let dim = selection.len();
    let matrix = SparseRowMat::try_new_from_triplets(dim, dim, &filtered)?;
    Ok((matrix, selection, old_to_new))
}

fn build_mat_from_rows(
    data: &[Vec<f64>],
    selection: &[usize],
) -> Result<Mat<f64>, Box<dyn std::error::Error>> {
    if selection.is_empty() {
        return Ok(Mat::zeros(0, 0));
    }

    let ncols = data
        .get(selection[0])
        .map(|row| row.len())
        .ok_or_else(|| "Selection references missing row".to_string())?;

    for &row_idx in selection {
        let row = data
            .get(row_idx)
            .ok_or_else(|| format!("Row {} missing in dense data", row_idx))?;
        if row.len() != ncols {
            return Err("Inconsistent column counts in dense data".into());
        }
    }

    let nrows = selection.len();
    Ok(Mat::from_fn(nrows, ncols, |i, j| data[selection[i]][j]))
}

fn load_matrix_triplets(
    path: &Path,
) -> Result<(usize, usize, Vec<(usize, usize, f64)>), Box<dyn std::error::Error>> {
    let mtx_data = matrix_market_rs::MtxData::<f64, 2>::from_file(path)?;
    let (nrows, ncols, coords, vals, symmetry) = match mtx_data {
        matrix_market_rs::MtxData::Sparse([nrows, ncols], coords, vals, symmetry) => {
            (nrows, ncols, coords, vals, symmetry)
        }
        _ => return Err("Only sparse Matrix Market matrices are supported".into()),
    };

    let mut triplets = Vec::with_capacity(coords.len());
    let symmetric = matches!(symmetry, matrix_market_rs::SymInfo::Symmetric);

    for (idx, [row, col]) in coords.iter().copied().enumerate() {
        let value = vals[idx];
        if value == 0.0 {
            continue;
        }
        triplets.push((row, col, value));
        if symmetric && row != col {
            triplets.push((col, row, value));
        }
    }

    Ok((nrows, ncols, triplets))
}

fn find_associated_vtk(base_path: &Path) -> Result<Option<PathBuf>, Box<dyn std::error::Error>> {
    let Some(dataset_name) = base_path.file_name() else {
        return Ok(None);
    };

    let mut current_dir = base_path.parent();
    while let Some(dir) = current_dir {
        let candidate = dir.join(dataset_name).with_extension("vtk");
        if candidate.is_file() {
            return Ok(Some(candidate));
        }
        current_dir = dir.parent();
    }

    Ok(None)
}
