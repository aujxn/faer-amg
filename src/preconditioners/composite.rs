use std::sync::Arc;

use faer::{
    dyn_stack::{MemStack, StackReq},
    linalg::temp_mat_scratch,
    matrix_free::{LinOp, Precond},
    prelude::ReborrowMut,
    Mat, MatMut, MatRef, Par,
};

#[derive(Clone, Debug)]
pub struct Composite {
    mat: Arc<dyn LinOp<f64> + Send>,
    components: Vec<Arc<dyn Precond<f64> + Send>>,
}

impl LinOp<f64> for Composite {
    fn nrows(&self) -> usize {
        self.mat.nrows()
    }

    fn ncols(&self) -> usize {
        self.mat.ncols()
    }

    fn apply_scratch(&self, rhs_ncols: usize, par: Par) -> StackReq {
        StackReq::any_of(
            &self
                .components
                .iter()
                .map(|comp| comp.apply_scratch(rhs_ncols, par))
                .collect::<Vec<StackReq>>(),
        )
        .and(temp_mat_scratch::<f64>(self.mat.nrows(), rhs_ncols))
    }

    fn apply(&self, out: MatMut<f64>, rhs: MatRef<f64>, par: Par, stack: &mut MemStack) {
        self.implementation(out, rhs, par, stack);
    }

    fn conj_apply(&self, out: MatMut<f64>, rhs: MatRef<f64>, par: Par, stack: &mut MemStack) {
        self.apply(out, rhs, par, stack);
    }
}

impl Composite {
    pub fn new(
        mat: Arc<dyn LinOp<f64> + Send>,
        first_component: Arc<dyn Precond<f64> + Send>,
    ) -> Self {
        Self {
            mat,
            components: vec![first_component],
        }
    }

    pub fn new_with_components(
        mat: Arc<dyn LinOp<f64> + Send>,
        components: Vec<Arc<dyn Precond<f64> + Send>>,
    ) -> Self {
        Self { mat, components }
    }

    fn implementation(&self, out: MatMut<f64>, rhs: MatRef<f64>, par: Par, stack: &mut MemStack) {
        let mut out = out;
        out.fill(0.0);
        let mut ws = Mat::zeros(out.nrows(), out.ncols());
        ws.copy_from(rhs);
        for component in self.components.iter().rev() {
            component.apply_in_place(ws.rb_mut(), par, stack);
            out += ws.as_ref();
            self.mat.apply(ws.as_mut(), out.as_ref(), par, stack);
            ws = rhs - ws;
        }
        for component in self.components.iter().skip(1) {
            component.apply_in_place(ws.rb_mut(), par, stack);
            out += ws.as_ref();
            self.mat.apply(ws.as_mut(), out.as_ref(), par, stack);
            ws = rhs - ws;
        }
    }

    pub fn push(&mut self, component: Arc<dyn Precond<f64> + Send>) {
        self.components.push(component);
    }

    pub fn components(&self) -> &Vec<Arc<dyn Precond<f64> + Send>> {
        &self.components
    }

    pub fn components_mut(&mut self) -> &mut Vec<Arc<dyn Precond<f64> + Send>> {
        &mut self.components
    }

    pub fn get_mat(&self) -> Arc<dyn LinOp<f64> + Send> {
        self.mat.clone()
    }
}
