use std::sync::Arc;

use faer::{
    dyn_stack::{MemStack, StackReq},
    matrix_free::{BiLinOp, BiPrecond, LinOp, Precond},
    reborrow::*,
    Mat, MatMut, MatRef, Par,
};
use faer_traits::RealField;

#[derive(Debug, Clone)]
pub struct MultiGrid<T: RealField> {
    operators: Vec<Arc<dyn LinOp<T> + Send>>,
    smoothers: Vec<Arc<dyn BiPrecond<T> + Send>>,
    interpolations: Vec<Arc<dyn LinOp<T> + Send>>,
    restrictions: Vec<Arc<dyn LinOp<T> + Send>>,
    cycle_type: usize,
}

const DEBUG: bool = false;

impl<T: RealField> MultiGrid<T> {
    pub fn new(
        finest_op: Arc<dyn LinOp<T> + Send>,
        smoother: Arc<dyn BiPrecond<T> + Send>,
    ) -> Self {
        Self {
            operators: vec![finest_op],
            smoothers: vec![smoother],
            interpolations: Vec::new(),
            restrictions: Vec::new(),
            cycle_type: 1,
        }
    }

    pub fn with_cycle_type(mut self, mu: usize) -> Self {
        self.cycle_type = mu;
        self
    }

    pub fn add_level(
        &mut self,
        op: Arc<dyn LinOp<T> + Send>,
        smoother: Arc<dyn BiPrecond<T> + Send>,
        r: Arc<dyn LinOp<T> + Send>,
        p: Arc<dyn LinOp<T> + Send>,
    ) {
        self.operators.push(op);
        self.smoothers.push(smoother);
        self.interpolations.push(p);
        self.restrictions.push(r);
    }

    fn init_cycle(
        &self,
        mut out: MatMut<'_, T>,
        rhs: MatRef<'_, T>,
        par: Par,
        stack: &mut MemStack,
    ) {
        let mut v = Mat::zeros(out.nrows(), out.ncols());
        self.cycle(v.as_mut(), rhs, 0, par, stack);
        out += v;
    }

    fn cycle(
        &self,
        mut v: MatMut<'_, T>,
        f: MatRef<'_, T>,
        level: usize,
        par: Par,
        stack: &mut MemStack,
    ) {
        let smoother = &self.smoothers[level];
        if level == self.operators.len() - 1 {
            smoother.apply(v, f, par, stack);
            return;
        }
        let mut work = Mat::zeros(v.nrows(), v.ncols());
        let op = &self.operators[level];

        forward_smooth(v.rb_mut(), f, op.clone(), smoother.clone(), par, stack, 1);
        if DEBUG {
            op.apply(work.rb_mut(), v.as_ref(), par, stack);
            work = f - work.rb();
            let norm = work.norm_l2();
            for _ in 0..level + 1 {
                print!("\t");
            }
            println!("{:?}", norm);
        }

        if level < self.operators.len() - 1 {
            let restrict = &self.restrictions[level];
            let interp = &self.interpolations[level];
            let mut v_coarse = Mat::zeros(restrict.nrows(), v.ncols());
            let mut f_coarse = Mat::zeros(restrict.nrows(), v.ncols());

            // compute fine residual and restrict to coarse residual
            op.apply(work.rb_mut(), v.as_ref(), par, stack);
            work = f - work.rb();
            restrict.apply(f_coarse.as_mut(), work.as_ref(), par, stack);

            for _ in 0..self.cycle_type {
                self.cycle(v_coarse.as_mut(), f_coarse.as_ref(), level + 1, par, stack);
            }

            interp.apply(work.rb_mut(), v_coarse.as_ref(), par, stack);
            v += &work;
            backward_smooth(v.rb_mut(), f, op.clone(), smoother.clone(), par, stack, 1);
            if DEBUG {
                op.apply(work.rb_mut(), v.as_ref(), par, stack);
                work = f - work.rb();
                let norm = work.norm_l2();
                for _ in 0..level + 1 {
                    print!("\t");
                }
                println!("{:?}", norm);
            }
        }
    }
}

fn forward_smooth<T: RealField>(
    x: MatMut<T>,
    b: MatRef<T>,
    op: Arc<dyn LinOp<T>>,
    pc: Arc<dyn BiPrecond<T>>,
    par: Par,
    stack: &mut MemStack,
    max_iter: usize,
) {
    let mut work = Mat::zeros(x.nrows(), x.ncols());
    let mut x = x;
    // first iteration of forward `x` is 0 so residual is `b`
    pc.apply(x.rb_mut(), b, par, stack);
    for _ in 1..max_iter {
        op.apply(work.rb_mut(), x.rb(), par, stack);
        let mut r = b - &work;
        pc.apply_in_place(r.rb_mut(), par, stack);
        x += r;
    }
}

fn backward_smooth<T: RealField>(
    x: MatMut<T>,
    b: MatRef<T>,
    op: Arc<dyn LinOp<T>>,
    pc: Arc<dyn BiPrecond<T>>,
    par: Par,
    stack: &mut MemStack,
    max_iter: usize,
) {
    let mut work = Mat::zeros(x.nrows(), x.ncols());
    let mut x = x;
    for _ in 0..max_iter {
        op.apply(work.rb_mut(), x.rb(), par, stack);
        let mut r = b - &work;
        pc.transpose_apply_in_place(r.rb_mut(), par, stack);
        x += r;
    }
}

impl<T: RealField> LinOp<T> for MultiGrid<T> {
    // TODO: low level API
    fn apply_scratch(&self, rhs_ncols: usize, par: Par) -> StackReq {
        let _rhs_ncols = rhs_ncols;
        let _par = par;
        StackReq::EMPTY
    }

    fn nrows(&self) -> usize {
        if self.operators.is_empty() {
            unreachable!("Cannot determine dimension of partially uninitialized MultiGrid.");
        }
        self.operators[0].nrows()
    }

    fn ncols(&self) -> usize {
        if self.operators.is_empty() {
            unreachable!("Cannot determine dimension of partially uninitialized MultiGrid.");
        }
        self.operators[0].ncols()
    }

    fn apply(&self, out: MatMut<'_, T>, rhs: MatRef<'_, T>, par: Par, stack: &mut MemStack) {
        let mut out = out;
        out.fill(T::zero());
        self.init_cycle(out, rhs, par, stack);
    }

    fn conj_apply(&self, out: MatMut<'_, T>, rhs: MatRef<'_, T>, par: Par, stack: &mut MemStack) {
        // TODO: only real multigrid for now
        self.apply(out, rhs, par, stack);
    }
}

impl<T: RealField> BiLinOp<T> for MultiGrid<T> {
    fn transpose_apply_scratch(&self, rhs_ncols: usize, par: Par) -> StackReq {
        // TODO : only symmetric multigrid
        self.apply_scratch(rhs_ncols, par)
    }

    fn transpose_apply(
        &self,
        out: MatMut<'_, T>,
        rhs: MatRef<'_, T>,
        par: Par,
        stack: &mut MemStack,
    ) {
        // TODO : only symmetric multigrid
        self.apply(out, rhs, par, stack);
    }

    fn adjoint_apply(
        &self,
        out: MatMut<'_, T>,
        rhs: MatRef<'_, T>,
        par: Par,
        stack: &mut MemStack,
    ) {
        // TODO : only symmetric multigrid
        self.conj_apply(out, rhs, par, stack);
    }
}

// TODO: auto impl are fine for now but custom would be better
impl<T: RealField> Precond<T> for MultiGrid<T> {}
impl<T: RealField> BiPrecond<T> for MultiGrid<T> {}
