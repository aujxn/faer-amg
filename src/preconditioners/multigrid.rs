use std::sync::Arc;

use faer::{
    dyn_stack::{MemStack, StackReq},
    matrix_free::{BiPrecond, LinOp, Precond},
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

        smooth(v.rb_mut(), f, op.clone(), smoother.clone(), par, stack, 1);
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
            smooth(v.rb_mut(), f, op.clone(), smoother.clone(), par, stack, 1);
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

fn smooth<T: RealField>(
    x: MatMut<T>,
    b: MatRef<T>,
    op: Arc<dyn LinOp<T>>,
    pc: Arc<dyn BiPrecond<T>>,
    par: Par,
    stack: &mut MemStack,
    max_iter: usize,
) {
    let mut work = Mat::zeros(x.nrows(), x.ncols());
    let mut iter = 0;
    let mut x = x;
    loop {
        op.apply(work.rb_mut(), x.rb(), par, stack);
        let mut r = b - &work;
        iter += 1;
        pc.apply_in_place(r.rb_mut(), par, stack);
        x += r;
        if iter == max_iter {
            break;
        }
    }
}

impl<T: RealField> LinOp<T> for MultiGrid<T> {
    // TODO: remove allocations in apply and learn how to use this stack
    fn apply_scratch(&self, rhs_ncols: usize, par: Par) -> StackReq {
        let _rhs_ncols = rhs_ncols;
        let _par = par;
        StackReq::EMPTY
    }

    fn nrows(&self) -> usize {
        if self.operators.is_empty() {
            panic!("Cannot determine dimension of partially uninitialized MultiGrid.");
        }
        self.operators[0].nrows()
    }

    fn ncols(&self) -> usize {
        if self.operators.is_empty() {
            panic!("Cannot determine dimension of partially uninitialized MultiGrid.");
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

impl<T: RealField> Precond<T> for MultiGrid<T> {
    fn apply_in_place_scratch(&self, rhs_ncols: usize, par: Par) -> StackReq {
        let _rhs_ncols = rhs_ncols;
        let _par = par;
        StackReq::EMPTY
    }
    fn apply_in_place(&self, rhs: MatMut<'_, T>, par: Par, stack: &mut MemStack) {
        let mut f = Mat::zeros(rhs.nrows(), rhs.ncols());
        self.apply(f.rb_mut(), rhs.rb(), par, stack);
        let mut rhs = rhs;
        rhs.copy_from(f);
    }
    fn conj_apply_in_place(&self, rhs: MatMut<'_, T>, par: Par, stack: &mut MemStack) {
        self.apply_in_place(rhs, par, stack);
    }
}
