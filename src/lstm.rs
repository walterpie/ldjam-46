use std::collections::HashMap;

use ggez::GameResult;

use nalgebra::{DMatrix, DVector};

use serde::{Deserialize, Serialize};

use rand::prelude::*;
use rand_distr::StandardNormal;

use crate::data::{Entity, GameData};
use crate::nn::{sigmoid, sigmoid_der, softmax};

const EPSILON: f32 = 1e-8;

#[derive(Debug, Clone)]
pub struct Adam {
    pub dwf: DMatrix<f32>,
    pub dbf: DVector<f32>,
    pub dwi: DMatrix<f32>,
    pub dbi: DVector<f32>,
    pub dwc: DMatrix<f32>,
    pub dbc: DVector<f32>,
    pub dwo: DMatrix<f32>,
    pub dbo: DVector<f32>,
}

impl Adam {
    pub fn new(n_a: usize, n_x: usize) -> Self {
        Self {
            dwf: DMatrix::zeros(n_a, n_a + n_x),
            dbf: DVector::zeros(n_a),
            dwi: DMatrix::zeros(n_a, n_a + n_x),
            dbi: DVector::zeros(n_a),
            dwc: DMatrix::zeros(n_a, n_a + n_x),
            dbc: DVector::zeros(n_a),
            dwo: DMatrix::zeros(n_a, n_a + n_x),
            dbo: DVector::zeros(n_a),
        }
    }
}

#[derive(Debug, Clone)]
pub struct LstmCache {
    pub a_next: DVector<f32>,
    pub c_next: DVector<f32>,
    pub a_prev: DVector<f32>,
    pub c_prev: DVector<f32>,
    pub xt: DVector<f32>,
    pub parameters: Parameters,
    pub ft: DVector<f32>,
    pub it: DVector<f32>,
    pub cct: DVector<f32>,
    pub ot: DVector<f32>,
}

#[derive(Debug, Clone)]
pub struct LstmCell {
    pub a_next: DVector<f32>,
    pub c_next: DVector<f32>,
    pub yt_pred: DVector<f32>,
    pub cache: LstmCache,
}

#[derive(Debug, Clone)]
pub struct CellGradients {
    pub dxt: DVector<f32>,
    pub da_prev: DVector<f32>,
    pub dc_prev: DVector<f32>,
    pub dwf: DMatrix<f32>,
    pub dwi: DMatrix<f32>,
    pub dwc: DMatrix<f32>,
    pub dwo: DMatrix<f32>,
    pub dbf: DVector<f32>,
    pub dbi: DVector<f32>,
    pub dbc: DVector<f32>,
    pub dbo: DVector<f32>,
}

#[derive(Debug, Clone)]
pub struct Gradients {
    pub dx: DMatrix<f32>,
    pub da_prev: DVector<f32>,
    pub dc_prev: DVector<f32>,
    pub dwf: DMatrix<f32>,
    pub dwi: DMatrix<f32>,
    pub dwc: DMatrix<f32>,
    pub dwo: DMatrix<f32>,
    pub dbf: DVector<f32>,
    pub dbi: DVector<f32>,
    pub dbc: DVector<f32>,
    pub dbo: DVector<f32>,
}

#[derive(Debug, Clone)]
pub struct Parameters {
    pub wf: DMatrix<f32>,
    pub bf: DVector<f32>,
    pub wi: DMatrix<f32>,
    pub bi: DVector<f32>,
    pub wc: DMatrix<f32>,
    pub bc: DVector<f32>,
    pub wo: DMatrix<f32>,
    pub bo: DVector<f32>,
    pub wy: DMatrix<f32>,
    pub by: DVector<f32>,
}

impl Parameters {
    pub fn new(n_a: usize, n_x: usize) -> Self {
        let cols = n_a;
        let rows = n_a + n_x;

        let mut rng = thread_rng();

        let mut wf = vec![0.0; cols * rows];
        for i in 0..cols * rows {
            wf[i] = rng.sample(StandardNormal);
        }
        let wf = DMatrix::from_vec(cols, rows, wf);
        let mut bf = vec![0.0; cols];
        for i in 0..cols {
            bf[i] = rng.sample(StandardNormal);
        }
        let bf = DVector::from_vec(bf);
        let mut wi = vec![0.0; cols * rows];
        for i in 0..cols * rows {
            wi[i] = rng.sample(StandardNormal);
        }
        let wi = DMatrix::from_vec(cols, rows, wi);
        let mut bi = vec![0.0; cols];
        for i in 0..cols {
            bi[i] = rng.sample(StandardNormal);
        }
        let bi = DVector::from_vec(bi);
        let mut wc = vec![0.0; cols * rows];
        for i in 0..cols * rows {
            wc[i] = rng.sample(StandardNormal);
        }
        let wc = DMatrix::from_vec(cols, rows, wc);
        let mut bc = vec![0.0; cols];
        for i in 0..cols {
            bc[i] = rng.sample(StandardNormal);
        }
        let bc = DVector::from_vec(bc);
        let mut wo = vec![0.0; cols * rows];
        for i in 0..cols * rows {
            wo[i] = rng.sample(StandardNormal);
        }
        let wo = DMatrix::from_vec(cols, rows, wo);
        let mut bo = vec![0.0; cols];
        for i in 0..cols {
            bo[i] = rng.sample(StandardNormal);
        }
        let bo = DVector::from_vec(bo);
        let mut wy = vec![0.0; cols * rows];
        for i in 0..cols * rows {
            wy[i] = rng.sample(StandardNormal);
        }
        let wy = DMatrix::from_vec(cols, rows, wy);
        let mut by = vec![0.0; cols];
        for i in 0..cols {
            by[i] = rng.sample(StandardNormal);
        }
        let by = DVector::from_vec(by);

        Self {
            wf,
            bf,
            wi,
            bi,
            wc,
            bc,
            wo,
            bo,
            wy,
            by,
        }
    }

    pub fn len() -> usize {
        5
    }
}

pub struct Network {
    da: DMatrix<f32>,
    dc: DMatrix<f32>,
    x: Vec<DVector<f32>>,
    caches: Vec<LstmCache>,
    shape: (usize, usize, usize),
    a0: DVector<f32>,
    parameters: Parameters,
}

impl Network {
    pub fn new((n_a, n_x, m): (usize, usize, usize), a0: DVector<f32>) -> Self {
        let mut rng = thread_rng();

        let mut da = vec![0.0; n_a * m];
        for i in 0..n_a * m {
            da[i] = rng.sample(StandardNormal);
        }
        let da = DMatrix::from_vec(n_a, m, da);
        let mut dc = vec![0.0; n_a * m];
        for i in 0..n_a * m {
            dc[i] = rng.sample(StandardNormal);
        }
        let dc = DMatrix::from_vec(n_a, m, dc);
        let x = Vec::new();
        let caches = Vec::new();
        let shape = (n_a, n_x, m);
        let parameters = Parameters::new(n_a, n_x);

        Self {
            da,
            dc,
            x,
            caches,
            shape,
            a0,
            parameters,
        }
    }

    pub fn feedforward(&mut self, input: DVector<f32>) -> DVector<f32> {
        self.x.push(input);
        let (y, caches) = forward(
            (self.shape.0, self.shape.2),
            &mut self.x,
            &self.a0,
            &self.parameters,
        );
        self.caches = caches;
        let mut y_pred = DVector::zeros(y.nrows());
        let j = y.ncols() - 1;
        for i in 0..y.nrows() {
            y_pred[i] = y[(i, j)];
        }
        y_pred
    }

    pub fn update(
        &mut self,
        v: &mut Adam,
        s: &mut Adam,
        t: f32,
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
    ) {
        let gradients = backward(&self.da, &self.dc, &self.x, &self.caches);
        let mut v_corrected = Adam::new(self.shape.0, self.shape.1);
        let mut s_corrected = Adam::new(self.shape.0, self.shape.1);

        v.dwf = &v.dwf * beta1 + &gradients.dwf * (1.0_f32 - beta1);
        v.dbf = &v.dbf * beta1 + &gradients.dbf * (1.0_f32 - beta1);

        v_corrected.dwf = &v.dwf * (1.0_f32 - beta1.powf(t)).recip();
        v_corrected.dbf = &v.dbf * (1.0_f32 - beta1.powf(t)).recip();

        s.dwf = &s.dwf * beta2 + gradients.dwf.map(|x| x * x) * (1.0_f32 - beta2);
        s.dbf = &s.dbf * beta2 + gradients.dbf.map(|x| x * x) * (1.0_f32 - beta2);

        s_corrected.dwf = &s.dwf * (1.0_f32 - beta2.powf(t)).recip();
        s_corrected.dbf = &s.dbf * (1.0_f32 - beta2.powf(t)).recip();

        self.parameters.wf -= (v_corrected.dwf * learning_rate)
            .component_div(&s_corrected.dwf.map(|x| x.sqrt() + EPSILON));
        self.parameters.bf -= (v_corrected.dbf * learning_rate)
            .component_div(&s_corrected.dbf.map(|x| x.sqrt() + EPSILON));

        v.dwc = &v.dwc * beta1 + &gradients.dwc * (1.0_f32 - beta1);
        v.dbc = &v.dbc * beta1 + &gradients.dbc * (1.0_f32 - beta1);

        v_corrected.dwc = &v.dwc * (1.0_f32 - beta1.powf(t)).recip();
        v_corrected.dbc = &v.dbc * (1.0_f32 - beta1.powf(t)).recip();

        s.dwc = &s.dwc * beta2 + gradients.dwc.map(|x| x * x) * (1.0_f32 - beta2);
        s.dbc = &s.dbc * beta2 + gradients.dbc.map(|x| x * x) * (1.0_f32 - beta2);

        s_corrected.dwc = &s.dwc * (1.0_f32 - beta2.powf(t)).recip();
        s_corrected.dbc = &s.dbc * (1.0_f32 - beta2.powf(t)).recip();

        self.parameters.wc -= (v_corrected.dwc * learning_rate)
            .component_div(&s_corrected.dwc.map(|x| x.sqrt() + EPSILON));
        self.parameters.bc -= (v_corrected.dbc * learning_rate)
            .component_div(&s_corrected.dbc.map(|x| x.sqrt() + EPSILON));

        v.dwi = &v.dwi * beta1 + &gradients.dwi * (1.0_f32 - beta1);
        v.dbi = &v.dbi * beta1 + &gradients.dbi * (1.0_f32 - beta1);

        v_corrected.dwi = &v.dwi * (1.0_f32 - beta1.powf(t)).recip();
        v_corrected.dbi = &v.dbi * (1.0_f32 - beta1.powf(t)).recip();

        s.dwi = &s.dwi * beta2 + gradients.dwi.map(|x| x * x) * (1.0_f32 - beta2);
        s.dbi = &s.dbi * beta2 + gradients.dbi.map(|x| x * x) * (1.0_f32 - beta2);

        s_corrected.dwi = &s.dwi * (1.0_f32 - beta2.powf(t)).recip();
        s_corrected.dbi = &s.dbi * (1.0_f32 - beta2.powf(t)).recip();

        self.parameters.wi -= (v_corrected.dwi * learning_rate)
            .component_div(&s_corrected.dwi.map(|x| x.sqrt() + EPSILON));
        self.parameters.bi -= (v_corrected.dbi * learning_rate)
            .component_div(&s_corrected.dbi.map(|x| x.sqrt() + EPSILON));

        v.dwo = &v.dwo * beta1 + &gradients.dwo * (1.0_f32 - beta1);
        v.dbo = &v.dbo * beta1 + &gradients.dbo * (1.0_f32 - beta1);

        v_corrected.dwo = &v.dwo * (1.0_f32 - beta1.powf(t)).recip();
        v_corrected.dbo = &v.dbo * (1.0_f32 - beta1.powf(t)).recip();

        s.dwo = &s.dwo * beta2 + gradients.dwo.map(|x| x * x) * (1.0_f32 - beta2);
        s.dbo = &s.dbo * beta2 + gradients.dbo.map(|x| x * x) * (1.0_f32 - beta2);

        s_corrected.dwo = &s.dwo * (1.0_f32 - beta2.powf(t)).recip();
        s_corrected.dbo = &s.dbo * (1.0_f32 - beta2.powf(t)).recip();

        self.parameters.wo -= (v_corrected.dwo * learning_rate)
            .component_div(&s_corrected.dwc.map(|x| x.sqrt() + EPSILON));
        self.parameters.bo -= (v_corrected.dbo * learning_rate)
            .component_div(&s_corrected.dbc.map(|x| x.sqrt() + EPSILON));
    }
}

fn cell_forward(
    xt: &DVector<f32>,
    a_prev: &DVector<f32>,
    c_prev: &DVector<f32>,
    parameters: &Parameters,
) -> LstmCell {
    let (n_x, _) = xt.shape();
    let (_, n_a) = parameters.wy.shape();

    let mut concat = DVector::zeros(n_a + n_x);
    for i in 0..n_a {
        concat[i] = a_prev[i]
    }
    for i in n_a..n_a + n_x {
        concat[i] = xt[i - n_a]
    }

    let ft: DVector<f32> = (&parameters.wf * &concat + &parameters.bf).map(|f| sigmoid(f));
    let it: DVector<f32> = (&parameters.wi * &concat + &parameters.bi).map(|f| sigmoid(f));
    let cct: DVector<f32> = (&parameters.wc * &concat + &parameters.bc).map(|f| f.tanh());
    let c_next: DVector<f32> = (&ft * c_prev) + (&it * &cct);
    let ot: DVector<f32> = (&parameters.wo * &concat + &parameters.bo).map(|f| sigmoid(f));
    let a_next: DVector<f32> = &ot * c_next.map(|f| f.tanh());

    let yt_pred: DVector<f32> = softmax(&parameters.wy * &a_next + &parameters.by);

    let cache = LstmCache {
        a_next: a_next.clone(),
        c_next: c_next.clone(),
        a_prev: a_prev.clone(),
        c_prev: c_prev.clone(),
        ft,
        it,
        cct,
        ot,
        xt: xt.clone(),
        parameters: parameters.clone(),
    };
    LstmCell {
        a_next: a_next.clone(),
        c_next: c_next.clone(),
        yt_pred,
        cache,
    }
}

fn cell_backward(
    da_next: &DVector<f32>,
    dc_next: &DVector<f32>,
    cache: &LstmCache,
) -> CellGradients {
    let n_a = cache.a_next.nrows();

    let dot = da_next * cache.c_next.map(|f| f.tanh()) * &cache.ot * (cache.ot.map(|f| 1.0 - f));
    let dcct = (dc_next * &cache.it
        + &cache.ot * cache.c_next.map(|f| 1.0 - f.tanh() * f.tanh()) * &cache.it * da_next)
        * cache.cct.map(|f| 1.0 - f * f);
    let dit = (dc_next * &cache.cct
        + &cache.ot * cache.c_next.map(|f| 1.0 - f.tanh() * f.tanh()) * &cache.cct * da_next)
        * &cache.it
        * cache.it.map(|f| 1.0 - f);
    let dft = (dc_next * &cache.c_prev
        + &cache.ot * cache.c_next.map(|f| 1.0 - f.tanh() * f.tanh()) * &cache.c_prev * da_next)
        * &cache.ft
        * cache.ft.map(|f| 1.0 - f);

    let concat = cache
        .a_prev
        .iter()
        .copied()
        .chain(cache.xt.iter().copied())
        .collect();
    let concat = DVector::from_vec(concat);
    let concat_t = concat.transpose();

    let dwf = &dft * &concat_t;
    let dwi = &dit * &concat_t;
    let dwc = &dcct * &concat_t;
    let dwo = &dot * &concat_t;
    let dbf = (0..dft.nrows())
        .map(|_| (0..dft.ncols()).map(|j| dft[j]).sum())
        .collect::<Vec<_>>();
    let dbf = DVector::from_vec(dbf);
    let dbi = (0..dit.nrows())
        .map(|_| (0..dit.ncols()).map(|j| dft[j]).sum())
        .collect::<Vec<_>>();
    let dbi = DVector::from_vec(dbi);
    let dbc = (0..dcct.nrows())
        .map(|_| (0..dcct.ncols()).map(|j| dft[j]).sum())
        .collect::<Vec<_>>();
    let dbc = DVector::from_vec(dbc);
    let dbo = (0..dot.nrows())
        .map(|_| (0..dot.ncols()).map(|j| dft[j]).sum())
        .collect::<Vec<_>>();
    let dbo = DVector::from_vec(dbo);

    let mut tmp_wft = Vec::new();
    for i in 0..cache.parameters.wf.nrows() {
        for j in 0..n_a {
            tmp_wft.push(cache.parameters.wf[(j, i)]);
        }
    }
    let tmp_wft = DMatrix::from_vec(n_a, cache.parameters.wf.nrows(), tmp_wft);

    let mut tmp_wit = Vec::new();
    for i in 0..cache.parameters.wi.nrows() {
        for j in 0..n_a {
            tmp_wit.push(cache.parameters.wi[(j, i)]);
        }
    }
    let tmp_wit = DMatrix::from_vec(n_a, cache.parameters.wi.nrows(), tmp_wit);

    let mut tmp_wct = Vec::new();
    for i in 0..cache.parameters.wc.nrows() {
        for j in 0..n_a {
            tmp_wct.push(cache.parameters.wc[(j, i)]);
        }
    }
    let tmp_wct = DMatrix::from_vec(n_a, cache.parameters.wc.nrows(), tmp_wct);

    let mut tmp_wot = Vec::new();
    for i in 0..cache.parameters.wo.nrows() {
        for j in 0..n_a {
            tmp_wot.push(cache.parameters.wo[(j, i)]);
        }
    }
    let tmp_wot = DMatrix::from_vec(n_a, cache.parameters.wo.nrows(), tmp_wot);

    let mut tmp_wyt = Vec::new();
    for i in 0..cache.parameters.wo.nrows() {
        for j in 0..n_a {
            tmp_wyt.push(cache.parameters.wy[(j, i)]);
        }
    }
    let tmp_wyt = DMatrix::from_vec(n_a, cache.parameters.wy.nrows(), tmp_wyt);

    let da_prev = &tmp_wft * &dft + &tmp_wit * &dit + &tmp_wct * &dcct + &tmp_wot * &dot;
    let dc_prev = dc_next * &cache.ft
        + &cache.ot * cache.c_next.map(|f| 1.0 - f.tanh() * f.tanh()) * &cache.ft * da_next;

    let mut tmp_wft = Vec::new();
    for i in 0..cache.parameters.wf.nrows() {
        for j in n_a..cache.parameters.wf.ncols() {
            tmp_wft.push(cache.parameters.wf[(j, i)]);
        }
    }
    let tmp_wft = DMatrix::from_vec(n_a, cache.parameters.wf.nrows(), tmp_wft);

    let mut tmp_wit = Vec::new();
    for i in 0..cache.parameters.wi.nrows() {
        for j in n_a..cache.parameters.wi.ncols() {
            tmp_wit.push(cache.parameters.wc[(j, i)]);
        }
    }
    let tmp_wit = DMatrix::from_vec(n_a, cache.parameters.wi.nrows(), tmp_wit);

    let mut tmp_wct = Vec::new();
    for i in 0..cache.parameters.wc.nrows() {
        for j in n_a..cache.parameters.wc.ncols() {
            tmp_wct.push(cache.parameters.wc[(j, i)]);
        }
    }
    let tmp_wct = DMatrix::from_vec(n_a, cache.parameters.wc.nrows(), tmp_wct);

    let mut tmp_wot = Vec::new();
    for i in 0..cache.parameters.wo.nrows() {
        for j in n_a..cache.parameters.wo.ncols() {
            tmp_wot.push(cache.parameters.wo[(j, i)]);
        }
    }
    let tmp_wot = DMatrix::from_vec(n_a, cache.parameters.wo.nrows(), tmp_wot);

    let mut tmp_wyt = Vec::new();
    for i in 0..cache.parameters.wy.nrows() {
        for j in n_a..cache.parameters.wy.ncols() {
            tmp_wyt.push(cache.parameters.wy[(j, i)]);
        }
    }
    let tmp_wyt = DMatrix::from_vec(n_a, cache.parameters.wy.nrows(), tmp_wyt);

    let dxt = &tmp_wft * &dft + &tmp_wit * &dit + &tmp_wct * &dcct + &tmp_wot * &dot;

    CellGradients {
        dxt,
        da_prev,
        dc_prev,
        dwf,
        dwi,
        dwc,
        dwo,
        dbf,
        dbi,
        dbc,
        dbo,
    }
}

fn forward(
    (_, m): (usize, usize),
    x: &mut Vec<DVector<f32>>,
    a0: &DVector<f32>,
    parameters: &Parameters,
) -> (DMatrix<f32>, Vec<LstmCache>) {
    let mut caches = Vec::new();

    let (n_y, n_a) = parameters.wy.shape();

    let mut a = DMatrix::zeros(n_a, m);
    let mut c = DMatrix::zeros(n_a, m);
    let mut y = DMatrix::zeros(n_y, m);

    let mut a_next = a0.clone();
    let mut c_next = DVector::zeros(a_next.nrows());

    for t in 0..n_a {
        let cell = cell_forward(&x[t], &a_next, &c_next, parameters);
        for i in 0..cell.a_next.nrows() {
            a[(i, t)] = cell.a_next[i];
        }
        for i in 0..cell.yt_pred.nrows() {
            y[(i, t)] = cell.yt_pred[i];
        }
        for i in 0..cell.c_next.nrows() {
            c[(i, t)] = cell.c_next[i];
        }
        a_next = cell.a_next;
        c_next = cell.c_next;
        caches.push(cell.cache);
    }

    (y, caches)
}

fn backward(
    da: &DMatrix<f32>,
    dc: &DMatrix<f32>,
    x: &[DVector<f32>],
    caches: &[LstmCache],
) -> Gradients {
    let (n_a, t_x) = da.shape();
    let cache = &caches[0];
    let (n_x, m) = cache.xt.shape();

    let mut dx = DMatrix::zeros(n_x, m);
    let da_prev = DVector::zeros(n_a);
    let dc_prev = DVector::zeros(n_a);
    let mut dwf = DMatrix::zeros(n_a, n_a + n_x);
    let mut dwi = DMatrix::zeros(n_a, n_a + n_x);
    let mut dwc = DMatrix::zeros(n_a, n_a + n_x);
    let mut dwo = DMatrix::zeros(n_a, n_a + n_x);
    let mut dbf = DVector::zeros(n_a);
    let mut dbi = DVector::zeros(n_a);
    let mut dbc = DVector::zeros(n_a);
    let mut dbo = DVector::zeros(n_a);

    let mut gradients = CellGradients {
        dxt: DVector::zeros(n_x),
        da_prev: DVector::zeros(n_a),
        dc_prev: DVector::zeros(n_a),
        dwf: DMatrix::zeros(n_a, n_a + n_x),
        dwi: DMatrix::zeros(n_a, n_a + n_x),
        dwc: DMatrix::zeros(n_a, n_a + n_x),
        dwo: DMatrix::zeros(n_a, n_a + n_x),
        dbf: DVector::zeros(n_a),
        dbi: DVector::zeros(n_a),
        dbc: DVector::zeros(n_a),
        dbo: DVector::zeros(n_a),
    };
    for t in (0..t_x).rev() {
        let mut da_next = DVector::zeros(n_a);
        for i in 0..da.nrows() {
            da_next[i] = da[(i, t)];
        }
        gradients = cell_backward(&da_next, &dc_prev, &caches[t]);
        for i in 0..gradients.dxt.nrows() {
            dx[(i, t)] = gradients.dxt[i];
            dwf += &gradients.dwf;
            dwi += &gradients.dwi;
            dwc += &gradients.dwc;
            dwo += &gradients.dwo;
            dbf += &gradients.dbf;
            dbi += &gradients.dbi;
            dbc += &gradients.dbc;
            dbo += &gradients.dbo;
        }
    }
    let da0 = gradients.da_prev;

    Gradients {
        dx,
        da_prev,
        dc_prev,
        dwf,
        dwi,
        dwc,
        dwo,
        dbf,
        dbi,
        dbc,
        dbo,
    }
}
