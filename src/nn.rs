use std::iter;

use ggez::GameResult;

use nalgebra::{DMatrix, DVector};

use serde::{Deserialize, Serialize};

use rand::prelude::*;
use rand_distr::StandardNormal;

use crate::data::{Entity, GameData};

pub fn softmax(x: DVector<f32>) -> DVector<f32> {
    let max = x.max();
    let e = x.map(|e| e - max).map(|f| f.exp());
    &e * e.sum().recip()
}

pub fn sigmoid(n: f32) -> f32 {
    (1.0 + n.exp()).recip()
}

pub fn sigmoid_der(n: f32) -> f32 {
    let sig = sigmoid(n);
    sig * (1.0 - sig)
}

pub fn cost(result: DMatrix<f32>, desired: DMatrix<f32>) -> f32 {
    let diff = result - desired;
    let prod = diff.component_mul(&diff);
    prod.sum()
}

pub fn nabla_w_l(act: &DVector<f32>, delta: &DVector<f32>) -> DMatrix<f32> {
    let mut output: DMatrix<f32> = DMatrix::zeros(act.nrows(), delta.nrows());
    for i in 0..delta.nrows() {
        for j in 0..act.nrows() {
            output[(i, j)] = act[j] * delta[i];
        }
    }
    output
}

#[derive(Debug, Clone, PartialEq)]
pub struct Inputs {
    pub input: DVector<f32>,
}

impl Inputs {
    pub fn new(n: usize) -> Self {
        Self {
            input: DVector::zeros(n),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Outputs {
    pub output: DVector<f32>,
}

impl Outputs {
    pub fn new(n: usize) -> Self {
        Self {
            output: DVector::zeros(n),
        }
    }
}

/// Rnn-ish thing, not scientifically gud
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Network {
    cache: DVector<f32>,
    weights: Vec<DMatrix<f32>>,
    biases: Vec<DVector<f32>>,
}

impl Network {
    pub fn new(layers: &[usize]) -> Network {
        let last = *layers.last().unwrap();
        let mut weights = Vec::with_capacity(layers.len() - 1);
        let mut biases = Vec::with_capacity(layers.len() - 1);
        let iter = iter::once(layers[0] + last)
            .chain(layers[1..layers.len() - 1].iter().copied())
            .zip(layers[1..].iter().copied());
        let mut rng = thread_rng();
        for (input, output) in iter {
            let mut vec = Vec::with_capacity(output * input);
            for _ in 0..output * input {
                vec.push(rng.sample(StandardNormal));
            }
            weights.push(DMatrix::from_vec(output, input, vec));
            let mut vec = Vec::with_capacity(output);
            for _ in 0..output {
                vec.push(rng.sample(StandardNormal));
            }
            biases.push(DVector::from_vec(vec));
        }
        let cache = DVector::zeros(last);
        Network {
            cache,
            weights,
            biases,
        }
    }

    pub fn feedforward(&mut self, layer: DVector<f32>) -> DVector<f32> {
        let layer = self.cache.iter().chain(&layer).copied().collect();
        let mut layer = DVector::from_vec(layer);
        for (w, b) in self.weights.iter().zip(&self.biases) {
            let result = w * layer + b;
            layer = result.map(sigmoid);
        }
        self.cache = layer.clone();
        layer
    }

    pub fn update(&mut self, input: &DVector<f32>, desired: &DVector<f32>, eta: f32) {
        let mut nabla_b = Vec::new();
        let mut nabla_w = Vec::new();
        self.backprop(&mut nabla_b, &mut nabla_w, input, desired);

        let iter = self
            .weights
            .iter_mut()
            .zip(nabla_w)
            .zip(self.biases.iter_mut().zip(nabla_b));
        for ((w0, w1), (b0, b1)) in iter {
            let w = w1 * eta;
            let b = b1 * eta;
            *w0 -= w;
            *b0 -= b;
        }
    }

    pub fn backprop(
        &mut self,
        nabla_b: &mut Vec<DVector<f32>>,
        nabla_w: &mut Vec<DMatrix<f32>>,
        input: &DVector<f32>,
        desired: &DVector<f32>,
    ) {
        let mut activations = Vec::with_capacity(self.weights.len() + 1);
        activations[0] = input.clone();

        let mut activation = 0;

        let mut zs = Vec::with_capacity(self.weights.len());

        for (w, b) in self.weights.iter().zip(&self.biases) {
            let z = w * &activations[activation] + b;
            activations.push(z.map(sigmoid));
            activation += 1;
            zs.push(z);
        }

        let tmp1 = &activations[activation] - desired;
        let tmp2 = zs.last().unwrap().map(sigmoid_der);
        let delta = tmp1.component_mul(&tmp2);
        nabla_w.push(nabla_w_l(activations.last().unwrap(), &delta));
        nabla_b.push(delta);
        let len = self.weights.len();
        for l in 2..len + 1 {
            let z = &zs[len - l];
            let der = z.map(sigmoid_der);
            let tmp = self.weights[len - l + 1].transpose();
            let a = tmp * &nabla_b[l - 1];
            let delta = a.component_mul(&der);
            nabla_w.push(nabla_w_l(&activations[len - l], &delta));
            nabla_b.push(delta);
        }
        nabla_w.reverse();
        nabla_b.reverse();
    }
}

pub fn nn_system<I>(data: &mut GameData, entities: I) -> GameResult<()>
where
    I: IntoIterator<Item = Entity>,
{
    for e in entities {
        let input = data[e.component::<Inputs>()].input.clone();
        let network = &mut data[e.component::<Network>()];

        let output = network.feedforward(input);

        data[e.component::<Outputs>()].output = output;
    }
    Ok(())
}
