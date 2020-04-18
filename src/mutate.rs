use ggez::graphics::Color;

use nalgebra::{DMatrix, DVector, Vector2};

use rand::random;

pub trait Mutate {
    fn mutate(&self, other: &Self, factor: f32, chance: f32, mutation: f32) -> Self;
}

impl Mutate for f32 {
    fn mutate(&self, other: &Self, factor: f32, chance: f32, mutation: f32) -> Self {
        let mut result = *self * factor + *other * (1.0 - factor);
        if random::<f32>() < chance {
            result *= random::<f32>() * 2.0 * mutation + (1.0 - mutation);
        }
        result
    }
}

impl Mutate for Color {
    fn mutate(&self, other: &Self, factor: f32, chance: f32, mutation: f32) -> Self {
        let r = self.r.mutate(&other.r, factor, chance, mutation);
        let g = self.g.mutate(&other.g, factor, chance, mutation);
        let b = self.b.mutate(&other.b, factor, chance, mutation);
        let a = self.a.mutate(&other.a, factor, chance, mutation);
        Color::new(r, g, b, a)
    }
}

impl Mutate for Vector2<f32> {
    fn mutate(&self, other: &Self, factor: f32, chance: f32, mutation: f32) -> Self {
        let x = self.x.mutate(&other.x, factor, chance, mutation);
        let y = self.y.mutate(&other.y, factor, chance, mutation);
        Vector2::new(x, y)
    }
}

impl Mutate for DMatrix<f32> {
    fn mutate(&self, other: &Self, factor: f32, chance: f32, mutation: f32) -> Self {
        assert_eq!(self.nrows(), other.nrows());
        assert_eq!(self.ncols(), other.ncols());

        let mut result = DMatrix::zeros(self.nrows(), self.ncols());

        for i in 0..self.nrows() {
            for j in 0..self.ncols() {
                result[(i, j)] = self[(i, j)].mutate(&other[(i, j)], factor, chance, mutation);
            }
        }

        result
    }
}

impl Mutate for DVector<f32> {
    fn mutate(&self, other: &Self, factor: f32, chance: f32, mutation: f32) -> Self {
        assert_eq!(self.nrows(), other.nrows());

        let mut result = DVector::zeros(self.nrows());

        for i in 0..self.nrows() {
            result[i] = self[i].mutate(&other[i], factor, chance, mutation);
        }

        result
    }
}
