use std::f32;

use ggez::timer;
use ggez::{Context, GameResult};

use nalgebra::Vector2;

use crate::creature::*;
use crate::data::{Entity, GameData};

/// Should be stored in an array of structs
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Body {
    pub radius: f32,
    pub mass: f32,
    pub rmass: f32,
    pub restitution: f32,
}

impl Body {
    pub fn new(radius: f32, mass: f32, restitution: f32) -> Self {
        let rmass = if mass == 0.0 { 0.0 } else { mass.recip() };
        Self {
            radius,
            mass,
            rmass,
            restitution,
        }
    }
}

/// Informs the physics engine of collisions
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Manifold {
    a: Entity,
    b: Entity,
    normal: Vector2<f32>,
    penetration: f32,
}

/// Resolves a manifold generated with `gen_manifold`
pub fn resolve(data: &mut GameData, m: &Manifold) {
    let rv =
        data[m.b.component::<Velocity>()].velocity - data[m.a.component::<Velocity>()].velocity;
    let veln = rv.dot(&m.normal);

    if veln > 0.0 {
        return;
    }

    let a = data[m.a.component::<Body>()];
    let b = data[m.b.component::<Body>()];

    let e = a.restitution.min(b.restitution);

    let mut j = -(1.0 + e) * veln;
    j /= a.rmass + b.rmass;
    let j = j;

    let impulse = m.normal * j;

    data[m.a.component::<Velocity>()].velocity -= impulse * a.rmass;
    data[m.b.component::<Velocity>()].velocity += impulse * b.rmass;
}

/// Corrects position using some pre-set PERCENT and SLOP
pub fn correct(data: &mut GameData, m: &Manifold) {
    const PERCENT: f32 = 0.2;
    const SLOP: f32 = 0.02;

    let a = data[m.a.component::<Body>()];
    let b = data[m.b.component::<Body>()];

    let correction = m.normal * (m.penetration - SLOP).max(0.0) / (a.rmass + b.rmass) * PERCENT;
    data[m.a.component::<Position>()].position -= correction * a.rmass;
    data[m.b.component::<Position>()].position += correction * b.rmass;
}

pub fn gen_manifold(data: &mut GameData, a: Entity, b: Entity) -> Option<Manifold> {
    let a_body = data[a.component::<Body>()];
    let b_body = data[b.component::<Body>()];

    let n = data[b.component::<Position>()].position - data[a.component::<Position>()].position;

    let r = b_body.radius + a_body.radius;
    let r2 = r * r;

    let dist2 = n.magnitude_squared();
    if dist2 > r2 {
        return None;
    }

    let dist = dist2.sqrt();

    let penetration;
    let normal;

    if dist > f32::EPSILON {
        penetration = r - dist;
        normal = n / dist;
    } else {
        penetration = a_body.radius;
        normal = Vector2::new(1.0, 0.0);
    }

    Some(Manifold {
        a,
        b,
        normal,
        penetration,
    })
}

pub fn physics_system<I1, I2>(
    ctx: &mut Context,
    data: &mut GameData,
    left: I1,
    right: I2,
) -> GameResult<()>
where
    I1: IntoIterator<Item = Entity> + Clone,
    I2: IntoIterator<Item = Entity> + Clone,
{
    for a in left.clone() {
        for b in right.clone() {
            if let Some(m) = gen_manifold(data, a, b) {
                resolve(data, &m);
                correct(data, &m);
            }
        }
    }
    let delta = timer::delta(ctx);
    let delta = delta.as_secs() as f32 + delta.subsec_nanos() as f32 / 1000000000.0;
    for a in left.clone() {
        let vel = data[a.component::<Velocity>()].velocity;
        let pos = &mut data[a.component::<Position>()].position;
        *pos += vel * delta;
    }
    Ok(())
}
