use std::f32;

use ggez::timer;
use ggez::{Context, GameResult};

use nalgebra::{DVector, Vector2};

use ordered_float::OrderedFloat;

use crate::creature::*;
use crate::data::{Entity, GameData};
use crate::nn::{Inputs, Outputs};
use crate::SPEED;

pub const VIEW_DISTANCE: f32 = 300.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ray {
    p1: Vector2<f32>,
    p2: Vector2<f32>,
}

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

pub fn raycast<I>(data: &GameData, ray: &Ray, entities: I) -> Option<(Entity, f32)>
where
    I: IntoIterator<Item = Entity>,
{
    let b = ray.p2 - ray.p1;
    let mut result = None;
    let mut min_dist = f32::INFINITY;
    for e in entities {
        let pos = data[e.component::<Position>()].position;
        let radius = data[e.component::<Body>()].radius;
        let a = pos - ray.p1;
        let dot = a.dot(&b);
        let len = b.magnitude();
        let t = dot / len;
        if t < 0.0 || t >= 1.0 {
            continue;
        }
        let len2 = len * len;
        let a1 = b * (dot / len2);
        let a2 = a - a1;
        if a2.magnitude_squared() > radius * radius {
            continue;
        }

        let dist = a1.magnitude();
        if dist < min_dist {
            min_dist = dist;
            result = Some(e);
        }
    }
    result.map(|r| (r, min_dist))
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

                let c1 = data[m.a.component::<Creature>()];
                let c2 = data[m.b.component::<Creature>()];
                if c1.timeout >= 0.0 || c2.timeout >= 0.0 || c1.kind != c2.kind {
                    continue;
                }

                mate(ctx, data, m.a, m.b)?;
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

pub fn input_system<I>(data: &mut GameData, entities: I) -> GameResult<()>
where
    I: IntoIterator<Item = Entity> + Clone,
{
    for e in entities.clone() {
        let p1 = data[e.component::<Position>()].position;
        let d = data[e.component::<Direction>()].direction;
        let (y, x) = d.sin_cos();
        let p2 = Vector2::new(x, y) * VIEW_DISTANCE;
        let ray = Ray { p1, p2 };
        let r1 = raycast(data, &ray, entities.clone());
        let (y, x) = (d + 45.0_f32.to_radians()).sin_cos();
        let p2 = Vector2::new(x, y) * VIEW_DISTANCE;
        let ray = Ray { p1, p2 };
        let r2 = raycast(data, &ray, entities.clone());
        let (y, x) = (d - 45.0_f32.to_radians()).sin_cos();
        let p2 = Vector2::new(x, y) * VIEW_DISTANCE;
        let ray = Ray { p1, p2 };
        let r3 = raycast(data, &ray, entities.clone());
        let mut inputs = vec![0.0; 6];
        if let Some((e, d)) = r1 {
            let kind = data[e.component::<Creature>()].kind;
            inputs[0] = kind.as_f32();
            inputs[3] = d / VIEW_DISTANCE;
        }
        if let Some((e, d)) = r2 {
            let kind = data[e.component::<Creature>()].kind;
            inputs[1] = kind.as_f32();
            inputs[4] = d / VIEW_DISTANCE;
        }
        if let Some((e, d)) = r3 {
            let kind = data[e.component::<Creature>()].kind;
            inputs[2] = kind.as_f32();
            inputs[5] = d / VIEW_DISTANCE;
        }
        data[e.component::<Inputs>()].input = DVector::from_vec(inputs);
    }
    Ok(())
}

pub fn output_system<I>(data: &mut GameData, entities: I) -> GameResult<()>
where
    I: IntoIterator<Item = Entity> + Clone,
{
    for e in entities.clone() {
        let output = &data[e.component::<Outputs>()].output;
        let (index, _) = output
            .iter()
            .enumerate()
            .max_by_key(|(_, x)| OrderedFloat::from(**x))
            .unwrap();
        let angle = (360.0 / 8.0 * index as f32).to_radians();
        let (y, x) = angle.sin_cos();
        let new_velocity = Velocity::new(x * SPEED, y * SPEED);
        let new_direction = angle;
        data[e.component::<Velocity>()] = new_velocity;
        data[e.component::<Direction>()].direction = new_direction;
    }
    Ok(())
}
