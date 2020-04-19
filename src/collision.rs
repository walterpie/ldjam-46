use std::f32;

use ggez::timer;
use ggez::{Context, GameResult};

use nalgebra::{DVector, Vector2};

use ordered_float::OrderedFloat;

use rand::random;

use crate::creature::*;
use crate::data::{Entity, GameData, Has};
use crate::nn::{Inputs, Outputs};
use crate::{HEIGHT, RADIUS, SPEED, WIDTH};

pub const VIEW_DISTANCE: f32 = 10000.0;

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

pub fn raycast<I>(data: &GameData, ray: &Ray, this: Entity, entities: I) -> Option<(Entity, f32)>
where
    I: IntoIterator<Item = Entity>,
{
    let b = ray.p2 - ray.p1;
    let mut result = None;
    let mut min_dist = f32::INFINITY;
    for e in entities {
        if e == this {
            continue;
        }

        let pos = data[e.component::<Position>()].position;
        let radius = data[e.component::<Body>()].radius;
        let a = pos - ray.p1;
        let dot = a.dot(&b);
        let len2 = b.magnitude_squared();
        let t = dot / len2;
        if t < 0.0 || t >= 1.0 {
            continue;
        }
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
            if a == b {
                continue;
            }

            if !data.has(a.component::<Body>()) || !data.has(b.component::<Body>()) {
                continue;
            }

            if let Some(m) = gen_manifold(data, a, b) {
                resolve(data, &m);
                correct(data, &m);

                if data.has(m.a.component::<Creature>()) && data.has(m.b.component::<Creature>()) {
                    let c1 = data[m.a.component::<Creature>()];
                    let c2 = data[m.b.component::<Creature>()];
                    match (c1.kind, c2.kind) {
                        (Kind::Vegan, Kind::Vegan) => {}
                        (Kind::Vegan, Kind::Carnivorous) => {
                            data[m.b.component::<Creature>()].hunger -= FOOD;
                            data.delete(m.a);
                            data.lazy.remove(m.a);
                            continue;
                        }
                        (Kind::Carnivorous, Kind::Vegan) => {
                            data[m.a.component::<Creature>()].hunger -= FOOD;
                            data.delete(m.b);
                            data.lazy.remove(m.b);
                            continue;
                        }
                        (Kind::Carnivorous, Kind::Carnivorous) => {}
                    }
                    if c1.timeout >= 0.0 || c2.timeout >= 0.0 {
                        continue;
                    }

                    mate(ctx, data, m.a, m.b)?;
                } else if data.has(m.a.component::<Creature>()) && data.has(m.b.component::<Food>())
                {
                    data[m.a.component::<Creature>()].hunger -= FOOD;
                    data.delete(m.b);
                    data.lazy.remove(m.b);
                } else if data.has(m.a.component::<Food>()) && data.has(m.b.component::<Creature>())
                {
                    data[m.b.component::<Creature>()].hunger -= FOOD;
                    data.delete(m.a);
                    data.lazy.remove(m.a);
                }
            }
        }
    }
    let delta = timer::delta(ctx);
    let delta = delta.as_secs() as f32 + delta.subsec_nanos() as f32 / 1000000000.0;
    for a in left.clone() {
        if !data.has(a.component::<Velocity>()) || !data.has(a.component::<Position>()) {
            continue;
        }
        let vel = data[a.component::<Velocity>()].velocity;
        let pos = &mut data[a.component::<Position>()].position;
        *pos += vel * delta;
        if pos.x < -RADIUS {
            pos.x += WIDTH + RADIUS;
        } else if pos.x > WIDTH + RADIUS {
            pos.x -= WIDTH + RADIUS;
        }
        if pos.y < -RADIUS {
            pos.y += HEIGHT + RADIUS;
        } else if pos.y > HEIGHT + RADIUS {
            pos.y -= HEIGHT + RADIUS;
        }
    }
    Ok(())
}

pub fn input_system<I1, I2>(data: &mut GameData, creatures: I1, all: I2) -> GameResult<()>
where
    I1: IntoIterator<Item = Entity>,
    I2: IntoIterator<Item = Entity> + Clone,
{
    for e in creatures {
        let this = e;
        let p1 = data[e.component::<Position>()].position;
        let d = data[e.component::<Direction>()].direction;
        let mut inputs = vec![1.0; RAY_COUNT * 2];
        for i in 0..RAY_COUNT {
            let f = i as f32 / (RAY_COUNT as f32 - 1.0);
            let d = -FOV_2 * f + d + FOV_2 * f;
            let (y, x) = d.sin_cos();
            let p2 = Vector2::new(x, y) * VIEW_DISTANCE;
            let ray = Ray { p1, p2 };
            let result = raycast(data, &ray, e, all.clone());
            if let Some((e, d)) = result {
                let kind = match data[this.component::<Creature>()].kind {
                    Kind::Vegan => {
                        if data.has(e.component::<Food>()) {
                            1.0
                        } else if data[e.component::<Creature>()].kind == Kind::Vegan {
                            0.5
                        } else {
                            0.0
                        }
                    }
                    Kind::Carnivorous => {
                        if data.has(e.component::<Food>()) {
                            0.0
                        } else if data[e.component::<Creature>()].kind == Kind::Vegan {
                            1.0
                        } else {
                            0.5
                        }
                    }
                };

                inputs[i * 2] = kind;
                inputs[i * 2 + 1] = d / VIEW_DISTANCE;
            }
        }
        data[e.component::<Inputs>()].input = DVector::from_vec(inputs);
    }
    Ok(())
}

pub fn output_system<I>(data: &mut GameData, entities: I) -> GameResult<()>
where
    I: IntoIterator<Item = Entity>,
{
    for e in entities {
        let output = &data[e.component::<Outputs>()].output;
        let (mut index, _) = output
            .iter()
            .enumerate()
            .max_by_key(|(_, x)| OrderedFloat::from(**x))
            .unwrap();
        if output.iter().all(|x| *x == 1.0) {
            index = random::<usize>() % DIR_COUNT;
        }
        let angle = (360.0 / DIR_COUNT as f32 * index as f32).to_radians();
        let (y, x) = angle.sin_cos();
        let new_velocity = Velocity::new(x * SPEED, y * SPEED);
        let new_direction = angle;
        data[e.component::<Velocity>()] = new_velocity;
        data[e.component::<Direction>()].direction = new_direction;
    }
    Ok(())
}
