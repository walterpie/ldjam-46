use ggez::{Context, GameResult};

use nalgebra::Vector2;

use rand::random;

use serde::{Deserialize, Serialize};

use crate::collision::Body;
use crate::data::{Entity, GameData, Insert};
use crate::draw::Draw;
use crate::mutate::Mutate;
use crate::nn::{Desired, Inputs, Network, Outputs};
use crate::TIME_FACTOR;

pub const VEGAN_CHILDREN: f32 = 3.0;
pub const VEGAN_TIMEOUT: f32 = 15.0 / TIME_FACTOR;
pub const CARNIVORE_CHILDREN: f32 = 1.0;
pub const CARNIVORE_TIMEOUT: f32 = 40.0 / TIME_FACTOR;
pub const FOV_2: f32 = 180.0;
pub const RAY_COUNT: usize = 8;
pub const DIR_COUNT: usize = 16;
pub const VEGAN_STARVE: f32 = 180.0 / TIME_FACTOR;
pub const CARNIVORE_STARVE: f32 = 60.0 / TIME_FACTOR;
pub const VEGAN_NUTRITION: f32 = 2.0;
pub const CARNIVORE_NUTRITION: f32 = 3.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Food;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Kind {
    /// Corresponds to 0.0
    Vegan,
    /// Corresponds to 1.0
    Carnivorous,
}

/// Should be stored in an array of structs
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Creature {
    /// Can be either vegan or carnivorous
    pub kind: Kind,
    /// Once this reaches 1.0, the creature dies
    pub hunger: f32,
    /// If this is below 0.0, the creature can mate
    pub timeout: f32,
    /// This is the time spent living
    pub life: f32,
}

impl Creature {
    pub fn new(kind: Kind) -> Self {
        let timeout = match kind {
            Kind::Vegan => VEGAN_TIMEOUT,
            Kind::Carnivorous => CARNIVORE_TIMEOUT,
        };

        Self {
            kind,
            hunger: 0.0,
            timeout,
            life: 0.0,
        }
    }
}

/// Should be stored in an array of structs
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Position {
    pub position: Vector2<f32>,
}

impl Position {
    pub fn new(x: f32, y: f32) -> Self {
        Self {
            position: Vector2::new(x, y),
        }
    }
}

/// Should be stored in an array of structs
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Velocity {
    pub velocity: Vector2<f32>,
}

impl Velocity {
    pub fn new(x: f32, y: f32) -> Self {
        Self {
            velocity: Vector2::new(x, y),
        }
    }
}

/// Should be stored in an array of structs
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Direction {
    pub direction: f32,
}

impl Direction {
    pub fn new(direction: f32) -> Self {
        Self { direction }
    }
}

pub const M_FACTOR: f32 = 0.5;
pub const M_CHANCE: f32 = 0.05;
pub const M_MUTATION: f32 = 0.10;

pub fn mate(ctx: &mut Context, data: &mut GameData, a: Entity, b: Entity) -> GameResult<()> {
    let timeout = match data[a.component::<Creature>()].kind {
        Kind::Vegan => VEGAN_TIMEOUT,
        Kind::Carnivorous => CARNIVORE_TIMEOUT,
    };

    data[a.component::<Creature>()].timeout = timeout;
    data[b.component::<Creature>()].timeout = timeout;

    let min_children = 1.0;
    let max_children = match data[a.component::<Creature>()].kind {
        Kind::Vegan => VEGAN_CHILDREN,
        Kind::Carnivorous => CARNIVORE_CHILDREN,
    };

    let children = min_children + random::<f32>() * (max_children - min_children);
    let children = children.round() as usize;
    for _ in 0..children {
        let apos = data[a.component::<Position>()].position;
        let bpos = data[b.component::<Position>()].position;
        let position = (apos + bpos) * 0.5;
        let x = position[0];
        let y = position[1];
        let e = data.lazy.add_entity();
        let radius = data[a.component::<Body>()].radius.mutate(
            &data[b.component::<Body>()].radius,
            M_FACTOR,
            M_CHANCE,
            M_MUTATION,
        );
        let mass = data[a.component::<Body>()].mass.mutate(
            &data[b.component::<Body>()].mass,
            M_FACTOR,
            M_CHANCE,
            M_MUTATION,
        );
        let restitution = data[a.component::<Body>()].restitution.mutate(
            &data[b.component::<Body>()].restitution,
            M_FACTOR,
            M_CHANCE,
            M_MUTATION,
        );
        let color = data[a.component::<Draw>()].color.mutate(
            &data[b.component::<Draw>()].color,
            M_FACTOR,
            M_CHANCE,
            M_MUTATION,
        );
        let kind = data[a.component::<Creature>()].kind;
        data.lazy.insert(e, Creature::new(kind));
        data.lazy.insert(e, Position::new(x, y));
        data.lazy.insert(e, Velocity::new(0.0, 0.0));
        data.lazy.insert(e, Direction::new(0.0));
        data.lazy.insert(e, Body::new(radius, mass, restitution));
        data.lazy.insert(e, Draw::creature(ctx, radius, color)?);
        data.lazy
            .insert(e, Network::new(&[RAY_COUNT * 2, 24, 20, DIR_COUNT]));
        data.lazy.insert(e, Inputs::new(RAY_COUNT * 2));
        data.lazy.insert(e, Outputs::new(DIR_COUNT));
        data.lazy.insert(e, Desired::new(DIR_COUNT));
    }

    Ok(())
}
