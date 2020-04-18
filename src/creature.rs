use ggez::{Context, GameResult};

use nalgebra::Vector2;

use crate::collision::Body;
use crate::data::{Entity, GameData, Insert};
use crate::draw::Draw;
use crate::mutate::Mutate;
use crate::nn::{Inputs, Network, Outputs};

pub const DEFAULT_TIMEOUT: f32 = 10.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Kind {
    /// Corresponds to 0.0
    Vegan,
    /// Corresponds to 1.0
    Carnivorous,
}

impl Kind {
    pub fn as_f32(self) -> f32 {
        match self {
            Kind::Vegan => 0.0,
            Kind::Carnivorous => 1.0,
        }
    }
}

/// Should be stored in an array of structs
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Creature {
    /// Can be either vegan or carnivorous
    pub kind: Kind,
    /// Once this reaches 1.0, the creature dies
    pub hunger: f32,
    /// If this is below 0.0, the creature can mate
    pub timeout: f32,
}

impl Creature {
    pub fn new(kind: Kind) -> Self {
        Self {
            kind,
            hunger: 0.0,
            timeout: DEFAULT_TIMEOUT,
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
pub const M_CHANCE: f32 = 0.5;
pub const M_MUTATION: f32 = 0.05;

pub fn mate(ctx: &mut Context, data: &mut GameData, a: Entity, b: Entity) -> GameResult<()> {
    data[a.component::<Creature>()].timeout = DEFAULT_TIMEOUT;
    data[b.component::<Creature>()].timeout = DEFAULT_TIMEOUT;

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
    data.lazy.insert(e, Draw::circle(ctx, radius, color)?);
    data.lazy.insert(e, Network::new(&[6, 6, 8]));
    data.lazy.insert(e, Inputs::new(6));
    data.lazy.insert(e, Outputs::new(8));

    Ok(())
}
