use nalgebra::Vector2;

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
