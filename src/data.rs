use std::marker::PhantomData;
use std::ops::{Index, IndexMut};

use crate::collision::*;
use crate::creature::*;
use crate::draw::*;

pub trait Insert<T> {
    fn insert(&mut self, e: Entity, t: T);
}

/// A collection of all the components
#[derive(Debug, PartialEq)]
pub struct GameData {
    entity: usize,
    positions: Vec<Option<Position>>,
    velocities: Vec<Option<Velocity>>,
    bodies: Vec<Option<Body>>,
    draw: Vec<Option<Draw>>,
}

impl GameData {
    pub fn new() -> Self {
        Self {
            entity: 0,
            positions: Vec::new(),
            velocities: Vec::new(),
            bodies: Vec::new(),
            draw: Vec::new(),
        }
    }

    pub fn add_entity(&mut self) -> Entity {
        self.positions.push(None);
        self.velocities.push(None);
        self.bodies.push(None);
        self.draw.push(None);

        let e = Entity { idx: self.entity };
        self.entity += 1;
        e
    }
}

/// And index into the SOAs representing entities
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Entity {
    idx: usize,
}

impl Entity {
    pub fn component<T>(&self) -> Component<T> {
        Component {
            idx: self.idx,
            _phantom: PhantomData,
        }
    }
}

/// Used to index into the corresponding `Vec<T>` in a `GameData`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Component<T> {
    idx: usize,
    _phantom: PhantomData<T>,
}

impl Index<Component<Position>> for GameData {
    type Output = Position;

    fn index(&self, idx: Component<Position>) -> &Self::Output {
        self.positions[idx.idx]
            .as_ref()
            .expect("entity doesn't have component")
    }
}

impl IndexMut<Component<Position>> for GameData {
    fn index_mut(&mut self, idx: Component<Position>) -> &mut Self::Output {
        self.positions[idx.idx]
            .as_mut()
            .expect("entity doesn't have component")
    }
}

impl Insert<Position> for GameData {
    fn insert(&mut self, e: Entity, t: Position) {
        self.positions[e.idx] = Some(t);
    }
}

impl Index<Component<Velocity>> for GameData {
    type Output = Velocity;

    fn index(&self, idx: Component<Velocity>) -> &Self::Output {
        self.velocities[idx.idx]
            .as_ref()
            .expect("entity doesn't have component")
    }
}

impl IndexMut<Component<Velocity>> for GameData {
    fn index_mut(&mut self, idx: Component<Velocity>) -> &mut Self::Output {
        self.velocities[idx.idx]
            .as_mut()
            .expect("entity doesn't have component")
    }
}

impl Insert<Velocity> for GameData {
    fn insert(&mut self, e: Entity, t: Velocity) {
        self.velocities[e.idx] = Some(t);
    }
}

impl Index<Component<Body>> for GameData {
    type Output = Body;

    fn index(&self, idx: Component<Body>) -> &Self::Output {
        self.bodies[idx.idx]
            .as_ref()
            .expect("entity doesn't have component")
    }
}

impl IndexMut<Component<Body>> for GameData {
    fn index_mut(&mut self, idx: Component<Body>) -> &mut Self::Output {
        self.bodies[idx.idx]
            .as_mut()
            .expect("entity doesn't have component")
    }
}

impl Insert<Body> for GameData {
    fn insert(&mut self, e: Entity, t: Body) {
        self.bodies[e.idx] = Some(t);
    }
}

impl Index<Component<Draw>> for GameData {
    type Output = Draw;

    fn index(&self, idx: Component<Draw>) -> &Self::Output {
        self.draw[idx.idx]
            .as_ref()
            .expect("entity doesn't have component")
    }
}

impl IndexMut<Component<Draw>> for GameData {
    fn index_mut(&mut self, idx: Component<Draw>) -> &mut Self::Output {
        self.draw[idx.idx]
            .as_mut()
            .expect("entity doesn't have component")
    }
}

impl Insert<Draw> for GameData {
    fn insert(&mut self, e: Entity, t: Draw) {
        self.draw[e.idx] = Some(t);
    }
}
