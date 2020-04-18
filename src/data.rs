use std::marker::PhantomData;
use std::ops::{Index, IndexMut};

use crate::collision::*;
use crate::creature::*;
use crate::draw::*;
use crate::lazy::*;
use crate::nn::{Inputs, Network, Outputs};

pub trait Insert<T> {
    fn insert(&mut self, e: Entity, t: T);
}

/// A collection of all the components
#[derive(Debug, PartialEq)]
pub struct GameData {
    entity: usize,
    creatures: Vec<Option<Creature>>,
    positions: Vec<Option<Position>>,
    velocities: Vec<Option<Velocity>>,
    directions: Vec<Option<Direction>>,
    bodies: Vec<Option<Body>>,
    draw: Vec<Option<Draw>>,
    nns: Vec<Option<Network>>,
    inputs: Vec<Option<Inputs>>,
    outputs: Vec<Option<Outputs>>,
    pub lazy: LazyUpdate,
}

impl GameData {
    pub fn new() -> Self {
        Self {
            entity: 0,
            creatures: Vec::new(),
            positions: Vec::new(),
            velocities: Vec::new(),
            directions: Vec::new(),
            bodies: Vec::new(),
            draw: Vec::new(),
            nns: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            lazy: LazyUpdate::new(),
        }
    }

    pub fn add_entity(&mut self) -> Entity {
        self.creatures.push(None);
        self.positions.push(None);
        self.velocities.push(None);
        self.directions.push(None);
        self.bodies.push(None);
        self.draw.push(None);
        self.nns.push(None);
        self.inputs.push(None);
        self.outputs.push(None);

        let e = Entity { idx: self.entity };
        self.entity += 1;
        e
    }

    pub fn commit(&mut self) -> Vec<Entity> {
        let delta = self.lazy.entity;
        let result = (self.entity..self.entity + delta)
            .map(|idx| Entity { idx })
            .collect();
        self.entity += self.lazy.entity;
        self.lazy.entity = 0;
        self.creatures.extend(self.lazy.creatures.drain(..));
        self.positions.extend(self.lazy.positions.drain(..));
        self.velocities.extend(self.lazy.velocities.drain(..));
        self.directions.extend(self.lazy.directions.drain(..));
        self.bodies.extend(self.lazy.bodies.drain(..));
        self.draw.extend(self.lazy.draw.drain(..));
        self.nns.extend(self.lazy.nns.drain(..));
        self.inputs.extend(self.lazy.inputs.drain(..));
        self.outputs.extend(self.lazy.outputs.drain(..));
        result
    }
}

/// And index into the SOAs representing entities
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Entity {
    pub idx: usize,
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

impl Index<Component<Creature>> for GameData {
    type Output = Creature;

    fn index(&self, idx: Component<Creature>) -> &Self::Output {
        self.creatures[idx.idx]
            .as_ref()
            .expect("entity doesn't have component")
    }
}

impl IndexMut<Component<Creature>> for GameData {
    fn index_mut(&mut self, idx: Component<Creature>) -> &mut Self::Output {
        self.creatures[idx.idx]
            .as_mut()
            .expect("entity doesn't have component")
    }
}

impl Insert<Creature> for GameData {
    fn insert(&mut self, e: Entity, t: Creature) {
        self.creatures[e.idx] = Some(t);
    }
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

impl Index<Component<Direction>> for GameData {
    type Output = Direction;

    fn index(&self, idx: Component<Direction>) -> &Self::Output {
        self.directions[idx.idx]
            .as_ref()
            .expect("entity doesn't have component")
    }
}

impl IndexMut<Component<Direction>> for GameData {
    fn index_mut(&mut self, idx: Component<Direction>) -> &mut Self::Output {
        self.directions[idx.idx]
            .as_mut()
            .expect("entity doesn't have component")
    }
}

impl Insert<Direction> for GameData {
    fn insert(&mut self, e: Entity, t: Direction) {
        self.directions[e.idx] = Some(t);
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

impl Index<Component<Network>> for GameData {
    type Output = Network;

    fn index(&self, idx: Component<Network>) -> &Self::Output {
        self.nns[idx.idx]
            .as_ref()
            .expect("entity doesn't have component")
    }
}

impl IndexMut<Component<Network>> for GameData {
    fn index_mut(&mut self, idx: Component<Network>) -> &mut Self::Output {
        self.nns[idx.idx]
            .as_mut()
            .expect("entity doesn't have component")
    }
}

impl Insert<Network> for GameData {
    fn insert(&mut self, e: Entity, t: Network) {
        self.nns[e.idx] = Some(t);
    }
}

impl Index<Component<Inputs>> for GameData {
    type Output = Inputs;

    fn index(&self, idx: Component<Inputs>) -> &Self::Output {
        self.inputs[idx.idx]
            .as_ref()
            .expect("entity doesn't have component")
    }
}

impl IndexMut<Component<Inputs>> for GameData {
    fn index_mut(&mut self, idx: Component<Inputs>) -> &mut Self::Output {
        self.inputs[idx.idx]
            .as_mut()
            .expect("entity doesn't have component")
    }
}

impl Insert<Inputs> for GameData {
    fn insert(&mut self, e: Entity, t: Inputs) {
        self.inputs[e.idx] = Some(t);
    }
}

impl Index<Component<Outputs>> for GameData {
    type Output = Outputs;

    fn index(&self, idx: Component<Outputs>) -> &Self::Output {
        self.outputs[idx.idx]
            .as_ref()
            .expect("entity doesn't have component")
    }
}

impl IndexMut<Component<Outputs>> for GameData {
    fn index_mut(&mut self, idx: Component<Outputs>) -> &mut Self::Output {
        self.outputs[idx.idx]
            .as_mut()
            .expect("entity doesn't have component")
    }
}

impl Insert<Outputs> for GameData {
    fn insert(&mut self, e: Entity, t: Outputs) {
        self.outputs[e.idx] = Some(t);
    }
}