use std::marker::PhantomData;
use std::ops::{Index, IndexMut};

use crate::collision::*;
use crate::creature::*;
use crate::data::{Entity, Insert};
use crate::draw::*;
use crate::nn::{Inputs, Network, Outputs};

/// A collection of lazily evaluated components
#[derive(Debug, PartialEq)]
pub struct LazyUpdate {
    pub remove: Vec<Entity>,
    pub entity: usize,
    pub creatures: Vec<Option<Creature>>,
    pub foods: Vec<Option<Food>>,
    pub positions: Vec<Option<Position>>,
    pub velocities: Vec<Option<Velocity>>,
    pub directions: Vec<Option<Direction>>,
    pub bodies: Vec<Option<Body>>,
    pub draw: Vec<Option<Draw>>,
    pub nns: Vec<Option<Network>>,
    pub inputs: Vec<Option<Inputs>>,
    pub outputs: Vec<Option<Outputs>>,
}

impl LazyUpdate {
    pub fn new() -> Self {
        Self {
            entity: 0,
            remove: Vec::new(),
            creatures: Vec::new(),
            foods: Vec::new(),
            positions: Vec::new(),
            velocities: Vec::new(),
            directions: Vec::new(),
            bodies: Vec::new(),
            draw: Vec::new(),
            nns: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }

    pub fn add_entity(&mut self) -> Entity {
        self.creatures.push(None);
        self.foods.push(None);
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

    pub fn remove(&mut self, e: Entity) {
        self.remove.push(e);
    }
}

/// Used to index into the corresponding `Vec<T>` in a `LazyUpdate`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Component<T> {
    idx: usize,
    _phantom: PhantomData<T>,
}

impl Index<Component<Creature>> for LazyUpdate {
    type Output = Creature;

    fn index(&self, idx: Component<Creature>) -> &Self::Output {
        self.creatures[idx.idx]
            .as_ref()
            .expect("entity doesn't have component")
    }
}

impl IndexMut<Component<Creature>> for LazyUpdate {
    fn index_mut(&mut self, idx: Component<Creature>) -> &mut Self::Output {
        self.creatures[idx.idx]
            .as_mut()
            .expect("entity doesn't have component")
    }
}

impl Insert<Creature> for LazyUpdate {
    fn insert(&mut self, e: Entity, t: Creature) {
        self.creatures[e.idx] = Some(t);
    }
}

impl Index<Component<Food>> for LazyUpdate {
    type Output = Food;

    fn index(&self, idx: Component<Food>) -> &Self::Output {
        self.foods[idx.idx]
            .as_ref()
            .expect("entity doesn't have component")
    }
}

impl IndexMut<Component<Food>> for LazyUpdate {
    fn index_mut(&mut self, idx: Component<Food>) -> &mut Self::Output {
        self.foods[idx.idx]
            .as_mut()
            .expect("entity doesn't have component")
    }
}

impl Insert<Food> for LazyUpdate {
    fn insert(&mut self, e: Entity, t: Food) {
        self.foods[e.idx] = Some(t);
    }
}

impl Index<Component<Position>> for LazyUpdate {
    type Output = Position;

    fn index(&self, idx: Component<Position>) -> &Self::Output {
        self.positions[idx.idx]
            .as_ref()
            .expect("entity doesn't have component")
    }
}

impl IndexMut<Component<Position>> for LazyUpdate {
    fn index_mut(&mut self, idx: Component<Position>) -> &mut Self::Output {
        self.positions[idx.idx]
            .as_mut()
            .expect("entity doesn't have component")
    }
}

impl Insert<Position> for LazyUpdate {
    fn insert(&mut self, e: Entity, t: Position) {
        self.positions[e.idx] = Some(t);
    }
}

impl Index<Component<Velocity>> for LazyUpdate {
    type Output = Velocity;

    fn index(&self, idx: Component<Velocity>) -> &Self::Output {
        self.velocities[idx.idx]
            .as_ref()
            .expect("entity doesn't have component")
    }
}

impl IndexMut<Component<Velocity>> for LazyUpdate {
    fn index_mut(&mut self, idx: Component<Velocity>) -> &mut Self::Output {
        self.velocities[idx.idx]
            .as_mut()
            .expect("entity doesn't have component")
    }
}

impl Insert<Velocity> for LazyUpdate {
    fn insert(&mut self, e: Entity, t: Velocity) {
        self.velocities[e.idx] = Some(t);
    }
}

impl Index<Component<Direction>> for LazyUpdate {
    type Output = Direction;

    fn index(&self, idx: Component<Direction>) -> &Self::Output {
        self.directions[idx.idx]
            .as_ref()
            .expect("entity doesn't have component")
    }
}

impl IndexMut<Component<Direction>> for LazyUpdate {
    fn index_mut(&mut self, idx: Component<Direction>) -> &mut Self::Output {
        self.directions[idx.idx]
            .as_mut()
            .expect("entity doesn't have component")
    }
}

impl Insert<Direction> for LazyUpdate {
    fn insert(&mut self, e: Entity, t: Direction) {
        self.directions[e.idx] = Some(t);
    }
}

impl Index<Component<Body>> for LazyUpdate {
    type Output = Body;

    fn index(&self, idx: Component<Body>) -> &Self::Output {
        self.bodies[idx.idx]
            .as_ref()
            .expect("entity doesn't have component")
    }
}

impl IndexMut<Component<Body>> for LazyUpdate {
    fn index_mut(&mut self, idx: Component<Body>) -> &mut Self::Output {
        self.bodies[idx.idx]
            .as_mut()
            .expect("entity doesn't have component")
    }
}

impl Insert<Body> for LazyUpdate {
    fn insert(&mut self, e: Entity, t: Body) {
        self.bodies[e.idx] = Some(t);
    }
}

impl Index<Component<Draw>> for LazyUpdate {
    type Output = Draw;

    fn index(&self, idx: Component<Draw>) -> &Self::Output {
        self.draw[idx.idx]
            .as_ref()
            .expect("entity doesn't have component")
    }
}

impl IndexMut<Component<Draw>> for LazyUpdate {
    fn index_mut(&mut self, idx: Component<Draw>) -> &mut Self::Output {
        self.draw[idx.idx]
            .as_mut()
            .expect("entity doesn't have component")
    }
}

impl Insert<Draw> for LazyUpdate {
    fn insert(&mut self, e: Entity, t: Draw) {
        self.draw[e.idx] = Some(t);
    }
}

impl Index<Component<Network>> for LazyUpdate {
    type Output = Network;

    fn index(&self, idx: Component<Network>) -> &Self::Output {
        self.nns[idx.idx]
            .as_ref()
            .expect("entity doesn't have component")
    }
}

impl IndexMut<Component<Network>> for LazyUpdate {
    fn index_mut(&mut self, idx: Component<Network>) -> &mut Self::Output {
        self.nns[idx.idx]
            .as_mut()
            .expect("entity doesn't have component")
    }
}

impl Insert<Network> for LazyUpdate {
    fn insert(&mut self, e: Entity, t: Network) {
        self.nns[e.idx] = Some(t);
    }
}

impl Index<Component<Inputs>> for LazyUpdate {
    type Output = Inputs;

    fn index(&self, idx: Component<Inputs>) -> &Self::Output {
        self.inputs[idx.idx]
            .as_ref()
            .expect("entity doesn't have component")
    }
}

impl IndexMut<Component<Inputs>> for LazyUpdate {
    fn index_mut(&mut self, idx: Component<Inputs>) -> &mut Self::Output {
        self.inputs[idx.idx]
            .as_mut()
            .expect("entity doesn't have component")
    }
}

impl Insert<Inputs> for LazyUpdate {
    fn insert(&mut self, e: Entity, t: Inputs) {
        self.inputs[e.idx] = Some(t);
    }
}

impl Index<Component<Outputs>> for LazyUpdate {
    type Output = Outputs;

    fn index(&self, idx: Component<Outputs>) -> &Self::Output {
        self.outputs[idx.idx]
            .as_ref()
            .expect("entity doesn't have component")
    }
}

impl IndexMut<Component<Outputs>> for LazyUpdate {
    fn index_mut(&mut self, idx: Component<Outputs>) -> &mut Self::Output {
        self.outputs[idx.idx]
            .as_mut()
            .expect("entity doesn't have component")
    }
}

impl Insert<Outputs> for LazyUpdate {
    fn insert(&mut self, e: Entity, t: Outputs) {
        self.outputs[e.idx] = Some(t);
    }
}
