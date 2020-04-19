use std::collections::HashSet;
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};

use crate::collision::*;
use crate::creature::*;
use crate::draw::*;
use crate::lazy::*;
use crate::nn::{Desired, Inputs, Network, Outputs};

pub trait Has<T> {
    fn has(&self, c: Component<T>) -> bool;
}

pub trait Insert<T> {
    fn insert(&mut self, e: Entity, t: T);
}

/// A collection of all the components
#[derive(Debug, PartialEq)]
pub struct GameData {
    entity: usize,
    delete: HashSet<Entity>,
    creatures: Vec<Option<Creature>>,
    foods: Vec<Option<Food>>,
    positions: Vec<Option<Position>>,
    velocities: Vec<Option<Velocity>>,
    directions: Vec<Option<Direction>>,
    bodies: Vec<Option<Body>>,
    draw: Vec<Option<Draw>>,
    nns: Vec<Option<Network>>,
    inputs: Vec<Option<Inputs>>,
    outputs: Vec<Option<Outputs>>,
    desired: Vec<Option<Desired>>,
    pub lazy: LazyUpdate,
}

impl GameData {
    pub fn new() -> Self {
        Self {
            entity: 0,
            delete: HashSet::new(),
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
            desired: Vec::new(),
            lazy: LazyUpdate::new(),
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
        self.desired.push(None);

        let e = Entity { idx: self.entity };
        self.entity += 1;
        e
    }

    /// This does not immediately remove the entity, it only marks it for
    /// deletion
    pub fn delete(&mut self, e: Entity) {
        self.delete.insert(e);
    }

    pub fn commit(&mut self) -> (Vec<Entity>, Vec<Entity>) {
        let delta = self.lazy.entity;
        let mut remove = Vec::new();
        let result = (self.entity..self.entity + delta)
            .map(|idx| Entity { idx })
            .collect();
        self.entity += self.lazy.entity;
        self.lazy.entity = 0;
        self.creatures.extend(self.lazy.creatures.drain(..));
        self.foods.extend(self.lazy.foods.drain(..));
        self.positions.extend(self.lazy.positions.drain(..));
        self.velocities.extend(self.lazy.velocities.drain(..));
        self.directions.extend(self.lazy.directions.drain(..));
        self.bodies.extend(self.lazy.bodies.drain(..));
        self.draw.extend(self.lazy.draw.drain(..));
        self.nns.extend(self.lazy.nns.drain(..));
        self.inputs.extend(self.lazy.inputs.drain(..));
        self.outputs.extend(self.lazy.outputs.drain(..));
        self.desired.extend(self.lazy.desired.drain(..));
        for e in self.lazy.remove.drain(..) {
            self.creatures[e.idx] = None;
            self.foods[e.idx] = None;
            self.positions[e.idx] = None;
            self.velocities[e.idx] = None;
            self.directions[e.idx] = None;
            self.bodies[e.idx] = None;
            self.draw[e.idx] = None;
            self.nns[e.idx] = None;
            self.inputs[e.idx] = None;
            self.outputs[e.idx] = None;
            self.desired[e.idx] = None;
            remove.push(e);
        }
        (result, remove)
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

impl Has<Creature> for GameData {
    fn has(&self, c: Component<Creature>) -> bool {
        if self.delete.contains(&Entity { idx: c.idx }) {
            return false;
        }

        self.creatures[c.idx].is_some()
    }
}

impl Insert<Creature> for GameData {
    fn insert(&mut self, e: Entity, t: Creature) {
        self.creatures[e.idx] = Some(t);
    }
}

impl Index<Component<Food>> for GameData {
    type Output = Food;

    fn index(&self, idx: Component<Food>) -> &Self::Output {
        self.foods[idx.idx]
            .as_ref()
            .expect("entity doesn't have component")
    }
}

impl IndexMut<Component<Food>> for GameData {
    fn index_mut(&mut self, idx: Component<Food>) -> &mut Self::Output {
        self.foods[idx.idx]
            .as_mut()
            .expect("entity doesn't have component")
    }
}

impl Has<Food> for GameData {
    fn has(&self, c: Component<Food>) -> bool {
        if self.delete.contains(&Entity { idx: c.idx }) {
            return false;
        }

        self.foods[c.idx].is_some()
    }
}

impl Insert<Food> for GameData {
    fn insert(&mut self, e: Entity, t: Food) {
        self.foods[e.idx] = Some(t);
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

impl Has<Position> for GameData {
    fn has(&self, c: Component<Position>) -> bool {
        if self.delete.contains(&Entity { idx: c.idx }) {
            return false;
        }

        self.positions[c.idx].is_some()
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

impl Has<Velocity> for GameData {
    fn has(&self, c: Component<Velocity>) -> bool {
        if self.delete.contains(&Entity { idx: c.idx }) {
            return false;
        }

        self.velocities[c.idx].is_some()
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

impl Has<Direction> for GameData {
    fn has(&self, c: Component<Direction>) -> bool {
        if self.delete.contains(&Entity { idx: c.idx }) {
            return false;
        }

        self.directions[c.idx].is_some()
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

impl Has<Body> for GameData {
    fn has(&self, c: Component<Body>) -> bool {
        if self.delete.contains(&Entity { idx: c.idx }) {
            return false;
        }

        self.bodies[c.idx].is_some()
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

impl Has<Draw> for GameData {
    fn has(&self, c: Component<Draw>) -> bool {
        if self.delete.contains(&Entity { idx: c.idx }) {
            return false;
        }

        self.draw[c.idx].is_some()
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

impl Has<Network> for GameData {
    fn has(&self, c: Component<Network>) -> bool {
        if self.delete.contains(&Entity { idx: c.idx }) {
            return false;
        }

        self.nns[c.idx].is_some()
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

impl Has<Inputs> for GameData {
    fn has(&self, c: Component<Inputs>) -> bool {
        if self.delete.contains(&Entity { idx: c.idx }) {
            return false;
        }

        self.inputs[c.idx].is_some()
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

impl Has<Outputs> for GameData {
    fn has(&self, c: Component<Outputs>) -> bool {
        if self.delete.contains(&Entity { idx: c.idx }) {
            return false;
        }

        self.outputs[c.idx].is_some()
    }
}

impl Insert<Outputs> for GameData {
    fn insert(&mut self, e: Entity, t: Outputs) {
        self.outputs[e.idx] = Some(t);
    }
}

impl Index<Component<Desired>> for GameData {
    type Output = Desired;

    fn index(&self, idx: Component<Desired>) -> &Self::Output {
        self.desired[idx.idx]
            .as_ref()
            .expect("entity doesn't have component")
    }
}

impl IndexMut<Component<Desired>> for GameData {
    fn index_mut(&mut self, idx: Component<Desired>) -> &mut Self::Output {
        self.desired[idx.idx]
            .as_mut()
            .expect("entity doesn't have component")
    }
}

impl Has<Desired> for GameData {
    fn has(&self, c: Component<Desired>) -> bool {
        if self.delete.contains(&Entity { idx: c.idx }) {
            return false;
        }

        self.desired[c.idx].is_some()
    }
}

impl Insert<Desired> for GameData {
    fn insert(&mut self, e: Entity, t: Desired) {
        self.desired[e.idx] = Some(t);
    }
}
