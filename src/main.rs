use std::fs;
use std::path::Path;
use std::process;

use ggez::conf::WindowMode;
use ggez::event::{self, EventHandler};
use ggez::graphics::{self, Color};
use ggez::timer;
use ggez::{Context, ContextBuilder, GameResult};

use rand::random;

use self::collision::Body;
use self::creature::*;
use self::data::{Entity, GameData, Insert};
use self::draw::Draw;
use self::nn::{Inputs, Network, Outputs};

pub mod collision;
pub mod creature;
pub mod data;
pub mod draw;
pub mod lazy;
pub mod lstm;
pub mod mutate;
pub mod nn;

pub const DPI_FACTOR: f32 = 1.0 / 3.166;
pub const WIDTH: f32 = 1280.0 * DPI_FACTOR;
pub const HEIGHT: f32 = 720.0 * DPI_FACTOR;
pub const RADIUS: f32 = 16.0;
pub const SPEED: f32 = 100.0;
pub const CREATURE_COUNT: usize = 40;
pub const FOOD_COUNT: usize = 15;
pub const FOOD_TIMEOUT: f32 = 8.0;
pub const VEGAN_RATIO: f32 = 0.66;

enum State {
    Game,
}

struct GameState {
    data: GameData,
    foods: Vec<Entity>,
    creatures: Vec<Entity>,
    food_timeout: f32,
}

impl GameState {
    pub fn new(ctx: &mut Context) -> GameResult<Self> {
        let mut data = GameData::new();
        let mut foods = Vec::new();
        let mut creatures = Vec::new();
        for _ in 0..FOOD_COUNT {
            let e = data.add_entity();
            let radius = random::<f32>() * RADIUS * DPI_FACTOR;
            let color = Color::new(random::<f32>(), random::<f32>(), random::<f32>(), 1.0);
            data.insert(e, Food);
            data.insert(
                e,
                Position::new(random::<f32>() * WIDTH, random::<f32>() * HEIGHT),
            );
            data.insert(e, Velocity::new(0.0, 0.0));
            data.insert(e, Body::new(radius, random::<f32>(), random::<f32>()));
            data.insert(e, Draw::circle(ctx, radius, color)?);
            foods.push(e)
        }

        let mut new_count = CREATURE_COUNT;

        let path: &Path = "alive.bin".as_ref();
        if path.exists() {
            let encoded = fs::read("alive.bin").expect("couldn't load top 20 from `alive.bin`");
            let top: Vec<(Creature, Network)> =
                bincode::deserialize(&encoded).expect("couldn't deserialize top 20");

            new_count -= top.len();

            for (creature, network) in top {
                let e = data.add_entity();
                let radius = random::<f32>() * RADIUS * DPI_FACTOR;
                let color = if creature.kind == Kind::Vegan {
                    Color::new(0.0, random::<f32>(), random::<f32>(), 1.0)
                } else {
                    Color::new(random::<f32>(), 0.0, random::<f32>(), 1.0)
                };
                data.insert(e, creature);
                data.insert(
                    e,
                    Position::new(random::<f32>() * WIDTH, random::<f32>() * HEIGHT),
                );
                data.insert(e, Velocity::new(0.0, 0.0));
                data.insert(e, Direction::new(0.0));
                data.insert(e, Body::new(radius, random::<f32>(), random::<f32>()));
                data.insert(e, Draw::circle(ctx, radius, color)?);
                data.insert(e, network);
                data.insert(e, Inputs::new(RAY_COUNT * 2));
                data.insert(e, Outputs::new(DIR_COUNT));
                creatures.push(e)
            }
        }

        for _ in 0..new_count {
            let e = data.add_entity();
            let radius = random::<f32>() * RADIUS * DPI_FACTOR;
            let color;
            let kind = if random::<f32>() < VEGAN_RATIO {
                color = Color::new(0.0, random::<f32>(), random::<f32>(), 1.0);
                Kind::Vegan
            } else {
                color = Color::new(random::<f32>(), 0.0, random::<f32>(), 1.0);
                Kind::Carnivorous
            };
            data.insert(e, Creature::new(kind));
            data.insert(
                e,
                Position::new(random::<f32>() * WIDTH, random::<f32>() * HEIGHT),
            );
            data.insert(e, Velocity::new(0.0, 0.0));
            data.insert(e, Direction::new(0.0));
            data.insert(e, Body::new(radius, random::<f32>(), random::<f32>()));
            data.insert(e, Draw::circle(ctx, radius, color)?);
            data.insert(e, Network::new(&[RAY_COUNT * 2, 6, 6, DIR_COUNT]));
            data.insert(e, Inputs::new(RAY_COUNT * 2));
            data.insert(e, Outputs::new(DIR_COUNT));
            creatures.push(e)
        }

        Ok(Self {
            data,
            foods,
            creatures,
            food_timeout: 0.0,
        })
    }
}

impl EventHandler for GameState {
    fn update(&mut self, ctx: &mut Context) -> GameResult<()> {
        let delta = timer::delta(ctx);
        let delta = delta.as_secs() as f32 + delta.subsec_nanos() as f32 / 1000000000.0;
        self.food_timeout += delta;
        if self.food_timeout > FOOD_TIMEOUT {
            self.food_timeout -= FOOD_TIMEOUT;
            for _ in 0..FOOD_COUNT {
                let e = self.data.add_entity();
                let radius = random::<f32>() * RADIUS * DPI_FACTOR;
                let color = Color::new(random::<f32>(), random::<f32>(), random::<f32>(), 1.0);
                self.data.insert(e, Food);
                self.data.insert(
                    e,
                    Position::new(random::<f32>() * WIDTH, random::<f32>() * HEIGHT),
                );
                self.data.insert(e, Velocity::new(0.0, 0.0));
                self.data
                    .insert(e, Body::new(radius, random::<f32>(), random::<f32>()));
                self.data.insert(e, Draw::circle(ctx, radius, color)?);
                self.foods.push(e)
            }
        }
        for e in self.creatures.iter().copied() {
            self.data[e.component::<Creature>()].timeout -= delta;
            self.data[e.component::<Creature>()].life += delta;
            self.data[e.component::<Creature>()].hunger += delta;
            if self.data[e.component::<Creature>()].hunger > STARVE {
                self.data.delete(e);
                self.data.lazy.remove(e);
            }
        }

        let (_, remove) = self.data.commit();
        for r in remove {
            let pos = self.creatures.iter().position(|e| *e == r);
            if let Some(pos) = pos {
                self.creatures.remove(pos);
                continue;
            }
        }

        collision::physics_system(
            ctx,
            &mut self.data,
            self.creatures.iter().chain(&self.foods).copied(),
            self.creatures.iter().chain(&self.foods).copied(),
        )?;

        let (add, remove) = self.data.commit();
        for r in remove {
            let pos = self.creatures.iter().position(|e| *e == r);
            if let Some(pos) = pos {
                self.creatures.remove(pos);
                continue;
            }
            let pos = self.foods.iter().position(|e| *e == r);
            if let Some(pos) = pos {
                self.foods.remove(pos);
                continue;
            }
        }
        self.creatures.extend(add);

        collision::input_system(
            &mut self.data,
            self.creatures.iter().copied(),
            self.creatures.iter().chain(&self.foods).copied(),
        )?;
        nn::nn_system(&mut self.data, self.creatures.iter().copied())?;
        collision::output_system(&mut self.data, self.creatures.iter().copied())?;

        Ok(())
    }

    fn draw(&mut self, ctx: &mut Context) -> GameResult<()> {
        graphics::clear(ctx, graphics::BLACK);

        draw::draw_system(ctx, &self.data, self.foods.iter().copied())?;
        draw::draw_system(ctx, &self.data, self.creatures.iter().copied())?;

        graphics::present(ctx)?;
        Ok(())
    }

    fn quit_event(&mut self, _ctx: &mut Context) -> bool {
        // filler
        let mut top = vec![(Creature::new(Kind::Vegan), Network::new(&[1])); CREATURE_COUNT];
        for e in self.creatures.iter().copied() {
            for top in top.iter_mut() {
                if self.data[e.component::<Creature>()].life > top.0.life {
                    *top = (
                        self.data[e.component::<Creature>()],
                        self.data[e.component::<Network>()].clone(),
                    );
                    break;
                }
            }
        }

        let encoded: Vec<u8> = bincode::serialize(&top).expect("couldn't serialize top 20");

        fs::write("alive.bin", &encoded).expect("couldn't save top 20 to `alive.bin`");

        false
    }
}

struct Game {
    game: GameState,
    state: State,
}

impl Game {
    pub fn new(ctx: &mut Context) -> GameResult<Game> {
        Ok(Game {
            game: GameState::new(ctx)?,
            state: State::Game,
        })
    }
}

impl EventHandler for Game {
    fn update(&mut self, ctx: &mut Context) -> GameResult<()> {
        match self.state {
            State::Game => self.game.update(ctx),
        }
    }

    fn draw(&mut self, ctx: &mut Context) -> GameResult<()> {
        match self.state {
            State::Game => self.game.draw(ctx),
        }
    }

    fn quit_event(&mut self, ctx: &mut Context) -> bool {
        match self.state {
            State::Game => self.game.quit_event(ctx),
        }
    }
}

fn main() {
    let (mut ctx, mut event_loop) =
        ContextBuilder::new("ldjam-46", "Szymon \"pi\" Walter <waltersz@protonmail.com>")
            .window_mode(WindowMode {
                // NOTE: the `* DPI_FACTOR` part should only be necessary on my fucked laptop
                width: WIDTH,
                height: HEIGHT,
                ..Default::default()
            })
            .build()
            .expect("couldn't build game context");

    let mut game = Game::new(&mut ctx).expect("couldn't create game");

    if let Err(err) = event::run(&mut ctx, &mut event_loop, &mut game) {
        eprintln!("{}", err);
        process::exit(1);
    }
}
