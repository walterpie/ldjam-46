use std::env;
use std::fs;
use std::path::Path;
use std::process;

use ggez::audio::{SoundSource, Source};
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
use self::nn::{Desired, Inputs, Network, Outputs};

pub mod collision;
pub mod creature;
pub mod data;
pub mod draw;
pub mod lazy;
pub mod mutate;
pub mod nn;

pub const TIME_FACTOR: f32 = 2.5;
pub const GEN_TIME: f32 = 72.0 / TIME_FACTOR;
pub const DPI_FACTOR: f32 = 1.0 / 3.166;
pub const WIDTH: f32 = 1920.0 * DPI_FACTOR;
pub const HEIGHT: f32 = 1080.0 * DPI_FACTOR;
pub const FOOD_MIN_RADIUS: f32 = 10.0;
pub const FOOD_MAX_RADIUS: f32 = 20.0;
pub const VEGAN_MIN_RADIUS: f32 = 15.0;
pub const VEGAN_MAX_RADIUS: f32 = 30.0;
pub const CARNIVORE_MIN_RADIUS: f32 = 7.0;
pub const CARNIVORE_MAX_RADIUS: f32 = 14.0;
pub const MAX_RADIUS: f32 = VEGAN_MAX_RADIUS;
pub const CARNIVORE_SPEED: f32 = 40.0 * TIME_FACTOR;
pub const VEGAN_SPEED: f32 = 100.0 * TIME_FACTOR;
pub const TOP_COUNT: usize = 10;
pub const CREATURE_COUNT: usize = 100;
pub const FOOD_COUNT: usize = 30;
pub const FOOD_TIMEOUT: f32 = 1.0 / TIME_FACTOR;
pub const CARNIVORE_RATIO: f32 = 0.06;

enum State {
    Game,
}

struct GameState {
    generation: usize,
    time: f32,
    data: GameData,
    foods: Vec<Entity>,
    creatures: Vec<Entity>,
    food_timeout: f32,
}

impl GameState {
    pub fn new(ctx: &mut Context, generation: usize) -> GameResult<Self> {
        let mut data = GameData::new();
        let mut foods = Vec::new();
        let mut creatures = Vec::new();
        for _ in 0..FOOD_COUNT {
            let e = data.add_entity();
            let radius = (FOOD_MIN_RADIUS + random::<f32>() * (FOOD_MAX_RADIUS - FOOD_MIN_RADIUS))
                * DPI_FACTOR;
            let color = random::<f32>();
            let color = Color::new(color, color, color, 1.0);
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

        let mut carnivores = (CREATURE_COUNT as f32 * CARNIVORE_RATIO) as usize;

        let mut args = env::args();
        args.next();
        if let Some(path) = args.next() {
            println!("{:?}", path);
            let path: &Path = path.as_ref();
            if path.exists() {
                let encoded = fs::read(path).expect("couldn't load top 10");
                let top: Vec<(Creature, Network)> =
                    bincode::deserialize(&encoded).expect("couldn't deserialize top 10");

                new_count -= top.len();

                for (creature, network) in top {
                    let e = data.add_entity();
                    let radius = if creature.kind == Kind::Vegan {
                        (VEGAN_MIN_RADIUS + random::<f32>() * (VEGAN_MAX_RADIUS - VEGAN_MIN_RADIUS))
                            * DPI_FACTOR
                    } else {
                        carnivores -= 1;
                        (CARNIVORE_MIN_RADIUS
                            + random::<f32>() * (CARNIVORE_MAX_RADIUS - CARNIVORE_MIN_RADIUS))
                            * DPI_FACTOR
                    };
                    let color = if creature.kind == Kind::Vegan {
                        Color::new(0.0, random::<f32>(), random::<f32>() * 0.2, 1.0)
                    } else {
                        Color::new(random::<f32>(), 0.0, random::<f32>() * 0.2, 1.0)
                    };
                    data.insert(e, creature);
                    data.insert(
                        e,
                        Position::new(random::<f32>() * WIDTH, random::<f32>() * HEIGHT),
                    );
                    data.insert(e, Velocity::new(0.0, 0.0));
                    data.insert(e, Direction::new(0.0));
                    data.insert(e, Body::new(radius, random::<f32>(), random::<f32>()));
                    data.insert(e, Draw::creature(ctx, radius, color)?);
                    data.insert(e, network);
                    data.insert(e, Inputs::new(RAY_COUNT * 2));
                    data.insert(e, Outputs::new(DIR_COUNT));
                    data.insert(e, Desired::new(DIR_COUNT));
                    creatures.push(e)
                }
            }
        }

        for _ in 0..new_count {
            let e = data.add_entity();
            let color;
            let kind = if carnivores == 0 {
                color = Color::new(0.0, random::<f32>(), random::<f32>() * 0.2, 1.0);
                Kind::Vegan
            } else {
                carnivores -= 1;
                color = Color::new(random::<f32>(), 0.0, random::<f32>() * 0.2, 1.0);
                Kind::Carnivorous
            };
            let radius = if kind == Kind::Vegan {
                (VEGAN_MIN_RADIUS + random::<f32>() * (VEGAN_MAX_RADIUS - VEGAN_MIN_RADIUS))
                    * DPI_FACTOR
            } else {
                (CARNIVORE_MIN_RADIUS
                    + random::<f32>() * (CARNIVORE_MAX_RADIUS - CARNIVORE_MIN_RADIUS))
                    * DPI_FACTOR
            };
            data.insert(e, Creature::new(kind));
            data.insert(
                e,
                Position::new(random::<f32>() * WIDTH, random::<f32>() * HEIGHT),
            );
            data.insert(e, Velocity::new(0.0, 0.0));
            data.insert(e, Direction::new(0.0));
            data.insert(e, Body::new(radius, random::<f32>(), random::<f32>()));
            data.insert(e, Draw::creature(ctx, radius, color)?);
            data.insert(e, Network::new(&[RAY_COUNT * 2, 24, 20, DIR_COUNT]));
            data.insert(e, Inputs::new(RAY_COUNT * 2));
            data.insert(e, Outputs::new(DIR_COUNT));
            data.insert(e, Desired::new(DIR_COUNT));
            creatures.push(e)
        }

        Ok(Self {
            generation,
            time: 0.0,
            data,
            foods,
            creatures,
            food_timeout: 0.0,
        })
    }
}

impl EventHandler for GameState {
    fn update(&mut self, ctx: &mut Context) -> GameResult<()> {
        let delta = timer::duration_to_f64(timer::delta(ctx)) as f32;
        self.time += delta;

        if self.time > GEN_TIME {
            *self = GameState::new(ctx, self.generation + 1)?;
            return Ok(());
        }

        self.food_timeout += delta;
        if self.food_timeout > FOOD_TIMEOUT {
            self.food_timeout -= FOOD_TIMEOUT;
            for _ in 0..FOOD_COUNT {
                let e = self.data.add_entity();
                let radius = (FOOD_MIN_RADIUS
                    + random::<f32>() * (FOOD_MAX_RADIUS - FOOD_MIN_RADIUS))
                    * DPI_FACTOR;
                let color = random::<f32>();
                let color = Color::new(color, color, color, 1.0);
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
            let starve = match self.data[e.component::<Creature>()].kind {
                Kind::Carnivorous => CARNIVORE_STARVE,
                Kind::Vegan => VEGAN_STARVE,
            };
            if self.data[e.component::<Creature>()].hunger > starve {
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
        let mut top = vec![(Creature::new(Kind::Vegan), Network::new(&[1, 1])); TOP_COUNT];
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

        let encoded: Vec<u8> = bincode::serialize(&top).expect("couldn't serialize top 10");

        fs::write(format!("gen{}.bin", self.generation), &encoded).expect("couldn't save top 10");

        false
    }
}

struct Game {
    _sound: Source,
    game: GameState,
    state: State,
}

impl Game {
    pub fn new(ctx: &mut Context) -> GameResult<Game> {
        let mut sound = Source::new(ctx, "/ldjam.mp3")?;
        sound.set_repeat(true);
        sound.play()?;
        Ok(Game {
            _sound: sound,
            game: GameState::new(ctx, 0)?,
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
