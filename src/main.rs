use std::process;

use ggez::conf::WindowMode;
use ggez::event::{self, EventHandler};
use ggez::graphics::{self, Color};
use ggez::{Context, ContextBuilder, GameResult};

use rand::random;

use self::collision::Body;
use self::creature::*;
use self::data::{Entity, GameData, Insert};
use self::draw::Draw;

pub mod collision;
pub mod creature;
pub mod data;
pub mod draw;

pub const DPI_FACTOR: f32 = 1.0 / 3.166;
pub const WIDTH: f32 = 1280.0 * DPI_FACTOR;
pub const HEIGHT: f32 = 720.0 * DPI_FACTOR;
pub const RADIUS: f32 = 64.0;

enum State {
    Game,
}

struct GameState {
    data: GameData,
    creatures: Vec<Entity>,
}

impl GameState {
    pub fn new(ctx: &mut Context) -> GameResult<GameState> {
        let mut data = GameData::new();
        let mut creatures = Vec::new();
        for _ in 0..100 {
            let e = data.add_entity();
            let radius = random::<f32>() * RADIUS * DPI_FACTOR;
            let color = Color::new(
                random::<f32>(),
                random::<f32>(),
                random::<f32>(),
                random::<f32>(),
            );
            data.insert(
                e,
                Position::new(random::<f32>() * WIDTH, random::<f32>() * HEIGHT),
            );
            let vx = random::<f32>() * 2.0 - 1.0;
            let vy = random::<f32>() * 2.0 - 1.0;
            data.insert(e, Velocity::new(vx * 32.0, vy * 32.0));
            data.insert(e, Body::new(radius, random::<f32>(), random::<f32>()));
            data.insert(e, Draw::circle(ctx, radius, color)?);
            creatures.push(e)
        }

        Ok(GameState { data, creatures })
    }
}

impl EventHandler for GameState {
    fn update(&mut self, ctx: &mut Context) -> GameResult<()> {
        collision::physics_system(
            ctx,
            &mut self.data,
            self.creatures.iter().copied(),
            self.creatures.iter().copied(),
        )?;

        Ok(())
    }

    fn draw(&mut self, ctx: &mut Context) -> GameResult<()> {
        graphics::clear(ctx, graphics::BLACK);

        draw::draw_system(ctx, &self.data, self.creatures.iter().copied())?;

        graphics::present(ctx)?;
        Ok(())
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
