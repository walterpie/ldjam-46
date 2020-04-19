use ggez::graphics::{self, Color, DrawMode, DrawParam, Mesh, MeshBuilder};
use ggez::{Context, GameResult};

use crate::creature::{Direction, Position};
use crate::data::Has;
use crate::data::{Entity, GameData};
use crate::DPI_FACTOR;

/// Should be stored in an array of structs
#[derive(Debug, Clone, PartialEq)]
pub struct Draw {
    mesh: Mesh,
    pub color: Color,
}

impl Draw {
    pub fn circle(ctx: &mut Context, radius: f32, color: Color) -> GameResult<Self> {
        let mesh = MeshBuilder::new()
            .circle(DrawMode::fill(), [0.0, 0.0], radius, 0.25, color)
            .build(ctx)?;
        Ok(Self { mesh, color })
    }

    pub fn creature(ctx: &mut Context, radius: f32, color: Color) -> GameResult<Self> {
        let mesh = MeshBuilder::new()
            .circle(DrawMode::fill(), [0.0, 0.0], radius, 0.25, color)
            .line(&[[0.0, 0.0], [2.0 * radius, 0.0]], 4.0 * DPI_FACTOR, color)?
            .build(ctx)?;
        Ok(Self { mesh, color })
    }
}

pub fn draw_system<I>(ctx: &mut Context, data: &GameData, iter: I) -> GameResult<()>
where
    I: IntoIterator<Item = Entity>,
{
    for e in iter {
        let position = data[e.component::<Position>()].position;
        let mesh = &data[e.component::<Draw>()].mesh;
        let rotation = if data.has(e.component::<Direction>()) {
            data[e.component::<Direction>()].direction
        } else {
            0.0
        };
        graphics::draw(
            ctx,
            mesh,
            DrawParam::new()
                .dest([position.x, position.y])
                .offset([0.0, 0.0])
                .rotation(rotation)
                .scale([1.0, 1.0]),
        )?;
    }
    Ok(())
}
