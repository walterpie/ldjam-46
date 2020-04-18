use ggez::graphics::{self, Color, DrawMode, DrawParam, Mesh, MeshBuilder};
use ggez::{Context, GameResult};

use crate::creature::Position;
use crate::data::{Entity, GameData};

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
}

pub fn draw_system<I>(ctx: &mut Context, data: &GameData, iter: I) -> GameResult<()>
where
    I: IntoIterator<Item = Entity>,
{
    for e in iter {
        let position = data[e.component::<Position>()].position;
        let mesh = &data[e.component::<Draw>()].mesh;
        graphics::draw(
            ctx,
            mesh,
            DrawParam::new()
                .dest([position.x, position.y])
                .offset([0.0, 0.0])
                .rotation(0.0)
                .scale([1.0, 1.0]),
        )?;
    }
    Ok(())
}
