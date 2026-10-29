use std::convert::Infallible;

use geo_types::{Coord, LineString, Polygon};

use crate::polygonize::VectorFeature;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MapGrid {
    pub min_x: f64,
    pub max_y: f64,
    pub resolution: f64,
    pub epsg: u32,
}

pub fn pixel_to_map(features: &[VectorFeature], grid: &MapGrid) -> Vec<VectorFeature> {
    let Ok(mapped) = map_coordinates(features, |column, row| {
        Ok::<_, Infallible>((
            grid.min_x + column * grid.resolution,
            grid.max_y - row * grid.resolution,
        ))
    });
    mapped
}

pub fn map_coordinates<E>(
    features: &[VectorFeature],
    map: impl Fn(f64, f64) -> Result<(f64, f64), E>,
) -> Result<Vec<VectorFeature>, E> {
    let map_ring = |ring: &LineString<f64>| {
        ring.coords()
            .map(|coord| map(coord.x, coord.y).map(|(x, y)| Coord { x, y }))
            .collect::<Result<LineString<f64>, E>>()
    };
    features
        .iter()
        .map(|feature| {
            let exterior = map_ring(feature.geometry.exterior())?;
            let interiors = feature
                .geometry
                .interiors()
                .iter()
                .map(map_ring)
                .collect::<Result<Vec<_>, E>>()?;
            Ok(VectorFeature {
                class_id: feature.class_id,
                geometry: Polygon::new(exterior, interiors),
                area_px: feature.area_px,
                confidence: feature.confidence,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pixel_outline_maps_to_grid_corners() {
        let pixel_square = LineString::from(vec![(2.0, 3.0), (3.0, 3.0), (3.0, 4.0), (2.0, 4.0)]);
        let features = vec![VectorFeature {
            class_id: 1,
            geometry: Polygon::new(pixel_square, vec![]),
            area_px: 1.0,
            confidence: 0.75,
        }];
        let grid = MapGrid {
            min_x: 100.0,
            max_y: 50.0,
            resolution: 0.5,
            epsg: 3857,
        };

        let mapped = pixel_to_map(&features, &grid);

        let corners: Vec<(f64, f64)> = mapped[0]
            .geometry
            .exterior()
            .coords()
            .map(|c| (c.x, c.y))
            .collect();
        assert_eq!(
            corners,
            vec![
                (101.0, 48.5),
                (101.5, 48.5),
                (101.5, 48.0),
                (101.0, 48.0),
                (101.0, 48.5),
            ]
        );
        assert_eq!(mapped[0].area_px, 1.0);
        assert_eq!(mapped[0].class_id, 1);
    }
}
