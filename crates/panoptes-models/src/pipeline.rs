//! Processing pipeline — orchestrate end-to-end inference workflows.

use panoptes_core::inference::{InferenceEngine, InferenceResult};
use panoptes_core::model::ModelConfig;
use panoptes_core::tensor::ImageTensor;
use panoptes_raster::window::{WindowConfig, extract_tiles};
use panoptes_vector::polygonize::{VectorFeature, polygonize_all};

/// A complete inference pipeline.
pub struct Pipeline {
    /// Model configuration.
    pub config: ModelConfig,
    /// Tile extraction configuration.
    pub window_config: WindowConfig,
    /// Minimum polygon area (in pixels).
    pub min_area: f64,
}

/// Result from processing a single tile.
#[derive(Debug)]
pub struct TileResult {
    /// Tile origin.
    pub origin_x: u32,
    pub origin_y: u32,
    /// Inference result for this tile.
    pub inference: InferenceResult,
}

/// Result from processing an entire image.
pub struct ImageResult {
    /// Per-tile results.
    pub tile_results: Vec<TileResult>,
    /// Extracted vector features (polygonized from all tiles).
    pub features: Vec<VectorFeature>,
}

impl Pipeline {
    /// Create a new pipeline with default window settings.
    pub fn new(config: ModelConfig) -> Self {
        let tile_size = config.input.width;
        Self {
            config,
            window_config: WindowConfig::new(tile_size, tile_size / 4),
            min_area: 10.0,
        }
    }

    /// Process an image through the pipeline using the given engine.
    pub fn process(&self, image: &ImageTensor, engine: &dyn InferenceEngine) -> ImageResult {
        let tiles = extract_tiles(image, &self.window_config);

        let tile_results: Vec<TileResult> = tiles
            .iter()
            .map(|tile| {
                let result = engine.predict(&tile.data, &self.config);
                TileResult {
                    origin_x: tile.origin_x,
                    origin_y: tile.origin_y,
                    inference: result,
                }
            })
            .collect();

        // Collect vector features from segmentation results
        let mut all_features = Vec::new();
        for tr in &tile_results {
            if let InferenceResult::Segmentation { mask, confidence } = &tr.inference
                && let Ok(features) = polygonize_all(
                    mask,
                    self.config.classes.len(),
                    self.min_area,
                    Some(confidence),
                )
            {
                // Offset features to global coordinates
                all_features.extend(features.into_iter().map(|mut f| {
                    use geo_types::Coord;
                    let offset_x = tr.origin_x as f64;
                    let offset_y = tr.origin_y as f64;
                    let exterior: Vec<Coord<f64>> = f
                        .geometry
                        .exterior()
                        .0
                        .iter()
                        .map(|c| Coord {
                            x: c.x + offset_x,
                            y: c.y + offset_y,
                        })
                        .collect();
                    f.geometry =
                        geo_types::Polygon::new(geo_types::LineString::from(exterior), vec![]);
                    f
                }));
            }
        }

        ImageResult {
            tile_results,
            features: all_features,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::building_segmentation;
    use ndarray::Array3;
    use panoptes_core::inference::ThresholdEngine;

    #[test]
    fn test_pipeline_creation() {
        let config = building_segmentation();
        let pipeline = Pipeline::new(config);
        assert_eq!(pipeline.window_config.tile_width, 512);
        assert_eq!(pipeline.window_config.overlap, 128);
    }

    #[test]
    fn test_pipeline_process() {
        let config = building_segmentation();
        let pipeline = Pipeline::new(config);
        let engine = ThresholdEngine::new(vec![128.0]);

        // Create an image large enough for at least one tile
        let image = Array3::from_elem((3, 512, 512), 200.0_f32);
        let result = pipeline.process(&image, &engine);
        assert!(!result.tile_results.is_empty());
    }
}
