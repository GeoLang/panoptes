//! Processing pipeline — orchestrate end-to-end inference workflows.

use ndarray::Array2;
use panoptes_core::inference::{InferenceEngine, InferenceResult};
use panoptes_core::model::ModelConfig;
use panoptes_core::tensor::{ImageTensor, ProbabilityMap, SegmentationMask};
use panoptes_raster::window::{WindowConfig, extract_tiles};
use panoptes_vector::polygonize::{VectorFeature, polygonize_all};

// pixels no whole tile covers get a value no class id takes
const UNCOVERED: u8 = u8::MAX;

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

        let shape = image.shape();
        let (mask, confidence) = stitch_tiles(&tile_results, shape[1], shape[2]);
        let features = if tile_results.is_empty() {
            Vec::new()
        } else {
            polygonize_all(
                &mask,
                self.config.classes.len(),
                self.min_area,
                Some(&confidence),
            )
            .expect("whole tiles make a nonempty mask")
        };

        ImageResult {
            tile_results,
            features,
        }
    }
}

// each pixel takes its class from the tile whose center is nearest
fn stitch_tiles(
    tile_results: &[TileResult],
    height: usize,
    width: usize,
) -> (SegmentationMask, ProbabilityMap) {
    let mut mask = SegmentationMask::from_elem((height, width), UNCOVERED);
    let mut confidence = ProbabilityMap::zeros((height, width));
    let mut center_distance = Array2::from_elem((height, width), f32::INFINITY);
    for tile in tile_results {
        let InferenceResult::Segmentation {
            mask: tile_mask,
            confidence: tile_confidence,
        } = &tile.inference
        else {
            continue;
        };
        let (tile_height, tile_width) = (tile_mask.shape()[0], tile_mask.shape()[1]);
        for row in 0..tile_height {
            let y = tile.origin_y as usize + row;
            if y >= height {
                break;
            }
            let row_distance = (row as f32 + 0.5 - tile_height as f32 / 2.0).abs();
            for column in 0..tile_width {
                let x = tile.origin_x as usize + column;
                if x >= width {
                    break;
                }
                let column_distance = (column as f32 + 0.5 - tile_width as f32 / 2.0).abs();
                let distance = row_distance.max(column_distance);
                if distance < center_distance[[y, x]] {
                    center_distance[[y, x]] = distance;
                    mask[[y, x]] = tile_mask[[row, column]];
                    confidence[[y, x]] = tile_confidence[[row, column]];
                }
            }
        }
    }
    (mask, confidence)
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

    #[test]
    fn a_feature_across_tile_overlaps_is_traced_once() {
        let mut pipeline = Pipeline::new(building_segmentation());
        pipeline.window_config = WindowConfig::new(256, 64);
        let engine = ThresholdEngine::new(vec![128.0]);
        let mut image = Array3::from_elem((3, 600, 600), 10.0_f32);
        for row in 230..330 {
            for column in 220..320 {
                image[[1, row, column]] = 250.0;
            }
        }

        let result = pipeline.process(&image, &engine);

        let squares: Vec<_> = result.features.iter().filter(|f| f.class_id == 1).collect();
        assert_eq!(squares.len(), 1, "one square, not one per overlapping tile");
        assert_eq!(squares[0].area_px, 10_000.0);
    }
}
