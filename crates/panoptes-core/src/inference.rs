//! Inference engine — model loading and prediction execution.
//!
//! This module provides a trait-based inference abstraction plus a threshold engine that
//! stands in when no model file is available. Real inference is `crate::onnx::OnnxEngine`,
//! behind the `onnx` feature.

use ndarray::Array2;

use crate::model::{ModelConfig, TaskType};
use crate::tensor::{ImageTensor, ProbabilityMap, SegmentationMask};

/// A bounding box detection result.
#[derive(Debug, Clone)]
pub struct Detection {
    pub class_id: u8,
    pub confidence: f32,
    /// Bounding box: (x_min, y_min, x_max, y_max) in pixel coordinates.
    pub bbox: [f32; 4],
}

/// Inference result variants.
#[derive(Debug, Clone)]
pub enum InferenceResult {
    /// Semantic segmentation: class mask + confidence map.
    Segmentation {
        mask: SegmentationMask,
        confidence: ProbabilityMap,
    },
    /// Object detection: list of detections.
    Detection { detections: Vec<Detection> },
    /// Binary classification.
    Classification { class_id: u8, confidence: f32 },
}

/// Trait for model inference backends.
pub trait InferenceEngine: Send + Sync {
    /// Run inference on a single image tensor (CHW format, [0-255] range).
    fn predict(&self, input: &ImageTensor, config: &ModelConfig) -> InferenceResult;

    /// Backend name.
    fn name(&self) -> &str;
}

/// Threshold-based stand-in engine, for smoke-testing the pipeline when no model is
/// available. Per-channel thresholding, no learned model, so its output says nothing
/// about the imagery.
pub struct ThresholdEngine {
    /// Per-channel thresholds — pixels above threshold are classified as that channel's class.
    pub thresholds: Vec<f32>,
}

impl ThresholdEngine {
    pub fn new(thresholds: Vec<f32>) -> Self {
        Self { thresholds }
    }

    /// Simple threshold: classify by dominant channel.
    pub fn default_rgb() -> Self {
        Self {
            thresholds: vec![128.0, 128.0, 128.0],
        }
    }
}

impl InferenceEngine for ThresholdEngine {
    fn predict(&self, input: &ImageTensor, config: &ModelConfig) -> InferenceResult {
        match config.task {
            TaskType::Segmentation => {
                let shape = input.shape();
                let (c, h, w) = (shape[0], shape[1], shape[2]);
                let num_classes = config.num_classes();

                let mut mask = Array2::zeros((h, w));
                let mut conf = Array2::zeros((h, w));
                for y in 0..h {
                    for x in 0..w {
                        // pick the channel with highest value above its threshold
                        let mut best_class = 0usize;
                        let mut best_val = 0.0_f32;
                        for ci in 0..c.min(num_classes) {
                            let val = input[[ci, y, x]];
                            let threshold = self.thresholds.get(ci).copied().unwrap_or(128.0);
                            if val > threshold && val > best_val {
                                best_val = val;
                                best_class = ci;
                            }
                        }
                        mask[[y, x]] = best_class as u8;
                        // honest confidence: normalized intensity of the winning channel
                        conf[[y, x]] = (best_val / 255.0).clamp(0.0, 1.0);
                    }
                }

                InferenceResult::Segmentation {
                    mask,
                    confidence: conf,
                }
            }
            TaskType::Detection => {
                // Simple: find bright regions and output bounding boxes
                let shape = input.shape();
                let (_, h, w) = (shape[0], shape[1], shape[2]);
                let mut detections = Vec::new();

                // Scan in a grid
                let cell_h = h / 4;
                let cell_w = w / 4;
                for gy in 0..4 {
                    for gx in 0..4 {
                        let y_start = gy * cell_h;
                        let x_start = gx * cell_w;
                        let mut sum = 0.0_f32;
                        let mut count = 0;
                        for y in y_start..y_start + cell_h {
                            for x in x_start..x_start + cell_w {
                                sum += input[[0, y, x]];
                                count += 1;
                            }
                        }
                        let avg = sum / count as f32;
                        if avg > self.thresholds.first().copied().unwrap_or(128.0) {
                            detections.push(Detection {
                                class_id: 1,
                                confidence: avg / 255.0,
                                bbox: [
                                    x_start as f32,
                                    y_start as f32,
                                    (x_start + cell_w) as f32,
                                    (y_start + cell_h) as f32,
                                ],
                            });
                        }
                    }
                }

                InferenceResult::Detection { detections }
            }
            TaskType::Classification | TaskType::ChangeDetection => {
                // Simple: average brightness determines class
                let mean = input.mean().unwrap_or(0.0);
                let threshold = self.thresholds.first().copied().unwrap_or(128.0);
                let class_id = if mean > threshold { 1 } else { 0 };
                InferenceResult::Classification {
                    class_id,
                    confidence: if class_id == 1 {
                        mean / 255.0
                    } else {
                        1.0 - mean / 255.0
                    },
                }
            }
        }
    }

    fn name(&self) -> &str {
        "threshold"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{ClassDef, InputSpec};
    use ndarray::Array3;

    fn test_config() -> ModelConfig {
        ModelConfig {
            name: "test".to_string(),
            version: "1.0".to_string(),
            task: TaskType::Segmentation,
            input: InputSpec {
                channels: 3,
                height: 8,
                width: 8,
                imagenet_normalize: false,
            },
            classes: vec![
                ClassDef {
                    id: 0,
                    name: "bg".to_string(),
                    color: [0, 0, 0],
                },
                ClassDef {
                    id: 1,
                    name: "building".to_string(),
                    color: [255, 0, 0],
                },
                ClassDef {
                    id: 2,
                    name: "road".to_string(),
                    color: [0, 255, 0],
                },
            ],
            confidence_threshold: 0.5,
            model_path: None,
        }
    }

    #[test]
    fn test_threshold_segmentation() {
        let engine = ThresholdEngine::default_rgb();
        let config = test_config();

        // Create a bright red image (channel 0 = 200, others = 50)
        let mut input = Array3::from_elem((3, 8, 8), 50.0_f32);
        input.slice_mut(ndarray::s![0, .., ..]).fill(200.0);

        let result = engine.predict(&input, &config);
        match result {
            InferenceResult::Segmentation { mask, confidence } => {
                // Most pixels should be class 0 (first channel above threshold)
                assert_eq!(mask.shape(), &[8, 8]);
                assert_eq!(confidence.shape(), &[8, 8]);
            }
            _ => panic!("Expected segmentation result"),
        }
    }

    #[test]
    fn test_threshold_detection() {
        let engine = ThresholdEngine::new(vec![100.0]);
        let mut config = test_config();
        config.task = TaskType::Detection;

        // Bright image - should produce detections
        let input = Array3::from_elem((3, 8, 8), 200.0_f32);
        let result = engine.predict(&input, &config);
        match result {
            InferenceResult::Detection { detections } => {
                assert!(!detections.is_empty());
            }
            _ => panic!("Expected detection result"),
        }
    }
}
