//! Model catalog — pre-defined model configurations for common tasks.
//!
//! Only [`building_segmentation`] has published weights: `panoptes-buildings-v1.onnx`,
//! a release asset that has to be downloaded into the working directory. Every other
//! entry is metadata only, with a `model_path` of `None`, so to run those tasks pass
//! your own ONNX segmentation model with the CLI `--model path/to.onnx`.

use panoptes_core::model::{ClassDef, InputSpec, ModelConfig, TaskType};

/// Create a building segmentation model configuration.
pub fn building_segmentation() -> ModelConfig {
    ModelConfig {
        name: "panoptes-buildings-v1".to_string(),
        version: "1.0.0".to_string(),
        task: TaskType::Segmentation,
        input: InputSpec {
            width: 512,
            height: 512,
            channels: 3,
            imagenet_normalize: false,
        },
        classes: vec![
            ClassDef {
                id: 0,
                name: "background".to_string(),
                color: [0, 0, 0],
            },
            ClassDef {
                id: 1,
                name: "building".to_string(),
                color: [255, 0, 0],
            },
        ],
        confidence_threshold: 0.5,
        model_path: Some("panoptes-buildings-v1.onnx".to_string()),
    }
}

/// Create a road segmentation model configuration.
pub fn road_segmentation() -> ModelConfig {
    ModelConfig {
        name: "panoptes-roads-v1".to_string(),
        version: "1.0.0".to_string(),
        task: TaskType::Segmentation,
        input: InputSpec {
            width: 512,
            height: 512,
            channels: 3,
            imagenet_normalize: true,
        },
        classes: vec![
            ClassDef {
                id: 0,
                name: "background".to_string(),
                color: [0, 0, 0],
            },
            ClassDef {
                id: 1,
                name: "road".to_string(),
                color: [255, 255, 0],
            },
        ],
        confidence_threshold: 0.4,
        model_path: None,
    }
}

/// Create a land cover classification model configuration.
pub fn land_cover_classification() -> ModelConfig {
    ModelConfig {
        name: "panoptes-landcover-v1".to_string(),
        version: "1.0.0".to_string(),
        task: TaskType::Segmentation,
        input: InputSpec {
            width: 256,
            height: 256,
            channels: 3,
            imagenet_normalize: true,
        },
        classes: vec![
            ClassDef {
                id: 0,
                name: "water".to_string(),
                color: [0, 0, 255],
            },
            ClassDef {
                id: 1,
                name: "vegetation".to_string(),
                color: [0, 255, 0],
            },
            ClassDef {
                id: 2,
                name: "bare_soil".to_string(),
                color: [139, 69, 19],
            },
            ClassDef {
                id: 3,
                name: "built_up".to_string(),
                color: [128, 128, 128],
            },
            ClassDef {
                id: 4,
                name: "agriculture".to_string(),
                color: [255, 255, 0],
            },
        ],
        confidence_threshold: 0.3,
        model_path: None,
    }
}

/// Create a vegetation index model configuration.
pub fn vegetation_detection() -> ModelConfig {
    ModelConfig {
        name: "panoptes-vegetation-v1".to_string(),
        version: "1.0.0".to_string(),
        task: TaskType::Segmentation,
        input: InputSpec {
            width: 512,
            height: 512,
            channels: 3,
            imagenet_normalize: true,
        },
        classes: vec![
            ClassDef {
                id: 0,
                name: "non_vegetation".to_string(),
                color: [128, 128, 128],
            },
            ClassDef {
                id: 1,
                name: "trees".to_string(),
                color: [0, 128, 0],
            },
            ClassDef {
                id: 2,
                name: "shrubs".to_string(),
                color: [0, 255, 0],
            },
            ClassDef {
                id: 3,
                name: "grass".to_string(),
                color: [144, 238, 144],
            },
        ],
        confidence_threshold: 0.4,
        model_path: None,
    }
}

/// Create a change detection model configuration.
pub fn change_detection() -> ModelConfig {
    ModelConfig {
        name: "panoptes-change-v1".to_string(),
        version: "1.0.0".to_string(),
        task: TaskType::ChangeDetection,
        input: InputSpec {
            width: 256,
            height: 256,
            channels: 6, // 2x RGB
            imagenet_normalize: true,
        },
        classes: vec![
            ClassDef {
                id: 0,
                name: "no_change".to_string(),
                color: [0, 0, 0],
            },
            ClassDef {
                id: 1,
                name: "change".to_string(),
                color: [255, 0, 255],
            },
        ],
        confidence_threshold: 0.5,
        model_path: None,
    }
}

/// List all available model configurations.
pub fn list_models() -> Vec<ModelConfig> {
    vec![
        building_segmentation(),
        road_segmentation(),
        land_cover_classification(),
        vegetation_detection(),
        change_detection(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_list_models() {
        let models = list_models();
        assert_eq!(models.len(), 5);
        assert_eq!(models[0].name, "panoptes-buildings-v1");
    }

    #[test]
    fn test_building_model_config() {
        let config = building_segmentation();
        assert_eq!(config.classes.len(), 2);
        assert_eq!(config.input.width, 512);
        assert!(matches!(config.task, TaskType::Segmentation));
        assert_eq!(
            config.model_path.as_deref(),
            Some("panoptes-buildings-v1.onnx")
        );
        assert!(!config.input.imagenet_normalize);
    }

    #[test]
    fn test_change_detection_config() {
        let config = change_detection();
        assert_eq!(config.input.channels, 6);
        assert!(matches!(config.task, TaskType::ChangeDetection));
    }
}
