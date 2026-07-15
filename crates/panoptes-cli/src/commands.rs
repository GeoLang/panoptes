//! CLI command implementations.

use std::path::{Path, PathBuf};

use panoptes_core::confidence;
use panoptes_core::inference::ThresholdEngine;
use panoptes_core::model::{ClassDef, InputSpec, ModelConfig, TaskType};
use panoptes_core::tensor::ImageTensor;
use panoptes_models::catalog;
use panoptes_models::pipeline::{ImageResult, Pipeline};
use panoptes_raster::change::detect_change;
use panoptes_raster::tile::load_image;
use panoptes_raster::window::WindowConfig;
use panoptes_vector::geojson_io::to_geojson_string;
use panoptes_vector::polygonize::polygonize_class;

/// Map a catalog model name to its configuration.
fn catalog_config(model: &str) -> Option<ModelConfig> {
    match model {
        "buildings" => Some(catalog::building_segmentation()),
        "roads" => Some(catalog::road_segmentation()),
        "vegetation" => Some(catalog::vegetation_detection()),
        "landcover" => Some(catalog::land_cover_classification()),
        _ => None,
    }
}

/// Binary segmentation config for a user-provided ONNX model.
fn generic_onnx_config(
    model: &str,
    tile_size: usize,
    confidence_threshold: f32,
    imagenet_normalize: bool,
) -> ModelConfig {
    ModelConfig {
        name: format!("user-onnx ({model})"),
        version: "user".to_string(),
        task: TaskType::Segmentation,
        input: InputSpec {
            channels: 3,
            height: tile_size,
            width: tile_size,
            imagenet_normalize,
        },
        classes: vec![
            ClassDef {
                id: 0,
                name: "background".to_string(),
                color: [0, 0, 0],
            },
            ClassDef {
                id: 1,
                name: "foreground".to_string(),
                color: [255, 0, 0],
            },
        ],
        confidence_threshold,
        model_path: Some(model.to_string()),
    }
}

/// Resolve `--model` into a config plus an optional local ONNX file to run.
///
/// A catalog name (`buildings`, ...) yields catalog metadata; it only runs ONNX if its
/// local file exists (catalog models ship no weights yet). Anything else is treated as a
/// path to a user ONNX segmentation model.
fn resolve_model(
    model: &str,
    tile_size: usize,
    confidence_threshold: f32,
    imagenet_normalize: bool,
) -> Result<(ModelConfig, Option<PathBuf>), String> {
    if let Some(config) = catalog_config(model) {
        let onnx_path = config
            .model_path
            .as_ref()
            .map(PathBuf::from)
            .filter(|p| p.exists());
        Ok((config, onnx_path))
    } else {
        let path = PathBuf::from(model);
        if path.exists() {
            let config =
                generic_onnx_config(model, tile_size, confidence_threshold, imagenet_normalize);
            Ok((config, Some(path)))
        } else {
            Err(format!(
                "Unknown model '{model}': not a catalog name and not an existing file.\nRun `panoptes models` to list catalog models, or pass a path to an .onnx file."
            ))
        }
    }
}

/// Run segmentation with the selected engine, reporting which one actually ran.
fn run_segmentation(
    pipeline: &Pipeline,
    image: &ImageTensor,
    engine: &str,
    onnx_path: Option<&Path>,
) -> ImageResult {
    let want_onnx = match engine {
        "threshold" => false,
        "onnx" => true,
        "auto" => onnx_path.is_some(),
        other => {
            eprintln!("Unknown engine '{other}', use one of: auto, onnx, threshold");
            std::process::exit(1);
        }
    };

    if want_onnx {
        #[cfg(feature = "onnx")]
        {
            use panoptes_core::inference::InferenceEngine;
            use panoptes_core::onnx::OnnxEngine;
            match onnx_path {
                Some(p) => match OnnxEngine::load(p, pipeline.config.clone()) {
                    Ok(eng) => {
                        println!("Engine: {} (model: {})", eng.name(), p.display());
                        return pipeline.process(image, &eng);
                    }
                    Err(e) => {
                        eprintln!("Failed to load ONNX model {}: {e}", p.display());
                        std::process::exit(1);
                    }
                },
                None => {
                    eprintln!("--engine onnx requires an ONNX model; pass --model path/to.onnx");
                    std::process::exit(1);
                }
            }
        }
        #[cfg(not(feature = "onnx"))]
        {
            if engine == "onnx" {
                eprintln!("this build has no ONNX support; rebuild with `--features onnx`");
                std::process::exit(1);
            }
            eprintln!(
                "note: an ONNX model was provided but this build lacks ONNX support; using threshold engine"
            );
        }
    }

    let eng = ThresholdEngine::new(vec![128.0]);
    println!("Engine: threshold (heuristic, no ONNX inference)");
    pipeline.process(image, &eng)
}

#[allow(clippy::too_many_arguments)]
pub fn segment(
    input: &Path,
    output: &Path,
    model: &str,
    engine: &str,
    tile_size: usize,
    min_area: f64,
    confidence_threshold: f32,
    imagenet_normalize: bool,
) {
    let (config, onnx_path) =
        match resolve_model(model, tile_size, confidence_threshold, imagenet_normalize) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("{e}");
                std::process::exit(1);
            }
        };

    let image = match load_image(input) {
        Ok(img) => img,
        Err(e) => {
            eprintln!("Failed to load image: {}", e);
            std::process::exit(1);
        }
    };

    println!("Loaded image: {:?}", image.shape());
    println!("Using model: {}", config.name);

    let mut pipeline = Pipeline::new(config);
    pipeline.window_config = WindowConfig::new(tile_size, tile_size / 4);
    pipeline.min_area = min_area;

    let result = run_segmentation(&pipeline, &image, engine, onnx_path.as_deref());

    println!("Processed {} tiles", result.tile_results.len());
    println!("Extracted {} features", result.features.len());

    let geojson = to_geojson_string(&result.features);
    if let Err(e) = std::fs::write(output, &geojson) {
        eprintln!("Failed to write output: {}", e);
        std::process::exit(1);
    }
    println!("Output written to: {}", output.display());
}

pub fn change(before: &Path, after: &Path, output: &Path, threshold: f32) {
    let before_img = match load_image(before) {
        Ok(img) => img,
        Err(e) => {
            eprintln!("Failed to load before image: {}", e);
            std::process::exit(1);
        }
    };

    let after_img = match load_image(after) {
        Ok(img) => img,
        Err(e) => {
            eprintln!("Failed to load after image: {}", e);
            std::process::exit(1);
        }
    };

    println!("Before image: {:?}", before_img.shape());
    println!("After image: {:?}", after_img.shape());

    let result = detect_change(&before_img, &after_img, threshold);
    println!("Change ratio: {:.1}%", result.change_ratio * 100.0);

    // Polygonize the change mask
    let features = match polygonize_class(&result.change_mask, 1, 10.0, None) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Polygonization failed: {}", e);
            std::process::exit(1);
        }
    };

    println!("Detected {} change regions", features.len());

    let geojson = to_geojson_string(&features);
    if let Err(e) = std::fs::write(output, &geojson) {
        eprintln!("Failed to write output: {}", e);
        std::process::exit(1);
    }
    println!("Output written to: {}", output.display());
}

pub fn list_models() {
    let models = catalog::list_models();
    println!("Catalog models ({}):", models.len());
    println!();
    for model in &models {
        println!(
            "  {} ({})",
            model.name,
            format!("{:?}", model.task).to_lowercase()
        );
        println!(
            "    Input: {}x{}x{}",
            model.input.width, model.input.height, model.input.channels
        );
        println!(
            "    Classes: {}",
            model
                .classes
                .iter()
                .map(|c| c.name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );
        println!("    Threshold: {}", model.confidence_threshold);
        match &model.model_path {
            Some(p) => println!("    Status: weights at {p}"),
            None => println!("    Status: planned (no weights published)"),
        }
        println!();
    }
    println!("These entries are metadata only; no weights are published yet.");
    println!("To run inference today, pass your own ONNX segmentation model:");
    println!("  panoptes segment --input img.png --output out.geojson --model path/to/model.onnx");
}

pub fn evaluate(prediction: &Path, ground_truth: &Path, num_classes: usize) {
    let pred_img = match load_image(prediction) {
        Ok(img) => img,
        Err(e) => {
            eprintln!("Failed to load prediction: {}", e);
            std::process::exit(1);
        }
    };

    let gt_img = match load_image(ground_truth) {
        Ok(img) => img,
        Err(e) => {
            eprintln!("Failed to load ground truth: {}", e);
            std::process::exit(1);
        }
    };

    // Use first channel as mask
    let pred_shape = pred_img.shape();
    let gt_shape = gt_img.shape();
    let (h, w) = (pred_shape[1], pred_shape[2]);

    let pred_mask = ndarray::Array2::from_shape_fn((h, w), |(y, x)| pred_img[[0, y, x]] as u8);
    let gt_mask = ndarray::Array2::from_shape_fn((gt_shape[1], gt_shape[2]), |(y, x)| {
        gt_img[[0, y, x]] as u8
    });

    let acc = confidence::pixel_accuracy(&pred_mask, &gt_mask);
    let miou = confidence::mean_iou(&pred_mask, &gt_mask, num_classes);

    println!("Evaluation Results:");
    println!("  Pixel Accuracy: {:.2}%", acc * 100.0);
    println!("  Mean IoU: {:.4}", miou);
    println!();

    for class_id in 0..num_classes {
        let class_iou = confidence::iou(&pred_mask, &gt_mask, class_id as u8);
        println!("  Class {} IoU: {:.4}", class_id, class_iou);
    }
}
