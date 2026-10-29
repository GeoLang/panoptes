//! CLI command implementations.

use std::path::{Path, PathBuf};

use geoplumb::caps::{CapsPattern, CapsSet, Constraint, RasterPattern, SetField};
use geoplumb::{Bbox, Chunk, Crs, Engine, Transform, WindowReq};
use geoplumb_server::config::Config;
use geoplumb_server::{LayerGraph, build_layer_graph, parse_time};
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
use panoptes_vector::georeference::{MapGrid, map_coordinates, pixel_to_map};
use panoptes_vector::polygonize::{VectorFeature, polygonize_class};

const README_MODEL_SECTION: &str = "https://github.com/GeoLang/panoptes#model-weights";

const BUILDINGS_WEIGHTS_RELEASE: &str =
    "https://github.com/GeoLang/panoptes/releases/tag/weights-buildings-v1";

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
/// local file exists. `buildings` names a published release asset the user has to
/// download, the other catalog entries have no weights at all. Anything else is treated
/// as a path to a user ONNX segmentation model.
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
                "Unknown model '{model}': not a catalog name and not an existing file.\nThe only published weights are for 'buildings', downloaded from\n{BUILDINGS_WEIGHTS_RELEASE}\nFor any other task, pass a path to your own .onnx file.\nSee {README_MODEL_SECTION}"
            ))
        }
    }
}

/// What to do about a selected model that has no ONNX file on disk.
fn missing_weights_hint(config: &ModelConfig) -> String {
    match &config.model_path {
        Some(file) => {
            format!("Download {file} into the working directory from\n{BUILDINGS_WEIGHTS_RELEASE}")
        }
        None => {
            "No weights are published for it, so pass --model path/to.onnx with a model you supply."
                .to_string()
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
                    eprintln!(
                        "--engine onnx needs an ONNX model file, and none was found for '{}'.\n{}\nSee {README_MODEL_SECTION}",
                        pipeline.config.name,
                        missing_weights_hint(&pipeline.config)
                    );
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

    if engine == "auto" && onnx_path.is_none() {
        eprintln!(
            "warning: no ONNX model file was found for '{}', so nothing was segmented by a model.\nFalling back to the threshold heuristic, whose output is a smoke test of the tiling and\npolygonization path, not a prediction about this image.\n{}\nSee {README_MODEL_SECTION}",
            pipeline.config.name,
            missing_weights_hint(&pipeline.config)
        );
    }

    let eng = ThresholdEngine::new(vec![128.0]);
    println!("Engine: threshold (heuristic, no ONNX inference)");
    pipeline.process(image, &eng)
}

pub enum ImageSource {
    File(PathBuf),
    Layer(LayerRequest),
}

pub struct LayerRequest {
    pub layer_file: PathBuf,
    pub layer: String,
    pub bbox: Bbox,
    pub crs: u32,
    pub resolution: f64,
    pub time: Option<String>,
}

struct GeoreferencedImage {
    tensor: ImageTensor,
    grid: MapGrid,
    nodata_cells: usize,
}

const LAYER_CACHE_BYTES: usize = 512 << 20;

// an explicit reproject resamples a layer that is already in the crs
struct CrsDemand(Crs);

impl Transform for CrsDemand {
    fn identity(&self) -> String {
        format!("CrsDemand crs={}", self.0)
    }

    fn constraint(&self) -> Constraint {
        Constraint::Identity(CapsSet::one(CapsPattern::Raster(RasterPattern {
            crs: SetField::one(self.0),
            ..RasterPattern::default()
        })))
    }

    fn plan(&self, out: &WindowReq) -> WindowReq {
        *out
    }

    fn compute(&self, out: &WindowReq, input: &Chunk) -> geoplumb::Result<Chunk> {
        Ok(Chunk::Raster(input.raster()?.crop_to(&out.bbox)))
    }
}

fn pull_layer_image(request: &LayerRequest, channels: usize) -> Result<GeoreferencedImage, String> {
    let time = parse_time(request.time.as_deref())?;
    let path = &request.layer_file;
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let config = Config::parse(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    let Some(layer) = config.layers.iter().find(|l| l.name == request.layer) else {
        let names: Vec<&str> = config.layers.iter().map(|l| l.name.as_str()).collect();
        return Err(format!(
            "{} has no layer named {}, it has: {}",
            path.display(),
            request.layer,
            names.join(", ")
        ));
    };
    let LayerGraph {
        mut graph, node, ..
    } = build_layer_graph(layer)?;
    let node = graph.add_transform(node, Box::new(CrsDemand(Crs(request.crs))));
    let engine = Engine::new(graph, LAYER_CACHE_BYTES).map_err(|e| e.to_string())?;
    let runtime = tokio::runtime::Runtime::new().map_err(|e| e.to_string())?;
    let window = WindowReq {
        bbox: request.bbox,
        resolution: request.resolution,
        time,
    };
    let raster = runtime
        .block_on(engine.pull(node, window))
        .and_then(|chunk| chunk.into_raster())
        .map_err(|e| e.to_string())?;

    let band_count = raster.bands.band_count();
    if band_count != channels {
        return Err(format!(
            "the layer has {band_count} bands and the model takes {channels}"
        ));
    }
    let (height, width) = (raster.height(), raster.width());
    let mut tensor = ImageTensor::zeros((channels, height, width));
    let mut nodata = vec![false; height * width];
    for (channel, band) in raster.bands.bands().iter().enumerate() {
        for (index, &value) in band.data().iter().enumerate() {
            if band.is_nodata(value) {
                nodata[index] = true;
                continue;
            }
            tensor[[channel, index / width, index % width]] = value as f32;
        }
    }
    let nodata_cells = nodata.iter().filter(|&&cell| cell).count();
    Ok(GeoreferencedImage {
        tensor,
        grid: MapGrid {
            min_x: raster.bbox.min_x,
            max_y: raster.bbox.max_y,
            resolution: raster.resolution,
            epsg: raster.crs.0,
        },
        nodata_cells,
    })
}

fn georeference(features: &[VectorFeature], grid: &MapGrid) -> Result<Vec<VectorFeature>, String> {
    let mapped = pixel_to_map(features, grid);
    if grid.epsg == Crs::WGS84.0 {
        return Ok(mapped);
    }
    let source = Crs(grid.epsg).authority();
    let transform = projicio_core::Transform::new(&source, &Crs::WGS84.authority())
        .map_err(|e| format!("{source} to EPSG:4326: {e}"))?;
    map_coordinates(&mapped, |x, y| transform.convert(x, y))
        .map_err(|e| format!("{source} to EPSG:4326: {e}"))
}

#[allow(clippy::too_many_arguments)]
pub fn segment(
    source: &ImageSource,
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

    let (image, grid) = match source {
        ImageSource::File(input) => match load_image(input) {
            Ok(img) => (img, None),
            Err(e) => {
                eprintln!("Failed to load image: {}", e);
                std::process::exit(1);
            }
        },
        ImageSource::Layer(request) => match pull_layer_image(request, config.input.channels) {
            Ok(pulled) => {
                if pulled.nodata_cells > 0 {
                    println!(
                        "{} nodata cells in the pulled window were set to 0",
                        pulled.nodata_cells
                    );
                }
                (pulled.tensor, Some(pulled.grid))
            }
            Err(e) => {
                eprintln!("Failed to pull layer {}: {e}", request.layer);
                std::process::exit(1);
            }
        },
    };

    println!("Loaded image: {:?}", image.shape());
    println!("Using model: {}", config.name);

    let mut pipeline = Pipeline::new(config);
    pipeline.window_config = WindowConfig::new(tile_size, tile_size / 4);
    pipeline.min_area = min_area;

    let result = run_segmentation(&pipeline, &image, engine, onnx_path.as_deref());

    println!("Processed {} tiles", result.tile_results.len());
    println!("Extracted {} features", result.features.len());

    if result.tile_results.is_empty() {
        eprintln!(
            "warning: the image is smaller than --tile-size {tile_size}, so no tile was processed and the output is empty"
        );
    }

    let features = match &grid {
        None => result.features,
        Some(grid) => match georeference(&result.features, grid) {
            Ok(features) => features,
            Err(e) => {
                eprintln!("Failed to georeference features: {e}");
                std::process::exit(1);
            }
        },
    };

    let geojson = to_geojson_string(&features);
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
            Some(p) if Path::new(p).exists() => println!("    Status: weights at {p}"),
            Some(p) => println!("    Status: published, expects {p} in the working directory"),
            None => println!("    Status: planned (no weights published)"),
        }
        println!();
    }
    println!("Weights are published for buildings only, as a release asset to download into the");
    println!("working directory:");
    println!("  {BUILDINGS_WEIGHTS_RELEASE}");
    println!("  panoptes segment --input img.png --output out.geojson --model buildings");
    println!();
    println!("The other entries are metadata only, and naming one runs the threshold heuristic");
    println!("rather than segmentation. For those tasks, pass an ONNX model you supply:");
    println!("  panoptes segment --input img.png --output out.geojson --model path/to/model.onnx");
    println!("See {README_MODEL_SECTION}");
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
