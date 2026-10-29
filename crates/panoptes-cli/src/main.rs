//! # panoptes-cli
//!
//! Command-line interface for AI feature extraction from geospatial imagery.

use clap::{ArgGroup, Parser, Subcommand};
use geoplumb::Bbox;
use std::path::PathBuf;

mod commands;

use commands::{ImageSource, LayerRequest};

#[derive(Parser)]
#[command(name = "panoptes")]
#[command(
    about = "AI feature extraction from geospatial imagery (experimental, weights are a separate download)"
)]
#[command(version)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Segment an image with the downloaded buildings weights or an ONNX model you supply.
    #[command(group(
        ArgGroup::new("image_source")
            .required(true)
            .multiple(false)
            .args(["input", "layer_file"])
    ))]
    Segment {
        /// Input image path.
        #[arg(short, long)]
        input: Option<PathBuf>,
        #[arg(
            long,
            requires = "layer",
            requires = "bbox",
            requires = "resolution",
            help = "geoplumb-server layer file to read the image from, in place of --input"
        )]
        layer_file: Option<PathBuf>,
        #[arg(
            long,
            requires = "layer_file",
            help = "Name of the layer in --layer-file"
        )]
        layer: Option<String>,
        #[arg(
            long,
            requires = "layer_file",
            allow_hyphen_values = true,
            value_parser = parse_bbox,
            value_name = "MIN_X,MIN_Y,MAX_X,MAX_Y",
            help = "Window to pull from the layer, in --crs units"
        )]
        bbox: Option<Bbox>,
        #[arg(
            long,
            default_value = "3857",
            help = "EPSG code of the grid the window is pulled on"
        )]
        crs: u32,
        #[arg(
            long,
            requires = "layer_file",
            value_parser = parse_resolution,
            help = "Pixel size in --crs units, near the resolution the model was trained at"
        )]
        resolution: Option<f64>,
        #[arg(
            long,
            requires = "layer_file",
            value_name = "START/END",
            help = "RFC 3339 interval for a time-varying layer, the layer's own when absent"
        )]
        time: Option<String>,
        /// Output GeoJSON path.
        #[arg(short, long)]
        output: PathBuf,
        /// Path to a .onnx model, or a catalog name (buildings, roads, vegetation, landcover). Only buildings has weights, and only once panoptes-buildings-v1.onnx is in the working directory; the rest fall back to the threshold heuristic.
        #[arg(short, long, default_value = "buildings")]
        model: String,
        /// Inference engine: auto (onnx if a model file is available, else threshold), onnx, threshold.
        #[arg(short, long, default_value = "auto")]
        engine: String,
        /// Tile size in pixels.
        #[arg(long, default_value = "512")]
        tile_size: usize,
        /// Minimum feature area in pixels.
        #[arg(long, default_value = "10")]
        min_area: f64,
        /// Confidence threshold for a user ONNX model's foreground class.
        #[arg(long, default_value = "0.5")]
        confidence: f32,
        /// Apply ImageNet normalization for a user ONNX model (else scale to \[0,1\]).
        #[arg(long)]
        imagenet_normalize: bool,
    },
    /// Detect change between two images.
    Change {
        /// Before image path.
        #[arg(long)]
        before: PathBuf,
        /// After image path.
        #[arg(long)]
        after: PathBuf,
        /// Output GeoJSON path.
        #[arg(short, long)]
        output: PathBuf,
        /// Change detection threshold (0.0-1.0).
        #[arg(short, long, default_value = "0.3")]
        threshold: f32,
    },
    /// List available models.
    Models,
    /// Evaluate predictions against ground truth.
    Evaluate {
        /// Prediction mask image.
        #[arg(short, long)]
        prediction: PathBuf,
        /// Ground truth mask image.
        #[arg(short, long)]
        ground_truth: PathBuf,
        /// Number of classes.
        #[arg(short, long)]
        num_classes: usize,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Segment {
            input,
            layer_file,
            layer,
            bbox,
            crs,
            resolution,
            time,
            output,
            model,
            engine,
            tile_size,
            min_area,
            confidence,
            imagenet_normalize,
        } => {
            let source = match (input, layer_file) {
                (Some(input), None) => ImageSource::File(input),
                (None, Some(layer_file)) => ImageSource::Layer(LayerRequest {
                    layer_file,
                    layer: layer.expect("clap requires --layer with --layer-file"),
                    bbox: bbox.expect("clap requires --bbox with --layer-file"),
                    crs,
                    resolution: resolution.expect("clap requires --resolution with --layer-file"),
                    time,
                }),
                _ => unreachable!("the image_source group takes exactly one of its two arguments"),
            };
            commands::segment(
                &source,
                &output,
                &model,
                &engine,
                tile_size,
                min_area,
                confidence,
                imagenet_normalize,
            )
        }
        Commands::Change {
            before,
            after,
            output,
            threshold,
        } => commands::change(&before, &after, &output, threshold),
        Commands::Models => commands::list_models(),
        Commands::Evaluate {
            prediction,
            ground_truth,
            num_classes,
        } => commands::evaluate(&prediction, &ground_truth, num_classes),
    }
}

fn parse_bbox(text: &str) -> Result<Bbox, String> {
    let values = text
        .split(',')
        .map(|value| {
            value
                .trim()
                .parse::<f64>()
                .map_err(|e| format!("{value:?}: {e}"))
        })
        .collect::<Result<Vec<f64>, String>>()?;
    let &[min_x, min_y, max_x, max_y] = values.as_slice() else {
        return Err(format!(
            "expected four comma-separated numbers, got {}",
            values.len()
        ));
    };
    let ordered = min_x < max_x && min_y < max_y;
    if !ordered || values.iter().any(|value| !value.is_finite()) {
        return Err("expected finite numbers with each min below its max".to_string());
    }
    Ok(Bbox::new(min_x, min_y, max_x, max_y))
}

fn parse_resolution(text: &str) -> Result<f64, String> {
    let resolution = text.parse::<f64>().map_err(|e| e.to_string())?;
    if !(resolution.is_finite() && resolution > 0.0) {
        return Err(format!("{resolution} is not a positive pixel size"));
    }
    Ok(resolution)
}
