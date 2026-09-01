//! # panoptes-cli
//!
//! Command-line interface for AI feature extraction from geospatial imagery.

use clap::{Parser, Subcommand};
use std::path::PathBuf;

mod commands;

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
    Segment {
        /// Input image path.
        #[arg(short, long)]
        input: PathBuf,
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
            output,
            model,
            engine,
            tile_size,
            min_area,
            confidence,
            imagenet_normalize,
        } => commands::segment(
            &input,
            &output,
            &model,
            &engine,
            tile_size,
            min_area,
            confidence,
            imagenet_normalize,
        ),
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
