# Panoptes

[![CI](https://github.com/GeoLang/panoptes/actions/workflows/ci.yml/badge.svg)](https://github.com/GeoLang/panoptes/actions)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)

**AI Feature Extraction from Geospatial Imagery**

Panoptes is a Rust library and CLI tool for segmenting satellite and aerial imagery and writing the result out as vectors. Segmentation runs an ONNX model you supply, change detection is pixel differencing. The vectors are bounding boxes of connected components, not feature outlines, see [Limitations](#limitations).

## Experimental: no trained weights are published

This repo is experimental. No model weights ship with it, none are published anywhere,
and nothing in the catalog is trained. Segmentation does not work out of the box.

To run real inference you must supply your own ONNX segmentation model and pass it with
`--model path/to/model.onnx`, in a build made with `--features onnx`, on a machine with
ONNX Runtime >= 1.20 installed. [Model contract](#model-contract) states the exact input
and output shapes the file has to have.

Without a model file, `panoptes segment` runs a threshold heuristic instead. It is a
smoke test for the tiling, polygonization and GeoJSON path, not feature extraction, and
its output is not a prediction about the imagery.

## Features

- **Semantic Segmentation** — Per-pixel classification, driven by a user-supplied ONNX model
- **Change Detection** — Temporal comparison by pixel differencing, no model involved
- **Vector Output** — Automatic polygonization of predictions to GeoJSON, as one bounding box per connected component
- **Sliding Window** — Tiled processing with configurable overlap, so the model input is bounded whatever the image size
- **Quality Metrics** — IoU, mean IoU, and pixel accuracy against ground truth
- **GDAL-Free** — Pure Rust image decoding, no system dependencies for the core build

## Architecture

```
panoptes-core       Core types: tensors, model configs, inference traits, metrics
panoptes-raster     Tile I/O, sliding windows, pyramids, change detection
panoptes-vector     Polygonization, simplification, GeoJSON export
panoptes-models     Model catalog (metadata only) and processing pipeline
panoptes-cli        Command-line interface
```

## Quick Start

```bash
# Run your own ONNX segmentation model (build with --features onnx)
panoptes segment --input image.png --output out.geojson --model path/to/model.onnx

# Detect changes between two dates
panoptes change --before 2022.tif --after 2024.tif --output changes.geojson

# List catalog models (metadata only, see below)
panoptes models

# Evaluate prediction accuracy
panoptes evaluate --prediction pred.tif --ground-truth gt.tif --num-classes 5
```

## Inference

`panoptes segment` chooses an engine and always reports which one ran:

- `--model path/to.onnx` runs the model through ONNX Runtime (build with `--features onnx`).
- A catalog name (`buildings`, `roads`, ...) runs ONNX only if that model's weights exist
  locally. None are published, so these always fall back to the threshold heuristic, and
  say so on stderr.
- `--engine threshold` forces the heuristic. `--engine onnx` requires a model file and
  exits with an error if there is none.

ONNX support is gated behind the `onnx` cargo feature and uses ort's dynamically loaded
runtime. Install ONNX Runtime (>= 1.20) and, if it is not on a standard path, set
`ORT_DYLIB_PATH` to the `libonnxruntime.so`.

Per-polygon `confidence` in the GeoJSON output is the mean model probability over the
polygon's pixels (the threshold engine reports the mean above-threshold intensity).

### Model contract

A `--model` file has to be an ONNX graph that takes one float tensor and returns one
float tensor.

Input: `[1, 3, tile_size, tile_size]` in NCHW order, where `tile_size` is the
`--tile-size` value (default 512). Pixels are scaled to `[0, 1]`, or ImageNet-normalized
if you pass `--imagenet-normalize`. The input is bound by position, so its name does not
matter.

Output, first output of the graph, one of:

- `[1, 1, H, W]` or `[1, H, W]`, read as a foreground probability map. A pixel is
  foreground when its value is at or above `--confidence` (default 0.5).
- `[1, K, H, W]`, read as per-class logits. Softmax over the channel axis, then argmax
  picks the class.

Anything else fails with an unexpected output shape error.

A `--model` path is always treated as a two-class (background, foreground) model, so a
`[1, K, H, W]` output with `K > 2` runs, but only classes 0 and 1 are polygonized and the
rest are dropped. Both classes are written to the GeoJSON, background included, tagged by
`class_id`.

## Limitations

- **No trained weights, so there is no accuracy number for anything here.** Nothing in
  this repo has been evaluated against a benchmark dataset.
- **Every polygon written is an axis-aligned rectangle.** Polygonization flood-fills
  connected components and emits each one's 5-point bounding box, so no GeoJSON panoptes
  writes traces a feature outline. Douglas-Peucker simplification is implemented in
  `panoptes-vector` and nothing calls it.
- There is no multi-resolution or scale-invariant detection. The pyramid builder in
  `panoptes-raster` has no caller: inference runs at one scale and change detection is a
  single-scale pixel difference.
- Tiling bounds the model input, not memory. Loading decodes the whole file into a
  full-image f32 array before any tiling, so a 50k x 50k image needs roughly 30 GB.
- Tile-parallel inference across rayon, with overlap blending on merge, exists in the
  library and has no caller outside its own test. The CLI path is a sequential map that
  polygonizes each tile on its own.
- Tiles are whole tiles only. An image smaller than `--tile-size` in either dimension
  yields no tiles at all, and edge remainders are dropped rather than padded.
- **Output coordinates are pixel space, not world coordinates.** Nothing reads a
  geotransform, so the GeoJSON carries no CRS and its coordinates are image row/column
  values. Reprojecting to real-world coordinates is on you for now.
- COG reading does not decode pixels and nothing reaches it. The module parses the IFD
  chain and tile offsets, and a tile read hands back the raw bytes: the compression field
  is recorded and never acted on, so there is no decompression and no pixel decode. It is
  local-file only too, with no HTTP or S3 client. Image loading goes through the `image`
  crate instead, so the CLI never enters this path.
- Satellite preprocessing (DN to TOA, DOS1, pan-sharpening, the spectral indices, band
  compositing) is unreachable. `SatelliteImage` is constructed nowhere in the workspace,
  including tests, and the only file reader calls `to_rgb8()`, so every image becomes
  3-channel 8-bit RGB and no NIR or SWIR band can enter. The `Sensor` enum is a bare tag
  with no calibration coefficients, whatever `satellite.rs`'s rustdoc says about
  Sentinel-2 and Landsat 8/9.
- Object detection exists as a library result type, produced only by the threshold
  heuristic. There is no detection CLI command and no non-maximum suppression.
- `panoptes-raster::explain` provides occlusion sensitivity and a saliency map derived
  from confidence. Grad-CAM is declared in the enum but not implemented.

## Catalog Models

Planned entries, and there is no work underway on them. These are metadata only (input
size, classes, thresholds), **no weights exist**. Naming one of them runs the threshold
heuristic. Use `--model <file.onnx>` to run a segmentation model you supply.

| Model | Task | Classes | Status |
|-------|------|---------|--------|
| `panoptes-buildings-v1` | Segmentation | background, building | planned |
| `panoptes-roads-v1` | Segmentation | background, road | planned |
| `panoptes-landcover-v1` | Segmentation | water, vegetation, bare_soil, built_up, agriculture | planned |
| `panoptes-vegetation-v1` | Segmentation | non_vegetation, trees, shrubs, grass | planned |
| `panoptes-change-v1` | Change Detection | no_change, change | planned |

## Building

```bash
cargo build --release              # core build, no ONNX
cargo build --release --features onnx -p panoptes-cli   # with ONNX inference
```

## Testing

```bash
cargo test                                     # 44 tests, no ONNX
cargo test -p panoptes-models --features onnx  # end-to-end ONNX pipeline test
```

The ONNX test builds a tiny synthetic model in-process and runs it through the real
engine, so it proves the inference path end to end without any weights. It needs a local
ONNX Runtime, which CI installs so the test runs on every push.

## License

AGPL-3.0-or-later, see [LICENSE](LICENSE).

Copyright (C) 2026 Grok Image Compression Inc.
