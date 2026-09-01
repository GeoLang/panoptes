# Panoptes

[![CI](https://github.com/GeoLang/panoptes/actions/workflows/ci.yml/badge.svg)](https://github.com/GeoLang/panoptes/actions)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)

**AI Feature Extraction from Geospatial Imagery**

Panoptes is a Rust library and CLI tool for segmenting satellite and aerial imagery and writing the result out as vectors. Segmentation runs an ONNX model, either the published building weights or one you supply; change detection is pixel differencing. The vectors are traced pixel-edge outlines of connected components, see [Limitations](#limitations).

## Model weights

Building segmentation is the one task with published weights. The asset
`panoptes-buildings-v1.onnx` (~80 MB) is attached to the
[`weights-buildings-v1` release](https://github.com/GeoLang/panoptes/releases/tag/weights-buildings-v1).
It is a UNet++ with an EfficientNet-B4 encoder, exported to ONNX opset 17 from the
Hugging Face model
[`giswqs/whu-building-unetplusplus-efficientnet-b4`](https://huggingface.co/giswqs/whu-building-unetplusplus-efficientnet-b4),
whose weights are Apache-2.0. Training data is the WHU Building Dataset, which states no
license and asks to be cited.

Download the asset into the working directory, then:

```bash
panoptes segment --input img.png --output out.geojson --model buildings

# or point at the file wherever it is
panoptes segment --input img.png --output out.geojson --model /path/to/panoptes-buildings-v1.onnx
```

Tile size is 512 and pixels are scaled to `[0, 1]`, so do not pass `--imagenet-normalize`.
The build needs `--features onnx` and a local ONNX Runtime >= 1.20.

The model's upstream author reports IoU 0.9054 and Dice 0.9503 on the 1,228-tile WHU test
set. That is their number, not ours. The training imagery is 0.3 m aerial, so on
lower-resolution imagery (1 m NAIP, say) it still finds large buildings but misses more
small houses.

Roads, landcover, vegetation and change have no weights, published or planned. Naming one
runs a threshold heuristic instead: a smoke test for the tiling, polygonization and
GeoJSON path, not feature extraction, and its output is not a prediction about the
imagery. To run one of those tasks, supply your own ONNX segmentation model with
`--model path/to/model.onnx`. [Model contract](#model-contract) states the exact input and
output shapes the file has to have.

## Features

- **Semantic Segmentation** — Per-pixel classification, driven by the published building weights or an ONNX model you supply
- **Change Detection** — Temporal comparison by pixel differencing, no model involved
- **Vector Output** — Automatic polygonization of predictions to GeoJSON, as one traced outline per connected component
- **Sliding Window** — Tiled processing with configurable overlap, so the model input is bounded whatever the image size
- **Quality Metrics** — IoU, mean IoU, and pixel accuracy against ground truth
- **GDAL-Free** — Pure Rust image decoding, no system dependencies for the core build

## Architecture

```
panoptes-core       Core types: tensors, model configs, inference traits, metrics
panoptes-raster     Tile I/O, sliding windows, pyramids, change detection
panoptes-vector     Polygonization, simplification, GeoJSON export
panoptes-models     Model catalog and processing pipeline
panoptes-cli        Command-line interface
```

## Quick Start

```bash
# Segment buildings, with panoptes-buildings-v1.onnx downloaded into this directory
panoptes segment --input image.png --output out.geojson --model buildings

# Run your own ONNX segmentation model (build with --features onnx)
panoptes segment --input image.png --output out.geojson --model path/to/model.onnx

# Detect changes between two dates
panoptes change --before 2022.tif --after 2024.tif --output changes.geojson

# List catalog models and where their weights come from
panoptes models

# Evaluate prediction accuracy
panoptes evaluate --prediction pred.tif --ground-truth gt.tif --num-classes 5
```

## Inference

`panoptes segment` chooses an engine and always reports which one ran:

- `--model path/to.onnx` runs the model through ONNX Runtime (build with `--features onnx`).
- A catalog name (`buildings`, `roads`, ...) runs ONNX only if that model's weights exist
  locally. `buildings` needs `panoptes-buildings-v1.onnx` in the working directory; the
  other names have no weights, so they fall back to the threshold heuristic and say so on
  stderr.
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

- **The only accuracy number here is the buildings model author's**: IoU 0.9054 on the
  WHU test set, reported upstream and not reproduced by us. Nothing else in this repo has
  been evaluated against a benchmark dataset, and the other catalog entries have no
  weights at all.
- **Outlines are pixel edges and holes are filled.** Polygonization traces each
  connected component's outer boundary along pixel edges, so polygons are stair-stepped
  at pixel resolution and carry no interior rings: a building with a courtyard comes out
  solid. Douglas-Peucker simplification is implemented in `panoptes-vector` and nothing
  calls it.
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

`panoptes-buildings-v1` has published weights, see [Model weights](#model-weights). The
rest are planned entries with no work underway: metadata only (input size, classes,
thresholds), **no weights exist**. Naming one of them runs the threshold heuristic. Use
`--model <file.onnx>` to run a segmentation model you supply.

| Model | Task | Classes | Status |
|-------|------|---------|--------|
| `panoptes-buildings-v1` | Segmentation | background, building | published |
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
