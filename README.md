# Panoptes

[![CI](https://github.com/GeoLang/panoptes/actions/workflows/ci.yml/badge.svg)](https://github.com/GeoLang/panoptes/actions)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)

**AI Feature Extraction from Geospatial Imagery**

Panoptes is a Rust library and CLI tool for extracting vector features from satellite and aerial imagery using AI-powered segmentation and change analysis.

**No model weights ship with this repo.** Bring your own ONNX segmentation model and
pass it with `--model path/to/model.onnx`. Without one, `panoptes segment` falls back to
a threshold heuristic that is useful for smoke-testing the pipeline, not for real
extraction. See [Inference](#inference) below.

## Features

- **Semantic Segmentation** — Per-pixel classification, driven by a user-supplied ONNX model
- **Change Detection** — Temporal comparison to identify what changed between two images
- **Vector Output** — Automatic polygonization of predictions to GeoJSON, with Douglas-Peucker simplification
- **Multi-Resolution Analysis** — Image pyramid processing for scale-invariant detection
- **Sliding Window** — Efficient tiled processing of large imagery with configurable overlap
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
# Run a real ONNX segmentation model
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
- A catalog name (`buildings`, `roads`, ...) runs ONNX only if that model's weights
  exist locally; none are published yet, so these fall back to the threshold heuristic.
- `--engine threshold` forces the heuristic; `--engine onnx` requires a model file.

ONNX support is gated behind the `onnx` cargo feature and uses ort's dynamically loaded
runtime. Install ONNX Runtime (>= 1.20) and, if it is not on a standard path, set
`ORT_DYLIB_PATH` to the `libonnxruntime.so`.

Per-polygon `confidence` in the GeoJSON output is the mean model probability over the
polygon's pixels (the threshold engine reports the mean above-threshold intensity).

## Limitations

- **Output coordinates are pixel space, not world coordinates.** Nothing reads a
  geotransform, so the GeoJSON carries no CRS and its coordinates are image row/column
  values. Reprojecting to real-world coordinates is on you for now.
- COG reading is local-file only. There is no HTTP or S3 client, so remote range
  requests are not supported.
- Object detection exists as a library result type, produced only by the threshold
  heuristic. There is no detection CLI command and no non-maximum suppression.
- `panoptes-raster::explain` provides occlusion sensitivity and a saliency map derived
  from confidence. Grad-CAM is declared in the enum but not implemented.

## Catalog Models

Planned entries. These are metadata only (input size, classes, thresholds); **no weights
are published yet**. Use `--model <file.onnx>` to run any user-provided segmentation model.

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
cargo test                                     # 45 tests, no ONNX
cargo test -p panoptes-models --features onnx  # end-to-end ONNX pipeline test
```

## License

AGPL-3.0-or-later, see [LICENSE](LICENSE).

Copyright (C) 2026 Grok Image Compression Inc.
