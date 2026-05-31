# Panoptes

[![CI](https://github.com/GeoLang/panoptes/actions/workflows/ci.yml/badge.svg)](https://github.com/GeoLang/panoptes/actions)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)

**AI Feature Extraction from Geospatial Imagery**

Panoptes is a Rust library and CLI tool for extracting vector features from satellite and aerial imagery using AI-powered segmentation, detection, and change analysis.

## Features

- **Semantic Segmentation** — Per-pixel classification of buildings, roads, vegetation, water bodies
- **Object Detection** — Bounding-box extraction of discrete features
- **Change Detection** — Temporal comparison to identify what changed between two images
- **Vector Output** — Automatic polygonization of predictions to GeoJSON
- **Multi-Resolution Analysis** — Image pyramid processing for scale-invariant detection
- **Sliding Window** — Efficient tiled processing of large imagery with configurable overlap
- **Quality Metrics** — IoU, pixel accuracy, and confidence scoring
- **GDAL-Free** — Pure Rust image decoding (no system dependencies)

## Architecture

```
panoptes-core       Core types: tensors, model configs, inference traits, metrics
panoptes-raster     Tile I/O, sliding windows, pyramids, change detection
panoptes-vector     Polygonization, simplification, GeoJSON export
panoptes-models     Pre-built model catalog and processing pipeline
panoptes-cli        Command-line interface
```

## Quick Start

```bash
# Segment buildings from an aerial image
panoptes segment --input image.tif --output buildings.geojson --model buildings

# Detect changes between two dates
panoptes change --before 2022.tif --after 2024.tif --output changes.geojson

# List available models
panoptes models

# Evaluate prediction accuracy
panoptes evaluate --prediction pred.tif --ground-truth gt.tif --num-classes 5
```

## Available Models

| Model | Task | Classes |
|-------|------|---------|
| `panoptes-buildings-v1` | Segmentation | background, building |
| `panoptes-roads-v1` | Segmentation | background, road |
| `panoptes-landcover-v1` | Segmentation | water, vegetation, bare_soil, built_up, agriculture |
| `panoptes-vegetation-v1` | Segmentation | non_vegetation, trees, shrubs, grass |
| `panoptes-change-v1` | Change Detection | no_change, change |

## Building

```bash
cargo build --release
```

## Testing

```bash
cargo test
```

## License

AGPL-3.0-or-later
