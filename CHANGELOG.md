# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased] - 2026-09-01

### Added

- building segmentation weights, published as the `panoptes-buildings-v1.onnx` asset of
  the `weights-buildings-v1` release: unet++ with an efficientnet-b4 encoder, exported
  from the apache-2.0 hugging face model `giswqs/whu-building-unetplusplus-efficientnet-b4`
  and trained on the whu building dataset
- `scripts/export_buildings_onnx.py`, the torch to onnx export that produced the asset
- readme, landing page and `panoptes models` say where the buildings weights come from,
  what they were trained on, and that roads, landcover, vegetation and change still have
  none

### Fixed

- the buildings catalog entry claimed imagenet normalization; the model is trained on
  plain [0, 1] scaling, so predictions through `--model buildings` were fed the wrong
  input range

### Changed

- the buildings catalog entry's `model_path` is `panoptes-buildings-v1.onnx`, resolved
  from the working directory, so `--model buildings` runs onnx once the asset is
  downloaded and falls back to the threshold heuristic until then
- polygonization traces each component's outer boundary along pixel edges instead of
  emitting its bounding box; holes are still filled and simplification is still uncalled

## [Unreleased] - 2026-08-13

### Changed

- readme leads with the experimental label: no trained weights exist, segmentation does
  not work out of the box, and here is exactly what you must supply
- readme documents the model contract, the input and output tensor shapes a `--model`
  file has to have
- segment warns on stderr, naming the model and pointing at the readme, when it falls
  back to the threshold heuristic because no weights exist
- segment warns when the image is smaller than the tile size and the output is empty
- `--engine onnx` with no model file, and an unknown `--model`, point at the readme
- cli help, crate docs and the landing page stop implying a pre-trained model
- readme records that change detection is pixel differencing and that no accuracy number
  exists for anything here
- workspace description says what panoptes does instead of "geospatial monitoring"
- ci runs the onnx inference test: a job installs ONNX Runtime 1.20.1 from the official
  release and runs clippy and the test suite with `--features onnx`, so the end-to-end
  inference path is covered instead of compiling to nothing

## [0.1.0] - 2026-05-30

### Added

- Initial release.
