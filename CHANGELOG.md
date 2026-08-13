# Changelog

All notable changes to this project will be documented in this file.

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
