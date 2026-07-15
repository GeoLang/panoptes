//! End-to-end ONNX inference test.
//!
//! Builds a tiny valid ONNX model (a single `Sigmoid` node) by hand-encoding the
//! `ModelProto` protobuf, then runs a bright-blob image through the real pipeline
//! (OnnxEngine -> segmentation -> polygonize) with no network access.
//!
//! Run with: `cargo test -p panoptes-models --features onnx`.
//! Requires a `libonnxruntime.so` (>= 1.20); set `ORT_DYLIB_PATH` if it is not on
//! a standard path.

#![cfg(feature = "onnx")]

use ndarray::Array3;
use panoptes_core::inference::{InferenceResult, ThresholdEngine};
use panoptes_core::model::{ClassDef, InputSpec, ModelConfig, TaskType};
use panoptes_core::onnx::OnnxEngine;
use panoptes_models::pipeline::Pipeline;
use panoptes_raster::window::WindowConfig;
use std::io::Write;

// minimal protobuf wire-format writers (varint + length-delimited)

fn put_varint(buf: &mut Vec<u8>, mut v: u64) {
    loop {
        let mut byte = (v & 0x7f) as u8;
        v >>= 7;
        if v != 0 {
            byte |= 0x80;
        }
        buf.push(byte);
        if v == 0 {
            break;
        }
    }
}

fn tag(field: u32, wire: u32) -> u64 {
    ((field as u64) << 3) | wire as u64
}

fn put_varint_field(buf: &mut Vec<u8>, field: u32, value: u64) {
    put_varint(buf, tag(field, 0));
    put_varint(buf, value);
}

fn put_bytes_field(buf: &mut Vec<u8>, field: u32, data: &[u8]) {
    put_varint(buf, tag(field, 2));
    put_varint(buf, data.len() as u64);
    buf.extend_from_slice(data);
}

fn put_string_field(buf: &mut Vec<u8>, field: u32, s: &str) {
    put_bytes_field(buf, field, s.as_bytes());
}

/// A float tensor `ValueInfoProto` (elem_type only, dynamic shape).
fn value_info(name: &str) -> Vec<u8> {
    // TypeProto.Tensor { elem_type = 1 (FLOAT) }
    let mut tensor_type = Vec::new();
    put_varint_field(&mut tensor_type, 1, 1);
    // TypeProto { tensor_type = <above> }
    let mut type_proto = Vec::new();
    put_bytes_field(&mut type_proto, 1, &tensor_type);
    // ValueInfoProto { name, type }
    let mut vi = Vec::new();
    put_string_field(&mut vi, 1, name);
    put_bytes_field(&mut vi, 2, &type_proto);
    vi
}

/// Build a `ModelProto` with one `Sigmoid` node: input -> output.
fn build_sigmoid_onnx() -> Vec<u8> {
    // NodeProto { input=["input"], output=["output"], op_type="Sigmoid" }
    let mut node = Vec::new();
    put_string_field(&mut node, 1, "input");
    put_string_field(&mut node, 2, "output");
    put_string_field(&mut node, 4, "Sigmoid");

    // GraphProto { node, name, input, output }
    let mut graph = Vec::new();
    put_bytes_field(&mut graph, 1, &node);
    put_string_field(&mut graph, 2, "g");
    put_bytes_field(&mut graph, 11, &value_info("input"));
    put_bytes_field(&mut graph, 12, &value_info("output"));

    // OperatorSetIdProto { version = 13 } (default domain)
    let mut opset = Vec::new();
    put_varint_field(&mut opset, 2, 13);

    // ModelProto { ir_version = 7, opset_import, graph }
    let mut model = Vec::new();
    put_varint_field(&mut model, 1, 7);
    put_bytes_field(&mut model, 8, &opset);
    put_bytes_field(&mut model, 7, &graph);
    model
}

fn seg_config(size: usize) -> ModelConfig {
    ModelConfig {
        name: "test-sigmoid".to_string(),
        version: "1.0".to_string(),
        task: TaskType::Segmentation,
        input: InputSpec {
            channels: 3,
            height: size,
            width: size,
            imagenet_normalize: false,
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
                color: [0, 255, 0],
            },
            ClassDef {
                id: 2,
                name: "other".to_string(),
                color: [0, 0, 255],
            },
        ],
        confidence_threshold: 0.5,
        model_path: None,
    }
}

/// Dark background with a bright blob in channel 1 (the foreground class).
fn blob_image(size: usize, blob: std::ops::Range<usize>) -> Array3<f32> {
    let mut img = Array3::from_elem((3, size, size), 10.0_f32);
    for y in blob.clone() {
        for x in blob.clone() {
            img[[1, y, x]] = 240.0;
        }
    }
    img
}

fn class1_confidences(features: &[panoptes_vector::polygonize::VectorFeature]) -> Vec<f32> {
    features
        .iter()
        .filter(|f| f.class_id == 1)
        .map(|f| f.confidence)
        .collect()
}

#[test]
fn onnx_pipeline_produces_real_confidence() {
    let size = 16;
    let model_bytes = build_sigmoid_onnx();
    let mut model_file = tempfile::Builder::new()
        .suffix(".onnx")
        .tempfile()
        .expect("create tempfile");
    model_file.write_all(&model_bytes).expect("write model");
    model_file.flush().expect("flush model");

    let config = seg_config(size);
    let engine = OnnxEngine::load(model_file.path(), config.clone()).expect("load onnx model");

    let mut pipeline = Pipeline::new(config);
    pipeline.window_config = WindowConfig::new(size, 0);
    pipeline.min_area = 4.0;

    let image = blob_image(size, 4..12);
    let result = pipeline.process(&image, &engine);

    // correct output shape: the tile's segmentation mask matches the tile size
    assert_eq!(result.tile_results.len(), 1);
    match &result.tile_results[0].inference {
        InferenceResult::Segmentation { mask, confidence } => {
            assert_eq!(mask.shape(), &[size, size]);
            assert_eq!(confidence.shape(), &[size, size]);
        }
        other => panic!("expected segmentation, got {other:?}"),
    }

    // polygonize produces at least one polygon over the blob
    let onnx_conf = class1_confidences(&result.features);
    assert!(
        !onnx_conf.is_empty(),
        "expected >= 1 foreground polygon from the ONNX model"
    );

    // every confidence is a real probability in (0, 1), not the old hardcoded 1.0
    for &c in &onnx_conf {
        assert!(c > 0.0 && c < 1.0, "onnx confidence {c} not in (0,1)");
        assert!((c - 1.0).abs() > 1e-3, "onnx confidence must not be 1.0");
    }

    // the threshold engine on the same image yields a different confidence
    let threshold = ThresholdEngine::new(vec![128.0]);
    let threshold_result = pipeline.process(&image, &threshold);
    let threshold_conf = class1_confidences(&threshold_result.features);
    assert!(
        !threshold_conf.is_empty(),
        "expected the threshold engine to also find a foreground polygon"
    );
    assert!(
        (onnx_conf[0] - threshold_conf[0]).abs() > 1e-2,
        "onnx confidence {} should differ from threshold confidence {}",
        onnx_conf[0],
        threshold_conf[0]
    );
}
