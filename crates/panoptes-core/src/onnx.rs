//! ONNX Runtime inference engine — run real models via ort.
//!
//! Enable with `--features onnx`. Uses ort's `load-dynamic` backend, so the
//! ONNX Runtime shared library is loaded at runtime. Point `ORT_DYLIB_PATH` at
//! a `libonnxruntime.so` (>= 1.20) if it is not on the default search path.

use std::path::Path;
use std::sync::Mutex;

use ndarray::Array3;
use ort::session::Session;
use ort::value::Tensor;
use thiserror::Error;

use crate::inference::{Detection, InferenceEngine, InferenceResult};
use crate::model::{ModelConfig, TaskType};
use crate::tensor::{
    ImageTensor, ProbabilityMap, argmax_mask, confidence_map, imagenet_normalize, softmax_chw,
};

/// Errors from ONNX inference.
#[derive(Debug, Error)]
pub enum OnnxError {
    #[error("ONNX Runtime error: {0}")]
    Runtime(#[from] ort::Error),
    #[error("Model file not found: {0}")]
    ModelNotFound(String),
    #[error("ONNX Runtime library not found: {0}")]
    RuntimeNotFound(String),
    #[error("Unexpected output shape: {0}")]
    OutputShape(String),
}

/// Resolve the ONNX Runtime shared library before ort touches it.
///
/// ort's `load-dynamic` backend deadlocks (rc.12) if the dylib fails to load: its error
/// path re-enters the API init lock. So validate a real path up front. An explicit
/// `ORT_DYLIB_PATH` wins; otherwise probe common locations and set it.
fn ensure_runtime() -> Result<(), OnnxError> {
    if let Some(existing) = std::env::var_os("ORT_DYLIB_PATH") {
        let path = std::path::PathBuf::from(&existing);
        if existing.is_empty() || path.exists() {
            return Ok(());
        }
        return Err(OnnxError::RuntimeNotFound(format!(
            "ORT_DYLIB_PATH points to a missing file: {}",
            path.display()
        )));
    }

    let candidates = [
        "/lib64/libonnxruntime.so",
        "/lib64/libonnxruntime.so.1",
        "/usr/lib64/libonnxruntime.so",
        "/usr/lib64/libonnxruntime.so.1",
        "/usr/lib/x86_64-linux-gnu/libonnxruntime.so",
        "/usr/lib/x86_64-linux-gnu/libonnxruntime.so.1",
        "/usr/local/lib/libonnxruntime.so",
        "/usr/local/lib/libonnxruntime.so.1",
    ];
    for c in candidates {
        if std::path::Path::new(c).exists() {
            // safe: set before any ort call; load happens once behind ort's init lock
            unsafe { std::env::set_var("ORT_DYLIB_PATH", c) };
            return Ok(());
        }
    }

    Err(OnnxError::RuntimeNotFound(
        "could not find libonnxruntime (>= 1.20); install ONNX Runtime or set ORT_DYLIB_PATH"
            .to_string(),
    ))
}

/// Leak one reference to the global ONNX Runtime environment.
///
/// ort releases the environment from a `.fini_array` handler at process exit, but with the
/// dynamically loaded runtime that release can run after the runtime's own destructors and
/// segfault. Holding an extra strong reference keeps the refcount above zero so the final
/// release is skipped; the environment is reclaimed by the OS at exit anyway.
fn leak_ort_environment() {
    use std::sync::Once;
    static LEAK: Once = Once::new();
    LEAK.call_once(|| {
        if let Ok(env) = ort::environment::current() {
            std::mem::forget(env);
        }
    });
}

/// ONNX Runtime-based inference engine.
pub struct OnnxEngine {
    // ort's Session::run takes &mut self, so guard it for the &self trait method.
    session: Mutex<Session>,
    config: ModelConfig,
}

impl OnnxEngine {
    /// Load an ONNX model from disk (CPU execution provider).
    pub fn load(model_path: &Path, config: ModelConfig) -> Result<Self, OnnxError> {
        if !model_path.exists() {
            return Err(OnnxError::ModelNotFound(model_path.display().to_string()));
        }

        ensure_runtime()?;
        let session = Session::builder()?.commit_from_file(model_path)?;
        leak_ort_environment();
        Ok(Self {
            session: Mutex::new(session),
            config,
        })
    }

    /// Run inference on a single CHW image tensor.
    pub fn infer(&self, input: &ImageTensor) -> Result<InferenceResult, OnnxError> {
        let shape = input.shape();
        let (c, h, w) = (shape[0], shape[1], shape[2]);

        // preprocess into a normalized NCHW float buffer
        let normalized = if self.config.input.imagenet_normalize {
            imagenet_normalize(input)
        } else {
            input / 255.0
        };
        let data: Vec<f32> = normalized.iter().copied().collect();
        let input_tensor = Tensor::from_array((vec![1i64, c as i64, h as i64, w as i64], data))?;

        // the run outputs borrow the session guard, so extract everything before dropping it
        let mut session = self.session.lock().expect("ONNX session mutex poisoned");
        let outputs = session.run(ort::inputs![input_tensor])?;
        let (out_shape, out_data) = outputs[0].try_extract_tensor::<f32>()?;
        let out_shape: &[i64] = out_shape;

        match self.config.task {
            TaskType::Segmentation | TaskType::ChangeDetection => {
                self.parse_segmentation(out_shape, out_data)
            }
            TaskType::Detection => self.parse_detection(out_shape, out_data),
            TaskType::Classification => self.parse_classification(out_data),
        }
    }

    /// Parse a segmentation output into a class mask + probability map.
    ///
    /// Accepts either a single-channel foreground-probability map `[N,1,H,W]` (or
    /// `[N,H,W]`) or multi-class logits `[N,K,H,W]`.
    fn parse_segmentation(
        &self,
        out_shape: &[i64],
        out_data: &[f32],
    ) -> Result<InferenceResult, OnnxError> {
        let (channels, out_h, out_w) = match *out_shape {
            [_, k, hh, ww] => (k as usize, hh as usize, ww as usize),
            [_, hh, ww] => (1, hh as usize, ww as usize),
            _ => {
                return Err(OnnxError::OutputShape(format!("{out_shape:?}")));
            }
        };

        if channels <= 1 {
            // single-channel foreground probability map
            let threshold = self.config.confidence_threshold;
            let mut mask = ndarray::Array2::zeros((out_h, out_w));
            let mut confidence: ProbabilityMap = ndarray::Array2::zeros((out_h, out_w));
            for y in 0..out_h {
                for x in 0..out_w {
                    let p = out_data[y * out_w + x];
                    let foreground = p >= threshold;
                    mask[[y, x]] = foreground as u8;
                    // confidence of the predicted class
                    confidence[[y, x]] = if foreground { p } else { 1.0 - p };
                }
            }
            return Ok(InferenceResult::Segmentation { mask, confidence });
        }

        // multi-class logits: softmax over the channel axis, then argmax
        let logits = Array3::from_shape_fn((channels, out_h, out_w), |(ci, y, x)| {
            out_data[(ci * out_h + y) * out_w + x]
        });
        let probs = softmax_chw(&logits);
        let mask = argmax_mask(&probs);
        let confidence = confidence_map(&probs);

        Ok(InferenceResult::Segmentation { mask, confidence })
    }

    fn parse_detection(
        &self,
        out_shape: &[i64],
        out_data: &[f32],
    ) -> Result<InferenceResult, OnnxError> {
        // output: [1, N, 6] (x1, y1, x2, y2, confidence, class_id)
        let (num_detections, stride) = match *out_shape {
            [_, n, s] => (n as usize, s as usize),
            _ => return Err(OnnxError::OutputShape(format!("{out_shape:?}"))),
        };
        if stride < 6 {
            return Err(OnnxError::OutputShape(format!("{out_shape:?}")));
        }

        let mut detections = Vec::new();
        for i in 0..num_detections {
            let base = i * stride;
            let conf = out_data[base + 4];
            if conf >= self.config.confidence_threshold {
                detections.push(Detection {
                    class_id: out_data[base + 5] as u8,
                    confidence: conf,
                    bbox: [
                        out_data[base],
                        out_data[base + 1],
                        out_data[base + 2],
                        out_data[base + 3],
                    ],
                });
            }
        }

        Ok(InferenceResult::Detection { detections })
    }

    fn parse_classification(&self, out_data: &[f32]) -> Result<InferenceResult, OnnxError> {
        let mut max_idx = 0u8;
        let mut max_val = f32::NEG_INFINITY;
        for (i, &v) in out_data.iter().enumerate() {
            if v > max_val {
                max_val = v;
                max_idx = i as u8;
            }
        }

        Ok(InferenceResult::Classification {
            class_id: max_idx,
            confidence: max_val,
        })
    }
}

impl InferenceEngine for OnnxEngine {
    fn predict(&self, input: &ImageTensor, _config: &ModelConfig) -> InferenceResult {
        match self.infer(input) {
            Ok(result) => result,
            Err(e) => {
                eprintln!("ONNX inference failed: {e}");
                InferenceResult::Detection { detections: vec![] }
            }
        }
    }

    fn name(&self) -> &str {
        "onnx-runtime"
    }
}
