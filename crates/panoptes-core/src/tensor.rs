//! Tensor utilities — wrapping ndarray for image tensor operations.

use ndarray::{Array2, Array3, Array4};

/// Image tensor in CHW format (channels, height, width).
pub type ImageTensor = Array3<f32>;

/// Batch tensor in NCHW format (batch, channels, height, width).
pub type BatchTensor = Array4<f32>;

/// Segmentation mask (height x width) with class indices.
pub type SegmentationMask = Array2<u8>;

/// Probability map (height x width) with confidence scores.
pub type ProbabilityMap = Array2<f32>;

/// Normalize pixel values from [0, 255] to [0.0, 1.0].
pub fn normalize(tensor: &ImageTensor) -> ImageTensor {
    tensor / 255.0
}

/// Apply ImageNet-style normalization (mean subtraction + std division).
pub fn imagenet_normalize(tensor: &ImageTensor) -> ImageTensor {
    let mean = [0.485_f32, 0.456, 0.406];
    let std = [0.229_f32, 0.224, 0.225];

    let mut result = tensor.clone();
    for c in 0..3.min(tensor.shape()[0]) {
        result
            .slice_mut(ndarray::s![c, .., ..])
            .mapv_inplace(|v| (v / 255.0 - mean[c]) / std[c]);
    }
    result
}

/// Convert HWC (height, width, channels) to CHW format.
pub fn hwc_to_chw(hwc: &Array3<f32>) -> ImageTensor {
    hwc.view().permuted_axes([2, 0, 1]).to_owned()
}

/// Convert CHW to HWC format.
pub fn chw_to_hwc(chw: &ImageTensor) -> Array3<f32> {
    chw.view().permuted_axes([1, 2, 0]).to_owned()
}

/// Create a batch from multiple images.
pub fn stack_batch(images: &[ImageTensor]) -> BatchTensor {
    let n = images.len();
    if n == 0 {
        return Array4::zeros((0, 0, 0, 0));
    }
    let shape = images[0].shape();
    let (c, h, w) = (shape[0], shape[1], shape[2]);
    let mut batch = Array4::zeros((n, c, h, w));
    for (i, img) in images.iter().enumerate() {
        batch.slice_mut(ndarray::s![i, .., .., ..]).assign(img);
    }
    batch
}

/// Turn a single logit into a probability.
pub fn sigmoid(logit: f32) -> f32 {
    1.0 / (1.0 + (-logit).exp())
}

/// Apply softmax along axis 0 (channel axis) of a CHW tensor to get per-pixel class probabilities.
pub fn softmax_chw(tensor: &ImageTensor) -> ImageTensor {
    let shape = tensor.shape();
    let (c, h, w) = (shape[0], shape[1], shape[2]);
    let mut result = Array3::zeros((c, h, w));

    for y in 0..h {
        for x in 0..w {
            let mut max_val = f32::NEG_INFINITY;
            for ci in 0..c {
                max_val = max_val.max(tensor[[ci, y, x]]);
            }
            let mut sum = 0.0_f32;
            for ci in 0..c {
                let exp = (tensor[[ci, y, x]] - max_val).exp();
                result[[ci, y, x]] = exp;
                sum += exp;
            }
            for ci in 0..c {
                result[[ci, y, x]] /= sum;
            }
        }
    }
    result
}

/// Convert softmax output to a class mask (argmax along channel axis).
pub fn argmax_mask(probabilities: &ImageTensor) -> SegmentationMask {
    let shape = probabilities.shape();
    let (c, h, w) = (shape[0], shape[1], shape[2]);
    let mut mask = Array2::zeros((h, w));

    for y in 0..h {
        for x in 0..w {
            let mut best_class = 0u8;
            let mut best_prob = f32::NEG_INFINITY;
            for ci in 0..c {
                if probabilities[[ci, y, x]] > best_prob {
                    best_prob = probabilities[[ci, y, x]];
                    best_class = ci as u8;
                }
            }
            mask[[y, x]] = best_class;
        }
    }
    mask
}

/// Extract confidence map (max probability per pixel) from softmax output.
pub fn confidence_map(probabilities: &ImageTensor) -> ProbabilityMap {
    let shape = probabilities.shape();
    let (c, h, w) = (shape[0], shape[1], shape[2]);
    let mut conf = Array2::zeros((h, w));

    for y in 0..h {
        for x in 0..w {
            let mut max_prob = 0.0_f32;
            for ci in 0..c {
                max_prob = max_prob.max(probabilities[[ci, y, x]]);
            }
            conf[[y, x]] = max_prob;
        }
    }
    conf
}

/// Resize a 2D array (simple nearest-neighbor).
pub fn resize_mask(mask: &SegmentationMask, new_h: usize, new_w: usize) -> SegmentationMask {
    let (h, w) = (mask.shape()[0], mask.shape()[1]);
    let mut result = Array2::zeros((new_h, new_w));
    for y in 0..new_h {
        for x in 0..new_w {
            let src_y = (y * h) / new_h;
            let src_x = (x * w) / new_w;
            result[[y, x]] = mask[[src_y, src_x]];
        }
    }
    result
}

/// Pad a tensor to a target size (zero-padding on right/bottom).
pub fn pad_to(tensor: &ImageTensor, target_h: usize, target_w: usize) -> ImageTensor {
    let shape = tensor.shape();
    let (c, h, w) = (shape[0], shape[1], shape[2]);
    let mut padded = Array3::zeros((c, target_h, target_w));
    let copy_h = h.min(target_h);
    let copy_w = w.min(target_w);
    padded
        .slice_mut(ndarray::s![.., ..copy_h, ..copy_w])
        .assign(&tensor.slice(ndarray::s![.., ..copy_h, ..copy_w]));
    padded
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn test_normalize() {
        let t = Array3::from_elem((3, 2, 2), 255.0_f32);
        let n = normalize(&t);
        assert!((n[[0, 0, 0]] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_hwc_chw_roundtrip() {
        let chw = Array3::from_shape_fn((3, 4, 5), |(c, h, w)| (c * 20 + h * 5 + w) as f32);
        let hwc = chw_to_hwc(&chw);
        let back = hwc_to_chw(&hwc);
        assert_eq!(chw, back);
    }

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!((sigmoid(0.3) - 0.574_442_5).abs() < 1e-6);
        assert!(sigmoid(-20.0) < 1e-6);
        assert!(sigmoid(20.0) > 1.0 - 1e-6);
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let t = Array3::from_shape_fn((4, 3, 3), |(c, y, x)| c as f32 + y as f32 - x as f32);
        let probs = softmax_chw(&t);
        // Each pixel's probabilities should sum to ~1.0
        for y in 0..3 {
            for x in 0..3 {
                let sum: f32 = (0..4).map(|c| probs[[c, y, x]]).sum();
                assert!((sum - 1.0).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn test_argmax_mask() {
        // Channel 2 has highest values
        let mut t = Array3::zeros((3, 2, 2));
        t[[2, 0, 0]] = 10.0;
        t[[2, 0, 1]] = 10.0;
        t[[2, 1, 0]] = 10.0;
        t[[0, 1, 1]] = 10.0; // except this pixel
        let probs = softmax_chw(&t);
        let mask = argmax_mask(&probs);
        assert_eq!(mask[[0, 0]], 2);
        assert_eq!(mask[[1, 1]], 0);
    }

    #[test]
    fn test_stack_batch() {
        let img1 = Array3::ones((3, 4, 4));
        let img2 = Array3::from_elem((3, 4, 4), 2.0);
        let batch = stack_batch(&[img1, img2]);
        assert_eq!(batch.shape(), &[2, 3, 4, 4]);
        assert_eq!(batch[[0, 0, 0, 0]], 1.0);
        assert_eq!(batch[[1, 0, 0, 0]], 2.0);
    }
}
