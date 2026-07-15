//! Polygonization — convert segmentation masks to vector polygons.

use geo_types::{Coord, LineString, Polygon};
use ndarray::Array2;
use thiserror::Error;

use panoptes_core::tensor::{ProbabilityMap, SegmentationMask};

/// Errors during polygonization.
#[derive(Debug, Error)]
pub enum PolygonizeError {
    #[error("Empty mask")]
    EmptyMask,
    #[error("Invalid class id: {0}")]
    InvalidClass(u8),
}

/// A vectorized feature extracted from a segmentation mask.
#[derive(Debug, Clone)]
pub struct VectorFeature {
    /// Class ID this feature belongs to.
    pub class_id: u8,
    /// The polygon geometry.
    pub geometry: Polygon<f64>,
    /// Area in pixel units.
    pub area_px: f64,
    /// Confidence (mean confidence within the polygon region).
    pub confidence: f32,
}

/// Extract polygon boundaries for a specific class from a segmentation mask.
///
/// Uses a simple contour-tracing approach: finds connected regions of the target class
/// and generates bounding polygons.
///
/// When `confidence` is provided, each feature's confidence is the mean probability over
/// the region's pixels. When it is `None`, confidence defaults to 1.0.
pub fn polygonize_class(
    mask: &SegmentationMask,
    class_id: u8,
    min_area: f64,
    confidence: Option<&ProbabilityMap>,
) -> Result<Vec<VectorFeature>, PolygonizeError> {
    let (h, w) = (mask.shape()[0], mask.shape()[1]);
    if h == 0 || w == 0 {
        return Err(PolygonizeError::EmptyMask);
    }

    // Label connected components using simple flood-fill
    let mut visited = Array2::from_elem((h, w), false);
    let mut features = Vec::new();

    for start_y in 0..h {
        for start_x in 0..w {
            if visited[[start_y, start_x]] || mask[[start_y, start_x]] != class_id {
                continue;
            }

            // Flood-fill to find connected component
            let mut stack = vec![(start_y, start_x)];
            let mut min_x = start_x;
            let mut max_x = start_x;
            let mut min_y = start_y;
            let mut max_y = start_y;
            let mut pixel_count = 0u64;
            let mut conf_sum = 0.0_f32;

            while let Some((y, x)) = stack.pop() {
                if y >= h || x >= w || visited[[y, x]] || mask[[y, x]] != class_id {
                    continue;
                }
                visited[[y, x]] = true;
                pixel_count += 1;
                if let Some(conf) = confidence {
                    conf_sum += conf[[y, x]];
                }

                min_x = min_x.min(x);
                max_x = max_x.max(x);
                min_y = min_y.min(y);
                max_y = max_y.max(y);

                // 4-connectivity
                if y > 0 {
                    stack.push((y - 1, x));
                }
                if y + 1 < h {
                    stack.push((y + 1, x));
                }
                if x > 0 {
                    stack.push((y, x - 1));
                }
                if x + 1 < w {
                    stack.push((y, x + 1));
                }
            }

            let area = pixel_count as f64;
            if area < min_area {
                continue;
            }

            // Create bounding polygon from the convex hull approximation (bbox for now)
            let polygon = Polygon::new(
                LineString::from(vec![
                    Coord {
                        x: min_x as f64,
                        y: min_y as f64,
                    },
                    Coord {
                        x: max_x as f64 + 1.0,
                        y: min_y as f64,
                    },
                    Coord {
                        x: max_x as f64 + 1.0,
                        y: max_y as f64 + 1.0,
                    },
                    Coord {
                        x: min_x as f64,
                        y: max_y as f64 + 1.0,
                    },
                    Coord {
                        x: min_x as f64,
                        y: min_y as f64,
                    },
                ]),
                vec![],
            );

            let feature_confidence = match confidence {
                Some(_) if pixel_count > 0 => conf_sum / pixel_count as f32,
                _ => 1.0,
            };

            features.push(VectorFeature {
                class_id,
                geometry: polygon,
                area_px: area,
                confidence: feature_confidence,
            });
        }
    }

    Ok(features)
}

/// Extract polygons for all classes in a mask.
pub fn polygonize_all(
    mask: &SegmentationMask,
    num_classes: usize,
    min_area: f64,
    confidence: Option<&ProbabilityMap>,
) -> Result<Vec<VectorFeature>, PolygonizeError> {
    let mut all_features = Vec::new();
    for class_id in 0..num_classes {
        let features = polygonize_class(mask, class_id as u8, min_area, confidence)?;
        all_features.extend(features);
    }
    Ok(all_features)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_polygonize_single_class() {
        let mut mask = Array2::zeros((10, 10));
        // Fill a 5x5 block with class 1
        for y in 2..7 {
            for x in 2..7 {
                mask[[y, x]] = 1u8;
            }
        }
        let features = polygonize_class(&mask, 1, 1.0, None).unwrap();
        assert_eq!(features.len(), 1);
        assert_eq!(features[0].class_id, 1);
        assert!((features[0].area_px - 25.0).abs() < 1e-5);
    }

    #[test]
    fn test_polygonize_mean_confidence() {
        let mut mask = Array2::zeros((10, 10));
        let mut conf = Array2::zeros((10, 10));
        for y in 2..7 {
            for x in 2..7 {
                mask[[y, x]] = 1u8;
                conf[[y, x]] = 0.8;
            }
        }
        let features = polygonize_class(&mask, 1, 1.0, Some(&conf)).unwrap();
        assert_eq!(features.len(), 1);
        assert!((features[0].confidence - 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_polygonize_min_area_filter() {
        let mut mask = Array2::zeros((10, 10));
        mask[[5, 5]] = 1;
        mask[[5, 6]] = 1;
        let features = polygonize_class(&mask, 1, 5.0, None).unwrap();
        assert_eq!(features.len(), 0); // Too small
    }

    #[test]
    fn test_polygonize_empty() {
        let mask = Array2::zeros((10, 10));
        let features = polygonize_class(&mask, 1, 1.0, None).unwrap();
        assert_eq!(features.len(), 0);
    }
}
