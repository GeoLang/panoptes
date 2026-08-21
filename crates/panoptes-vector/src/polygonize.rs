//! Polygonization — convert segmentation masks to vector polygons.

use std::collections::HashSet;

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
/// Finds 4-connected regions of the target class and traces the outline of each
/// component (union of unit pixel squares).
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

            let mut stack = vec![(start_y, start_x)];
            let mut pixels = Vec::new();
            let mut conf_sum = 0.0_f32;

            while let Some((y, x)) = stack.pop() {
                if y >= h || x >= w || visited[[y, x]] || mask[[y, x]] != class_id {
                    continue;
                }
                visited[[y, x]] = true;
                pixels.push((x, y));
                if let Some(conf) = confidence {
                    conf_sum += conf[[y, x]];
                }

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

            let area = pixels.len() as f64;
            if area < min_area {
                continue;
            }

            let polygon = outline_polygon(&pixels);

            let feature_confidence = match confidence {
                Some(_) if !pixels.is_empty() => conf_sum / pixels.len() as f32,
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

fn outline_polygon(pixels: &[(usize, usize)]) -> Polygon<f64> {
    let set: HashSet<(i32, i32)> = pixels.iter().map(|&(x, y)| (x as i32, y as i32)).collect();

    let mut edges: HashSet<(i32, i32, i32, i32)> = HashSet::new();
    for &(x, y) in &set {
        if !set.contains(&(x, y - 1)) {
            edges.insert((x, y, x + 1, y));
        }
        if !set.contains(&(x + 1, y)) {
            edges.insert((x + 1, y, x + 1, y + 1));
        }
        if !set.contains(&(x, y + 1)) {
            edges.insert((x + 1, y + 1, x, y + 1));
        }
        if !set.contains(&(x - 1, y)) {
            edges.insert((x, y + 1, x, y));
        }
    }

    let Some(&(start_x, start_y)) = set.iter().min_by_key(|p| (p.1, p.0)) else {
        return Polygon::new(LineString::from(Vec::<Coord<f64>>::new()), vec![]);
    };

    let mut cx = start_x;
    let mut cy = start_y;
    let mut dx = 1_i32;
    let mut dy = 0_i32;
    let start_edge = (cx, cy, dx, dy);
    let mut ring = Vec::new();

    for _ in 0..edges.len() {
        ring.push((cx, cy));
        cx += dx;
        cy += dy;
        let Some((ndx, ndy)) = [(-dy, dx), (dx, dy), (dy, -dx), (-dx, -dy)]
            .into_iter()
            .find(|&(ndx, ndy)| edges.contains(&(cx, cy, cx + ndx, cy + ndy)))
        else {
            break;
        };
        dx = ndx;
        dy = ndy;
        if (cx, cy, dx, dy) == start_edge {
            break;
        }
    }
    if let Some(&first) = ring.first()
        && ring.last() != Some(&first)
    {
        ring.push(first);
    }

    let ring = collapse_collinear(ring);
    let coords: Vec<Coord<f64>> = ring
        .into_iter()
        .map(|(x, y)| Coord {
            x: x as f64,
            y: y as f64,
        })
        .collect();
    Polygon::new(LineString::from(coords), vec![])
}

fn collapse_collinear(mut ring: Vec<(i32, i32)>) -> Vec<(i32, i32)> {
    if ring.len() >= 2 && ring.first() == ring.last() {
        ring.pop();
    }
    let n = ring.len();
    if n < 3 {
        if let Some(&first) = ring.first() {
            ring.push(first);
        }
        return ring;
    }
    let mut out = Vec::with_capacity(n + 1);
    for i in 0..n {
        let prev = ring[(i + n - 1) % n];
        let curr = ring[i];
        let next = ring[(i + 1) % n];
        let col_x = prev.0 == curr.0 && curr.0 == next.0;
        let col_y = prev.1 == curr.1 && curr.1 == next.1;
        if !col_x && !col_y {
            out.push(curr);
        }
    }
    if let Some(&first) = out.first() {
        out.push(first);
    }
    out
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
    use geo::Area;
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
        assert!((features[0].geometry.unsigned_area() - 25.0).abs() < 1e-5);
        assert_eq!(features[0].geometry.exterior().0.len(), 5);
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

    #[test]
    fn test_polygonize_l_shape_is_not_bbox() {
        let mut mask = Array2::zeros((10, 10));
        for y in 2..5 {
            mask[[y, 2]] = 1u8;
        }
        for x in 2..5 {
            mask[[4, x]] = 1u8;
        }
        let features = polygonize_class(&mask, 1, 1.0, None).unwrap();
        assert_eq!(features.len(), 1);
        assert!((features[0].area_px - 5.0).abs() < 1e-5);

        let poly_area = features[0].geometry.unsigned_area();
        assert!(
            (poly_area - 5.0).abs() < 1e-5,
            "polygon area {poly_area} should match pixel count, not bbox area 9"
        );
        let unique_verts = features[0].geometry.exterior().0.len().saturating_sub(1);
        assert!(
            unique_verts > 4,
            "L-shape outline must have more than 4 vertices, got {unique_verts}"
        );
    }
}
