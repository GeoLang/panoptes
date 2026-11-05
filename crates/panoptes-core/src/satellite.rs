//! Satellite image processing: DN to top-of-atmosphere reflectance, DOS1 dark
//! object subtraction, Brovey pan-sharpening, spectral indices and mosaic
//! compositing.
//!
//! `Sensor` is a label in `ImageMetadata` and has no effect on any result. This
//! module has no coefficients for any sensor: `dn_to_toa_reflectance` takes gain,
//! offset and solar irradiance as arguments in `CalibrationCoeffs`.

use std::collections::HashMap;

/// A multi-band raster image from a satellite sensor.
#[derive(Debug, Clone)]
pub struct SatelliteImage {
    /// Band data as f64 arrays (band_name -> values).
    pub bands: HashMap<String, Vec<f64>>,
    pub width: usize,
    pub height: usize,
    pub metadata: ImageMetadata,
}

/// Satellite image metadata.
#[derive(Debug, Clone)]
pub struct ImageMetadata {
    pub sensor: Sensor,
    pub acquisition_date: String,
    pub sun_elevation: f64,
    pub sun_azimuth: f64,
    pub cloud_cover_percent: Option<f64>,
    pub earth_sun_distance: f64,
    pub bbox: [f64; 4],
}

/// Sensor label for `ImageMetadata`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sensor {
    Sentinel2A,
    Sentinel2B,
    Landsat8,
    Landsat9,
    PlanetScope,
    WorldView3,
    Generic,
}

/// Radiometric calibration coefficients per band.
#[derive(Debug, Clone)]
pub struct CalibrationCoeffs {
    pub gain: f64,
    pub offset: f64,
    pub solar_irradiance: f64,
}

/// Convert raw DN to top-of-atmosphere (TOA) reflectance.
pub fn dn_to_toa_reflectance(
    image: &SatelliteImage,
    band: &str,
    coeffs: &CalibrationCoeffs,
) -> Result<Vec<f64>, String> {
    let data = image
        .bands
        .get(band)
        .ok_or_else(|| format!("band '{band}' not found"))?;

    let sun_zenith = 90.0 - image.metadata.sun_elevation;
    let cos_zenith = sun_zenith.to_radians().cos();
    let d_sq = image.metadata.earth_sun_distance.powi(2);

    let reflectance: Vec<f64> = data
        .iter()
        .map(|&dn| {
            let radiance = coeffs.gain * dn + coeffs.offset;
            (std::f64::consts::PI * radiance * d_sq) / (coeffs.solar_irradiance * cos_zenith)
        })
        .collect();

    Ok(reflectance)
}

pub fn dos1_correction(band_data: &[f64], dark_object_percentile: f64) -> Vec<f64> {
    let mut sorted: Vec<f64> = band_data
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let idx = ((sorted.len() as f64 * dark_object_percentile / 100.0) as usize)
        .min(sorted.len().saturating_sub(1));
    let path_radiance = sorted.get(idx).copied().unwrap_or(0.0);

    band_data
        .iter()
        .map(|&v| (v - path_radiance).max(0.0))
        .collect()
}

/// Input parameters for Brovey pan-sharpening.
pub struct PansharpenInput<'a> {
    pub pan: &'a [f64],
    pub red: &'a [f64],
    pub green: &'a [f64],
    pub blue: &'a [f64],
    pub pan_width: usize,
    pub pan_height: usize,
    pub ms_width: usize,
    pub ms_height: usize,
}

/// Pan-sharpening using Brovey transform.
/// Fuses a high-resolution panchromatic band with lower-resolution multispectral bands.
pub fn pansharpen_brovey(input: &PansharpenInput<'_>) -> Result<PansharpenedImage, String> {
    let PansharpenInput {
        pan,
        red,
        green,
        blue,
        pan_width,
        pan_height,
        ms_width,
        ms_height,
    } = *input;
    if pan.len() != pan_width * pan_height {
        return Err("pan dimensions mismatch".to_string());
    }
    if red.len() != ms_width * ms_height {
        return Err("multispectral dimensions mismatch".to_string());
    }

    let scale_x = ms_width as f64 / pan_width as f64;
    let scale_y = ms_height as f64 / pan_height as f64;

    let n = pan_width * pan_height;
    let mut out_red = Vec::with_capacity(n);
    let mut out_green = Vec::with_capacity(n);
    let mut out_blue = Vec::with_capacity(n);

    for y in 0..pan_height {
        for x in 0..pan_width {
            let ms_x = ((x as f64) * scale_x) as usize;
            let ms_y = ((y as f64) * scale_y) as usize;
            let ms_idx = ms_y.min(ms_height - 1) * ms_width + ms_x.min(ms_width - 1);
            let pan_idx = y * pan_width + x;

            let r = red.get(ms_idx).copied().unwrap_or(0.0);
            let g = green.get(ms_idx).copied().unwrap_or(0.0);
            let b = blue.get(ms_idx).copied().unwrap_or(0.0);
            let p = pan[pan_idx];

            let total = r + g + b;
            if total > 0.0 {
                out_red.push(r / total * p);
                out_green.push(g / total * p);
                out_blue.push(b / total * p);
            } else {
                out_red.push(0.0);
                out_green.push(0.0);
                out_blue.push(0.0);
            }
        }
    }

    Ok(PansharpenedImage {
        red: out_red,
        green: out_green,
        blue: out_blue,
        width: pan_width,
        height: pan_height,
    })
}

/// Result of pan-sharpening.
#[derive(Debug, Clone)]
pub struct PansharpenedImage {
    pub red: Vec<f64>,
    pub green: Vec<f64>,
    pub blue: Vec<f64>,
    pub width: usize,
    pub height: usize,
}

/// Common spectral indices.
pub mod indices {
    /// Normalized Difference Vegetation Index: (NIR - Red) / (NIR + Red)
    pub fn ndvi(nir: &[f64], red: &[f64]) -> Vec<f64> {
        nir.iter()
            .zip(red.iter())
            .map(|(&n, &r)| {
                let sum = n + r;
                if sum.abs() < f64::EPSILON {
                    0.0
                } else {
                    (n - r) / sum
                }
            })
            .collect()
    }

    /// Normalized Difference Water Index: (Green - NIR) / (Green + NIR)
    pub fn ndwi(green: &[f64], nir: &[f64]) -> Vec<f64> {
        green
            .iter()
            .zip(nir.iter())
            .map(|(&g, &n)| {
                let sum = g + n;
                if sum.abs() < f64::EPSILON {
                    0.0
                } else {
                    (g - n) / sum
                }
            })
            .collect()
    }

    /// Enhanced Vegetation Index: 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)
    pub fn evi(nir: &[f64], red: &[f64], blue: &[f64]) -> Vec<f64> {
        nir.iter()
            .zip(red.iter())
            .zip(blue.iter())
            .map(|((&n, &r), &b)| {
                let denom = n + 6.0 * r - 7.5 * b + 1.0;
                if denom.abs() < f64::EPSILON {
                    0.0
                } else {
                    2.5 * (n - r) / denom
                }
            })
            .collect()
    }

    /// Normalized Burn Ratio: (NIR - SWIR) / (NIR + SWIR)
    pub fn nbr(nir: &[f64], swir: &[f64]) -> Vec<f64> {
        nir.iter()
            .zip(swir.iter())
            .map(|(&n, &s)| {
                let sum = n + s;
                if sum.abs() < f64::EPSILON {
                    0.0
                } else {
                    (n - s) / sum
                }
            })
            .collect()
    }

    /// Soil Adjusted Vegetation Index: (NIR - Red) / (NIR + Red + L) * (1 + L)
    pub fn savi(nir: &[f64], red: &[f64], l: f64) -> Vec<f64> {
        nir.iter()
            .zip(red.iter())
            .map(|(&n, &r)| {
                let denom = n + r + l;
                if denom.abs() < f64::EPSILON {
                    0.0
                } else {
                    (n - r) / denom * (1.0 + l)
                }
            })
            .collect()
    }
}

/// Mosaic compositing — combine multiple overlapping images.
pub mod mosaic {
    /// Compositing method for overlapping pixels.
    #[derive(Debug, Clone, Copy)]
    pub enum CompositeMethod {
        /// Use value from last image (paint-over order)
        Last,
        /// Average of all valid values
        Average,
        /// Median of all valid values
        Median,
        /// Minimum value (best for cloud removal in some cases)
        Min,
        /// Maximum value (useful for NDVI composites)
        Max,
    }

    /// Composite multiple bands into a single mosaic using the given method.
    pub fn composite(layers: &[&[f64]], method: CompositeMethod) -> Vec<f64> {
        if layers.is_empty() {
            return Vec::new();
        }
        let len = layers[0].len();
        (0..len)
            .map(|i| {
                let values: Vec<f64> = layers
                    .iter()
                    .filter_map(|layer| {
                        let v = layer.get(i).copied().unwrap_or(f64::NAN);
                        if v.is_finite() { Some(v) } else { None }
                    })
                    .collect();
                if values.is_empty() {
                    return f64::NAN;
                }
                match method {
                    CompositeMethod::Last => *values.last().unwrap(),
                    CompositeMethod::Average => values.iter().sum::<f64>() / values.len() as f64,
                    CompositeMethod::Median => {
                        let mut sorted = values.clone();
                        sorted
                            .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                        sorted[sorted.len() / 2]
                    }
                    CompositeMethod::Min => values.iter().copied().fold(f64::INFINITY, f64::min),
                    CompositeMethod::Max => {
                        values.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                    }
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ndvi() {
        let nir = vec![0.5, 0.8, 0.3];
        let red = vec![0.1, 0.2, 0.3];
        let result = indices::ndvi(&nir, &red);
        assert!((result[0] - 0.6667).abs() < 0.01);
        assert!((result[1] - 0.6).abs() < 0.01);
        assert!((result[2] - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_dos1_correction() {
        let data = vec![100.0, 200.0, 50.0, 300.0, 150.0];
        let corrected = dos1_correction(&data, 1.0);
        // Minimum value should be near 0
        assert!(corrected.iter().all(|&v| v >= 0.0));
        assert!(corrected.iter().any(|&v| v < 1.0)); // dark object
    }

    #[test]
    fn test_pansharpen_brovey() {
        let pan = vec![100.0; 16]; // 4x4
        let red = vec![50.0; 4]; // 2x2
        let green = vec![30.0; 4];
        let blue = vec![20.0; 4];

        let input = PansharpenInput {
            pan: &pan,
            red: &red,
            green: &green,
            blue: &blue,
            pan_width: 4,
            pan_height: 4,
            ms_width: 2,
            ms_height: 2,
        };
        let result = pansharpen_brovey(&input).unwrap();
        assert_eq!(result.width, 4);
        assert_eq!(result.height, 4);
        assert_eq!(result.red.len(), 16);
        // Brovey: r/(r+g+b) * pan = 50/100 * 100 = 50
        assert!((result.red[0] - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_mosaic_average() {
        let layer1: Vec<f64> = vec![1.0, 2.0, f64::NAN];
        let layer2: Vec<f64> = vec![3.0, f64::NAN, 4.0];
        let result = mosaic::composite(&[&layer1, &layer2], mosaic::CompositeMethod::Average);
        assert!((result[0] - 2.0).abs() < 0.01); // avg(1,3)
        assert!((result[1] - 2.0).abs() < 0.01); // only layer1 valid
        assert!((result[2] - 4.0).abs() < 0.01); // only layer2 valid
    }
}
