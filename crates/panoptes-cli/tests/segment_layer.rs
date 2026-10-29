use std::path::Path;
use std::process::{Command, Output};

use terrano_core::{BandedRaster, CogParams, Raster, write_cog_bands};

const EARTH_RADIUS: f64 = 6_378_137.0;
const WEB_MERCATOR_EDGE: f64 = 20_037_508.342_789_244;
const PIXEL_SIZE: f64 = 10.0;
const SIDE_PX: usize = 600;
const SQUARE_COLUMN: usize = 220;
const SQUARE_ROW: usize = 230;
const SQUARE_SIDE_PX: usize = 100;

// the reproject to 3857 resamples onto a grid anchored at the web mercator corner
fn fixture_min_x() -> f64 {
    -WEB_MERCATOR_EDGE + 2_103_750.0 * PIXEL_SIZE
}

fn fixture_max_y() -> f64 {
    WEB_MERCATOR_EDGE - 1_403_750.0 * PIXEL_SIZE
}

fn write_square_cog(path: &Path) {
    let inside_square = |index: usize| {
        let (row, column) = (index / SIDE_PX, index % SIDE_PX);
        (SQUARE_ROW..SQUARE_ROW + SQUARE_SIDE_PX).contains(&row)
            && (SQUARE_COLUMN..SQUARE_COLUMN + SQUARE_SIDE_PX).contains(&column)
    };
    // the threshold engine labels a pixel with the index of its brightest band over 128
    let band = |bright: f64| {
        let data = (0..SIDE_PX * SIDE_PX)
            .map(|index| if inside_square(index) { bright } else { 10.0 })
            .collect();
        Raster::from_vec(SIDE_PX, SIDE_PX, data, PIXEL_SIZE, f64::NAN).unwrap()
    };
    let bands = BandedRaster::new(vec![band(10.0), band(250.0), band(10.0)]).unwrap();
    let params = CogParams {
        tile_width: 256,
        tile_height: 256,
        overview_levels: 0,
        epsg: 3857,
        origin_x: fixture_min_x(),
        origin_y: fixture_max_y(),
        pixel_width: PIXEL_SIZE,
        pixel_height: PIXEL_SIZE,
        deflate: true,
        ..CogParams::default()
    };
    let mut file = std::fs::File::create(path).unwrap();
    write_cog_bands(&bands, &params, &mut file).unwrap();
}

fn write_layer_file(directory: &Path, cog: &Path) -> std::path::PathBuf {
    let layer_file = directory.join("layers.toml");
    let text = format!(
        "[[layer]]\nname = \"square\"\nsource = {{ kind = \"cog\", path = {cog:?} }}\n\n\
         [[layer]]\nname = \"duplicate\"\nsource = {{ kind = \"cog\", path = {cog:?} }}\n"
    );
    std::fs::write(&layer_file, text).unwrap();
    layer_file
}

fn run_segment(directory: &Path, layer_file: &Path, layer: &str, bbox: &str) -> Output {
    Command::new(env!("CARGO_BIN_EXE_panoptes"))
        .current_dir(directory)
        .args(["segment", "--layer-file"])
        .arg(layer_file)
        .args(["--layer", layer, "--bbox", bbox])
        .args(["--resolution", &PIXEL_SIZE.to_string(), "--crs", "3857"])
        .args(["--engine", "threshold", "--tile-size", "256"])
        .args(["--output", "out.geojson"])
        .output()
        .unwrap()
}

fn lon_lat(x: f64, y: f64) -> (f64, f64) {
    let lon = x / EARTH_RADIUS * 180.0 / std::f64::consts::PI;
    let lat = (2.0 * (y / EARTH_RADIUS).exp().atan() - std::f64::consts::FRAC_PI_2) * 180.0
        / std::f64::consts::PI;
    (lon, lat)
}

#[test]
fn layer_square_comes_out_in_lon_lat() {
    let directory = tempfile::tempdir().unwrap();
    let cog = directory.path().join("square.tif");
    write_square_cog(&cog);
    let layer_file = write_layer_file(directory.path(), &cog);
    // an inset the pull aligns away, so georeferencing from the request lands 0.7 px off
    let inset = 0.7 * PIXEL_SIZE;
    let extent = PIXEL_SIZE * SIDE_PX as f64;
    let bbox = format!(
        "{},{},{},{}",
        fixture_min_x() + inset,
        fixture_max_y() - extent,
        fixture_min_x() + extent,
        fixture_max_y() - inset
    );

    let output = run_segment(directory.path(), &layer_file, "square", &bbox);
    assert!(
        output.status.success(),
        "segment failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let text = std::fs::read_to_string(directory.path().join("out.geojson")).unwrap();
    let geojson: serde_json::Value = serde_json::from_str(&text).unwrap();
    let largest_square = geojson["features"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|feature| feature["properties"]["class_id"] == 1)
        .max_by(|a, b| {
            let area =
                |feature: &serde_json::Value| feature["properties"]["area_px"].as_f64().unwrap();
            area(a).total_cmp(&area(b))
        })
        .expect("at least one class 1 polygon");
    let ring: Vec<(f64, f64)> = largest_square["geometry"]["coordinates"][0]
        .as_array()
        .unwrap()
        .iter()
        .map(|vertex| (vertex[0].as_f64().unwrap(), vertex[1].as_f64().unwrap()))
        .collect();
    let min_lon = ring.iter().map(|v| v.0).fold(f64::INFINITY, f64::min);
    let max_lon = ring.iter().map(|v| v.0).fold(f64::NEG_INFINITY, f64::max);
    let min_lat = ring.iter().map(|v| v.1).fold(f64::INFINITY, f64::min);
    let max_lat = ring.iter().map(|v| v.1).fold(f64::NEG_INFINITY, f64::max);

    let square_min_x = fixture_min_x() + SQUARE_COLUMN as f64 * PIXEL_SIZE;
    let square_max_x = square_min_x + SQUARE_SIDE_PX as f64 * PIXEL_SIZE;
    let square_max_y = fixture_max_y() - SQUARE_ROW as f64 * PIXEL_SIZE;
    let square_min_y = square_max_y - SQUARE_SIDE_PX as f64 * PIXEL_SIZE;
    let (expected_min_lon, expected_min_lat) = lon_lat(square_min_x, square_min_y);
    let (expected_max_lon, expected_max_lat) = lon_lat(square_max_x, square_max_y);
    let half_pixel = 0.5 * PIXEL_SIZE;
    let lon_tolerance = lon_lat(square_min_x + half_pixel, square_min_y).0 - expected_min_lon;
    let lat_tolerance_south = lon_lat(square_min_x, square_min_y + half_pixel).1 - expected_min_lat;
    let lat_tolerance_north = expected_max_lat - lon_lat(square_max_x, square_max_y - half_pixel).1;

    for (name, got, expected, tolerance) in [
        ("min lon", min_lon, expected_min_lon, lon_tolerance),
        ("max lon", max_lon, expected_max_lon, lon_tolerance),
        ("min lat", min_lat, expected_min_lat, lat_tolerance_south),
        ("max lat", max_lat, expected_max_lat, lat_tolerance_north),
    ] {
        assert!(
            (got - expected).abs() < tolerance,
            "{name} is {got}, expected {expected} within {tolerance}"
        );
    }
}

#[test]
fn unknown_layer_lists_the_layer_names() {
    let directory = tempfile::tempdir().unwrap();
    let cog = directory.path().join("square.tif");
    let layer_file = write_layer_file(directory.path(), &cog);

    let output = run_segment(directory.path(), &layer_file, "missing", "0,0,1,1");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("square") && stderr.contains("duplicate"),
        "stderr does not list the layers: {stderr}"
    );
}
