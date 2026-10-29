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

fn fixture_min_x() -> f64 {
    -WEB_MERCATOR_EDGE + 2_103_750.0 * PIXEL_SIZE
}

fn fixture_max_y() -> f64 {
    WEB_MERCATOR_EDGE - 1_403_750.0 * PIXEL_SIZE
}

fn write_square_cog(path: &Path, origin_x: f64, origin_y: f64) {
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
        origin_x,
        origin_y,
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

struct LonLatExtent {
    min_lon: f64,
    max_lon: f64,
    min_lat: f64,
    max_lat: f64,
}

fn largest_square_extent(directory: &Path) -> LonLatExtent {
    let text = std::fs::read_to_string(directory.join("out.geojson")).unwrap();
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
    LonLatExtent {
        min_lon: ring.iter().map(|v| v.0).fold(f64::INFINITY, f64::min),
        max_lon: ring.iter().map(|v| v.0).fold(f64::NEG_INFINITY, f64::max),
        min_lat: ring.iter().map(|v| v.1).fold(f64::INFINITY, f64::min),
        max_lat: ring.iter().map(|v| v.1).fold(f64::NEG_INFINITY, f64::max),
    }
}

#[test]
fn layer_square_comes_out_in_lon_lat() {
    let directory = tempfile::tempdir().unwrap();
    let cog = directory.path().join("square.tif");
    write_square_cog(&cog, fixture_min_x(), fixture_max_y());
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

    let extent_found = largest_square_extent(directory.path());

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
        (
            "min lon",
            extent_found.min_lon,
            expected_min_lon,
            lon_tolerance,
        ),
        (
            "max lon",
            extent_found.max_lon,
            expected_max_lon,
            lon_tolerance,
        ),
        (
            "min lat",
            extent_found.min_lat,
            expected_min_lat,
            lat_tolerance_south,
        ),
        (
            "max lat",
            extent_found.max_lat,
            expected_max_lat,
            lat_tolerance_north,
        ),
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

#[test]
fn layer_already_in_the_crs_keeps_its_own_grid() {
    let directory = tempfile::tempdir().unwrap();
    let cog = directory.path().join("square.tif");
    // a fraction of a pixel off the web mercator corner grid, which a reproject resamples onto
    let origin_x = fixture_min_x() + 0.37 * PIXEL_SIZE;
    let origin_y = fixture_max_y() - 0.37 * PIXEL_SIZE;
    write_square_cog(&cog, origin_x, origin_y);
    let layer_file = write_layer_file(directory.path(), &cog);
    let extent = PIXEL_SIZE * SIDE_PX as f64;
    let bbox = format!(
        "{},{},{},{}",
        origin_x,
        origin_y - extent,
        origin_x + extent,
        origin_y
    );

    let output = run_segment(directory.path(), &layer_file, "square", &bbox);
    assert!(
        output.status.success(),
        "segment failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let extent_found = largest_square_extent(directory.path());
    let square_min_x = origin_x + SQUARE_COLUMN as f64 * PIXEL_SIZE;
    let square_max_y = origin_y - SQUARE_ROW as f64 * PIXEL_SIZE;
    let side = SQUARE_SIDE_PX as f64 * PIXEL_SIZE;
    let (expected_min_lon, expected_min_lat) = lon_lat(square_min_x, square_max_y - side);
    let (expected_max_lon, expected_max_lat) = lon_lat(square_min_x + side, square_max_y);
    for (name, got, expected) in [
        ("min lon", extent_found.min_lon, expected_min_lon),
        ("max lon", extent_found.max_lon, expected_max_lon),
        ("min lat", extent_found.min_lat, expected_min_lat),
        ("max lat", extent_found.max_lat, expected_max_lat),
    ] {
        assert!(
            (got - expected).abs() < 1e-9,
            "{name} is {got}, expected {expected}"
        );
    }
}

fn write_gdal_layer_file(directory: &Path, cog: &Path) -> std::path::PathBuf {
    let layer_file = directory.join("gdal_layers.toml");
    let text =
        format!("[[layer]]\nname = \"square\"\nsource = {{ kind = \"gdal\", path = {cog:?} }}\n");
    std::fs::write(&layer_file, text).unwrap();
    layer_file
}

#[cfg(feature = "gdal")]
#[test]
fn gdal_layer_comes_out_where_the_cog_layer_does() {
    let directory = tempfile::tempdir().unwrap();
    let cog = directory.path().join("square.tif");
    write_square_cog(&cog, fixture_min_x(), fixture_max_y());
    let extent = PIXEL_SIZE * SIDE_PX as f64;
    let bbox = format!(
        "{},{},{},{}",
        fixture_min_x(),
        fixture_max_y() - extent,
        fixture_min_x() + extent,
        fixture_max_y()
    );
    let cog_layers = write_layer_file(directory.path(), &cog);
    let gdal_layers = write_gdal_layer_file(directory.path(), &cog);

    let cog_output = run_segment(directory.path(), &cog_layers, "square", &bbox);
    assert!(cog_output.status.success());
    let from_cog = std::fs::read_to_string(directory.path().join("out.geojson")).unwrap();
    let gdal_output = run_segment(directory.path(), &gdal_layers, "square", &bbox);
    assert!(
        gdal_output.status.success(),
        "segment failed: {}",
        String::from_utf8_lossy(&gdal_output.stderr)
    );
    let from_gdal = std::fs::read_to_string(directory.path().join("out.geojson")).unwrap();

    assert_eq!(from_gdal, from_cog);
}

#[cfg(not(feature = "gdal"))]
#[test]
fn gdal_layer_without_the_feature_names_it() {
    let directory = tempfile::tempdir().unwrap();
    let cog = directory.path().join("square.tif");
    let layer_file = write_gdal_layer_file(directory.path(), &cog);

    let output = run_segment(directory.path(), &layer_file, "square", "0,0,1,1");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("gdal feature"),
        "stderr does not name the gdal feature: {stderr}"
    );
}
