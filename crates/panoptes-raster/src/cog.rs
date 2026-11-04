//! Cloud-Optimized GeoTIFF (COG) metadata and raw tile reads.
//!
//! `parse_cog_metadata` reads the IFD chain and tile offsets from a `Read + Seek`
//! source, and `read_raw_tile` returns one tile as its stored bytes. This module
//! has no HTTP or S3 client, no decompression and no pixel decode. Files with a
//! TIFF magic number other than 42, BigTIFF included, return
//! `CogError::InvalidHeader`.

use std::io::{Read, Seek, SeekFrom};
use thiserror::Error;

/// Errors during COG operations.
#[derive(Debug, Error)]
pub enum CogError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Invalid TIFF header")]
    InvalidHeader,
    #[error("Unsupported compression: {0}")]
    UnsupportedCompression(u16),
    #[error("Tile not found at index {0}")]
    TileNotFound(usize),
    #[error("Invalid IFD offset")]
    InvalidIfd,
}

/// TIFF byte order.
#[derive(Debug, Clone, Copy, PartialEq)]
enum ByteOrder {
    LittleEndian,
    BigEndian,
}

/// Image File Directory entry.
#[derive(Debug, Clone)]
pub struct IfdEntry {
    pub tag: u16,
    pub data_type: u16,
    pub count: u32,
    pub value_offset: u32,
}

/// COG overview level metadata.
#[derive(Debug, Clone)]
pub struct OverviewLevel {
    /// IFD index (0 = full resolution).
    pub level: usize,
    /// Image width at this level.
    pub width: u32,
    /// Image height at this level.
    pub height: u32,
    /// Tile width.
    pub tile_width: u32,
    /// Tile height.
    pub tile_height: u32,
    /// Number of tiles in X.
    pub tiles_across: u32,
    /// Number of tiles in Y.
    pub tiles_down: u32,
    /// Byte offsets of each tile.
    pub tile_offsets: Vec<u64>,
    /// Byte lengths of each tile.
    pub tile_byte_counts: Vec<u64>,
}

/// COG file metadata (parsed from IFDs).
#[derive(Debug, Clone)]
pub struct CogMetadata {
    /// Overview levels (index 0 = full resolution).
    pub levels: Vec<OverviewLevel>,
    /// Number of bands/samples.
    pub samples_per_pixel: u16,
    /// Bits per sample.
    pub bits_per_sample: u16,
    /// Compression type.
    pub compression: u16,
}

/// Parse COG metadata from a seekable reader.
pub fn parse_cog_metadata<R: Read + Seek>(reader: &mut R) -> Result<CogMetadata, CogError> {
    // Read TIFF header
    let mut header = [0u8; 8];
    reader.read_exact(&mut header)?;

    let byte_order = match &header[0..2] {
        b"II" => ByteOrder::LittleEndian,
        b"MM" => ByteOrder::BigEndian,
        _ => return Err(CogError::InvalidHeader),
    };

    let magic = read_u16(&header[2..4], byte_order);
    if magic != 42 {
        return Err(CogError::InvalidHeader);
    }

    let mut ifd_offset = read_u32(&header[4..8], byte_order) as u64;
    let mut levels = Vec::new();
    let mut samples_per_pixel = 1u16;
    let mut bits_per_sample = 8u16;
    let mut compression = 1u16;

    // Walk IFD chain
    while ifd_offset != 0 {
        reader.seek(SeekFrom::Start(ifd_offset))?;

        let mut count_buf = [0u8; 2];
        reader.read_exact(&mut count_buf)?;
        let entry_count = read_u16(&count_buf, byte_order);

        let mut width = 0u32;
        let mut height = 0u32;
        let mut tile_width = 256u32;
        let mut tile_height = 256u32;
        let mut tile_offsets = Vec::new();
        let mut tile_byte_counts = Vec::new();

        for _ in 0..entry_count {
            let mut entry_buf = [0u8; 12];
            reader.read_exact(&mut entry_buf)?;

            let tag = read_u16(&entry_buf[0..2], byte_order);
            let data_type = read_u16(&entry_buf[2..4], byte_order);
            let count = read_u32(&entry_buf[4..8], byte_order);
            let value = read_u32(&entry_buf[8..12], byte_order);

            match tag {
                256 => width = value,                    // ImageWidth
                257 => height = value,                   // ImageLength
                258 => bits_per_sample = value as u16,   // BitsPerSample
                259 => compression = value as u16,       // Compression
                277 => samples_per_pixel = value as u16, // SamplesPerPixel
                322 => tile_width = value,               // TileWidth
                323 => tile_height = value,              // TileLength
                324 => {
                    // TileOffsets
                    tile_offsets =
                        read_offset_array(reader, value as u64, count, byte_order, data_type)?;
                }
                325 => {
                    // TileByteCounts
                    tile_byte_counts =
                        read_offset_array(reader, value as u64, count, byte_order, data_type)?;
                }
                _ => {}
            }
        }

        let tiles_across = width.div_ceil(tile_width);
        let tiles_down = height.div_ceil(tile_height);

        levels.push(OverviewLevel {
            level: levels.len(),
            width,
            height,
            tile_width,
            tile_height,
            tiles_across,
            tiles_down,
            tile_offsets,
            tile_byte_counts,
        });

        // Read next IFD offset
        let mut next_buf = [0u8; 4];
        reader.read_exact(&mut next_buf)?;
        ifd_offset = read_u32(&next_buf, byte_order) as u64;
    }

    Ok(CogMetadata {
        levels,
        samples_per_pixel,
        bits_per_sample,
        compression,
    })
}

/// Read a raw tile from a COG at a specific level and tile index.
pub fn read_raw_tile<R: Read + Seek>(
    reader: &mut R,
    metadata: &CogMetadata,
    level: usize,
    tile_index: usize,
) -> Result<Vec<u8>, CogError> {
    let overview = metadata
        .levels
        .get(level)
        .ok_or(CogError::TileNotFound(tile_index))?;

    let offset = *overview
        .tile_offsets
        .get(tile_index)
        .ok_or(CogError::TileNotFound(tile_index))?;
    let length = *overview
        .tile_byte_counts
        .get(tile_index)
        .ok_or(CogError::TileNotFound(tile_index))?;

    reader.seek(SeekFrom::Start(offset))?;
    let mut buf = vec![0u8; length as usize];
    reader.read_exact(&mut buf)?;

    Ok(buf)
}

/// Compute which tile indices cover a given pixel bounding box.
pub fn tiles_for_bounds(
    level: &OverviewLevel,
    min_x: u32,
    min_y: u32,
    max_x: u32,
    max_y: u32,
) -> Vec<usize> {
    let start_col = min_x / level.tile_width;
    let end_col = max_x.div_ceil(level.tile_width);
    let start_row = min_y / level.tile_height;
    let end_row = max_y.div_ceil(level.tile_height);

    let mut indices = Vec::new();
    for row in start_row..end_row.min(level.tiles_down) {
        for col in start_col..end_col.min(level.tiles_across) {
            indices.push((row * level.tiles_across + col) as usize);
        }
    }
    indices
}

// Helper functions

fn read_u16(buf: &[u8], order: ByteOrder) -> u16 {
    match order {
        ByteOrder::LittleEndian => u16::from_le_bytes([buf[0], buf[1]]),
        ByteOrder::BigEndian => u16::from_be_bytes([buf[0], buf[1]]),
    }
}

fn read_u32(buf: &[u8], order: ByteOrder) -> u32 {
    match order {
        ByteOrder::LittleEndian => u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]),
        ByteOrder::BigEndian => u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]),
    }
}

fn read_offset_array<R: Read + Seek>(
    reader: &mut R,
    value_offset: u64,
    count: u32,
    byte_order: ByteOrder,
    data_type: u16,
) -> Result<Vec<u64>, CogError> {
    let current_pos = reader.stream_position()?;

    if count == 1 {
        return Ok(vec![value_offset]);
    }

    reader.seek(SeekFrom::Start(value_offset))?;
    let mut offsets = Vec::with_capacity(count as usize);

    let bytes_per_entry = match data_type {
        3 => 2,  // SHORT
        4 => 4,  // LONG
        16 => 8, // LONG8 (BigTIFF)
        _ => 4,
    };

    for _ in 0..count {
        let mut buf = vec![0u8; bytes_per_entry];
        reader.read_exact(&mut buf)?;
        let val = match bytes_per_entry {
            2 => read_u16(&buf, byte_order) as u64,
            4 => read_u32(&buf, byte_order) as u64,
            8 => match byte_order {
                ByteOrder::LittleEndian => u64::from_le_bytes([
                    buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7],
                ]),
                ByteOrder::BigEndian => u64::from_be_bytes([
                    buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7],
                ]),
            },
            _ => 0,
        };
        offsets.push(val);
    }

    reader.seek(SeekFrom::Start(current_pos))?;
    Ok(offsets)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn create_minimal_tiff() -> Vec<u8> {
        let mut data = Vec::new();
        // Little-endian TIFF header
        data.extend_from_slice(b"II"); // Byte order
        data.extend_from_slice(&42u16.to_le_bytes()); // Magic
        data.extend_from_slice(&8u32.to_le_bytes()); // IFD offset

        // IFD with 4 entries
        data.extend_from_slice(&4u16.to_le_bytes()); // Entry count

        // Tag 256: ImageWidth = 512
        data.extend_from_slice(&256u16.to_le_bytes());
        data.extend_from_slice(&4u16.to_le_bytes()); // LONG
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&512u32.to_le_bytes());

        // Tag 257: ImageLength = 512
        data.extend_from_slice(&257u16.to_le_bytes());
        data.extend_from_slice(&4u16.to_le_bytes());
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&512u32.to_le_bytes());

        // Tag 322: TileWidth = 256
        data.extend_from_slice(&322u16.to_le_bytes());
        data.extend_from_slice(&4u16.to_le_bytes());
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&256u32.to_le_bytes());

        // Tag 323: TileLength = 256
        data.extend_from_slice(&323u16.to_le_bytes());
        data.extend_from_slice(&4u16.to_le_bytes());
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&256u32.to_le_bytes());

        // Next IFD offset = 0 (no more IFDs)
        data.extend_from_slice(&0u32.to_le_bytes());

        data
    }

    #[test]
    fn test_parse_minimal_tiff() {
        let data = create_minimal_tiff();
        let mut cursor = Cursor::new(data);
        let metadata = parse_cog_metadata(&mut cursor).unwrap();
        assert_eq!(metadata.levels.len(), 1);
        assert_eq!(metadata.levels[0].width, 512);
        assert_eq!(metadata.levels[0].height, 512);
        assert_eq!(metadata.levels[0].tile_width, 256);
        assert_eq!(metadata.levels[0].tiles_across, 2);
        assert_eq!(metadata.levels[0].tiles_down, 2);
    }

    #[test]
    fn test_tiles_for_bounds() {
        let level = OverviewLevel {
            level: 0,
            width: 1024,
            height: 1024,
            tile_width: 256,
            tile_height: 256,
            tiles_across: 4,
            tiles_down: 4,
            tile_offsets: vec![],
            tile_byte_counts: vec![],
        };

        let indices = tiles_for_bounds(&level, 0, 0, 512, 512);
        assert_eq!(indices.len(), 4); // 2x2 tiles
        assert_eq!(indices, vec![0, 1, 4, 5]);
    }
}
