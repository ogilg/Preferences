"""Repair truncated npz files by extracting readable arrays and re-saving."""

from __future__ import annotations

import argparse
import struct
import zipfile
from pathlib import Path

import numpy as np


def find_local_file_headers(data: bytes) -> list[tuple[int, str, int, int]]:
    """Find all local file headers in the zip data.

    Returns list of (offset, filename, compressed_size, uncompressed_size).
    """
    headers = []
    pos = 0
    while pos < len(data) - 30:
        # Look for local file header signature: PK\x03\x04
        if data[pos:pos+4] == b'PK\x03\x04':
            # Parse header
            version = struct.unpack('<H', data[pos+4:pos+6])[0]
            flags = struct.unpack('<H', data[pos+6:pos+8])[0]
            compression = struct.unpack('<H', data[pos+8:pos+10])[0]

            # Check for ZIP64 (sizes will be 0xFFFFFFFF)
            compressed_size = struct.unpack('<I', data[pos+18:pos+22])[0]
            uncompressed_size = struct.unpack('<I', data[pos+22:pos+26])[0]

            filename_len = struct.unpack('<H', data[pos+26:pos+28])[0]
            extra_len = struct.unpack('<H', data[pos+28:pos+30])[0]

            filename = data[pos+30:pos+30+filename_len].decode('utf-8')

            # Handle ZIP64 extra field
            extra_start = pos + 30 + filename_len
            extra_data = data[extra_start:extra_start+extra_len]

            if compressed_size == 0xFFFFFFFF or uncompressed_size == 0xFFFFFFFF:
                # Parse ZIP64 extra field
                extra_pos = 0
                while extra_pos < len(extra_data) - 4:
                    header_id = struct.unpack('<H', extra_data[extra_pos:extra_pos+2])[0]
                    field_size = struct.unpack('<H', extra_data[extra_pos+2:extra_pos+4])[0]
                    if header_id == 0x0001:  # ZIP64 extended info
                        field_data = extra_data[extra_pos+4:extra_pos+4+field_size]
                        if uncompressed_size == 0xFFFFFFFF and len(field_data) >= 8:
                            uncompressed_size = struct.unpack('<Q', field_data[0:8])[0]
                        if compressed_size == 0xFFFFFFFF and len(field_data) >= 16:
                            compressed_size = struct.unpack('<Q', field_data[8:16])[0]
                        break
                    extra_pos += 4 + field_size

            data_start = pos + 30 + filename_len + extra_len
            headers.append((pos, filename, compressed_size, uncompressed_size, data_start))
            pos = data_start + compressed_size
        else:
            pos += 1

    return headers


def extract_npy_from_bytes(data: bytes) -> np.ndarray:
    """Extract numpy array from raw .npy bytes."""
    import io
    return np.load(io.BytesIO(data), allow_pickle=True)


def repair_npz(input_path: Path, output_path: Path) -> dict[str, int]:
    """Repair a truncated npz file by extracting readable arrays.

    Returns dict with stats about what was recovered.
    """
    print(f"Reading {input_path}...")
    with open(input_path, 'rb') as f:
        data = f.read()

    print(f"File size: {len(data):,} bytes")

    headers = find_local_file_headers(data)
    print(f"Found {len(headers)} file entries")

    arrays = {}
    recovered = 0
    failed = 0

    for i, (offset, filename, comp_size, uncomp_size, data_start) in enumerate(headers):
        # Check if we have enough data
        data_end = data_start + comp_size
        if data_end > len(data):
            print(f"  {filename}: truncated (need {data_end:,}, have {len(data):,})")
            failed += 1
            continue

        try:
            array_data = data[data_start:data_end]
            arr = extract_npy_from_bytes(array_data)
            name = filename.replace('.npy', '')
            arrays[name] = arr
            print(f"  {filename}: shape={arr.shape}, dtype={arr.dtype}")
            recovered += 1
        except Exception as e:
            print(f"  {filename}: failed to parse - {e}")
            failed += 1

    if arrays:
        print(f"\nSaving {len(arrays)} arrays to {output_path}...")
        np.savez(output_path, **arrays)
        print("Done!")
    else:
        print("No arrays recovered!")

    return {"recovered": recovered, "failed": failed, "total": len(headers)}


def main():
    parser = argparse.ArgumentParser(description="Repair truncated npz files")
    parser.add_argument("input", type=Path, help="Input npz file (truncated)")
    parser.add_argument("--output", type=Path, help="Output path (default: input with _repaired suffix)")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: {args.input} not found")
        return 1

    output = args.output or args.input.with_stem(args.input.stem + "_repaired")

    stats = repair_npz(args.input, output)
    print(f"\nRecovered {stats['recovered']}/{stats['total']} arrays")

    return 0 if stats['recovered'] > 0 else 1


if __name__ == "__main__":
    exit(main())
