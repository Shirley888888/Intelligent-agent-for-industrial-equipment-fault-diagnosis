#!/usr/bin/env python3
"""Download the original ETT-small oil-temperature CSVs from the official GitHub repo.

The CSV payloads are saved byte-for-byte (no transformation). A local manifest records
SHA-256, row count, timestamp cadence, and upstream attribution.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

# Pin the upstream commit so future changes to the default branch cannot silently
# change a training corpus. Blob SHAs were read from the GitHub Contents API.
SOURCE_COMMIT = "1d16c8f4f943005d613b5bc962e9eeb06058cf07"
BASE_URL = f"https://raw.githubusercontent.com/zhouhaoyi/ETDataset/{SOURCE_COMMIT}/ETT-small"
FILES = {"ETTh1": "1h", "ETTh2": "1h", "ETTm1": "15min", "ETTm2": "15min"}
EXPECTED_GIT_BLOB_SHA1 = {
    "ETTh1": "a52c4925778c07c1ef1a2cf6fd01594919717d9e",
    "ETTh2": "6e3760778598fdee5208213d3fce122637307eec",
    "ETTm1": "62c9f979e2dc5b05d9cc5b38891e60e6baf77417",
    "ETTm2": "f9d46027c987d442d2d8f81308d1d8c4d1d35db8",
}
COLUMNS = ["date", "HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
SOURCE_REPO = "https://github.com/zhouhaoyi/ETDataset"
LICENSE = "CC BY-ND 4.0"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git_blob_sha1(path: Path) -> str:
    """Return Git's canonical blob SHA-1 (header + exact file bytes)."""
    size = path.stat().st_size
    h = hashlib.sha1(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_csv(path: Path, name: str) -> dict:
    if name not in FILES:
        raise ValueError(f"Unknown ETT dataset: {name}")
    expected_minutes = 60 if FILES[name] == "1h" else 15
    rows = 0
    first = last = None
    previous = None
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != COLUMNS:
            raise ValueError(f"{name}: columns must be exactly {COLUMNS}, got {reader.fieldnames}")
        for line_no, row in enumerate(reader, start=2):
            t = datetime.fromisoformat(row["date"])
            if previous is not None:
                delta = int((t - previous).total_seconds() // 60)
                if delta != expected_minutes:
                    raise ValueError(f"{name}: timestamp cadence error at line {line_no}: {delta} min")
            for col in COLUMNS[1:]:
                value = float(row[col])
                if not (value == value and abs(value) != float("inf")):
                    raise ValueError(f"{name}: non-finite {col} at line {line_no}")
            first = first or t
            last, previous = t, t
            rows += 1
    min_rows = 17_000 if expected_minutes == 60 else 65_000
    if rows < min_rows:
        raise ValueError(f"{name}: unexpectedly short dataset ({rows} rows; expected >= {min_rows})")
    actual_blob_sha1 = git_blob_sha1(path)
    expected_blob_sha1 = EXPECTED_GIT_BLOB_SHA1[name]
    if actual_blob_sha1 != expected_blob_sha1:
        raise ValueError(
            f"{name}: upstream Git blob checksum mismatch; expected {expected_blob_sha1}, "
            f"got {actual_blob_sha1}. Refusing to train on an unverified dataset."
        )
    return {"file": path.name, "rows": rows, "frequency": FILES[name],
            "first_timestamp": first.isoformat(), "last_timestamp": last.isoformat(),
            "sha256": sha256(path), "github_blob_sha1": actual_blob_sha1,
            "source_commit": SOURCE_COMMIT,
            "source_url": f"{BASE_URL}/{name}.csv"}


def download(name: str, output_dir: Path, force: bool = False) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{name}.csv"
    if target.exists() and not force:
        try:
            return validate_csv(target, name)
        except Exception:
            target.unlink(missing_ok=True)
    url = f"{BASE_URL}/{name}.csv"
    temp = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "oil-temperature-research/1.0"})
    last_error = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=60) as response, temp.open("wb") as out:
                if response.status != 200:
                    raise RuntimeError(f"HTTP {response.status} for {url}")
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    out.write(block)
            info = validate_csv(temp, name)
            os.replace(temp, target)
            return {**info, "file": target.name}
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, RuntimeError) as exc:
            last_error = exc
            temp.unlink(missing_ok=True)
            if attempt < 3:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to download {name} from {url}: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download original ETT-small transformer oil-temperature data")
    parser.add_argument("--output-dir", default="data/ett")
    parser.add_argument("--names", nargs="+", default=list(FILES), choices=list(FILES))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    out = Path(args.output_dir)
    records = []
    for name in args.names:
        record = download(name, out, args.force)
        records.append(record)
        print(f"[OK] {name}: rows={record['rows']} cadence={record['frequency']} sha256={record['sha256']}")
    manifest_path = out / "source_manifest.json"
    previous = {}
    if manifest_path.exists():
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8")).get("files", {})
        except Exception:
            previous = {}
    for item in records:
        previous[item["file"]] = item
    expected_files = {
        f"{name}.csv": {
            "frequency": frequency,
            "github_blob_sha1": EXPECTED_GIT_BLOB_SHA1[name],
            "source_url": f"{BASE_URL}/{name}.csv",
        }
        for name, frequency in FILES.items()
    }
    manifest = {"dataset": "ETT-small", "source_repository": SOURCE_REPO,
                "source_commit": SOURCE_COMMIT,
                "expected_files": expected_files,
                "license": LICENSE,
                "attribution": "Zhou et al., Informer, AAAI 2021; data collected with Beijing Guowang Fuda Science & Technology Development Company.",
                "files": previous}
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, manifest_path)
    print(f"Manifest: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
