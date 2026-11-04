#!/usr/bin/env python3
"""Download only the observed train/official-val archives; preserve existing data.

Uses ordinary public Google Drive download confirmation, never saved credentials.
Run this on the school server, not on the laptop, for the actual large downloads.
"""
from __future__ import annotations

import argparse
import hashlib
import html.parser
import http.cookiejar
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
import zlib


DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / "data_manifest" / "drive_manifest.json"


def log(message: str) -> None:
    print(time.strftime("%Y-%m-%dT%H:%M:%S%z"), message, flush=True)


class DownloadForm(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.forms: list[tuple[str, dict[str, str]]] = []
        self._active: tuple[str, dict[str, str]] | None = None

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == "form":
            self._active = (values.get("action", ""), {})
            self.forms.append(self._active)
        elif tag == "input" and self._active and values.get("name"):
            self._active[1][values["name"]] = values.get("value", "")

    def handle_endtag(self, tag):
        if tag == "form":
            self._active = None


def assert_google_url(url: str) -> None:
    parsed = urllib.parse.urlsplit(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or not (
        host == "google.com" or host.endswith(".google.com")
        or host == "googleusercontent.com" or host.endswith(".googleusercontent.com")
    ):
        raise ValueError("Download redirected outside the allowed HTTPS Google domains")


def open_download(opener, file_id: str, offset: int):
    """Return a binary streaming response, following the ordinary confirmation form."""
    url = "https://drive.google.com/uc?export=download&id=" + urllib.parse.quote(file_id, safe="")
    for _ in range(4):
        assert_google_url(url)
        headers = {"User-Agent": "fashion-tagging-baseline-data/1.0"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        response = opener.open(urllib.request.Request(url, headers=headers), timeout=90)
        assert_google_url(response.url)
        if "text/html" not in response.headers.get("Content-Type", "").lower():
            return response
        try:
            body = response.read(256 * 1024 + 1)
            if len(body) > 256 * 1024:
                raise RuntimeError("Oversized HTML response instead of the requested ZIP")
            parser = DownloadForm()
            parser.feed(body.decode("utf-8", errors="replace"))
            forms = []
            for action, fields in parser.forms:
                candidate = urllib.parse.urljoin(response.url, action)
                parsed = urllib.parse.urlsplit(candidate)
                if parsed.hostname == "drive.usercontent.google.com" and fields.get("id") == file_id:
                    forms.append((candidate, fields))
            if len(forms) != 1:
                raise RuntimeError("Drive did not return its normal public download form; check sharing or quota in Drive")
            action, fields = forms[0]
            url = action + ("&" if urllib.parse.urlsplit(action).query else "?") + urllib.parse.urlencode(fields)
        finally:
            response.close()
    raise RuntimeError("Too many Google Drive download confirmation steps")


def validate_resume_response(response, offset: int, expected_size: int) -> None:
    if offset:
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
        if response.status != 206 or not match:
            raise RuntimeError("Server did not honor Range; existing partial archive has been preserved")
        if int(match[1]) != offset or int(match[3]) != expected_size:
            raise RuntimeError("Content-Range does not match the saved partial archive and expected total size")
    elif response.status not in (200, 206):
        raise RuntimeError(f"Unexpected download HTTP status {response.status}")
    elif response.status == 206:
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
        if not match or int(match[1]) != 0 or int(match[3]) != expected_size:
            raise RuntimeError("Initial response has an unexpected Content-Range")
    length = response.headers.get("Content-Length")
    if length is not None and int(length) != expected_size - offset:
        raise RuntimeError("Download length disagrees with the observed Drive file size")


def safe_zip_members(zf: zipfile.ZipFile, spec: dict) -> list[zipfile.ZipInfo]:
    names: set[str] = set()
    members = zf.infolist()
    required = set(spec["metadata_files"])
    image_names = set()
    prefix = spec["zip_prefix"]
    for info in members:
        name = info.filename
        original = getattr(info, "orig_filename", name)
        parts = PurePosixPath(name).parts
        if name != original or "\x00" in name or "\\" in name or not parts or name.startswith("/"):
            raise ValueError("Unsafe ZIP path")
        if any(part in (".", "..") or ":" in part for part in parts) or parts[0] != prefix:
            raise ValueError(f"ZIP entry outside expected {prefix}/ root")
        if name in names:
            raise ValueError(f"Duplicate ZIP entry: {name}")
        names.add(name)
        if stat.S_ISLNK(info.external_attr >> 16):
            raise ValueError(f"ZIP symlink is not allowed: {name}")
        if info.is_dir():
            continue
        if name in required:
            continue
        if len(parts) == 3 and parts[1] == "images" and parts[2].lower().endswith(".jpg"):
            image_names.add(name)
        else:
            raise ValueError(f"Unexpected file in baseline archive: {name}")
    if not required.issubset(names):
        raise ValueError("Archive is missing observed label/README metadata files")
    if len(image_names) != spec["image_count"]:
        raise ValueError(f"Expected {spec['image_count']} images, found {len(image_names)}")
    files = [info for info in members if not info.is_dir()]
    if len(files) != spec["image_count"] + len(required):
        raise ValueError("Archive file inventory differs from the observed dataset")
    return members


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_archive(path: Path, spec: dict, check_crc: bool = True) -> dict:
    if path.stat().st_size != spec["size_bytes"]:
        raise ValueError(f"Existing archive has the wrong size and was preserved: {path}")
    with zipfile.ZipFile(path) as zf:
        members = safe_zip_members(zf, spec)
        if check_crc:
            bad = zf.testzip()
            if bad:
                raise ValueError(f"Archive CRC validation failed: {bad}")
    return {"file": str(path.resolve()), "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path), "crc_checked": check_crc,
            "entry_count": len(members), "image_count": spec["image_count"]}


def download_one(spec: dict, archive_dir: Path, retries: int = 4) -> tuple[Path, dict]:
    if not spec.get("allowed_for_baseline") or spec.get("zip_prefix") not in ("train", "val"):
        raise ValueError("Only the observed train and official-val archives may be downloaded")
    archive_dir.mkdir(parents=True, exist_ok=True)
    final = archive_dir / spec["name"]
    partial = archive_dir / (spec["name"] + ".part")
    if final.exists():
        log(f"Reusing and verifying existing archive: {final.name}")
        return final, validate_archive(final, spec)
    expected = spec["size_bytes"]
    if partial.exists() and partial.stat().st_size > expected:
        raise ValueError(f"Oversized partial archive preserved: {partial}")
    for attempt in range(retries):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == expected:
            break
        opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
        try:
            with open_download(opener, spec["id"], offset) as response:
                validate_resume_response(response, offset, expected)
                log(f"Downloading {spec['name']}: {offset:,}/{expected:,} bytes")
                mode = "ab" if offset else "xb"
                # An empty saved partial is safe to append; never truncate an existing file.
                if not offset and partial.exists():
                    mode = "ab"
                last_report = time.monotonic()
                with partial.open(mode) as stream:
                    count = offset
                    while True:
                        block = response.read(4 * 1024 * 1024)
                        if not block:
                            break
                        if count + len(block) > expected:
                            raise RuntimeError("Download exceeded expected archive size")
                        stream.write(block)
                        count += len(block)
                        if time.monotonic() - last_report >= 15:
                            log(f"{spec['name']}: {count:,}/{expected:,} bytes ({count / expected:.1%})")
                            last_report = time.monotonic()
                    stream.flush()
                    os.fsync(stream.fileno())
                if count != expected:
                    raise RuntimeError(f"Incomplete archive: {count:,}/{expected:,} bytes")
            break
        except (OSError, urllib.error.URLError, RuntimeError) as error:
            # The error class is useful; URLs may contain temporary query values, so omit str(error).
            log(f"Download attempt {attempt + 1}/{retries} failed ({type(error).__name__}); partial bytes preserved")
            if attempt + 1 == retries:
                raise RuntimeError(f"Download failed after {retries} attempts; retained {partial.name}") from None
            time.sleep(min(2 ** attempt, 8))
    log(f"Verifying CRC and SHA256: {partial.name}")
    record = validate_archive(partial, spec)
    # Atomic, non-overwriting promotion on the same filesystem.
    os.link(partial, final)
    partial.unlink()
    record["file"] = str(final.resolve())
    return final, record


def existing_file_matches(path: Path, info: zipfile.ZipInfo) -> bool:
    if path.is_symlink() or not path.is_file() or path.stat().st_size != info.file_size:
        return False
    crc = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            crc = zlib.crc32(block, crc)
    return crc & 0xFFFFFFFF == info.CRC


def extract_preserving(archive: Path, destination: Path, spec: dict) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise ValueError("Extraction root must not be a symlink")
    base = destination.resolve()
    extracted = reused = 0
    with zipfile.ZipFile(archive) as zf:
        members = safe_zip_members(zf, spec)
        for info in members:
            target = base.joinpath(*PurePosixPath(info.filename).parts)
            if not target.resolve().is_relative_to(base):
                raise ValueError("Extraction path escaped the dataset root")
            for parent in (target.parent, *target.parents):
                if parent == base.parent:
                    break
                if parent.is_symlink():
                    raise ValueError("Extraction through a symlink is not allowed")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() or target.is_symlink():
                if not existing_file_matches(target, info):
                    raise ValueError(f"Existing different file preserved: {target}")
                reused += 1
                continue
            temporary = target.with_name(target.name + f".extract-{os.getpid()}.part")
            try:
                with zf.open(info) as source, temporary.open("xb") as stream:
                    while True:
                        block = source.read(4 * 1024 * 1024)
                        if not block:
                            break
                        stream.write(block)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.link(temporary, target)  # refuses to overwrite even if another writer raced us
                extracted += 1
            finally:
                if temporary.exists():
                    temporary.unlink()
            if extracted % 5000 == 0:
                log(f"Extracted {extracted:,} new files from {archive.name}")
    return {"root": str((base / spec["zip_prefix"]).resolve()), "new_files": extracted, "reused_files": reused}


def write_json_preserving(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        stamp = time.strftime("%Y%m%dT%H%M%S")
        path = path.with_name(path.stem + f"_{stamp}_{os.getpid()}" + path.suffix)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    log(f"Saved actual download record: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--extract-dir", type=Path)
    parser.add_argument("--splits", nargs="+", choices=("train", "val"), default=("train", "val"))
    parser.add_argument("--retries", type=int, default=4)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    result = {"completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "manifest": str(args.manifest.resolve()),
              "excluded": ["public_test", "examples"], "archives": {}}
    for partition in args.splits:
        spec = manifest["files"][partition]
        archive, record = download_one(spec, args.archive_dir.resolve(), args.retries)
        if args.extract_dir:
            record["extraction"] = extract_preserving(archive, args.extract_dir, spec)
        result["archives"][partition] = record
        write_json_preserving(args.archive_dir / f"{partition}_download_record.json", record)
    result["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write_json_preserving(args.archive_dir / "download_record.json", result)


if __name__ == "__main__":
    main()
