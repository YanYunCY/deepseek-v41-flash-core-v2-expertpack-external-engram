#!/usr/bin/env python3
"""Prepare and run the published DeepSeek V4.1 external-Engram release.

Model assets remain in --data-root.  The only write into that tree is the
small engram.runtime.index file; source, build, receipts, and logs are below
--root.  "start" deliberately has no ModelScope dependency or download path.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import signal
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence
from urllib.request import Request, urlopen

REPO = "Yanyunawa/DeepSeek-V4.1-Flash-MXFP4-GGUF"
RELEASE_ID = "20260916-external-engram"
REVISION = "master"
REMOTE_RELEASE = "deployment/20260916/release.json"
ROLES = {"core", "expert_manifest", "engram_manifest", "expert", "engram",
         "source", "source_hashes"}
ROLE_COUNTS = {"core": 1, "expert_manifest": 1, "engram_manifest": 1,
               "expert": 49, "engram": 104, "source": 1, "source_hashes": 1}
CHUNK = 8 * 1024 * 1024
RESERVE = 4 * 1024 ** 3


class DeploymentError(RuntimeError):
    """An unsafe or incomplete deployment state."""


@dataclass(frozen=True)
class Asset:
    path: str
    size: int
    sha256: str
    role: str


def die(message: str) -> None:
    raise DeploymentError(message)


def safe_rel(value: Any, label: str) -> str:
    """Return a no-whitespace, normalized POSIX relative path."""
    if not isinstance(value, str) or not value:
        die(f"{label} must be a non-empty relative POSIX path")
    if "\\" in value or any(char.isspace() for char in value):
        die(f"{label} must not contain whitespace or backslashes: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(p in ("", ".", "..") for p in path.parts):
        die(f"{label} is not a safe relative path: {value!r}")
    if "/".join(path.parts) != value:
        die(f"{label} is not normalized: {value!r}")
    return value


# Compatibility name retained for simple staging tests.
safe_remote_path = safe_rel


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        die(f"cannot read {label} ({path}): {exc}")


def atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        stream.write(data)
        temporary = Path(stream.name)
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, (json.dumps(value, sort_keys=True, indent=2) + "\n").encode())


def valid_sha(value: Any, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64 or value != value.lower()
            or any(c not in "0123456789abcdef" for c in value)):
        die(f"{label} must be a lowercase SHA256 digest")
    return value


def positive_int(value: Any, label: str, zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if zero else 1):
        die(f"{label} must be an integer {'>= 0' if zero else '> 0'}")
    return value


def parse_asset(value: Any, index: int) -> Asset:
    if not isinstance(value, Mapping):
        die(f"assets[{index}] must be an object")
    role = value.get("role")
    if role not in ROLES:
        die(f"assets[{index}].role is invalid: {role!r}")
    return Asset(safe_rel(value.get("path"), f"assets[{index}].path"),
                 positive_int(value.get("size"), f"assets[{index}].size"),
                 valid_sha(value.get("sha256"), f"assets[{index}].sha256"), role)


def parse_row(value: Any, index: int, engram: Mapping[str, Asset]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        die(f"engram_rows[{index}] must be an object")
    names = ("layer", "row_start", "row_count", "weight_path", "weight_row_bytes",
             "scale_path", "scale_row_bytes")
    if any(name not in value for name in names):
        die(f"engram_rows[{index}] is missing a required field")
    row = {
        "layer": positive_int(value["layer"], f"engram_rows[{index}].layer"),
        "row_start": positive_int(value["row_start"], f"engram_rows[{index}].row_start", True),
        "row_count": positive_int(value["row_count"], f"engram_rows[{index}].row_count"),
        "weight_path": safe_rel(value["weight_path"], f"engram_rows[{index}].weight_path"),
        "weight_row_bytes": positive_int(value["weight_row_bytes"], f"engram_rows[{index}].weight_row_bytes"),
        "scale_path": safe_rel(value["scale_path"], f"engram_rows[{index}].scale_path"),
        "scale_row_bytes": positive_int(value["scale_row_bytes"], f"engram_rows[{index}].scale_row_bytes"),
    }
    if row["weight_path"] == row["scale_path"] or row["weight_path"] not in engram or row["scale_path"] not in engram:
        die(f"engram_rows[{index}] must reference two distinct engram assets")
    if (engram[row["weight_path"]].size != row["row_count"] * row["weight_row_bytes"]
            or engram[row["scale_path"]].size != row["row_count"] * row["scale_row_bytes"]):
        die(f"engram_rows[{index}] byte sizes do not match the referenced assets")
    return row


def continuous_rows(rows: Sequence[Mapping[str, Any]], label: str) -> None:
    """Require each layer's chunks to cover an exact [0,total) range."""
    seen: set[str] = set()
    previous: dict[int, int] = {}
    for row in sorted(rows, key=lambda x: (x["layer"], x["row_start"])):
        layer = row["layer"]
        expected = previous.get(layer, 0)
        if row["row_start"] != expected:
            die(f"{label} layer {layer} has a gap or overlap at row {row['row_start']}")
        previous[layer] = expected + row["row_count"]
        for path in (row["weight_path"], row["scale_path"]):
            if path in seen:
                die(f"{label} reuses engram sidecar {path}")
            seen.add(path)


def validate_release(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        die("release descriptor must be a JSON object")
    if value.get("id") != RELEASE_ID or value.get("repo") != REPO:
        die(f"release id/repo must be {RELEASE_ID!r}/{REPO!r}")
    if value.get("revision") != REVISION:
        die(f"release revision must be {REVISION!r}")
    raw_assets = value.get("assets")
    if not isinstance(raw_assets, list):
        die("release assets must be an array")
    assets = [parse_asset(item, index) for index, item in enumerate(raw_assets)]
    if len({asset.path for asset in assets}) != len(assets):
        die("release contains duplicate asset paths")
    counts = {role: sum(asset.role == role for asset in assets) for role in ROLES}
    if counts != ROLE_COUNTS:
        die(f"release role counts must be {ROLE_COUNTS}, got {counts}")
    defaults = value.get("defaults")
    if not isinstance(defaults, Mapping):
        die("defaults must be an object")
    for name in ("gpu_cache_gib", "host_cache_gib"):
        number = defaults.get(name)
        if isinstance(number, bool) or not isinstance(number, int) or number < 0:
            die(f"defaults.{name} must be a non-negative integer")
    for name in ("io_threads", "threads", "batch", "context", "port"):
        positive_int(defaults.get(name), f"defaults.{name}")
    if "parallel" in defaults:
        positive_int(defaults["parallel"], "defaults.parallel")
    if defaults["port"] > 65535:
        die("defaults.port is outside the TCP range")
    raw_rows = value.get("engram_rows")
    if not isinstance(raw_rows, list) or len(raw_rows) != 52:
        die("release must contain exactly 52 engram_rows")
    engram = {asset.path: asset for asset in assets if asset.role == "engram"}
    rows = [parse_row(row, index, engram) for index, row in enumerate(raw_rows)]
    continuous_rows(rows, "release engram rows")
    used = {path for row in rows for path in (row["weight_path"], row["scale_path"])}
    if used != set(engram):
        die("release engram_rows must consume all 104 engram assets exactly once")
    return {"id": RELEASE_ID, "repo": REPO, "revision": REVISION, "assets": assets,
            "defaults": dict(defaults), "engram_rows": rows}


def under(base: Path, relative: str, label: str) -> Path:
    base = base.resolve()
    candidate = base / relative
    try:
        candidate.resolve(strict=False).relative_to(base)
    except ValueError:
        die(f"{label} resolves outside --data-root")
    current = base
    for part in PurePosixPath(relative).parts:
        current /= part
        if current.is_symlink():
            die(f"{label} uses a symlink: {current}")
    return candidate


def local_asset(data_root: Path, asset: Asset) -> Path:
    path = under(data_root, asset.path, asset.role)
    if path.is_symlink() or not path.is_file():
        die(f"missing or non-regular {asset.role}: {path}")
    return path


def stat_record(path: Path) -> dict[str, Any]:
    item = path.stat()
    return {"path": str(path), "size": item.st_size, "mtime_ns": item.st_mtime_ns,
            "ctime_ns": item.st_ctime_ns}


def verify_asset(data_root: Path, asset: Asset, old: Mapping[str, Any] | None,
                 rehash: bool) -> dict[str, Any]:
    path = local_asset(data_root, asset)
    before = stat_record(path)
    if (not rehash and isinstance(old, Mapping) and old.get("after") == before
            and old.get("sha256") == asset.sha256):
        return dict(old)
    actual = sha256_file(path)
    after = stat_record(path)
    if before != after:
        die(f"file changed while hashing: {asset.path}")
    if before["size"] != asset.size or actual != asset.sha256:
        die(f"size or SHA256 mismatch: {asset.path}")
    return {"role": asset.role, "sha256": actual, "before": before, "after": after}


def download_file(path: str, data_root: Path, revision: str) -> Path:
    try:
        from modelscope.hub.file_download import model_file_download
    except ImportError:
        die("prepare needs ModelScope: python3 -m pip install modelscope")
    try:
        result = model_file_download(model_id=REPO, file_path=path, revision=revision,
                                     local_dir=str(data_root))
    except Exception as exc:
        die(f"ModelScope download failed for {path}: {exc}")
    expected = under(data_root, path, "ModelScope result")
    returned = Path(result) if result else expected
    if not returned.is_absolute():
        returned = data_root / returned
    if returned.resolve(strict=False) != expected.resolve(strict=False):
        die(f"ModelScope returned unexpected file for {path}: {returned}")
    return expected


def download_assets(release: Mapping[str, Any], data_root: Path) -> None:
    assets: Sequence[Asset] = release["assets"]
    missing = [asset for asset in assets if not (under(data_root, asset.path, asset.role).is_file()
                                                 and not under(data_root, asset.path, asset.role).is_symlink())]
    required = sum(asset.size for asset in missing) + RESERVE
    if shutil.disk_usage(data_root).free < required:
        die(f"not enough space for missing assets plus 4 GiB reserve ({required} bytes)")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(download_file, asset.path, data_root, release["revision"]) for asset in missing]
        for future in futures:
            future.result()


def parse_expert_manifest(path: Path, experts: Sequence[Asset]) -> None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as exc:
        die(f"expert manifest is not UTF-8: {exc}")
    useful = [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]
    if not useful or useful[0].split() != ["DSEXP2", "2"]:
        die("expert manifest header must be exactly 'DSEXP2 2'")
    packs: list[tuple[int, str, int, str]] = []
    parent = PurePosixPath(next(asset.path for asset in experts if asset.role == "expert_manifest")).parent
    for line in useful[1:]:
        fields = line.split()
        if not fields or fields[0] != "PACK":
            continue
        if len(fields) != 5 or not fields[1].isdigit():
            die(f"invalid DSEXP2 PACK line: {line!r}")
        pack_id, filename, size, digest = int(fields[1]), fields[2], fields[3], fields[4]
        try:
            remote = safe_rel((parent / safe_rel(filename, "PACK filename")).as_posix(), "PACK path")
            packs.append((pack_id, remote, positive_int(int(size), "PACK size"), valid_sha(digest, "PACK sha256")))
        except ValueError:
            die(f"invalid DSEXP2 PACK size: {line!r}")
    if [item[0] for item in packs] != list(range(49)):
        die("DSEXP2 PACK ids must be contiguous 0 through 48")
    expected = {asset.path: asset for asset in experts if asset.role == "expert"}
    if len(packs) != len(expected):
        die("DSEXP2 PACK count does not match expert-role assets")
    for _, remote, size, digest in packs:
        asset = expected.get(remote)
        if asset is None or asset.size != size or asset.sha256 != digest:
            die(f"DSEXP2 PACK differs from release expert asset: {remote}")


def manifest_sidecar(value: Mapping[str, Any], name: str) -> tuple[str, int, int, str]:
    nested = value.get(name)
    if not isinstance(nested, Mapping):
        die(f"Engram {name} must be an object")
    path = safe_rel(nested.get("path", nested.get("remote_path")), f"Engram {name} path")
    row_bytes = positive_int(nested.get("row_bytes"), f"Engram {name}_row_bytes")
    size = positive_int(nested.get("size", nested.get("bytes")), f"Engram {name} size")
    digest = valid_sha(nested.get("sha256"), f"Engram {name} sha256")
    return path, row_bytes, size, digest


def parse_engram_manifest(path: Path) -> tuple[list[dict[str, Any]], dict[int, int]]:
    raw = read_json(path, "Engram manifest")
    if not isinstance(raw, Mapping) or not isinstance(raw.get("layers"), list):
        die("Engram manifest must contain a layers array")
    rows: list[dict[str, Any]] = []
    totals: dict[int, int] = {}
    for layer_value in raw["layers"]:
        if not isinstance(layer_value, Mapping) or not isinstance(layer_value.get("chunks"), list):
            die("Engram layer must contain chunks")
        layer = positive_int(layer_value.get("layer"), "Engram layer")
        if layer in totals:
            die(f"Engram manifest repeats layer {layer}")
        total = positive_int(layer_value.get("rows"), "Engram layer rows")
        totals[layer] = total
        for chunk in layer_value["chunks"]:
            if not isinstance(chunk, Mapping):
                die("Engram chunk must be an object")
            weight_path, weight_row_bytes, weight_size, weight_sha = manifest_sidecar(chunk, "weight")
            scale_path, scale_row_bytes, scale_size, scale_sha = manifest_sidecar(chunk, "scale")
            rows.append({"layer": layer,
                         "row_start": positive_int(chunk.get("row_start"), "Engram row_start", True),
                         "row_count": positive_int(chunk.get("row_count"), "Engram row_count"),
                         "weight_path": weight_path, "weight_row_bytes": weight_row_bytes,
                         "weight_size": weight_size, "weight_sha256": weight_sha,
                         "scale_path": scale_path, "scale_row_bytes": scale_row_bytes,
                         "scale_size": scale_size, "scale_sha256": scale_sha})
    if len(rows) != 52:
        die("Engram manifest must contain exactly 52 chunks")
    continuous_rows(rows, "Engram manifest")
    for layer, total in totals.items():
        actual = max(row["row_start"] + row["row_count"] for row in rows if row["layer"] == layer)
        if actual != total:
            die(f"Engram layer {layer} does not cover its stated rows")
    return rows, totals


def check_engram_manifest(path: Path, release: Mapping[str, Any]) -> None:
    manifest_rows, totals = parse_engram_manifest(path)
    release_rows = release["engram_rows"]
    def key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        return tuple(row[name] for name in ("layer", "row_start", "row_count", "weight_path",
                                             "weight_row_bytes", "scale_path", "scale_row_bytes"))
    if {key(row) for row in manifest_rows} != {key(row) for row in release_rows}:
        die("Engram manifest chunks do not exactly match release engram_rows")
    assets = {asset.path: asset for asset in release["assets"] if asset.role == "engram"}
    for row in manifest_rows:
        for name in ("weight", "scale"):
            asset = assets.get(row[name + "_path"])
            if asset is None or asset.size != row[name + "_size"] or asset.sha256 != row[name + "_sha256"]:
                die(f"Engram manifest {name} metadata differs from release asset: {row[name + '_path']}")
    for layer in totals:
        release_end = max(row["row_start"] + row["row_count"] for row in release_rows if row["layer"] == layer)
        if release_end != totals[layer]:
            die(f"release Engram rows do not cover manifest layer {layer}")


def source_hashes(path: Path) -> dict[str, str]:
    raw = read_json(path, "source hashes")
    if not isinstance(raw, Mapping):
        die("source hashes must be a JSON path-to-SHA256 object")
    return {safe_rel(key, "source hashes key"): valid_sha(value, f"source hash {key}")
            for key, value in raw.items()}


def safe_tar_members(archive: tarfile.TarFile) -> list[tarfile.TarInfo]:
    members = archive.getmembers()
    files: set[str] = set()
    for member in members:
        name = member.name[:-1] if member.isdir() and member.name.endswith("/") else member.name
        if name == "runtime":
            relative = ""
        elif name.startswith("runtime/"):
            relative = name[len("runtime/"):]
        else:
            die(f"source archive member is outside runtime/: {member.name!r}")
        if member.issym() or member.islnk() or not (member.isdir() or member.isfile()):
            die(f"source archive has unsafe member: {member.name!r}")
        if relative:
            safe_rel(relative, "source archive member")
        if member.isfile():
            if relative in files:
                die(f"source archive has duplicate member: {relative}")
            files.add(relative)
    return members


def extract_source(root: Path, archive_path: Path, hashes: Mapping[str, str],
                   source_sha: str, hashes_sha: str) -> None:
    target, marker = root / "runtime", root / "state" / "source.json"
    if target.exists():
        if not marker.is_file():
            die(f"refusing to overwrite unrecognized source tree: {target}")
        old = read_json(marker, "source marker")
        if old.get("source_sha256") != source_sha or old.get("source_hashes_sha256") != hashes_sha:
            die("recognized source tree belongs to a different release")
        verify_source_tree(target, hashes)
        return
    with tarfile.open(archive_path, "r:*") as archive:
        members = safe_tar_members(archive)
        files = {member.name[len("runtime/"):] for member in members if member.isfile()}
        if files != set(hashes):
            die("source archive files do not exactly match source_hashes JSON")
        for member in members:
            if member.isfile():
                stream = archive.extractfile(member)
                if stream is None or hashlib.sha256(stream.read()).hexdigest() != hashes[member.name[len("runtime/"):]]:
                    die(f"source hash mismatch: {member.name}")
        temporary = Path(tempfile.mkdtemp(prefix=".runtime-", dir=root))
        try:
            for member in members:
                rel = "" if member.name == "runtime" else member.name[len("runtime/"):]
                output = temporary / rel
                if member.isdir():
                    output.mkdir(parents=True, exist_ok=True)
                else:
                    output.parent.mkdir(parents=True, exist_ok=True)
                    source = archive.extractfile(member)
                    if source is None:
                        die(f"cannot extract {member.name}")
                    with output.open("xb") as stream:
                        shutil.copyfileobj(source, stream)
                    os.chmod(output, member.mode & 0o777)
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    atomic_json(marker, {"source_sha256": source_sha, "source_hashes_sha256": hashes_sha})


def verify_source_tree(source: Path, hashes: Mapping[str, str]) -> None:
    found: set[str] = set()
    for path in source.rglob("*"):
        relative = path.relative_to(source).as_posix()
        if path.is_symlink():
            die(f"recognized source tree contains a symlink: {relative}")
        if path.is_file():
            found.add(relative)
    if found != set(hashes):
        die("recognized source tree files differ from source_hashes JSON")
    for relative, expected in hashes.items():
        if sha256_file(source / relative) != expected:
            die(f"recognized source file hash mismatch: {relative}")


def build_runtime(root: Path, hashes_sha: str) -> Path:
    source, build = root / "runtime", root / "build-engram"
    marker = root / "state" / "build.json"
    if build.exists() and not marker.is_file():
        die(f"refusing to overwrite unrecognized build directory: {build}")
    if marker.is_file():
        build_state = read_json(marker, "build marker")
        if build_state.get("source_hashes_sha256") != hashes_sha:
            die("recognized build belongs to a different source release")
    build.mkdir(parents=True, exist_ok=True)
    atomic_json(marker, {"source_hashes_sha256": hashes_sha, "status": "building"})
    subprocess.run(["cmake", "-S", str(source), "-B", str(build), "-DGGML_HIP=ON",
                    "-DCMAKE_HIP_ARCHITECTURES=gfx942", "-DLLAMA_BUILD_UI=OFF",
                    "-DLLAMA_BUILD_APP=OFF", "-DLLAMA_USE_PREBUILT_UI=OFF",
                    "-DCMAKE_BUILD_TYPE=Release"], check=True)
    subprocess.run(["cmake", "--build", str(build), "--target", "llama-server", "-j8"], check=True)
    binary = build / "bin" / "llama-server"
    if binary.is_symlink() or not binary.is_file():
        die(f"build did not create llama-server: {binary}")
    atomic_json(marker, {"source_hashes_sha256": hashes_sha, "binary": str(binary)})
    return binary


def runtime_identity(binary: Path) -> dict[str, Any]:
    """Record the server plus every colocated shared object it can load."""
    bin_dir = binary.parent.resolve()
    files = [binary] + sorted(path for path in bin_dir.iterdir()
                              if path.name.startswith("lib") and ".so" in path.name)
    result: dict[str, Any] = {}
    for path in files:
        if not path.is_file():
            die(f"runtime product is not a regular file: {path}")
        target = path.resolve()
        try:
            target.relative_to(bin_dir)
        except ValueError:
            die(f"runtime library symlink leaves bin directory: {path}")
        result[path.name] = {"link": os.readlink(path) if path.is_symlink() else None,
                             "stat": stat_record(target), "sha256": sha256_file(target)}
    return result


def index_bytes(index: Path, data_root: Path, rows: Sequence[Mapping[str, Any]]) -> bytes:
    lines = ["DSEGRAM1"]
    for row in sorted(rows, key=lambda value: (value["layer"], value["row_start"])):
        weight = Path(os.path.relpath(data_root / row["weight_path"], index.parent)).as_posix()
        scale = Path(os.path.relpath(data_root / row["scale_path"], index.parent)).as_posix()
        lines.append(f"{row['layer']} {row['row_start']} {row['row_count']} {weight} "
                     f"{row['weight_row_bytes']} {scale} {row['scale_row_bytes']}")
    return ("\n".join(lines) + "\n").encode()


def write_index(index: Path, data_root: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    atomic_bytes(index, index_bytes(index, data_root, rows))


def proc_starttime(pid: int) -> int | None:
    try:
        return int(Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()[19])
    except (OSError, IndexError, ValueError):
        return None


def owns_process(identity: Mapping[str, Any]) -> bool:
    pid, started = identity.get("pid"), identity.get("starttime")
    if not isinstance(pid, int) or not isinstance(started, int):
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return proc_starttime(pid) == started


def port_available(port: int) -> bool:
    # A listener is never safe to replace.  Check it before enabling reuse,
    # which otherwise permits a bind through harmless server-side TIME_WAIT.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.25)
        if probe.connect_ex(("127.0.0.1", port)) == 0:
            return False
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def health_is_ready(host: str, port: int) -> bool:
    try:
        with urlopen(f"http://{host}:{port}/health", timeout=5) as response:
            return response.status == 200
    except Exception:
        return False


def http_post(port: int, endpoint: str, body: bytes) -> tuple[int, bytes]:
    request = Request(f"http://127.0.0.1:{port}{endpoint}", data=body, method="POST")
    request.add_header("Content-Type", "application/json")
    with urlopen(request, timeout=30) as response:
        return response.status, response.read()


def smoke_check(port: int) -> bytes:
    request = json.dumps({"messages": [{"role": "user", "content": "Reply with only OK."}],
                          "max_tokens": 32, "temperature": 0, "stream": False}).encode()
    status, body = http_post(port, "/v1/chat/completions", request)
    if status != 200 or not body:
        die("chat completion smoke check did not return HTTP 200 with content")
    try:
        value = json.loads(body)
        choice = value["choices"][0]
        content = choice["message"]["content"]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        die(f"chat completion smoke response is malformed: {exc}")
    if not isinstance(content, str) or not content.strip() or choice.get("finish_reason") != "stop":
        die("chat completion smoke response lacks content or stop finish_reason")
    return body


def runtime_command(binary: Path, core: Path, gpu: int, host: int, io_threads: int,
                    threads: int, batch: int, context: int, parallel: int, port: int) -> list[str]:
    """The fixed, benchmarked server shape; kept separate for offline tests."""
    return [str(binary), "-m", str(core), "--moe-stream-cache", str(gpu), "--moe-stream-l2", str(host),
            "--moe-stream-io-threads", str(io_threads), "--moe-stream-direct", "--no-warmup",
            "--fit", "off", "-ngl", "99", "-c", str(context), "-b", str(batch),
            "-ub", str(batch), "-np", str(parallel), "-t", str(threads), "--reasoning", "auto",
            "-lv", "4", "--host", "127.0.0.1", "--port", str(port)]


def prepare(args: argparse.Namespace) -> None:
    root = Path(args.root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root).expanduser().resolve() if args.data_root else root / "data"
    if args.data_root and (not data_root.is_dir() or data_root.is_symlink()):
        die(f"--data-root must be an existing non-symlink directory: {data_root}")
    if not args.data_root:
        data_root.mkdir(parents=True, exist_ok=True)
    if data_root.is_symlink():
        die(f"--data-root must not be a symlink: {data_root}")
    if any(char.isspace() for char in str(root)) or any(char.isspace() for char in str(data_root)):
        die("--root and --data-root cannot contain whitespace")
    if args.release:
        descriptor = Path(args.release).expanduser().read_bytes()
        remote_release = None
    else:
        remote_release = safe_rel(args.release_path, "--release-path")
        descriptor_path = download_file(remote_release, data_root, REVISION)
        descriptor = descriptor_path.read_bytes()
    try:
        release = validate_release(json.loads(descriptor.decode("utf-8")))
    except UnicodeDecodeError as exc:
        die(f"release descriptor is not UTF-8: {exc}")
    descriptor_sha = hashlib.sha256(descriptor).hexdigest()
    receipt_path = root / "state" / "receipt.json"
    old = read_json(receipt_path, "receipt") if receipt_path.is_file() else {}
    old_assets = old.get("assets") if (isinstance(old, Mapping)
                                       and old.get("release_sha256") == descriptor_sha
                                       and old.get("data_root") == str(data_root)) else {}
    download_assets(release, data_root)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        verified = {asset.path: future.result() for asset, future in
                    [(asset, pool.submit(verify_asset, data_root, asset,
                                         old_assets.get(asset.path) if isinstance(old_assets, Mapping) else None,
                                         args.rehash)) for asset in release["assets"]]}
    by_role = {asset.role: asset for asset in release["assets"]
               if asset.role not in ("expert", "engram")}
    parse_expert_manifest(local_asset(data_root, by_role["expert_manifest"]), release["assets"])
    check_engram_manifest(local_asset(data_root, by_role["engram_manifest"]), release)
    hashes = source_hashes(local_asset(data_root, by_role["source_hashes"]))
    extract_source(root, local_asset(data_root, by_role["source"]), hashes,
                   by_role["source"].sha256, by_role["source_hashes"].sha256)
    binary = build_runtime(root, by_role["source_hashes"].sha256)
    index = data_root / "engram.runtime.index"
    write_index(index, data_root, release["engram_rows"])
    receipt = {"release_sha256": descriptor_sha, "data_root": str(data_root),
               "assets": verified, "runtime": runtime_identity(binary), "index": str(index)}
    atomic_bytes(root / "state" / "release.json", descriptor)
    atomic_json(receipt_path, receipt)
    print(f"prepared {RELEASE_ID}; data stays in {data_root}")


def load_prepared(args: argparse.Namespace) -> tuple[Path, Path, dict[str, Any], dict[str, Any]]:
    root = Path(args.root).expanduser().resolve()
    state = root / "state"
    descriptor = (state / "release.json").read_bytes()
    release = validate_release(json.loads(descriptor.decode("utf-8")))
    receipt = read_json(state / "receipt.json", "receipt")
    data_root = Path(receipt.get("data_root", "")).resolve()
    if not data_root.is_dir() or (args.data_root and data_root != Path(args.data_root).expanduser().resolve()):
        die("prepared --data-root is unavailable or differs from the requested --data-root")
    if receipt.get("release_sha256") != hashlib.sha256(descriptor).hexdigest():
        die("prepared receipt does not match the local release descriptor")
    return root, data_root, release, receipt


def start(args: argparse.Namespace) -> None:
    root, data_root, release, receipt = load_prepared(args)
    process_path = root / "state" / "process.json"
    if process_path.is_file() and owns_process(read_json(process_path, "process identity")):
        die("a launcher-owned server is already running")
    records = receipt.get("assets")
    if not isinstance(records, Mapping):
        die("prepared receipt has no asset records")
    experts = [asset for asset in release["assets"] if asset.role == "expert"]
    if len(experts) != 49:
        die("ExpertPack SHA bypass requires exactly 49 expert assets")
    for asset in release["assets"]:
        record = records.get(asset.path)
        if not isinstance(record, Mapping) or record.get("after") != stat_record(local_asset(data_root, asset)):
            die(f"prepared asset is stale; run prepare: {asset.path}")
    binary = root / "build-engram" / "bin" / "llama-server"
    if binary.is_symlink() or not binary.is_file() or receipt.get("runtime") != runtime_identity(binary):
        die("llama-server identity is stale; run prepare")
    index = data_root / "engram.runtime.index"
    manifest = local_asset(data_root, next(asset for asset in release["assets"] if asset.role == "expert_manifest"))
    if index.is_symlink():
        die("Engram runtime index must not be a symlink")
    defaults = release["defaults"]
    gpu = defaults["gpu_cache_gib"] if args.gpu_cache_gib is None else args.gpu_cache_gib
    host = defaults["host_cache_gib"] if args.host_cache_gib is None else args.host_cache_gib
    io_threads = defaults["io_threads"] if args.io_threads is None else args.io_threads
    threads = defaults["threads"] if args.threads is None else args.threads
    batch = defaults["batch"] if args.batch is None else args.batch
    context = defaults["context"] if args.context is None else args.context
    parallel = defaults.get("parallel", 1) if args.parallel is None else args.parallel
    port = defaults["port"] if args.port is None else args.port
    if (gpu < 0 or host < 0
            or min(io_threads, threads, batch, context, parallel, port) <= 0
            or port > 65535):
        die("cache, thread, batch, context, and port overrides are out of range")
    if not port_available(port):
        die(f"127.0.0.1:{port} is already occupied")
    core = local_asset(data_root, next(asset for asset in release["assets"] if asset.role == "core"))
    command = runtime_command(binary, core, gpu, host, io_threads, threads, batch, context,
                              parallel, port)
    environment = os.environ.copy()
    for name in ("LLAMA_MOE_STREAM_PROFILE_OUT", "LLAMA_MOE_STREAM_PIN_LIST",
                 "LLAMA_MOE_STREAM_AHEAD", "LLAMA_MOE_STREAM_ROUTE_BIAS",
                 "LLAMA_MOE_STREAM_ORACLE", "LLAMA_MOE_STREAM_L2_GIB"):
        environment.pop(name, None)
    environment["LLAMA_ENGRAM_INDEX"] = str(index)
    environment["LLAMA_MOE_EXPERT_PACK_MANIFEST"] = str(manifest)
    environment["LLAMA_MOE_EXPERT_PACK_VERIFY_SHA256"] = "0"
    write_index(index, data_root, release["engram_rows"])
    if index.is_symlink() or index.read_bytes() != index_bytes(index, data_root, release["engram_rows"]):
        die("could not create a deterministic Engram runtime index")
    log = (root / "state" / "server.log").open("ab")
    child = subprocess.Popen(command, cwd=data_root, env=environment, stdout=log,
                             stderr=subprocess.STDOUT, start_new_session=True)
    identity = {
        "pid": child.pid,
        "starttime": proc_starttime(child.pid),
        "port": port,
        "runtime": {
            "context": context,
            "parallel": parallel,
            "gpu_cache_gib": gpu,
            "host_cache_gib": host,
            "batch": batch,
            "threads": threads,
            "io_threads": io_threads,
        },
    }
    if identity["starttime"] is None:
        child.terminate()
        log.close()
        die("could not capture child process start time")
    atomic_json(process_path, identity)
    try:
        deadline = time.monotonic() + args.ready_timeout
        while time.monotonic() < deadline:
            if child.poll() is not None or not owns_process(identity):
                die("llama-server stopped before health check passed")
            if health_is_ready("127.0.0.1", port):
                break
            time.sleep(1)
        else:
            die(f"/health did not become ready within {args.ready_timeout} seconds")
        atomic_bytes(root / "state" / "smoke.raw", smoke_check(port))
    except Exception:
        if owns_process(identity):
            child.terminate()
        raise
    finally:
        log.close()
    print(f"started pid={child.pid}, ready on 127.0.0.1:{port}")


def status(args: argparse.Namespace) -> None:
    path = Path(args.root).expanduser().resolve() / "state" / "process.json"
    if not path.is_file():
        print("stopped")
        return
    identity = read_json(path, "process identity")
    print(json.dumps({"pid": identity.get("pid"), "port": identity.get("port"),
                      "runtime": identity.get("runtime"), "running": owns_process(identity)},
                     sort_keys=True))


def stop(args: argparse.Namespace) -> None:
    path = Path(args.root).expanduser().resolve() / "state" / "process.json"
    if not path.is_file():
        print("not running")
        return
    identity = read_json(path, "process identity")
    if not owns_process(identity):
        print("record does not identify a running launcher-owned process")
        return
    os.kill(identity["pid"], signal.SIGTERM)
    deadline = time.monotonic() + 15
    while owns_process(identity) and time.monotonic() < deadline:
        time.sleep(.2)
    if owns_process(identity):
        os.kill(identity["pid"], signal.SIGKILL)
    path.unlink(missing_ok=True)
    print(f"stopped pid={identity['pid']}")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("all", "prepare", "start", "status", "stop"),
                        nargs="?", default="all")
    parser.add_argument("--root", default="/root/dsv41")
    parser.add_argument("--data-root")
    parser.add_argument("--release", help="local descriptor for staging tests")
    parser.add_argument("--release-path", default=REMOTE_RELEASE)
    parser.add_argument("--rehash", action="store_true")
    parser.add_argument("--gpu-cache-gib", type=int)
    parser.add_argument("--host-cache-gib", type=int)
    for name in ("io-threads", "threads", "batch", "context", "parallel", "port"):
        parser.add_argument("--" + name, dest=name.replace("-", "_"), type=int)
    parser.add_argument("--ready-timeout", type=int, default=600)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    try:
        if args.ready_timeout <= 0:
            die("--ready-timeout must be positive")
        if args.command == "all":
            prepare(args)
            start(args)
        elif args.command == "prepare":
            prepare(args)
        elif args.command == "start":
            start(args)
        elif args.command == "stop":
            stop(args)
        else:
            status(args)
        return 0
    except (DeploymentError, OSError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
