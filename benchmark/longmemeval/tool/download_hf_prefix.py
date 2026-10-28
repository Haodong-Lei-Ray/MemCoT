#!/usr/bin/env python3
"""Download a bounded archive prefix and extract complete leading indexes.

Needs the zstd CLI; no Python dependencies. Pins the public dataset revision.
Never downloads the full 33.7 GB archive. Returns 2 if the prefix is too short.
"""
import argparse
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
from urllib.request import Request, urlopen

URL = "https://huggingface.co/datasets/Rayleihaodong/MemCoT-longmemeval/resolve/176d2f5d7be0b9fc5bd131810643ed260d37dd1d/te3s.tar.zst"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mib", type=int, default=80)
    parser.add_argument("--indexes", type=int, default=1)
    parser.add_argument("--prefix-file", type=Path, help="Reuse an already downloaded prefix, without network access")
    args = parser.parse_args()
    if args.mib <= 0 or args.indexes <= 0:
        parser.error("--mib and --indexes must be positive")
    if not shutil.which("zstd"):
        parser.error("Install the zstd CLI first")
    args.output.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix_file or args.output / "te3s.prefix.zst.part"
    budget = args.mib * 1024 * 1024
    if not args.prefix_file:
        request = Request(URL + f"?prefix_bytes={budget}", headers={"Range": f"bytes=0-{budget - 1}"})
        with urlopen(request, timeout=60) as response:
            if response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes 0-{budget - 1}/"):
                raise RuntimeError("Server did not honor the bounded Range request; refusing download")
            remaining = budget
            with prefix.open("wb") as out:
                while remaining:
                    block = response.read(min(1024 * 1024, remaining))
                    if not block:
                        raise RuntimeError("Range response ended early")
                    out.write(block)
                    remaining -= len(block)
    completed, current, files = [], None, []
    proc = subprocess.Popen(["zstd", "-dc", str(prefix)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    truncated = False
    try:
        with tarfile.open(fileobj=proc.stdout, mode="r|") as archive:
            for member in archive:
                parts = PurePosixPath(member.name).parts
                if not parts or parts[0] != "te3s" or ".." in parts:
                    raise ValueError(f"Unexpected archive path: {member.name}")
                if len(parts) < 2:
                    continue
                group = parts[1]
                if current is not None and group != current:
                    completed.append(current)
                    if len(completed) >= args.indexes:
                        break
                current = group
                if member.isdir():
                    continue
                if not member.isfile() or len(parts) != 3:
                    raise ValueError(f"Unsupported archive member: {member.name}")
                target = args.output.joinpath(*parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    raise FileExistsError(f"Use a fresh output directory: {target}")
                temporary = target.with_name(target.name + ".partial")
                with archive.extractfile(member) as src, temporary.open("wb") as out:
                    shutil.copyfileobj(src, out)
                temporary.replace(target)
                files.append(member.name)
    except tarfile.ReadError:
        truncated = True
    finally:
        proc.stdout.close()
        if proc.poll() is None:
            proc.terminate()
        proc.wait()
    report = {"source": URL, "downloaded_bytes": 0 if args.prefix_file else budget,
              "prefix_bytes": prefix.stat().st_size, "complete_indexes": completed,
              "incomplete_index": current if current not in completed else None,
              "truncated_prefix": truncated, "extracted_files": files}
    (args.output / "download_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if len(completed) < args.indexes:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
