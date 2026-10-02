"""Read a bounded prefix of public 3DWay metadata and adapt paired 2D labels.

No third-party dependency or model loading is required. Example:
    python scripts/inspect_3dway_data.py --output data/public/3dway_sample.json

This is a metadata adapter, not a 3D dataset: calibration, collision labels,
and alternative routes for an identical scene are not present in these rows.
Use --archive only with an already downloaded upstream image archive; only
the explicitly referenced regular image files are copied, without extractall.
"""

import argparse
import json
import re
import tarfile
import urllib.request
from collections import defaultdict
from pathlib import Path, PurePosixPath


BASE_URL = "https://huggingface.co/datasets/liyy4586/3DWay-Data/resolve/main/"
POINT_OR_ACTION = re.compile(
    r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)"
    r"|<action>\s*(.*?)\s*</action>", re.DOTALL
)


def decode_prefix(data, limit):
    """Decode complete top-level objects; tolerate a truncated final row."""
    text = data.decode("utf-8", errors="ignore")
    decoder = json.JSONDecoder()
    pos = text.find("[") + 1
    records = []
    while pos > 0 and len(records) < limit:
        while pos < len(text) and text[pos] in " \r\n\t,":
            pos += 1
        if pos >= len(text) or text[pos] == "]":
            break
        try:
            value, pos = decoder.raw_decode(text, pos)
        except json.JSONDecodeError:
            break
        if not isinstance(value, dict):
            raise ValueError("Expected a JSON list of sample objects")
        records.append(value)
    return records


def read_metadata(source, max_bytes, max_records, timeout):
    """Stop reading as soon as enough rows are complete, within a byte cap."""
    if source.startswith(("https://", "http://")):
        request = urllib.request.Request(source, headers={
            "Range": "bytes=0-{}".format(max_bytes - 1),
            "User-Agent": "RouteSetDP-public-data-inspector/0.1",
        })
        stream = urllib.request.urlopen(request, timeout=timeout)
    else:
        stream = open(source, "rb")
    chunks = []
    bytes_read = 0
    with stream:
        while bytes_read < max_bytes:
            chunk = stream.read(min(32768, max_bytes - bytes_read))
            if not chunk:
                break
            chunks.append(chunk)
            bytes_read += len(chunk)
            rows = decode_prefix(b"".join(chunks), max_records)
            if len(rows) >= max_records:
                return rows, bytes_read
    return decode_prefix(b"".join(chunks), max_records), bytes_read


def parse_view(answer, view):
    tag = "ans_view{}".format(view)
    match = re.search(r"<{}>(.*?)</{}>".format(tag, tag), answer, re.DOTALL)
    if not match:
        raise ValueError("Missing {} in trajectory answer".format(tag))
    points, actions = [], []
    for token in POINT_OR_ACTION.finditer(match.group(1)):
        if token.group(1) is not None:
            point = [float(token.group(1)), float(token.group(2))]
            if not all(0.0 <= value <= 1.0 for value in point):
                raise ValueError("Public waypoint lies outside normalized [0,1]")
            points.append(point)
        else:
            actions.append({"waypoint_index": max(0, len(points) - 1),
                            "action": token.group(3)})
    if not points:
        raise ValueError("Empty waypoint view")
    return {"waypoints_2d_normalized": points, "gripper_events": actions}


def adapt_record(record):
    conversations = record.get("conversations", [])
    prompt = next((v.get("value", "") for v in conversations
                   if v.get("from") in ("human", "user")), "")
    answer = next((v.get("value", "") for v in conversations
                   if v.get("from") in ("gpt", "assistant")), "")
    instruction = re.search(r"<quest>(.*?)</quest>", prompt, re.DOTALL)
    image_paths = record.get("images", record.get("image", []))
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    if len(image_paths) != 2:
        raise ValueError("Expected exactly two image paths")
    views = [parse_view(answer, view) for view in (1, 2)]
    if len(views[0]["waypoints_2d_normalized"]) != len(views[1]["waypoints_2d_normalized"]):
        raise ValueError("Paired views have unequal waypoint counts")
    return {
        "source_id": record.get("id"),
        "instruction": instruction.group(1) if instruction else prompt,
        "image_paths": image_paths,
        "views": views,
        "camera_calibration": None,
        "supervision_type": "single_demonstration_paired_2d",
    }


def copy_referenced_images(archive_path, samples, output_root):
    """Copy only named regular image members; reject traversal and links."""
    root = output_root.resolve()
    wanted = set(path for sample in samples for path in sample["image_paths"])
    copied = []
    with tarfile.open(str(archive_path), "r:*") as archive:
        for member in archive:
            name = member.name[2:] if member.name.startswith("./") else member.name
            if name not in wanted or not member.isfile():
                continue
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts or ":" in name or "\\" in name:
                raise ValueError("Unsafe upstream image path: {}".format(name))
            destination = root.joinpath(*relative.parts).resolve()
            if root not in destination.parents:
                raise ValueError("Image escapes output directory")
            if destination.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, destination.open("wb") as target:
                while True:
                    chunk = source.read(65536)
                    if not chunk:
                        break
                    target.write(chunk)
            copied.append(name)
            wanted.discard(name)
            if not wanted:
                break
    return {"copied": copied, "missing": sorted(wanted)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", choices=("rlbench", "droid", "rh20t"), default="rlbench")
    parser.add_argument("--input", help="Local JSON metadata instead of public HTTPS URL")
    parser.add_argument("--max-bytes", type=int, default=262144)
    parser.add_argument("--max-records", type=int, default=12)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--archive", type=Path, help="Existing local image tar; no automatic large download")
    args = parser.parse_args()
    if args.max_bytes < 1 or args.max_records < 1:
        parser.error("Byte and record limits must be positive")
    source = args.input or BASE_URL + args.subset + ".json"
    records, bytes_read = read_metadata(source, args.max_bytes, args.max_records, args.timeout)
    if not records:
        raise RuntimeError("No complete rows within metadata prefix; increase --max-bytes")
    samples = [adapt_record(record) for record in records]
    routes_per_pair = defaultdict(set)
    for sample in samples:
        routes_per_pair[tuple(sample["image_paths"])].add(json.dumps(sample["views"], sort_keys=True))
    summary = {
        "source": source,
        "metadata_bytes_read": bytes_read,
        "records": len(samples),
        "unique_image_pairs": len(routes_per_pair),
        "distinct_routes_per_image_pair": [len(value) for value in routes_per_pair.values()],
        "record_keys": sorted(set(key for record in records for key in record)),
        "calibration_in_inspected_record": any("calib" in str(key).lower() or "intrinsic" in str(key).lower()
                                               or "extrinsic" in str(key).lower() for record in records for key in record),
        "scope": "bounded_prefix_only; not a full-dataset diversity audit",
    }
    payload = {"summary": summary, "samples": samples}
    if args.archive:
        if args.output is None:
            parser.error("--archive requires --output to specify a safe image destination")
        payload["image_copy"] = copy_referenced_images(args.archive, samples, args.output.parent / "images")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output:
        print("Saved normalized public samples to {}".format(args.output.resolve()))


if __name__ == "__main__":
    main()
