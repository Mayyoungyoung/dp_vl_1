"""Snapshot a requested TRAIN parent prefix plus four repeatedly used DEV parents.

Initialization failures occupy their original requested-prefix slots; they are
never skipped to replace them with successful parents. Source data is read only.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile

try:
    from .snapshot_observation_learning_curve import INPUT_KEYS, digest, row_digest, read_complete_rows, write_jsonl
except ImportError:
    from snapshot_observation_learning_curve import INPUT_KEYS, digest, row_digest, read_complete_rows, write_jsonl


def inventory(dataset, split, requested_count):
    dataset = Path(dataset).resolve()
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8-sig"))
    proposals = manifest["proposed_type_sequences"]
    if len(proposals) != 4 or {tuple(mode) for mode in proposals} != {(mode,) for mode in ("negative_x", "positive_x", "negative_y", "positive_y")}:
        raise ValueError("this snapshot requires exactly four declared single-obstacle side proposals")
    count, dev_count, seed = manifest["parents_requested"], manifest["dev_parents"], manifest["seed"]
    allowed_count = count - dev_count if split == "TRAIN" else dev_count
    start = seed if split == "TRAIN" else seed + count - dev_count
    if requested_count < 1 or requested_count > allowed_count:
        raise ValueError("requested parent count exceeds the source split")
    grouped = {key: defaultdict(list) for key in ("observations", "supervision", "attempts")}
    for key in grouped:
        for row in read_complete_rows(dataset / (key + ".jsonl")):
            grouped[key][row["parent_id"]].append(row)
    closed, incomplete = [], []
    for offset in range(requested_count):
        parent = "obstacle_reach_%06d" % (start + offset)
        observations, supervision, attempts = [grouped[key][parent] for key in grouped]
        setup_failures = [row for row in attempts if row.get("phase") == "parent_setup"]
        if setup_failures:
            if len(setup_failures) != 1 or len(attempts) != 1 or observations or supervision or setup_failures[0].get("success"):
                raise ValueError("inconsistent parent-setup failure record: " + parent)
            if setup_failures[0]["split"] != split:
                raise ValueError("setup failure split mismatch")
            closed.append(dict(parent_id=parent, split=split, setup_failed=True,
                               observations=[], supervision=[], attempts=attempts))
            continue
        expected = {parent + "_target%d" % target for target in range(3)}
        reason = None
        if len(observations) != 3 or {row["id"] for row in observations} != expected:
            reason = "three current-image/language observation rows not complete"
        elif any(set(row) != INPUT_KEYS or row["split"] != split for row in observations):
            raise ValueError("observation contract or source split mismatch: " + parent)
        elif len(supervision) != 3 or {row["id"] for row in supervision} != expected:
            reason = "three target supervision records not complete"
        elif any(row["split"] != split for row in supervision):
            raise ValueError("supervision split mismatch: " + parent)
        elif len(attempts) != 12 or {(row.get("input_id"), row.get("attempt")) for row in attempts} != {(identifier, number) for identifier in expected for number in range(4)}:
            reason = "all twelve declared proposal outcomes not complete"
        if reason:
            incomplete.append(dict(parent_id=parent, reason=reason))
            continue
        labels = {row["id"]: row for row in supervision}
        for row in attempts:
            if row["proposed_type_supervision_only"] != proposals[row["attempt"]]:
                raise ValueError("recorded proposal does not match fixed source budget")
        for identifier in expected:
            label = labels[identifier]
            routes = label.get("routes", [])
            actual = {row["route_file"]: row["actual_route_type"] for row in attempts if row["input_id"] == identifier and row["success"]}
            if len(set(routes)) != len(routes) or set(actual) != set(routes):
                raise ValueError("successful attempts and saved reference list disagree")
            if len(label.get("route_types", [])) != len(routes) or [actual[route] for route in routes] != label["route_types"]:
                raise ValueError("reference type labels differ from actual checked route outcomes")
            for path in [label["observation"], label["verification_only"]] + routes:
                if not (dataset / path).is_file():
                    raise ValueError("missing completed-parent artifact: " + path)
        closed.append(dict(parent_id=parent, split=split, setup_failed=False,
                           observations=observations, supervision=supervision, attempts=attempts))
    return manifest, closed, incomplete


def build(new_train, old_dev, output, train_parents=16, dev_parents=4):
    new_train, old_dev, output = map(lambda value: Path(value).resolve(), (new_train, old_dev, output))
    if output.exists():
        raise ValueError("snapshot already exists; never replace it")
    train_manifest, training, unfinished = inventory(new_train, "TRAIN", train_parents)
    dev_manifest, development, dev_unfinished = inventory(old_dev, "DEV_MODEL", dev_parents)
    if unfinished or dev_unfinished:
        raise ValueError("not ready: requested parent prefix contains unfinished parents: " + json.dumps(unfinished + dev_unfinished))
    if dev_manifest["parents_requested"] != dev_parents or dev_manifest["dev_parents"] != dev_parents:
        raise ValueError("reused DEV must be the explicitly declared complete four-parent pilot")
    if train_manifest["target_layout"] != dev_manifest["target_layout"] or train_manifest["acceptance"] != dev_manifest["acceptance"]:
        raise ValueError("TRAIN and reused DEV geometry setting/acceptance differ")
    selected = [(new_train, item) for item in training] + [(old_dev, item) for item in development]
    if len({item["parent_id"] for _, item in selected}) != len(selected):
        raise ValueError("TRAIN/DEV parent overlap")
    observations, supervision, attempts, provenance, file_hashes = [], [], [], [], {}
    for source, item in selected:
        parent_hashes = {str(path): digest(path) for path in sorted((source / item["parent_id"]).rglob("*")) if path.is_file()}
        for row in item["observations"]:
            changed = dict(row, image=str((source / row["image"]).resolve()))
            if not Path(changed["image"]).is_file():
                raise ValueError("missing closed parent image")
            observations.append(changed)
        for row in item["supervision"]:
            supervision.append(dict(row, observation=str((source / row["observation"]).resolve()),
                verification_only=str((source / row["verification_only"]).resolve()),
                routes=[str((source / path).resolve()) for path in row["routes"]]))
        attempts.extend(dict(row, source_dataset=str(source)) for row in item["attempts"])
        provenance.append(dict(parent_id=item["parent_id"], source_dataset=str(source), source_split=item["split"],
            setup_failed=item["setup_failed"], source_rows_sha256=row_digest(item), source_files_sha256=parent_hashes))
        file_hashes.update(parent_hashes)
    # Verify both frozen parent files and selected JSON rows. Concurrently
    # appended unrelated parents never change this snapshot's chosen prefix.
    for source, split, count, before in [(new_train, "TRAIN", train_parents, training), (old_dev, "DEV_MODEL", dev_parents, development)]:
        _, after, incomplete = inventory(source, split, count)
        if incomplete or row_digest(before) != row_digest(after):
            raise RuntimeError("closed source rows changed during snapshot")
    if any(digest(path) != expected for path, expected in file_hashes.items()):
        raise RuntimeError("closed source file changed during snapshot")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(output.name + ".staging")
    staging.mkdir(exist_ok=False)
    for name, rows in [("observations", observations), ("supervision", supervision), ("attempts", attempts)]:
        write_jsonl(staging / (name + ".jsonl"), rows)
    metadata = dict(created_at=datetime.now(timezone.utc).isoformat(),
        purpose="exploratory obstacle learning curve, repeatedly used DEV, not final confirmation",
        parent_selection="first requested continuous TRAIN seed prefix, INCLUDING initialization failures; no success-count replacement",
        train_parents_requested=train_parents, dev_parents_requested=dev_parents,
        train_observed_parents=sum(not item["setup_failed"] for item in training),
        dev_observed_parents=sum(not item["setup_failed"] for item in development),
        train_initialization_failures=[item for item in training if item["setup_failed"]],
        dev_initialization_failures=[item for item in development if item["setup_failed"]],
        train_semantic_examples=sum(row["split"] == "TRAIN" for row in supervision),
        train_reference_examples=sum(row["split"] == "TRAIN" and bool(row["routes"]) for row in supervision),
        dev_semantic_examples=sum(row["split"] == "DEV_MODEL" for row in supervision),
        dev_reference_examples=sum(row["split"] == "DEV_MODEL" and bool(row["routes"]) for row in supervision),
        source_split_unchanged=True, source_datasets=[str(new_train), str(old_dev)], source_parents=provenance,
        source_manifest_hashes={str(source / "manifest.json"): digest(source / "manifest.json") for source in (new_train, old_dev)},
        input_contract=sorted(INPUT_KEYS), input_manifest_sha256=digest(staging / "observations.jsonl"),
        script_sha256=digest(Path(__file__)),
        reused_dev_warning="All four pilot DEV parents informed collection and method choices. They are DEVELOPMENT, never locked TEST.",
        cache_policy="Build a new real frozen Qwen cache for this exact five-key observation manifest; no future/target/geometry labels enter condition tokens.")
    # Retain evaluator acceptance/type declarations while making snapshot split
    # denominators explicit. Seeds come from two source ranges, not one range.
    snapshot_manifest = dict(train_manifest, parents_requested=train_parents + dev_parents, dev_parents=dev_parents,
        seed=None, snapshot=True, snapshot_metadata=metadata,
        source_parent_ranges=dict(TRAIN=[item["parent_id"] for item in training], DEV_MODEL=[item["parent_id"] for item in development]))
    (staging / "manifest.json").write_text(json.dumps(snapshot_manifest, indent=2), encoding="utf-8")
    (staging / "snapshot_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    staging.rename(output)
    return {key: metadata[key] for key in ("train_parents_requested", "train_observed_parents", "dev_observed_parents",
        "train_semantic_examples", "train_reference_examples", "dev_semantic_examples", "dev_reference_examples", "input_manifest_sha256")}


def self_test():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        modes = [[mode] for mode in ("negative_x", "positive_x", "negative_y", "positive_y")]
        for name, split, seed, count in [("new", "TRAIN", 100, 2), ("dev", "DEV_MODEL", 200, 1)]:
            source = root / name; source.mkdir()
            (source / "manifest.json").write_text(json.dumps(dict(seed=seed, parents_requested=count,
                dev_parents=0 if split == "TRAIN" else count, proposed_type_sequences=modes,
                target_layout="fixture_only", acceptance={})))
            observations, supervision, attempts = [], [], []
            for index in range(count):
                parent = "obstacle_reach_%06d" % (seed + index); folder = source / parent; folder.mkdir()
                if name == "new" and index == 0:
                    attempts.append(dict(parent_id=parent, split=split, phase="parent_setup", success=False, error="fixture setup failure"))
                    continue
                for filename in ("front.png", "observation.npz", "verification_only.npz"):
                    (folder / filename).write_bytes(b"fixture, not research data")
                for target in range(3):
                    identifier = parent + "_target%d" % target
                    observations.append(dict(id=identifier, parent_id=parent, split=split, image=parent + "/front.png", instruction="fixture"))
                    supervision.append(dict(id=identifier, parent_id=parent, split=split, observation=parent + "/observation.npz",
                        verification_only=parent + "/verification_only.npz", routes=[], route_types=[]))
                    for proposal in range(4):
                        attempts.append(dict(parent_id=parent, input_id=identifier, attempt=proposal,
                            proposed_type_supervision_only=modes[proposal], success=False, actual_route_type=None))
            for key, rows in [("observations", observations), ("supervision", supervision), ("attempts", attempts)]:
                write_jsonl(source / (key + ".jsonl"), rows)
        source_rows = root / "new/attempts.jsonl"; original = source_rows.read_text()
        source_rows.write_text("\n".join(original.splitlines()[:-1]) + "\n")
        try:
            build(root / "new", root / "dev", root / "incomplete", 2, 1)
        except ValueError as error:
            assert "not ready" in str(error)
        else:
            raise AssertionError("incomplete 12th outcome must block snapshot")
        source_rows.write_text(original)
        result = build(root / "new", root / "dev", root / "ready", 2, 1)
        assert result["train_parents_requested"] == 2 and result["train_observed_parents"] == 1
        assert result["train_semantic_examples"] == 3 and result["train_reference_examples"] == 0
        assert len(read_complete_rows(root / "ready/observations.jsonl")) == 6
        assert all(set(row) == INPUT_KEYS for row in read_complete_rows(root / "ready/observations.jsonl"))
        assert source_rows.read_text() == original
    print(json.dumps(dict(self_test="passed", checks=["init failure retained in prefix denominator", "zero references retained",
        "twelfth proposal closure required", "input whitelist", "source unchanged", "atomic snapshot publication"])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-train", type=Path)
    parser.add_argument("--old-dev", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--train-parents", type=int, choices=(16, 32), default=16)
    parser.add_argument("--dev-parents", type=int, default=4)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif not args.new_train or not args.old_dev or not args.output:
        parser.error("new-train, old-dev and output required")
    else:
        print(json.dumps(build(args.new_train, args.old_dev, args.output, args.train_parents, args.dev_parents)))


if __name__ == "__main__":
    main()
