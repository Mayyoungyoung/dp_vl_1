"""Independent 2D -> 3D smoke check with upstream 3DWay camera calibration.

The XYZ grid is generated for this check, NOT a model prediction or an
annotation of the two public images. No action execution is evaluated.
"""

import argparse
import json
import struct
from pathlib import Path

import numpy as np


def projection_matrix(view):
    intrinsic = np.asarray(view["intrinsics"], dtype=np.float64)
    camera_to_world = np.asarray(view["camera_to_world"], dtype=np.float64)
    return intrinsic @ np.linalg.inv(camera_to_world)[:3, :]


def project(points, matrix):
    homogeneous = np.c_[points, np.ones(len(points))]
    camera_points = homogeneous @ matrix.T
    if np.any(camera_points[:, 2] <= 0):
        raise ValueError("Generated test point is behind a public camera")
    return camera_points[:, :2] / camera_points[:, 2:3]


def triangulate(points1, points2, matrix1, matrix2):
    result = []
    for first, second in zip(points1, points2):
        system = np.stack([
            first[0] * matrix1[2] - matrix1[0],
            first[1] * matrix1[2] - matrix1[1],
            second[0] * matrix2[2] - matrix2[0],
            second[1] * matrix2[2] - matrix2[1],
        ])
        _, _, right_vectors = np.linalg.svd(system)
        point = right_vectors[-1]
        if abs(point[3]) < 1e-12:
            raise ValueError("Triangulated point at infinity")
        result.append(point[:3] / point[3])
    return np.asarray(result)


def png_dimensions(path):
    with path.open("rb") as image:
        header = image.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Unexpected sample image format")
    return list(struct.unpack(">II", header[16:24]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, default=Path("data/3dway_example"))
    parser.add_argument("--output", type=Path, default=Path("reports/3dway_geometry_smoke.json"))
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    rig = json.loads((args.sample_dir / "camera_parameters.json").read_text(encoding="utf-8"))
    if rig["schema_version"] != 1 or rig["extrinsics_convention"] != "camera_to_world":
        raise ValueError("Unexpected upstream camera convention")
    views = [rig["views"][name] for name in ("view1", "view2")]
    dimensions = [png_dimensions(args.sample_dir / (name + ".png"))
                  for name in ("view1", "view2")]
    for actual, view in zip(dimensions, views):
        if actual != [view["width"], view["height"]]:
            raise ValueError("Image dimensions do not match public camera calibration")
    matrices = [projection_matrix(view) for view in views]
    points = np.asarray([[x, y, z] for x in (0.25, 0.35, 0.45)
                         for y in (-0.05, 0.0, 0.05) for z in (1.00, 1.10, 1.20)])
    pixels = [project(points, matrix) for matrix in matrices]
    for projected, view in zip(pixels, views):
        if not np.all((projected >= 0) & (projected < [view["width"], view["height"]])):
            raise ValueError("Generated test grid is outside public camera field of view")
    recovered = triangulate(pixels[0], pixels[1], matrices[0], matrices[1])
    errors = np.linalg.norm(recovered - points, axis=1)
    residuals = [np.linalg.norm(project(recovered, matrix) - target, axis=1)
                 for matrix, target in zip(matrices, pixels)]
    generator = np.random.RandomState(args.seed)
    noisy_pixels = [pixel + generator.normal(0, 1.0, pixel.shape) for pixel in pixels]
    noisy_recovered = triangulate(noisy_pixels[0], noisy_pixels[1], matrices[0], matrices[1])
    noisy_errors = np.linalg.norm(noisy_recovered - points, axis=1)
    centres = [np.asarray(view["camera_to_world"])[:3, 3] for view in views]
    report = {
        "source": "https://github.com/ziqin-h/3DWay/tree/main/3dway_policy/example_data",
        "test_scope": "public camera/image consistency + synthetic-point projection/triangulation; no model prediction, dataset route label, or robot execution",
        "camera_convention": rig["extrinsics_convention"],
        "public_image_dimensions": dimensions,
        "negative_focal_lengths_preserved": bool(all(np.asarray(view["intrinsics"])[0, 0] < 0 for view in views)),
        "camera_baseline_m": float(np.linalg.norm(centres[0] - centres[1])),
        "synthetic_test_points": len(points),
        "reconstruction_max_error_m": float(errors.max()),
        "reprojection_max_error_px": float(max(error.max() for error in residuals)),
        "noise_standard_deviation_px": 1.0,
        "noisy_reconstruction_mean_error_m": float(noisy_errors.mean()),
        "noisy_reconstruction_max_error_m": float(noisy_errors.max()),
        "seed": args.seed,
        "synthetic_points_xyz": points.tolist(),
        "projected_points_px": [pixel.tolist() for pixel in pixels],
        "passed": bool(errors.max() < 1e-8 and max(error.max() for error in residuals) < 1e-6),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ("synthetic_points_xyz", "projected_points_px")}, indent=2))
    if not report["passed"]:
        raise RuntimeError("Public geometry smoke test failed")


if __name__ == "__main__":
    main()
