#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calib_Step6_calibrate_cam_to_robot.py

<<실행 명령어>>
python Calib_Step6_calibrate_cam_to_robot.py \
    --input  pairs.json \
    --output calibration_result.json

---------------------------------------------------------------
Standalone Eye-to-Hand calibration: estimate the rigid transform
(4x4 homogeneous matrix) that maps points in the CAMERA frame to the
ROBOT BASE frame, given a set of 3D point correspondences.

<Algorithm>
-----------
Kabsch / Umeyama point-set alignment via SVD.
For N >= 3 correspondences this gives the closed-form least-squares
rotation R and translation t such that:

    robot_point ~= R @ camera_point + t

<Inputs>
--------
A single JSON file containing a list of correspondences. Each entry
must provide a 3D point in the camera frame and the matching 3D point
in the robot base frame, both in METERS:

    [
      {"camera_point": [x, y, z], "robot_point": [x, y, z]},
      {"camera_point": [x, y, z], "robot_point": [x, y, z]},
      ...
    ]

How to obtain the points (robot-agnostic)
-----------------------------------------
- Attach a fiducial (ArUco, checkerboard center, sphere center,
  touch-probe tip, etc.) at a known offset from the robot TCP.
- Move the robot to several poses spread across the workspace.
- At each pose, record:
    * camera_point: the fiducial's 3D position as seen by the camera
                    (e.g. ArUco tvec, or back-projected depth point)
    * robot_point:  the fiducial's 3D position in the robot base frame
                    (typically the TCP xyz, if the fiducial is at the
                    TCP origin; otherwise apply the known offset)
- Collect at least 3 pairs; 8-15 well-distributed poses is typical.

Outputs
-------
A JSON file with the rotation matrix, translation, full 4x4 transform,
and per-pair / aggregate error metrics in meters.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def load_pairs(path: Path):
    """Load and validate the correspondence list."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of correspondences.")

    camera_points = []
    robot_points = []
    for i, item in enumerate(data):
        if "camera_point" not in item or "robot_point" not in item:
            raise ValueError(
                f"Entry {i} is missing 'camera_point' or 'robot_point'."
            )
        cam = item["camera_point"]
        rob = item["robot_point"]
        if len(cam) < 3 or len(rob) < 3:
            raise ValueError(f"Entry {i} points must have 3 components.")
        camera_points.append([float(v) for v in cam[:3]])
        robot_points.append([float(v) for v in rob[:3]])

    if len(camera_points) < 3:
        raise ValueError(
            f"Need at least 3 correspondences, got {len(camera_points)}."
        )

    return (
        np.array(camera_points, dtype=np.float64),
        np.array(robot_points, dtype=np.float64),
    )


def estimate_rigid_transform(camera_points: np.ndarray, robot_points: np.ndarray):
    """Closed-form rotation+translation via SVD (Kabsch/Umeyama)."""
    centroid_camera = np.mean(camera_points, axis=0)
    centroid_robot = np.mean(robot_points, axis=0)

    centered_camera = camera_points - centroid_camera
    centered_robot = robot_points - centroid_robot

    covariance = centered_camera.T @ centered_robot
    u_mat, _, vt_mat = np.linalg.svd(covariance)

    rotation = vt_mat.T @ u_mat.T
    # Reflection fix
    if np.linalg.det(rotation) < 0:
        vt_mat[-1, :] *= -1
        rotation = vt_mat.T @ u_mat.T

    translation = centroid_robot - rotation @ centroid_camera

    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return rotation, translation, transform


def compute_errors(camera_points: np.ndarray, robot_points: np.ndarray, transform: np.ndarray):
    """Per-pair Euclidean residuals (meters) after applying the transform."""
    errors = []
    transformed_points = []
    for cam, rob in zip(camera_points, robot_points):
        cam_h = np.append(cam, 1.0)
        transformed = (transform @ cam_h)[:3]
        transformed_points.append(transformed.tolist())
        errors.append(float(np.linalg.norm(transformed - rob)))

    arr = np.array(errors, dtype=np.float64)
    return {
        "per_pair_errors_m": errors,
        "mean_error_m": float(np.mean(arr)),
        "max_error_m": float(np.max(arr)),
        "min_error_m": float(np.min(arr)),
        "rms_error_m": float(np.sqrt(np.mean(arr ** 2))),
        "transformed_camera_points_m": transformed_points,
    }


def build_result(input_path, camera_points, robot_points, rotation, translation, transform, error_info, threshold):
    return {
        "input_json": str(input_path),
        "num_pairs": int(len(camera_points)),
        "camera_points_m": camera_points.tolist(),
        "robot_points_m": robot_points.tolist(),
        "rotation_matrix": rotation.tolist(),
        "translation_m": translation.tolist(),
        "transformation_matrix_camera_to_robot": transform.tolist(),
        "mean_error_m": error_info["mean_error_m"],
        "max_error_m": error_info["max_error_m"],
        "min_error_m": error_info["min_error_m"],
        "rms_error_m": error_info["rms_error_m"],
        "per_pair_errors_m": error_info["per_pair_errors_m"],
        "transformed_camera_points_m": error_info["transformed_camera_points_m"],
        "mean_error_threshold_m": float(threshold),
        "calibration_success": error_info["mean_error_m"] <= threshold,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate the camera->robot rigid transform from 3D point correspondences."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input JSON with [{camera_point, robot_point}, ...] pairs (meters).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the calibration result JSON.",
    )
    parser.add_argument(
        "--max-mean-error",
        type=float,
        default=0.02,
        help="Mean error threshold in meters used to flag success/fail (default: 0.02).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    camera_points, robot_points = load_pairs(args.input)
    rotation, translation, transform = estimate_rigid_transform(camera_points, robot_points)
    error_info = compute_errors(camera_points, robot_points, transform)
    result = build_result(
        args.input,
        camera_points,
        robot_points,
        rotation,
        translation,
        transform,
        error_info,
        args.max_mean_error,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Pairs used: {result['num_pairs']}")
    print("Transformation matrix (camera -> robot base):")
    print(np.array(result["transformation_matrix_camera_to_robot"]))
    print(f"Mean error: {result['mean_error_m']:.6f} m")
    print(f"Max  error: {result['max_error_m']:.6f} m")
    print(f"RMS  error: {result['rms_error_m']:.6f} m")
    print(f"Calibration success: {result['calibration_success']}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
