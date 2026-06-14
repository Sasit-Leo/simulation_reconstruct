#!/usr/bin/env python3
"""
align_to_isaac.py — Shared Manhattan-world alignment for video_to_mesh.sh / video_to_scene.sh

Coordinate frame convention (Isaac Sim):
  Y = up (height direction, smallest PCA variance)
  X = longest wall (largest PCA variance)
  Z = other horizontal (middle PCA variance)

Alignment pipeline:
  1. PCA on input points → find 3 Manhattan axes
  2. Sort by variance: smallest→height(Z), largest→longest(X), middle→Y
  3. Sign correction (point Z toward +Z, X toward +X)
  4. R_zup: rotate PCA frame to world Z-up
  5. Flip detection: densest 20% band should be at bottom (floor)
  6. Y-up conversion: R_yup @ R_zup → Isaac Sim Y-up frame
  7. Apply rotation to mesh/tensor, save rotation.json

USD upAxis is set to "Z" for backward compatibility with existing outputs.
"""

import struct
import json
import numpy as np
from pathlib import Path


# ──────────────────────────────────────────────
# 1. COLMAP binary point cloud loader
# ──────────────────────────────────────────────
def load_colmap_points(points3d_bin_path: str) -> np.ndarray:
    """
    Parse COLMAP points3D.bin → (N, 3) float64 array.

    Binary format per point:
      - 8 bytes: point_id (uint64)
      - 24 bytes: XYZ (3 x float64)
      - 4 bytes: RGB (3 x uint8 + 1 pad)
      - 8 bytes: error (float64)
      - 8 bytes: track_length (uint64)
      - track_length * 8 bytes: image_id + point2D_idx (2 x int32 per track)
    """
    pts = []
    with open(points3d_bin_path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        for _ in range(n):
            data = f.read(43)  # id(8) + xyz(24) + rgb(4) + error(8) = 44? no: 8+24+3+1+8=44
            # Actually: id=8, xyz=3*8=24, rgb=3*1+1=4, error=8 = 44 bytes total
            # Wait the original code reads 43 bytes. Let me verify...
            # 8 (id) + 24 (xyz) + 4 (rgb padded) + 8 (error) = 44
            # But the offset for xyz is 8 (after id), so we read id(8) then unpack xyz at offset 8
            # Actually the original code: f.read(43) — this reads id(8)+xyz(24)+rgb(3)+pad(1)+error(8)-1?
            # Let me look at the original more carefully:
            # "data = f.read(43); x,y,z = struct.unpack_from('<ddd', data, 8)"
            # So: read 43 bytes, unpack 3 doubles starting at byte 8
            # byte 0-7: id, byte 8-31: xyz (24 bytes), byte 32-34: rgb (3 bytes), byte 35: pad
            # byte 36-43: error (8 bytes) → total 44? but read(43)...
            # Actually 8+24+3+1+8 = 44, so read(43) is off by 1. Maybe the COLMAP format doesn't have padding.
            # Let me check: COLMAP points3D.bin format:
            # point3D_id (uint64 = 8)
            # XYZ (3 x float64 = 24)
            # RGB (3 x uint8 = 3) — NO padding byte
            # error (float64 = 8)
            # Total fixed part: 8+24+3+8 = 43. Yes! 43 bytes before track data.
            x, y, z = struct.unpack_from('<ddd', data, 8)
            pts.append([x, y, z])
            track_len = struct.unpack('<Q', f.read(8))[0]
            f.seek(track_len * 8, 1)  # skip track entries (each: image_id(int32) + point2D_idx(int32) = 8 bytes)
    return np.array(pts, dtype=np.float64)


# ──────────────────────────────────────────────
# 2. Core Manhattan alignment
# ──────────────────────────────────────────────
def compute_manhattan_rotation(
    points: np.ndarray,
    do_flip_check: bool = True,
    rotation_threshold_deg: float = 2.0,
    flip_percentile_range: tuple = (10, 90),
    flip_band_fraction: float = 0.2,
) -> dict:
    """
    Compute Manhattan-world alignment rotation for Isaac Sim (Y-up convention).

    Algorithm:
      1. PCA (3 components) on points
      2. Sort components by variance ascending: 0=height, 1=middle, 2=longest
      3. Assign z_axis=comp[0], x_axis=comp[2], y_axis=cross(z,x)
      4. Sign correction: z→+Z hemisphere, x→+X hemisphere
      5. R_zup = stack rows → rotates PCA axes to world Z-up
      6. If angle < threshold: R_zup = I (skip near-identity)
      7. Flip detection: count points in bottom vs top 20% Z-band
         If top denser than bottom → flip Y and Z (diag(1,-1,-1))
      8. Y-up: R_yup = [[1,0,0],[0,0,-1],[0,1,0]], R_final = R_yup @ (flip @ R_zup)

    Returns dict with keys:
      R             — (3,3) final rotation matrix
      R_zup_only    — (3,3) intermediate Z-up rotation (before Y-up)
      points_rotated — (N,3) points in final frame
      angle_deg     — float, PCA Z vs world Z angle
      flipped       — bool, whether flip was applied
      flip_top_count  — int
      flip_bot_count  — int
      rotation_applied — bool
    """
    from sklearn.decomposition import PCA

    N = len(points)
    pca = PCA(n_components=3).fit(points)
    comps = pca.components_.copy()
    var = pca.explained_variance_.copy()

    # Sort by variance: ascending → 0=smallest(height), 1=middle, 2=largest(longest)
    order = np.argsort(var)

    z_axis = comps[order[0]].copy()  # smallest variance → height → Z-up
    x_axis = comps[order[2]].copy()  # largest variance → longest wall → X
    z_axis /= np.linalg.norm(z_axis)
    x_axis /= np.linalg.norm(x_axis)

    # Sign correction: ensure consistent hemisphere orientation
    if z_axis[2] < 0:
        z_axis = -z_axis
    if x_axis[0] < 0:
        x_axis = -x_axis

    # Right-handed frame: y = cross(z, x)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    # R_zup: rotate PCA-aligned vectors to world Z-up
    # Each row of R_zup is a new axis expressed in old coordinates
    R_zup = np.column_stack([x_axis, y_axis, z_axis]).T

    # Angle check
    angle = float(np.degrees(np.arccos(np.clip(np.dot(z_axis, [0, 0, 1]), -1.0, 1.0))))

    if angle <= rotation_threshold_deg:
        R_zup = np.eye(3)
        rotation_applied = False
    else:
        rotation_applied = True

    # ── Flip detection ──
    flipped = False
    flip_top_count = 0
    flip_bot_count = 0

    if do_flip_check and rotation_applied:
        p_temp = (R_zup @ points.T).T  # intermediate Z-up points
        zl = np.percentile(p_temp[:, 2], flip_percentile_range[0])
        zh = np.percentile(p_temp[:, 2], flip_percentile_range[1])
        band_width = (zh - zl) * flip_band_fraction

        bot = ((p_temp[:, 2] >= zl) & (p_temp[:, 2] < zl + band_width)).sum()
        top = ((p_temp[:, 2] > zh - band_width) & (p_temp[:, 2] <= zh)).sum()

        flip_top_count = int(top)
        flip_bot_count = int(bot)

        if top > bot:
            flip = np.array([[1, 0, 0],
                             [0, -1, 0],
                             [0, 0, -1]], dtype=np.float64)
            R_zup = flip @ R_zup
            flipped = True

    # ── Y-up conversion (Isaac Sim) ──
    R_yup = np.array([[1, 0, 0],
                      [0, 0, -1],
                      [0, 1, 0]], dtype=np.float64)
    R_final = R_yup @ R_zup

    points_rotated = (R_final @ points.T).T

    return {
        'R': R_final,
        'R_zup_only': R_zup,
        'points_rotated': points_rotated,
        'angle_deg': angle,
        'flipped': flipped,
        'flip_top_count': flip_top_count,
        'flip_bot_count': flip_bot_count,
        'rotation_applied': rotation_applied,
    }


# ──────────────────────────────────────────────
# 3. Apply rotation to mesh (trimesh)
# ──────────────────────────────────────────────
def apply_rotation_to_mesh(mesh_path: str, R: np.ndarray, output_path: str = None):
    """
    Load a trimesh .ply, rotate vertices by R, save.

    Args:
        mesh_path:   path to input PLY
        R:           (3,3) rotation matrix
        output_path: output path (overwrites input if None)
    """
    import trimesh

    mesh = trimesh.load(mesh_path)
    mesh.vertices = (R @ mesh.vertices.T).T
    out = output_path if output_path else mesh_path
    mesh.export(out)
    print(f'Mesh rotated: {len(mesh.vertices)} verts, {len(mesh.faces)} faces')


# ──────────────────────────────────────────────
# 4. Apply rotation to PyTorch tensor
# ──────────────────────────────────────────────
def apply_rotation_to_tensor(tensor, R: np.ndarray):
    """
    Rotate a (N, 3) PyTorch tensor by R.

    Args:
        tensor: torch.Tensor of shape (N, 3)
        R:      (3,3) rotation matrix

    Returns:
        New torch.Tensor, same dtype/device as input
    """
    import torch

    R_t = torch.from_numpy(R).float().to(tensor.device)
    return (R_t @ tensor.T).T


# ──────────────────────────────────────────────
# 5. rotation.json save / load
# ──────────────────────────────────────────────
def save_rotation_json(R: np.ndarray, path: str):
    """Save rotation matrix to JSON. Format: {"R": [[...], [...], [...]]}"""
    with open(path, 'w') as f:
        json.dump({'R': R.tolist()}, f)


def load_rotation_json(path: str) -> np.ndarray:
    """Load rotation matrix from JSON. Returns identity on failure."""
    try:
        with open(path, 'r') as f:
            d = json.load(f)
        return np.array(d['R'], dtype=np.float64)
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return np.eye(3, dtype=np.float64)


# ──────────────────────────────────────────────
# 6. USDA mesh export (PLY → USDA + collision)
# ──────────────────────────────────────────────
def export_mesh_usda(mesh_path: str, usda_path: str, up_axis: str = "Z"):
    """
    Export PLY mesh to USDA with collision API.

    Args:
        mesh_path: path to .ply file (already rotated to Isaac Sim frame)
        usda_path: output .usda path
        up_axis:   USD stage upAxis ("Y" or "Z", default "Z" for backward compat)
    """
    from pxr import Usd, UsdGeom, UsdPhysics
    import trimesh

    m = trimesh.load(mesh_path)

    stage = Usd.Stage.CreateNew(usda_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z if up_axis == "Z" else UsdGeom.Tokens.y)

    mesh = UsdGeom.Mesh.Define(stage, '/World/mesh')
    mesh.CreatePointsAttr().Set(m.vertices.astype(float).tolist())
    mesh.CreateFaceVertexCountsAttr().Set([3] * len(m.faces))
    mesh.CreateFaceVertexIndicesAttr().Set(m.faces.flatten().tolist())

    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    body = UsdPhysics.RigidBodyAPI.Apply(mesh.GetPrim())
    body.CreateKinematicEnabledAttr().Set(True)

    stage.GetRootLayer().Save()
    print(f'USDA: {usda_path} ({Path(usda_path).stat().st_size / 1024 / 1024:.0f} MB)')


# ──────────────────────────────────────────────
# 7. Ground collision plane (USD)
# ──────────────────────────────────────────────
def export_ground_collision_usda(
    points_rotated: np.ndarray,
    usda_path: str,
    margin: float = 2.0,
    ground_half_height: float = 0.02,
    up_axis: str = "Z",
):
    """
    Generate a thin ground collision plane from rotated points.

    Args:
        points_rotated: (N,3) points in the same frame as visual model
        usda_path:       output .usda path
        margin:          extra margin in meters on each side
        ground_half_height: half-thickness of ground slab
        up_axis:         USD stage upAxis ("Z" for backward compat)
    """
    from pxr import Usd, UsdGeom, UsdPhysics, Gf

    pts = points_rotated

    xmin = np.percentile(pts[:, 0], 5)
    xmax = np.percentile(pts[:, 0], 95)
    ymin = np.percentile(pts[:, 1], 5)
    ymax = np.percentile(pts[:, 1], 95)
    zmin = np.percentile(pts[:, 2], 5)

    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    w = (xmax - xmin) + margin * 2
    d = (ymax - ymin) + margin * 2

    stage = Usd.Stage.CreateNew(usda_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z if up_axis == "Z" else UsdGeom.Tokens.y)

    ground = UsdGeom.Cube.Define(stage, '/World/ground')
    ground.AddScaleOp().Set(Gf.Vec3f(float(w / 2), float(d / 2), float(ground_half_height)))
    ground.AddTranslateOp().Set(Gf.Vec3f(float(cx), float(cy), float(zmin)))

    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
    body = UsdPhysics.RigidBodyAPI.Apply(ground.GetPrim())
    body.CreateKinematicEnabledAttr().Set(True)

    stage.GetRootLayer().Save()
    print(f'Ground: {w:.1f}x{d:.1f}m at Z={zmin:.2f}')


# ──────────────────────────────────────────────
# 8. Combined scene (Gaussians + ground)
# ──────────────────────────────────────────────
def export_combined_scene_usda(
    usda_path: str,
    gaussian_subpath: str,
    ground_subpath: str,
):
    """
    Create a USD stage that layers Gaussian and ground USD files.

    Args:
        usda_path:         output .usda path
        gaussian_subpath:  relative path to the nurec USDZ
        ground_subpath:    relative path to the ground collision USDA
    """
    from pxr import Usd

    stage = Usd.Stage.CreateNew(usda_path)
    stage.GetRootLayer().subLayerPaths = [gaussian_subpath, ground_subpath]
    stage.SetDefaultPrim(stage.DefinePrim('/World', 'Xform'))
    stage.GetRootLayer().Save()
    print(f'Combined scene: {usda_path}')


# ──────────────────────────────────────────────
# CLI entry point (for standalone testing)
# ──────────────────────────────────────────────
if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python align_to_isaac.py <points3D.bin>")
        print("  Computes Manhattan rotation and prints summary.")
        sys.exit(0)

    pts = load_colmap_points(sys.argv[1])
    print(f"Loaded {len(pts):,} points from {sys.argv[1]}")

    result = compute_manhattan_rotation(pts, do_flip_check=True)
    R = result['R']

    print(f"  angle_deg = {result['angle_deg']:.1f}")
    print(f"  flipped   = {result['flipped']} (top={result['flip_top_count']}, bot={result['flip_bot_count']})")
    print(f"  rotation  = {result['rotation_applied']}")
    print(f"  R =\n{R}")
