#!/usr/bin/env python3
"""
Read LAMMPS dump trajectories and pack them into a single NPZ file.

Outputs:
  - trajectories: (M, T, N, 2) float array
  - types:        (M, N) int array
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
from tqdm import tqdm

MODE_PATHS = {
    "train": (
        Path("dataset") / "lmp_dumps",
        Path("dataset") / "trajectories_large.npz",
    ),
    "in_dst_test": (
        Path("dataset") / "lmp_dumps_inDistribution_test",
        Path("dataset") / "trajectories_inDistribution_test.npz",
    ),
    "test_left": (
        Path("dataset") / "lmp_dumps_test_left",
        Path("dataset") / "trajectories_test_left.npz",
    ),
    "test_mid": (
        Path("dataset") / "lmp_dumps_test_mid",
        Path("dataset") / "trajectories_test_mid.npz",
    ),
    "test_right": (
        Path("dataset") / "lmp_dumps_test_right",
        Path("dataset") / "trajectories_test_right.npz",
    ),
    "diff_dst_test": (
        Path("dataset") / "lmp_dumps_diffInitPos_test",
        Path("dataset") / "trajectories_diffInitPos_test.npz",
    ),
    "diff_N_test": (
        Path("dataset") / "lmp_dumps_diffN_test",
        Path("dataset") / "trajectories_diffN_test.npz",
    ),
}

def _parse_bounds_line(line: str) -> Tuple[float, float]:
    parts = line.split()
    if len(parts) < 2:
        raise ValueError(f"Bad BOX BOUNDS line: {line!r}")
    return float(parts[0]), float(parts[1])


def _iter_dump_frames(path: Path) -> Iterable[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Yield (ids, types, xy) per frame from a LAMMPS dump file.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        while True:
            line = f.readline()
            if not line:
                return
            if not line.startswith("ITEM: TIMESTEP"):
                continue

            _step = f.readline().strip()  # step value (unused)

            line = f.readline()
            if not line.startswith("ITEM: NUMBER OF ATOMS"):
                raise ValueError("Expected 'ITEM: NUMBER OF ATOMS'")
            n_atoms = int(f.readline().strip())

            line = f.readline()
            if not line.startswith("ITEM: BOX BOUNDS"):
                raise ValueError("Expected 'ITEM: BOX BOUNDS ...'")
            xlo, xhi = _parse_bounds_line(f.readline())
            ylo, yhi = _parse_bounds_line(f.readline())
            _zlo, _zhi = _parse_bounds_line(f.readline())

            line = f.readline().strip()
            if not line.startswith("ITEM: ATOMS"):
                raise ValueError("Expected 'ITEM: ATOMS ...'")
            cols = line.split()[2:]  # after "ITEM:", "ATOMS"
            col_index = {c: i for i, c in enumerate(cols)}

            if "id" not in col_index or "type" not in col_index:
                raise ValueError("Dump must include 'id' and 'type' columns.")

            x_key = next((k for k in ("x", "xu", "xsu", "xs") if k in col_index), None)
            y_key = next((k for k in ("y", "yu", "ysu", "ys") if k in col_index), None)
            if x_key is None or y_key is None:
                raise ValueError("Dump must include x/y (or xu/yu) or xs/ys columns.")

            rows = [f.readline().split() for _ in range(n_atoms)]
            if any(len(r) < len(cols) for r in rows):
                raise ValueError("Atom row has fewer columns than header indicates.")

            ids = np.array([int(r[col_index["id"]]) for r in rows], dtype=np.int32) # 
            types = np.array([int(r[col_index["type"]]) for r in rows], dtype=np.int32)
            x_raw = np.array([float(r[col_index[x_key]]) for r in rows], dtype=np.float32)
            y_raw = np.array([float(r[col_index[y_key]]) for r in rows], dtype=np.float32)

            if x_key.endswith("s") and y_key.endswith("s"):
                x = xlo + x_raw * (xhi - xlo)
                y = ylo + y_raw * (yhi - ylo)
            else:
                x, y = x_raw, y_raw

            xy = np.column_stack((x, y))
            yield ids, types, xy


def read_lammps_trajectory(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      positions: (T, N, 2)
      types:     (N,)
    """
    frames = []
    id_order = None
    types_ref = None
    n_atoms = None

    for ids, types, xy in _iter_dump_frames(path):
        if n_atoms is None:
            n_atoms = ids.size
        elif ids.size != n_atoms:
            raise ValueError(f"{path} has inconsistent atom counts across frames.")

        if id_order is None:
            id_order = np.sort(ids)
            if np.unique(id_order).size != n_atoms:
                raise ValueError(f"{path} contains duplicate atom ids.")

        idxs = np.searchsorted(id_order, ids)
        if np.any(idxs >= n_atoms) or np.any(id_order[idxs] != ids):
            raise ValueError(f"{path} has ids that differ from the first frame.")

        frame_xy = np.empty((n_atoms, 2), dtype=np.float32)
        frame_xy[idxs] = xy

        if types_ref is None:
            types_ref = np.empty(n_atoms, dtype=np.int32)
            types_ref[idxs] = types
        else:
            frame_types = np.empty(n_atoms, dtype=np.int32)
            frame_types[idxs] = types
            if not np.array_equal(frame_types, types_ref):
                raise ValueError(f"{path} has particle type changes across frames.")

        frames.append(frame_xy)

    if not frames:
        raise ValueError(f"No frames found in {path}")

    positions = np.stack(frames, axis=0)
    return positions, types_ref


def load_all_trajectories(
    input_dir: Path,
    pattern: str,
    workers: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    files = sorted(input_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched {pattern} in {input_dir}")

    trajs = []
    types_list = []

    def _append(file_path: Path, traj: np.ndarray, types: np.ndarray) -> None:
        if trajs:
            if traj.shape != trajs[0].shape:
                raise ValueError(
                    f"{file_path} has shape {traj.shape}, expected {trajs[0].shape}."
                )
            if types.shape != types_list[0].shape:
                raise ValueError(
                    f"{file_path} has types shape {types.shape}, "
                    f"expected {types_list[0].shape}."
                )
        trajs.append(traj)
        types_list.append(types)

    if workers is None or workers <= 1:
        iterator = (
            (file_path, read_lammps_trajectory(file_path))
            for file_path in tqdm(files, desc="Loading trajectories")
        )
        for file_path, (traj, types) in iterator:
            _append(file_path, traj, types)
    else:
        with mp.Pool(processes=workers) as pool:
            # Pool.imap preserves input order, keeping file order consistent.
            results = tqdm(
                pool.imap(read_lammps_trajectory, files),
                total=len(files),
                desc="Loading trajectories",
            )
            for file_path, (traj, types) in zip(files, results):
                _append(file_path, traj, types)

    trajectories = np.stack(trajs, axis=0)
    types_arr = np.stack(types_list, axis=0)
    filenames = [p.name for p in files]
    return trajectories, types_arr, filenames


def main() -> None:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Pack LAMMPS dump trajectories into NPZ.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=tuple(MODE_PATHS),
        default="train",
        help="Dataset preset to use for the default input/output paths.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory with LAMMPS dump files. Overrides the path implied by --mode.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.lammpstrj",
        help="Glob pattern to match dump files (default: *.lammpstrj)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output NPZ path. Overrides the path implied by --mode.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Number of worker processes (0/1 disables multiprocessing).",
    )

    args = parser.parse_args()
    mode_input_dir, mode_output = MODE_PATHS[args.mode]
    args.input_dir = args.input_dir or (script_dir / mode_input_dir)
    args.output = args.output or (script_dir / mode_output)

    trajectories, types, filenames = load_all_trajectories(
        args.input_dir,
        args.pattern,
        args.workers,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        trajectories=trajectories,
        types=types,
        filenames=np.array(filenames, dtype=object),
    )

    print(
        f"Saved {trajectories.shape[0]} trajectories to {args.output} "
        f"with shape {trajectories.shape} and types {types.shape}."
    )


if __name__ == "__main__":
    main()
