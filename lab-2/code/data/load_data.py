"""DR lab datasets. Downloads and local trees live under ``code/data/raw/`` (see README)."""

from __future__ import annotations

import os
import re
import shutil
import traceback
import urllib.request
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn, Optional

import h5py
import numpy as np
from PIL import Image
from sklearn.datasets import fetch_openml, make_circles, make_swiss_roll

_HERE = Path(__file__).resolve().parent
_RAW = _HERE / "raw"
_RAW.mkdir(parents=True, exist_ok=True)
_CACHE = _RAW / ".cache"
_CACHE.mkdir(parents=True, exist_ok=True)
_MNIST_CACHE = _CACHE / "openml"
_MNIST_CACHE.mkdir(parents=True, exist_ok=True)
_COIL_CACHE = _CACHE / "coil20"
_COIL_CACHE.mkdir(parents=True, exist_ok=True)


def _under_here(p: str | Path) -> Path:
    q = Path(p).expanduser()
    return q.resolve() if q.is_absolute() else (_HERE / q).resolve()


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    X: np.ndarray
    y: np.ndarray
    description: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "X", np.asarray(self.X, dtype=float))
        object.__setattr__(self, "y", np.asarray(self.y))

    def __repr__(self) -> str:
        return f"DatasetBundle({self.name!r}, X{self.X.shape}, y{self.y.shape})"


def load_swiss_roll(
    n_samples: int = 1_500, noise: float = 0.05, random_state: Optional[int] = 0
) -> DatasetBundle:
    X, color = make_swiss_roll(
        n_samples=n_samples, noise=noise, random_state=random_state
    )
    return DatasetBundle(
        "swiss_roll", X, color.astype(float), "Swiss roll; y = unroll coordinate."
    )


def make_circles_(
    n_samples: int = 1_000,
    *,
    noise: float = 0.05,
    factor: float = 0.5,
    random_state: Optional[int] = 0,
) -> DatasetBundle:
    X, y = make_circles(
        n_samples=n_samples,
        noise=noise,
        factor=factor,
        random_state=random_state,
    )
    return DatasetBundle("circles", X.astype(float), y.astype(int), "Nested circles.")


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response, tmp.open("wb") as stream:
        shutil.copyfileobj(response, stream)
    tmp.replace(destination)


def _download_mnist_h5(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = fetch_openml(
        "mnist_784",
        version=1,
        as_frame=False,
        data_home=str(_MNIST_CACHE),
    )
    X = np.asarray(data.data, dtype=float)
    y = np.asarray(data.target, dtype=int)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with h5py.File(tmp, "w") as f:
        f.create_dataset("X", data=X, compression="gzip", compression_opts=1)
        f.create_dataset("y", data=y, compression="gzip", compression_opts=1)
    tmp.replace(path)


_OBJ = re.compile(r"^obj(\d+)$", re.I)
_FLAT = re.compile(r"^obj(\d+)__(\d+)\.(png|pgm|ppm|jpg|jpeg)$", re.I)
_IMG_EXT = {".png", ".pgm", ".ppm", ".jpg", ".jpeg"}
_MIN_FLAT = 200


def _gray_vec(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=float).ravel()


def _coil_bundle(rows: list[np.ndarray], labels: list[int], root: Path) -> DatasetBundle:
    if not rows:
        raise ValueError(f"No images under {root}")
    w = {r.shape[0] for r in rows}
    if len(w) != 1:
        raise ValueError(f"Inconsistent image widths: {sorted(w)[:8]}")
    X, y = np.stack(rows), np.asarray(labels, dtype=int)
    return DatasetBundle("coil20", X, y, f"COIL-20 @ {root}; shape {X.shape}")


def _proc_layout_ok(root: Path) -> bool:
    return root.is_dir() and sum(
        1 for d in root.iterdir() if d.is_dir() and _OBJ.match(d.name)
    ) >= 20


def _unwrap_proc(root: Path) -> Path:
    cur = root.resolve()
    for _ in range(10):
        if _proc_layout_ok(cur):
            return cur
        subs = [p for p in cur.iterdir() if p.is_dir()]
        if len(subs) != 1:
            return cur
        cur = subs[0]
    return cur


def _flat_ok(root: Path) -> bool:
    if not root.is_dir():
        return False
    n = sum(1 for p in root.iterdir() if p.is_file() and _FLAT.match(p.name))
    return n >= _MIN_FLAT


def _load_coil_flat(root: Path) -> DatasetBundle:
    root = root.resolve()
    triples: list[tuple[int, int, Path]] = []
    for p in root.iterdir():
        if p.is_file() and (m := _FLAT.match(p.name)):
            triples.append((int(m.group(1)), int(m.group(2)), p))
    if len(triples) < _MIN_FLAT:
        raise ValueError(f"Too few flat COIL files in {root}")
    oids = {a for a, _, _ in triples}
    if oids != set(range(1, 21)):
        raise ValueError(f"Need obj ids 1..20, got {sorted(oids)}")
    triples.sort(key=lambda t: (t[0], t[1]))
    rows, labels = [], []
    for oid, _, p in triples:
        rows.append(_gray_vec(p))
        labels.append(oid - 1)
    return _coil_bundle(rows, labels, root)


def _load_coil_proc(root: Path) -> DatasetBundle:
    root = _unwrap_proc(root.resolve())
    if not _proc_layout_ok(root):
        raise ValueError(f"Not a COIL-20 tree (obj1..obj20): {root}")
    obj_dirs = sorted(
        (int(m.group(1)), d)
        for d in root.iterdir()
        if d.is_dir() and (m := _OBJ.match(d.name))
    )
    ids = [i for i, _ in obj_dirs]
    if len(ids) != 20 or sorted(ids) != list(range(1, 21)):
        raise ValueError(f"Expected obj1..obj20 under {root}")
    rows, labels = [], []
    for oid, d in obj_dirs:
        lab = oid - 1
        for p in sorted(
            x for x in d.iterdir() if x.is_file() and x.suffix.lower() in _IMG_EXT
        ):
            rows.append(_gray_vec(p))
            labels.append(lab)
    return _coil_bundle(rows, labels, root)


def _coil_candidates(extra: Optional[Path] = None) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()

    def add(p: Path) -> None:
        k = str(p.expanduser())
        if k not in seen:
            seen.add(k)
            out.append(Path(k))

    if extra is not None:
        add(extra)
    for env in ("COIL20_PROCESSED_DIR", "COIL20_DIR", "COIL20_IMAGE_DIR"):
        if v := os.environ.get(env):
            add(Path(v))
    for base in (_RAW, _COIL_CACHE, _HERE):
        for name in ("coil-20", "coil-20-proc", "COIL-20", "COIL20", "coil20"):
            add(base / name)
    for name in ("coil-20", "coil-20-proc", "COIL-20", "COIL20", "coil20"):
        add(Path.cwd() / name)
    return out


def _first_proc(cands: list[Path]) -> Optional[Path]:
    for raw in cands:
        if raw.is_dir():
            r = _unwrap_proc(raw)
            if _proc_layout_ok(r):
                return r.resolve()
    return None


def _resolve_coil_layout(root: Path) -> Optional[Path]:
    root = root.resolve()
    queue: list[Path] = [root]
    seen: set[Path] = set()
    while queue:
        cur = queue.pop(0)
        if cur in seen:
            continue
        seen.add(cur)
        if _proc_layout_ok(cur):
            return _unwrap_proc(cur)
        if _flat_ok(cur):
            return cur
        try:
            queue.extend(p for p in cur.iterdir() if p.is_dir())
        except OSError:
            continue
    return None


def _coil_hint(path: Path) -> str:
    try:
        names = sorted(p.name for p in path.iterdir())
    except OSError as e:
        return str(e)
    return f"entries ({len(names)}): {names[:30]}{'…' if len(names) > 30 else ''}"


_COIL_URL = (
    "http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/"
    "coil-20/coil-20-proc.zip"
)


def _download_coil20(root: Path = _COIL_CACHE) -> Path:
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "coil-20-proc.zip"
    if not archive.is_file():
        _download_file(_COIL_URL, archive)
    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(root)
    layout = _resolve_coil_layout(root)
    if layout is None:
        raise ValueError(f"Downloaded COIL-20 archive did not contain a usable layout: {root}")
    return layout


def _raise_coil_bad() -> NoReturn:
    cands = _coil_candidates(None)
    dirs = [p for p in cands if p.is_dir()]
    if not dirs:
        raise FileNotFoundError(
            f"COIL-20 not found. It will be downloaded automatically from {_COIL_URL} "
            f"when load_coil20() runs, or you can point COIL20_PROCESSED_DIR at an extracted tree."
        )
    for key in ("coil-20", "coil-20-proc"):
        p = _RAW / key
        if p.is_dir() and not (_proc_layout_ok(_unwrap_proc(p)) or _flat_ok(p)):
            raise ValueError(f"Invalid COIL layout at {p}: {_coil_hint(p)}")
    raise FileNotFoundError(
        f"No valid COIL-20 tree or flat objNN__M.* folder. Tried {_RAW}, {_HERE}, cwd."
    )


def load_coil20(processed_dir: Optional[str | Path] = None) -> DatasetBundle:
    if processed_dir is not None:
        p = _under_here(processed_dir)
        if not p.is_dir():
            raise FileNotFoundError(f"Not a directory: {p}")
        layout = _resolve_coil_layout(p)
        if layout is not None:
            if _proc_layout_ok(layout):
                return _load_coil_proc(layout)
            return _load_coil_flat(layout)
        raise ValueError(f"Bad COIL path {p}: {_coil_hint(p)}")

    cands = _coil_candidates(None)
    for raw in cands:
        if raw.is_dir():
            layout = _resolve_coil_layout(raw)
            if layout is not None:
                if _proc_layout_ok(layout):
                    return _load_coil_proc(layout)
                return _load_coil_flat(layout)
    try:
        layout = _download_coil20()
    except Exception as exc:
        raise FileNotFoundError(f"Failed to download COIL-20 from {_COIL_URL}: {exc}") from exc
    if _proc_layout_ok(layout):
        return _load_coil_proc(layout)
    return _load_coil_flat(layout)


# PyMVPA ``mnist.hdf5`` (https://www.pymvpa.org/datadb/mnist.html) — nested paths in the published file.
_MNIST_H5_KNOWN: tuple[tuple[str, str], ...] = (
    (
        "items/1/items/1/rcargs/items/0",
        "items/1/items/1/rcargs/items/1/items/0/items/1/rcargs/items/0",
    ),
    (
        "items/0/items/1/rcargs/items/0",
        "items/0/items/1/rcargs/items/1/items/0/items/1/rcargs/items/0",
    ),
)


def _mnist_h5_path(hdf5_path: Optional[str | Path]) -> Path:
    if hdf5_path is None:
        return (_RAW / "mnist.hdf5").resolve()
    q = Path(hdf5_path).expanduser()
    return q.resolve() if q.is_absolute() else (_RAW / q).resolve()


def _read_mnist_h5(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load 70_000 × 784 ``X`` and int ``y`` from PyMVPA-style or compatible HDF5."""
    if not path.is_file():
        _download_mnist_h5(path)
    with h5py.File(path, "r") as f:
        if "X" in f and "y" in f:
            X = np.asarray(f["X"], dtype=float)
            y = np.asarray(f["y"], dtype=int)
            if X.ndim == 2 and X.shape[1] == 784 and y.ndim == 1 and len(X) == len(y):
                return X, y
        if all(xp in f and yp in f for xp, yp in _MNIST_H5_KNOWN):
            X = np.vstack(
                [np.asarray(f[xp], dtype=float) for xp, _ in _MNIST_H5_KNOWN]
            )
            y = np.concatenate(
                [np.asarray(f[yp], dtype=int) for _, yp in _MNIST_H5_KNOWN]
            )
            return X, y

        xs: dict[int, np.ndarray] = {}
        ys: dict[int, np.ndarray] = {}

        def visit(_name: str, obj: h5py.Dataset | h5py.Group) -> None:
            if not isinstance(obj, h5py.Dataset) or obj.size == 0:
                return
            if obj.ndim == 2 and obj.shape[1] == 784:
                n = int(obj.shape[0])
                if n in (60000, 10000):
                    if n in xs:
                        raise ValueError(
                            f"Ambiguous MNIST HDF5: multiple {n}×784 datasets in {path}"
                        )
                    xs[n] = np.asarray(obj[...], dtype=float)
            elif obj.ndim == 1:
                n = int(obj.shape[0])
                if n not in (60000, 10000):
                    return
                arr = np.asarray(obj[...])
                if arr.dtype.kind not in "iu":
                    return
                arr = arr.astype(np.int64, copy=False)
                if arr.min() < 0 or arr.max() > 9:
                    return
                if n in ys:
                    return
                ys[n] = arr.astype(int, copy=False)

        f.visititems(visit)
        if set(xs) == {60000, 10000} and set(ys) == {60000, 10000}:
            X = np.vstack([xs[60000], xs[10000]])
            y = np.concatenate([ys[60000], ys[10000]])
            return X, y

    raise ValueError(
        f"Unrecognized MNIST HDF5 layout: {path}. "
        "Expected PyMVPA ``mnist.hdf5`` or (60000, 784) and (10000, 784) arrays "
        "with matching 1-D digit labels."
    )


def load_mnist(
    n_samples: Optional[int] = None,
    random_state: Optional[int] = 0,
    *,
    hdf5_path: Optional[str | Path] = None,
) -> DatasetBundle:
    """
    MNIST is cached under ``code/data/raw/mnist.hdf5`` and auto-downloaded from OpenML on first use.

    If ``hdf5_path`` is provided, the file is read from or written to that path instead of the default cache.
    """
    path = _mnist_h5_path(hdf5_path)
    X, y = _read_mnist_h5(path)
    if n_samples is not None and n_samples < len(X):
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X), size=n_samples, replace=False)
        X, y = X[idx], y[idx]
    return DatasetBundle(
        "mnist",
        X,
        y,
        f"MNIST from {path.name}; {X.shape[0]} × 784",
    )


ALL_DATASET_KEYS: tuple[str, ...] = (
    "circles",
    "swiss_roll",
    "mnist",
    "coil20",
)

_REGISTRY: Mapping[str, Callable[..., DatasetBundle]] = {
    "circles": make_circles_,
    "swiss_roll": load_swiss_roll,
    "mnist": load_mnist,
    "coil20": load_coil20,
    "coil_20": load_coil20,
    "coil-20": load_coil20,
}


def load_dataset(name: str, **kwargs) -> DatasetBundle:
    key = name.strip().lower().replace(" ", "_")
    if key not in _REGISTRY:
        raise ValueError(f"Unknown {name!r}. Pick from: {sorted(_REGISTRY)}")
    return _REGISTRY[key](**kwargs)


def _prefetch_all_datasets() -> int:
    tasks: list[tuple[str, Callable[[], DatasetBundle], bool]] = [
        ("circles", lambda: make_circles_(), False),
        ("swiss_roll", lambda: load_swiss_roll(), False),
        ("mnist", lambda: load_mnist(), False),
        ("coil20", load_coil20, False),
    ]
    print(f"Prefetch → caches under {_RAW}\n", flush=True)
    failed, skipped = [], []
    for label, fn, optional in tasks:
        print(f"[{label}] …", flush=True)
        try:
            b = fn()
        except FileNotFoundError as e:
            if optional:
                skipped.append(label)
                print(f"  skip: {e}", flush=True)
            else:
                failed.append(label)
                print(f"  fail: {e}", flush=True)
                traceback.print_exc()
        except Exception as e:
            failed.append(label)
            print(f"  fail: {e}", flush=True)
            traceback.print_exc()
        else:
            print(f"  ok {b.name} {b.X.shape}", flush=True)
    if skipped:
        print(f"\nOptional skipped: {skipped}", flush=True)
    if failed:
        print(f"\nFailed: {failed}", flush=True)
        return 1
    print("\nDone.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_prefetch_all_datasets())
