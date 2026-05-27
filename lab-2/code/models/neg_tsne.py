"""
Neg-t-SNE: thin wrapper around ``cne.CNE`` (contrastive-ne).

Spectrum via fixed Z_bar (Damrich et al., ICLR 2023). kNN graph: scikit-learn.
"""

from __future__ import annotations

import time
import warnings
from typing import Any, Optional

import numpy as np
from cne import CNE

from .base import BaseDR
from utils.knn_graph import symmetrized_knn_graph

__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_K",
    "DEFAULT_N_EPOCHS",
    "DEFAULT_NEG_SAMPLES",
    "DEFAULT_OPTIMIZER",
    "NegTSNE",
    "fit_neg_tsne_spectrum",
    "make_cne_kwargs",
    "symmetrized_knn_graph",
    "z_bar_from_s",
    "z_bars_full",
    "z_bars_spectrum",
    "z_ee",
    "z_tsne",
]

# defaults
DEFAULT_K = 15
DEFAULT_NEG_SAMPLES = 5
DEFAULT_BATCH_SIZE = 1024
DEFAULT_N_EPOCHS = 500
DEFAULT_OPTIMIZER = "sgd"


# Z_bar spectrum 
def z_tsne(n_samples: int) -> float:
    """t-SNE-like anchor (``s=0``): ``100 * n``."""
    return float(100 * n_samples)


def z_ee(n_samples: int, negative_samples: int = DEFAULT_NEG_SAMPLES) -> float:
    """UMAP-like anchor and early-exaggeration ``Z_bar``: ``n^2 / m``."""
    return float(n_samples**2 / negative_samples)


def _z_spectrum_bounds(
    n_samples: int,
    negative_samples: int = DEFAULT_NEG_SAMPLES,
) -> tuple[float, float]:
    """Return ``(Z_t-SNE-like, Z_UMAP-like)`` for log-interpolation."""
    return z_tsne(n_samples), z_ee(n_samples, negative_samples)


def _z_bar_log_interp(z_lo: float, z_hi: float, s: float) -> float:
    return float(z_lo * np.exp(float(s) * (np.log(z_hi) - np.log(z_lo))))


def z_bar_from_s(
    n_samples: int,
    negative_samples: int,
    s: float,
) -> float:
    """Log-interpolate ``Z_bar`` between t-SNE-like (``s=0``) and UMAP-like (``s=1``)."""
    z_lo, z_hi = _z_spectrum_bounds(n_samples, negative_samples)
    return _z_bar_log_interp(z_lo, z_hi, s)


def z_bars_spectrum(
    n_samples: int,
    negative_samples: int = DEFAULT_NEG_SAMPLES,
    *,
    n_points: int = 9,
) -> np.ndarray:
    """Sweep ``geomspace(100*n, n^2/m, n_points)`` along the paper spectrum."""
    z_lo, z_hi = _z_spectrum_bounds(n_samples, negative_samples)
    return np.geomspace(z_lo, z_hi, int(n_points))


def z_bars_full() -> np.ndarray:
    """Fixed ``logspace(4, 12)`` — not tied to paper anchors; prefer ``z_bars_spectrum``."""
    return np.logspace(4, 12, 9)


# CNE / graph helpers


def epoch_split(n_epochs: int) -> tuple[int, int]:
    """Split total epochs into early-exaggeration and main phases (1/3 + 2/3)."""
    ee = n_epochs // 3
    return ee, n_epochs - ee


def make_cne_kwargs(
    *,
    Z_bar: float,
    n_epochs: int,
    k: int = DEFAULT_K,
    negative_samples: int = DEFAULT_NEG_SAMPLES,
    batch_size: int = DEFAULT_BATCH_SIZE,
    optimizer: str = DEFAULT_OPTIMIZER,
    seed: int = 0,
    device: str = "auto",
    data_on_gpu: bool | str = False,
    use_keops: Optional[bool] = False,
    print_freq_epoch: Optional[int] = None,
) -> dict[str, Any]:
    """Keyword arguments for ``cne.CNE`` (``loss_mode='neg'``)."""
    if print_freq_epoch is None:
        print_freq_epoch = max(1, n_epochs // 5)
    return {
        "loss_mode": "neg",
        "Z_bar": float(Z_bar),
        "negative_samples": negative_samples,
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "optimizer": optimizer,
        "anneal_lr": True,
        "lr_min_factor": 0.0,
        "momentum": 0.0,
        "k": k,
        "seed": seed,
        "early_exaggeration": False,
        "force_resample": True,
        "loss_aggregation": "sum",
        "device": device,
        "data_on_gpu": data_on_gpu,
        "use_keops": use_keops,
        "print_freq_epoch": print_freq_epoch,
    }


def _resolve_graph(
    X: np.ndarray,
    k: int,
    graph: Optional[Any],
) -> Any:
    if graph is not None:
        return graph
    return symmetrized_knn_graph(X, k=k)


# NegTSNE model


class NegTSNE(BaseDR):
    """Non-parametric Neg-t-SNE via ``cne.CNE`` and sklearn kNN."""

    def __init__(
        self,
        n_components: int = 2,
        *,
        Z_bar: float,
        k: int = DEFAULT_K,
        negative_samples: int = DEFAULT_NEG_SAMPLES,
        batch_size: int = DEFAULT_BATCH_SIZE,
        n_epochs: int = DEFAULT_N_EPOCHS,
        optimizer: str = DEFAULT_OPTIMIZER,
        seed: int = 0,
        device: str = "auto",
        data_on_gpu: bool | str = False,
        use_keops: bool = False,
        center: bool = False,
        random_state: Optional[int] = None,
        **extra_cne_kwargs: Any,
    ) -> None:
        super().__init__(
            n_components=n_components,
            center=center,
            random_state=random_state if random_state is not None else seed,
        )
        self.Z_bar = float(Z_bar)
        self.k = k
        self.negative_samples = negative_samples
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.optimizer = optimizer
        self.seed = seed if random_state is None else int(random_state)
        self.device = device
        self.data_on_gpu = data_on_gpu
        self.use_keops = use_keops
        self.extra_cne_kwargs = extra_cne_kwargs

        self.embedding_: Optional[np.ndarray] = None
        self._cne: Optional[CNE] = None
        self._graph: Any = None

    def _build_cne(self, n_epochs: Optional[int] = None) -> CNE:
        kwargs = make_cne_kwargs(
            Z_bar=self.Z_bar,
            n_epochs=n_epochs if n_epochs is not None else self.n_epochs,
            k=self.k,
            negative_samples=self.negative_samples,
            batch_size=self.batch_size,
            optimizer=self.optimizer,
            seed=self.seed,
            device=self.device,
            data_on_gpu=self.data_on_gpu,
            use_keops=self.use_keops,
        )
        kwargs.update(self.extra_cne_kwargs)
        kwargs["embd_dim"] = self.n_components
        return CNE(**kwargs)

    def _fit(self, X: np.ndarray) -> None:
        self._fit_transform(X)

    def fit_transform(
        self,
        X: np.ndarray,
        init: Optional[np.ndarray] = None,
        graph: Optional[Any] = None,
    ) -> np.ndarray:
        X = self._validate(X)
        Xp = self._preprocess(X, fitting=True)
        t0 = time.perf_counter()
        Y = self._fit_transform(Xp, init=init, graph=graph)
        self._fit_time_ = time.perf_counter() - t0
        self._is_fitted = True
        return self._validate_output(Y)

    def _fit_transform(
        self,
        X: np.ndarray,
        init: Optional[np.ndarray] = None,
        graph: Optional[Any] = None,
    ) -> np.ndarray:
        resolved = _resolve_graph(
            X,
            self.k,
            graph if graph is not None else self._graph,
        )
        self._graph = resolved

        self._cne = self._build_cne()
        Y = self._cne.fit_transform(X, init=init, graph=resolved)
        self.embedding_ = np.asarray(Y, dtype=float)
        return self.embedding_

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if self.embedding_ is None or self._cne is None:
            raise RuntimeError("NegTSNE is not fitted.")
        if X.shape[0] == self._n_samples_fit:
            return self.embedding_.copy()
        warnings.warn(
            "NegTSNE is non-parametric and cannot embed new points. "
            "Returning the training embedding.",
            UserWarning,
            stacklevel=2,
        )
        return self.embedding_.copy()

    def get_params(self) -> dict[str, Any]:
        return {
            **super().get_params(),
            "Z_bar": self.Z_bar,
            "k": self.k,
            "negative_samples": self.negative_samples,
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
            "optimizer": self.optimizer,
            "seed": self.seed,
            "device": self.device,
            "data_on_gpu": self.data_on_gpu,
            "use_keops": self.use_keops,
        }


# Spectrum pipeline


def fit_neg_tsne_spectrum(
    X: np.ndarray,
    z_bars: Optional[np.ndarray] = None,
    *,
    y: Optional[np.ndarray] = None,
    k: int = DEFAULT_K,
    negative_samples: int = DEFAULT_NEG_SAMPLES,
    batch_size: int = DEFAULT_BATCH_SIZE,
    n_epochs: int = DEFAULT_N_EPOCHS,
    optimizer: str = DEFAULT_OPTIMIZER,
    seed: int = 0,
    device: str = "auto",
    data_on_gpu: bool | str = False,
    use_keops: bool = False,
    early_z_bar: Optional[float] = None,
    graph: Optional[Any] = None,
    verbose: bool = True,
) -> tuple[dict[float, np.ndarray], dict[float, float], np.ndarray]:
    """Early exaggeration at ``n^2/m``, then one ``NegTSNE`` fit per ``Z_bar``."""
    del y

    X = np.asarray(X, dtype=np.float32)
    n = X.shape[0]
    if z_bars is None:
        z_bars = z_bars_spectrum(n, negative_samples)
    z_bars = np.asarray(z_bars, dtype=float)

    if graph is None and verbose:
        print(f"Building kNN graph (k={k}) with sklearn...")
    graph = _resolve_graph(X, k, graph)

    ee_epochs, main_epochs = epoch_split(n_epochs)
    z_ee_val = z_ee(n, negative_samples) if early_z_bar is None else float(early_z_bar)

    if verbose:
        print(f"n={n:,} | EE: Z_bar={z_ee_val:.3e}, epochs={ee_epochs}")
    t0 = time.perf_counter()
    init_ee = NegTSNE(
        Z_bar=z_ee_val,
        k=k,
        negative_samples=negative_samples,
        batch_size=batch_size,
        n_epochs=ee_epochs,
        optimizer=optimizer,
        seed=seed,
        device=device,
        data_on_gpu=data_on_gpu,
        use_keops=use_keops,
        center=False,
    ).fit_transform(X, graph=graph)

    if verbose:
        print(f"EE finished in {(time.perf_counter() - t0) / 60:.1f} min")

    embeddings: dict[float, np.ndarray] = {}
    runtimes_min: dict[float, float] = {}

    for z in z_bars:
        z_key = float(z)
        if verbose:
            print(f"\n=== Z_bar = {z_key:.3e} ({main_epochs} epochs) ===")
        t1 = time.perf_counter()
        embd = NegTSNE(
            Z_bar=z_key,
            k=k,
            negative_samples=negative_samples,
            batch_size=batch_size,
            n_epochs=main_epochs,
            optimizer=optimizer,
            seed=seed,
            device=device,
            data_on_gpu=data_on_gpu,
            use_keops=use_keops,
            center=False,
        ).fit_transform(X, init=init_ee.copy(), graph=graph)
        embeddings[z_key] = embd
        runtimes_min[z_key] = (time.perf_counter() - t1) / 60
        if verbose:
            print(f"Done in {runtimes_min[z_key]:.1f} min")

    return embeddings, runtimes_min, init_ee
