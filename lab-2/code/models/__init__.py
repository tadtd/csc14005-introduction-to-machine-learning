"""Dimensionality reduction models used by the experiment notebooks."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .base import BaseDR
from .isomap import Isomap
from .kpca import KPCA
from .laplacian_eigen import LaplacianEigenmaps
from .lle import LLE
from .pca import PCA
from .t_sne import TSNE
from .umap import UMAP

_NEG_TSNE_EXPORTS = {
    "NegTSNE",
    "fit_neg_tsne_spectrum",
    "z_bar_from_s",
    "z_bars_full",
    "z_bars_spectrum",
    "z_ee",
    "z_tsne",
}


def __getattr__(name: str) -> Any:
    """Load Neg-t-SNE exports only when they are requested."""
    if name not in _NEG_TSNE_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    try:
        module = import_module(".neg_tsne", __name__)
    except ModuleNotFoundError as exc:
        if exc.name == "cne":
            raise ModuleNotFoundError(
                "NegTSNE requires the optional dependency 'contrastive-ne' "
                "(imported as 'cne'). Run `uv sync` in lab-2 before using "
                "models.neg_tsne or the contrastive notebook."
            ) from exc
        raise

    value = getattr(module, name)
    globals()[name] = value
    return value


__all__ = [
    "BaseDR",
    "Isomap",
    "KPCA",
    "LaplacianEigenmaps",
    "LLE",
    "NegTSNE",
    "PCA",
    "TSNE",
    "UMAP",
    "fit_neg_tsne_spectrum",
    "z_bar_from_s",
    "z_bars_full",
    "z_bars_spectrum",
    "z_ee",
    "z_tsne",
]
