"""Generate the PCA, KPCA, and Isomap results included in the report."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import pearsonr
from sklearn.datasets import load_wine, make_circles, make_swiss_roll
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

LAB_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = LAB_ROOT / "report" / "figures"
sys.path.insert(0, str(LAB_ROOT / "code"))

from models.isomap import Isomap  # noqa: E402
from models.kpca import KPCA  # noqa: E402
from models.pca import PCA  # noqa: E402

SEED = 42
GAMMAS = (0.5, 2, 8, 32, 128)
NEIGHBORS = (4, 8, 12, 20, 40)


def save_figure(figure: plt.Figure, filename: str) -> None:
    path = FIGURES_DIR / filename
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {path.relative_to(LAB_ROOT)}")


def run_pca_wine() -> None:
    X = StandardScaler().fit_transform(load_wine().data)
    explained: list[float] = []
    errors: list[float] = []

    for k in range(1, X.shape[1] + 1):
        model = PCA(n_components=k, random_state=SEED).fit(X)
        explained.append(float(model.explained_variance_ratio_.sum()))
        errors.append(model.reconstruction_error(X))

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    ks = np.arange(1, X.shape[1] + 1)
    axes[0].plot(ks, explained, marker="o")
    axes[0].set_title("PCA trên Wine (dữ liệu đã chuẩn hóa)")
    axes[0].set_xlabel("Số thành phần k")
    axes[0].set_ylabel("Phương sai tích lũy")
    axes[0].grid(alpha=0.35)
    axes[1].plot(ks, errors, marker="o", color="tab:red")
    axes[1].set_title("Lỗi tái tạo trung bình")
    axes[1].set_xlabel("Số thành phần k")
    axes[1].set_ylabel("Lỗi tái tạo trung bình")
    axes[1].grid(alpha=0.35)
    figure.tight_layout()
    save_figure(figure, "pca_wine_diagnostics.png")

    print("\nPCA on standardized Wine")
    for k in (1, 2, 13):
        print(f"  k={k:2d}: variance={explained[k - 1]:.4f}, error={errors[k - 1]:.12g}")


def make_extended_circles() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_original, y = make_circles(
        n_samples=1000,
        noise=0.05,
        factor=0.5,
        random_state=SEED,
    )
    noise = np.random.default_rng(SEED).normal(
        loc=0.0,
        scale=0.03,
        size=(X_original.shape[0], 8),
    )
    return X_original, np.column_stack((X_original, noise)), y


def run_kpca_circles() -> None:
    X_original, X, y = make_extended_circles()
    Y_pca = PCA(n_components=2, random_state=SEED).fit_transform(X)
    Y_linear = KPCA(n_components=2, kernel="linear", random_state=SEED).fit_transform(X)
    distance_difference = float(np.max(np.abs(pdist(Y_pca) - pdist(Y_linear))))

    scores: dict[float, float] = {}
    embeddings: dict[float, np.ndarray] = {}
    selected_model: KPCA | None = None
    for gamma in GAMMAS:
        model = KPCA(n_components=2, kernel="rbf", gamma=gamma, random_state=SEED)
        embeddings[gamma] = model.fit_transform(X)
        scores[gamma] = float(silhouette_score(embeddings[gamma], y))
        if gamma == 32:
            selected_model = model

    if selected_model is None:
        raise RuntimeError("Expected gamma=32 in GAMMAS.")

    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    panels = (
        (X_original, "Hai chiều gốc"),
        (Y_pca, "PCA: 10D xuống 2D"),
        (embeddings[32], "KPCA: 10D xuống 2D, gamma=32"),
    )
    for axis, (embedding, title) in zip(axes, panels):
        axis.scatter(embedding[:, 0], embedding[:, 1], c=y, cmap="coolwarm", s=9, alpha=0.85)
        axis.set_title(title)
        axis.set_xlabel("Thành phần 1")
        axis.set_ylabel("Thành phần 2")
        axis.grid(alpha=0.25)
    figure.suptitle("Giảm chiều dữ liệu hai vòng tròn có thêm 8 chiều nhiễu")
    figure.tight_layout()
    save_figure(figure, "kpca_circles_embedding.png")

    pca_score = float(silhouette_score(Y_pca, y))
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(GAMMAS, [scores[gamma] for gamma in GAMMAS], marker="o")
    axis.axhline(pca_score, color="tab:red", linestyle="--", label="PCA")
    axis.set_xscale("log")
    axis.set_title("Độ nhạy tham số của KPCA (10D xuống 2D)")
    axis.set_xlabel("Tham số gamma của kernel RBF")
    axis.set_ylabel("Silhouette theo nhãn tổng hợp")
    axis.grid(alpha=0.35)
    axis.legend()
    figure.tight_layout()
    save_figure(figure, "kpca_gamma_sensitivity.png")

    print("\nKPCA on extended Circles")
    print(f"  max pairwise-distance difference (PCA vs linear KPCA): {distance_difference:.3e}")
    print(f"  gamma=32 eigenvalues: {selected_model.eigenvalues_[0]:.4f}, {selected_model.eigenvalues_[1]:.4f}")
    print(f"  PCA silhouette: {pca_score:.4f}")
    for gamma in GAMMAS:
        print(f"  RBF gamma={gamma:>5}: silhouette={scores[gamma]:.4f}")


def run_isomap_swiss_roll() -> None:
    X, color = make_swiss_roll(n_samples=700, noise=0.05, random_state=SEED)
    embeddings: dict[int, np.ndarray] = {}
    correlations: dict[int, float] = {}
    raw_component_counts: dict[int, int] = {}
    models: dict[int, Isomap] = {}
    distances = cdist(X, X, metric="euclidean")

    for n_neighbors in NEIGHBORS:
        model = Isomap(n_neighbors=n_neighbors, n_components=2, random_state=SEED)
        graph = model._build_neighbor_graph(distances)
        raw_component_counts[n_neighbors] = int(connected_components(graph, directed=False)[0])
        embeddings[n_neighbors] = model.fit_transform(X)
        models[n_neighbors] = model
        correlations[n_neighbors] = float(
            pearsonr(
                squareform(model.geodesic_distances_, checks=False),
                pdist(embeddings[n_neighbors]),
            ).statistic
        )

    Y_pca = PCA(n_components=2, random_state=SEED).fit_transform(X)
    pca_correlation = float(
        pearsonr(
            squareform(models[12].geodesic_distances_, checks=False),
            pdist(Y_pca),
        ).statistic
    )

    figure = plt.figure(figsize=(14, 4))
    axis = figure.add_subplot(1, 3, 1, projection="3d")
    axis.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap="viridis", s=10)
    axis.set_title("Swiss roll gốc")
    axis.set_xlabel("x1")
    axis.set_ylabel("x2")
    axis.set_zlabel("x3")
    for position, (embedding, title) in enumerate(
        ((Y_pca, "PCA 2D"), (embeddings[12], "Isomap 2D")),
        start=2,
    ):
        axis = figure.add_subplot(1, 3, position)
        axis.scatter(embedding[:, 0], embedding[:, 1], c=color, cmap="viridis", s=10)
        axis.set_title(title)
        axis.set_xlabel("Thành phần 1")
        axis.set_ylabel("Thành phần 2")
        axis.grid(alpha=0.25)
    figure.tight_layout()
    save_figure(figure, "isomap_swiss_roll_embedding.png")

    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(NEIGHBORS, [correlations[n] for n in NEIGHBORS], marker="o", color="tab:green")
    axis.set_title("Độ nhạy số láng giềng của Isomap")
    axis.set_xlabel("Số láng giềng")
    axis.set_ylabel("Tương quan với khoảng cách geodesic")
    axis.grid(alpha=0.35)
    figure.tight_layout()
    save_figure(figure, "isomap_neighbor_sensitivity.png")

    print("\nIsomap on Swiss roll")
    print(f"  PCA against t=12 geodesics: {pca_correlation:.4f}")
    for n_neighbors in NEIGHBORS:
        print(
            f"  t={n_neighbors:2d}: correlation={correlations[n_neighbors]:.4f}, "
            f"components before repair={raw_component_counts[n_neighbors]}"
        )


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    run_pca_wine()
    run_kpca_circles()
    run_isomap_swiss_roll()


if __name__ == "__main__":
    main()
