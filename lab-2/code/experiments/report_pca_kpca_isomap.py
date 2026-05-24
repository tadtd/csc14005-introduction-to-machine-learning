"""Generate report figures and metrics for PCA, KPCA, and Isomap."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from data.load_data import load_dataset
from models import Isomap, KPCA, PCA


SEED = 42
FIGURE_DIR = ROOT / "report" / "figures"
KPCA_GAMMAS = [0.5, 2.0, 8.0, 32.0, 128.0]
ISOMAP_NEIGHBORS = [4, 8, 12, 20, 40]


def save_figure(fig: plt.Figure, name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def pca_experiment() -> dict[str, object]:
    bundle = load_dataset("wine")
    X = StandardScaler().fit_transform(bundle.X)
    dimensions = list(range(1, X.shape[1] + 1))
    errors: list[float] = []
    explained: list[float] = []
    for k in dimensions:
        model = PCA(n_components=k).fit(X)
        errors.append(model.reconstruction_error(X))
        explained.append(float(np.sum(model.explained_variance_ratio_)))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(dimensions, explained, marker="o", color="tab:blue")
    axes[0].set_xlabel("Số thành phần k")
    axes[0].set_ylabel("Phương sai tích lũy")
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(alpha=0.3)
    axes[1].plot(dimensions, errors, marker="o", color="tab:red")
    axes[1].set_xlabel("Số thành phần k")
    axes[1].set_ylabel("Lỗi tái tạo trung bình")
    axes[1].grid(alpha=0.3)
    fig.suptitle("PCA trên Wine (dữ liệu đã chuẩn hóa)")
    fig.tight_layout()
    save_figure(fig, "pca_wine_diagnostics.png")
    return {
        "explained_variance_k2": explained[1],
        "reconstruction_error_k1": errors[0],
        "reconstruction_error_k2": errors[1],
        "reconstruction_error_k13": errors[-1],
    }


def kpca_experiment() -> dict[str, object]:
    bundle = load_dataset("circles", n_samples=1000, noise=0.05, random_state=SEED)
    noise_dimensions = np.random.default_rng(SEED).normal(
        loc=0.0, scale=0.03, size=(bundle.X.shape[0], 8)
    )
    X = np.column_stack((bundle.X, noise_dimensions))

    pca_embedding = PCA(n_components=2).fit_transform(X)
    linear_embedding = KPCA(n_components=2, kernel="linear").fit_transform(X)
    selected_gamma = 32.0
    kpca = KPCA(n_components=2, kernel="rbf", gamma=selected_gamma).fit(X)
    kpca_embedding = kpca.transform(X)
    linear_distance_difference = float(
        np.max(np.abs(pdist(pca_embedding) - pdist(linear_embedding)))
    )
    gamma_scores = {}
    for gamma in KPCA_GAMMAS:
        embedding = KPCA(n_components=2, kernel="rbf", gamma=gamma).fit_transform(X)
        gamma_scores[str(gamma)] = float(silhouette_score(embedding, bundle.y))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, embedding, title in (
        (axes[0], bundle.X, "Hai chiều gốc"),
        (axes[1], pca_embedding, "PCA: 10D xuống 2D"),
        (axes[2], kpca_embedding, "KPCA: 10D xuống 2D, gamma=32"),
    ):
        ax.scatter(embedding[:, 0], embedding[:, 1], c=bundle.y, cmap="coolwarm", s=8)
        ax.set_title(title)
        ax.set_xlabel("Thành phần 1")
        ax.set_ylabel("Thành phần 2")
        ax.grid(alpha=0.2)
    fig.suptitle("Giảm chiều dữ liệu hai vòng tròn có thêm 8 chiều nhiễu")
    fig.tight_layout()
    save_figure(fig, "kpca_circles_embedding.png")

    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(KPCA_GAMMAS, [gamma_scores[str(gamma)] for gamma in KPCA_GAMMAS], marker="o")
    ax.axhline(
        silhouette_score(pca_embedding, bundle.y),
        color="tab:red",
        linestyle="--",
        label="PCA",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Tham số gamma của kernel RBF")
    ax.set_ylabel("Silhouette theo nhãn tổng hợp")
    ax.set_title("Độ nhạy tham số của KPCA (10D xuống 2D)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, "kpca_gamma_sensitivity.png")
    return {
        "input_dimension": int(X.shape[1]),
        "output_dimension": 2,
        "linear_kernel_max_distance_difference": linear_distance_difference,
        "pca_silhouette": float(silhouette_score(pca_embedding, bundle.y)),
        "selected_gamma": selected_gamma,
        "rbf_silhouette_by_gamma": gamma_scores,
        "rbf_top_eigenvalues": kpca.eigenvalues_.tolist(),
    }


def isomap_experiment() -> dict[str, object]:
    bundle = load_dataset("swiss_roll", n_samples=700, noise=0.05, random_state=SEED)
    pca_embedding = PCA(n_components=2).fit_transform(bundle.X)
    selected_neighbors = 12
    isomap = Isomap(n_components=2, n_neighbors=selected_neighbors).fit(bundle.X)
    isomap_embedding = isomap.transform(bundle.X)

    geodesic = isomap.geodesic_distances_
    if geodesic is None:
        raise RuntimeError("Isomap did not retain geodesic distances.")
    upper = np.triu_indices_from(geodesic, k=1)
    pca_distances = squareform(pdist(pca_embedding))[upper]
    isomap_distances = squareform(pdist(isomap_embedding))[upper]
    geodesic_distances = geodesic[upper]
    pca_corr = float(np.corrcoef(geodesic_distances, pca_distances)[0, 1])
    isomap_corr = float(np.corrcoef(geodesic_distances, isomap_distances)[0, 1])

    fig = plt.figure(figsize=(14, 4))
    ax0 = fig.add_subplot(131, projection="3d")
    ax0.scatter(bundle.X[:, 0], bundle.X[:, 1], bundle.X[:, 2], c=bundle.y, cmap="viridis", s=8)
    ax0.set_title("Swiss roll gốc")
    ax0.set_xlabel("x1")
    ax0.set_ylabel("x2")
    ax0.set_zlabel("x3")
    for index, (embedding, title) in enumerate(
        ((pca_embedding, "PCA 2D"), (isomap_embedding, "Isomap 2D")), start=2
    ):
        ax = fig.add_subplot(1, 3, index)
        ax.scatter(embedding[:, 0], embedding[:, 1], c=bundle.y, cmap="viridis", s=8)
        ax.set_title(title)
        ax.set_xlabel("Thành phần 1")
        ax.set_ylabel("Thành phần 2")
        ax.grid(alpha=0.2)
    fig.tight_layout()
    save_figure(fig, "isomap_swiss_roll_embedding.png")

    neighbor_correlations = {}
    for n_neighbors in ISOMAP_NEIGHBORS:
        model = Isomap(n_components=2, n_neighbors=n_neighbors).fit(bundle.X)
        embedding = model.transform(bundle.X)
        distances = model.geodesic_distances_
        if distances is None:
            raise RuntimeError("Isomap did not retain geodesic distances.")
        pairs = np.triu_indices_from(distances, k=1)
        neighbor_correlations[str(n_neighbors)] = float(
            np.corrcoef(
                distances[pairs],
                squareform(pdist(embedding))[pairs],
            )[0, 1]
        )

    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(
        ISOMAP_NEIGHBORS,
        [neighbor_correlations[str(value)] for value in ISOMAP_NEIGHBORS],
        marker="o",
        color="tab:green",
    )
    ax.set_xlabel("Số láng giềng")
    ax.set_ylabel("Tương quan với khoảng cách geodesic")
    ax.set_title("Độ nhạy số láng giềng của Isomap")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_figure(fig, "isomap_neighbor_sensitivity.png")
    return {
        "selected_neighbors": selected_neighbors,
        "geodesic_correlation_pca": pca_corr,
        "geodesic_correlation_isomap": isomap_corr,
        "isomap_correlation_by_neighbors": neighbor_correlations,
    }


def main() -> None:
    results = {
        "random_seed": SEED,
        "pca": pca_experiment(),
        "kpca": kpca_experiment(),
        "isomap": isomap_experiment(),
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
