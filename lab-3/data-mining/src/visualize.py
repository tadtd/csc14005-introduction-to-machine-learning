"""
Visualization Functions for Tensor Decomposition
==================================================

All visualization functions take COMPUTED DATA as input.
No data is generated inside these functions — they only plot
results that have been computed by the decomposition algorithms.

Each function documents:
- Input: what data it expects
- Meaning: what the plot represents
- Interpretation: how to read the results

Plot Style: Academic publication quality
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import warnings


# =============================================================================
# GLOBAL STYLE SETTINGS
# =============================================================================

def set_style():
    """Set publication-quality plot style."""
    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'figure.dpi': 150,
        'font.size': 11,
        'font.family': 'serif',
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'lines.linewidth': 1.5,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


# =============================================================================
# 1. CONVERGENCE PLOT
# =============================================================================

def plot_convergence(fit_history, title="CP-ALS Convergence", ax=None,
                     ylabel="Fit", xlabel="Iteration", label=None,
                     log_scale=False, save_path=None):
    """
    Plot convergence curve: fit value vs. iteration.

    Input
    -----
    - fit_history: list of fit values computed at each iteration
      (from cp_als or cp_opt results)

    Meaning
    -------
    Shows how the decomposition quality improves over iterations.
    A good convergence curve:
    - Rises steeply initially (rapid improvement)
    - Flattens eventually (convergence)

    Interpretation
    --------------
    - Steep rise → algorithm making progress
    - Plateau → algorithm has converged
    - Oscillations → possible numerical issues or rank too high
    - Never plateaus → may need more iterations

    Parameters
    ----------
    fit_history : list of float
        Fit values at each iteration.
    title : str
        Plot title.
    ax : matplotlib.axes.Axes or None
        Axes to plot on. If None, creates new figure.
    ylabel, xlabel : str
        Axis labels.
    label : str or None
        Legend label.
    log_scale : bool
        If True, plot 1-fit on log scale (useful for seeing convergence rate).
    save_path : str or None
        If provided, save figure to this path.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    set_style()

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    iterations = np.arange(1, len(fit_history) + 1)

    if log_scale:
        # Plot 1-fit (residual) on log scale
        residuals = 1.0 - np.array(fit_history)
        residuals = np.maximum(residuals, 1e-16)  # Avoid log(0)
        ax.semilogy(iterations, residuals, 'o-', markersize=3, label=label)
        ax.set_ylabel("1 - Fit (log scale)")
    else:
        ax.plot(iterations, fit_history, 'o-', markersize=3, label=label)
        ax.set_ylabel(ylabel)

    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    if label:
        ax.legend()

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved to {save_path}")

    return fig, ax


# =============================================================================
# 2. FACTOR MATRIX VISUALIZATION
# =============================================================================

def plot_factors(factors, mode_names=None, weights=None, title="CP Factors",
                 figsize=None, save_path=None):
    """
    Visualize factor matrices as heatmaps and/or line plots.

    Input
    -----
    - factors: list of factor matrices from CP decomposition
    - mode_names: names for each mode (e.g., ['Subjects', 'Channels', 'Time'])
    - weights: component weights λ

    Meaning
    -------
    Each factor matrix A^(n) shows the loading of each element along
    mode n on each component r. High absolute values indicate strong
    contribution to that component.

    Interpretation
    --------------
    - Mode 0 (Subjects): which subjects load heavily on each component
    - Mode 1 (Channels): spatial pattern of each component
    - Mode 2 (Time): temporal dynamics of each component

    Parameters
    ----------
    factors : list of np.ndarray
        Factor matrices.
    mode_names : list of str or None
        Names for each mode.
    weights : np.ndarray or None
        Component weights.
    title : str
        Overall title.
    figsize : tuple or None
        Figure size.
    save_path : str or None
        Save path.
    """
    set_style()

    N = len(factors)
    R = factors[0].shape[1]

    if mode_names is None:
        mode_names = [f"Mode {n}" for n in range(N)]

    if figsize is None:
        figsize = (4 * N, max(3, R * 0.8))

    fig, axes = plt.subplots(1, N, figsize=figsize)
    if N == 1:
        axes = [axes]

    for n in range(N):
        ax = axes[n]
        A = factors[n]

        # Heatmap
        im = ax.imshow(A, aspect='auto', cmap='RdBu_r',
                        vmin=-np.max(np.abs(A)), vmax=np.max(np.abs(A)))
        ax.set_title(f"{mode_names[n]}\n({A.shape[0]} × {R})")
        ax.set_xlabel("Component")
        ax.set_ylabel(f"{mode_names[n]} index")
        ax.set_xticks(range(R))
        ax.set_xticklabels([f"R{r+1}" for r in range(R)])
        plt.colorbar(im, ax=ax, shrink=0.8)

    if weights is not None:
        fig.suptitle(f"{title}\nWeights: [{', '.join(f'{w:.3f}' for w in weights)}]",
                     fontsize=13, y=1.02)
    else:
        fig.suptitle(title, fontsize=13, y=1.02)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)

    return fig, axes


# =============================================================================
# 3. RECONSTRUCTION ERROR VS RANK
# =============================================================================

def plot_reconstruction_error(ranks, errors, corcondia=None,
                               title="Reconstruction Error vs. Rank",
                               save_path=None):
    """
    Plot reconstruction error and CORCONDIA vs. CP rank.

    Input
    -----
    - ranks: list of rank values evaluated
    - errors: list of relative errors for each rank
    - corcondia: list of CORCONDIA values (optional)

    Meaning
    -------
    Shows how approximation quality improves with increasing rank.
    Used for rank selection ("elbow method").

    Interpretation
    --------------
    - Error decreases with rank (more components → better fit)
    - Look for "elbow": point where improvement slows down
    - CORCONDIA drops sharply when rank exceeds the true rank
    - Choose rank where: error is low AND CORCONDIA > 80%

    Parameters
    ----------
    ranks : list of int
    errors : list of float
    corcondia : list of float or None
    title : str
    save_path : str or None
    """
    set_style()

    if corcondia is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    else:
        fig, ax1 = plt.subplots(figsize=(7, 5))

    # Error plot
    ax1.plot(ranks, errors, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel("Rank (R)")
    ax1.set_ylabel("Relative Error")
    ax1.set_title("Reconstruction Error")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    # CORCONDIA plot
    if corcondia is not None:
        ax2.plot(ranks, corcondia, 'rs-', linewidth=2, markersize=8)
        ax2.axhline(y=80, color='gray', linestyle='--', alpha=0.7,
                     label='80% threshold')
        ax2.set_xlabel("Rank (R)")
        ax2.set_ylabel("CORCONDIA (%)")
        ax2.set_title("Core Consistency Diagnostic")
        ax2.set_ylim(-10, 110)
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax2.legend()

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)

    return fig


# =============================================================================
# 4. EEG COMPONENT VISUALIZATION
# =============================================================================

def plot_eeg_components(factors, channel_names, times=None, sfreq=None,
                        weights=None, title="EEG CP Components",
                        save_path=None):
    """
    Visualize CP components in the context of EEG data.

    Input
    -----
    - factors: [A_subjects, A_channels, A_time] from CP decomposition
    - channel_names: list of EEG channel names
    - times: time points (seconds) for temporal factor
    - sfreq: sampling frequency

    Meaning
    -------
    Row 1 (Spatial): Loading of each channel on each component
        → shows which brain regions are activated
    Row 2 (Temporal): Loading over time for each component
        → shows when activation occurs
    Row 3 (Subjects): Loading of each subject on each component
        → shows inter-subject variability

    Interpretation
    --------------
    - Spatial: high loadings on C3/C4 → motor cortex involvement
    - Temporal: peak around 0.5-1s → typical motor imagery response
    - Subjects: uniform loadings → consistent pattern across subjects

    Parameters
    ----------
    factors : list of np.ndarray
        [A_subjects (S×R), A_channels (C×R), A_time (T×R)].
    channel_names : list of str
        Channel names.
    times : np.ndarray or None
        Time points.
    sfreq : float or None
        Sampling frequency.
    weights : np.ndarray or None
    title : str
    save_path : str or None
    """
    set_style()

    R = factors[0].shape[1]
    n_rows = 3  # spatial, temporal, subjects

    fig, axes = plt.subplots(n_rows, R, figsize=(4 * R, 3 * n_rows))
    if R == 1:
        axes = axes.reshape(-1, 1)

    for r in range(R):
        weight_str = f" (λ={weights[r]:.2f})" if weights is not None else ""

        # --- Row 0: Subject loadings ---
        ax = axes[0, r]
        ax.bar(range(factors[0].shape[0]), factors[0][:, r],
               color='steelblue', alpha=0.8)
        ax.set_xlabel("Subject")
        ax.set_ylabel("Loading")
        ax.set_title(f"Component {r+1}{weight_str}\nSubjects")

        # --- Row 1: Channel (spatial) loadings ---
        ax = axes[1, r]
        loadings = factors[1][:, r]
        n_ch = len(loadings)

        # Bar plot với tên kênh
        colors = ['red' if v > 0 else 'blue' for v in loadings]
        bars = ax.barh(range(n_ch), loadings, color=colors, alpha=0.7)
        ax.set_yticks(range(0, n_ch, max(1, n_ch // 10)))
        if channel_names:
            ax.set_yticklabels([channel_names[i]
                                for i in range(0, n_ch, max(1, n_ch // 10))],
                               fontsize=7)
        ax.set_xlabel("Loading")
        ax.set_title(f"Spatial Pattern")
        ax.invert_yaxis()

        # --- Row 2: Temporal loadings ---
        ax = axes[2, r]
        if times is not None:
            ax.plot(times, factors[2][:, r], color='darkgreen', linewidth=1.5)
            ax.set_xlabel("Time (s)")
            ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5, label='Event')
        else:
            ax.plot(factors[2][:, r], color='darkgreen', linewidth=1.5)
            ax.set_xlabel("Time sample")
        ax.set_ylabel("Loading")
        ax.set_title(f"Temporal Pattern")

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)

    return fig, axes


# =============================================================================
# 5. TEMPORAL FACTOR VISUALIZATION
# =============================================================================

def plot_temporal_factors(factors_time, times=None, sfreq=None,
                          title="Temporal Factors",
                          component_labels=None, save_path=None):
    """
    Plot temporal factors as time-series.

    Input
    -----
    - factors_time: temporal factor matrix A^(time) of shape (T, R)
    - times: time array in seconds
    - sfreq: sampling frequency

    Meaning
    -------
    Each component's temporal loading shows the time course of that
    brain pattern.

    Interpretation
    --------------
    - Peaks/troughs indicate when the component is most active
    - Event-related: expect changes around time 0 (stimulus onset)
    - Oscillatory patterns may indicate rhythmic brain activity

    Parameters
    ----------
    factors_time : np.ndarray, shape (T, R)
    times : np.ndarray or None
    sfreq : float or None
    title : str
    component_labels : list of str or None
    save_path : str or None
    """
    set_style()

    R = factors_time.shape[1]

    fig, ax = plt.subplots(figsize=(10, 4))

    if times is None:
        if sfreq:
            times = np.arange(factors_time.shape[0]) / sfreq
        else:
            times = np.arange(factors_time.shape[0])

    colors = plt.cm.tab10(np.linspace(0, 1, R))

    for r in range(R):
        label = component_labels[r] if component_labels else f"Component {r+1}"
        ax.plot(times, factors_time[:, r], color=colors[r],
                linewidth=1.5, label=label)

    ax.set_xlabel("Time (s)" if sfreq else "Time sample")
    ax.set_ylabel("Loading")
    ax.set_title(title)
    ax.legend(loc='best')
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    if sfreq or (times is not None and len(times) > 0 and times[0] < 0):
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5, label='Event onset')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)

    return fig, ax


# =============================================================================
# 6. COMPARISON PLOT (CP-ALS vs CP-OPT)
# =============================================================================

def plot_comparison(results_dict, metric='fit', title="Method Comparison",
                    save_path=None):
    """
    Compare multiple decomposition methods on the same metric.

    Input
    -----
    - results_dict: dictionary {method_name: results_from_evaluate_ranks}
    - metric: 'fit', 'error', or 'corcondia'

    Meaning
    -------
    Side-by-side comparison of different decomposition methods
    across multiple ranks.

    Interpretation
    --------------
    - Higher fit → better approximation
    - Methods converging to same fit → algorithm-independent result
    - Different fits → one method may be stuck in local minimum

    Parameters
    ----------
    results_dict : dict
        {method_name: {'ranks': [...], 'fits': [...], ...}}
    metric : str
        'fit', 'error', or 'corcondia'
    title : str
    save_path : str or None
    """
    set_style()

    fig, ax = plt.subplots(figsize=(8, 5))

    metric_key = {
        'fit': 'fits',
        'error': 'errors',
        'corcondia': 'corcondia'
    }[metric]

    metric_label = {
        'fit': 'Fit Score',
        'error': 'Relative Error',
        'corcondia': 'CORCONDIA (%)'
    }[metric]

    markers = ['o', 's', '^', 'D', 'v', '<', '>']
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_dict)))

    for i, (method_name, results) in enumerate(results_dict.items()):
        ax.plot(results['ranks'], results[metric_key],
                marker=markers[i % len(markers)],
                color=colors[i],
                linewidth=2, markersize=8,
                label=method_name)

    ax.set_xlabel("Rank (R)")
    ax.set_ylabel(metric_label)
    ax.set_title(title)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend()

    if metric == 'corcondia':
        ax.axhline(y=80, color='gray', linestyle='--', alpha=0.7)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)

    return fig, ax


# =============================================================================
# 7. PARAFAC2 VISUALIZATION
# =============================================================================

def plot_parafac2_results(V, weights, projections, channel_names=None,
                           subject_ids=None, title="PARAFAC2 Results",
                           save_path=None):
    """
    Visualize PARAFAC2 decomposition results.

    Input
    -----
    - V: shared loading matrix (channels × R)
    - weights: per-slice weights (K × R)
    - projections: per-slice projections

    Meaning
    -------
    - V (shared): common spatial pattern across all subjects
    - weights: how strongly each subject expresses each component
    - projections: subject-specific patterns (respecting PARAFAC2 constraint)

    Parameters
    ----------
    V : np.ndarray, (J, R)
    weights : np.ndarray, (K, R)
    projections : list of np.ndarray
    channel_names : list of str or None
    subject_ids : list or None
    title : str
    save_path : str or None
    """
    set_style()

    R = V.shape[1]

    fig, axes = plt.subplots(2, R, figsize=(4 * R, 8))
    if R == 1:
        axes = axes.reshape(-1, 1)

    for r in range(R):
        # Row 0: Shared loading V
        ax = axes[0, r]
        loadings = V[:, r]
        n_ch = len(loadings)
        colors = ['red' if v > 0 else 'blue' for v in loadings]
        ax.barh(range(n_ch), loadings, color=colors, alpha=0.7)
        if channel_names:
            ax.set_yticks(range(0, n_ch, max(1, n_ch // 10)))
            ax.set_yticklabels([channel_names[i]
                                for i in range(0, n_ch, max(1, n_ch // 10))],
                               fontsize=7)
        ax.set_xlabel("Loading")
        ax.set_title(f"Component {r+1}\nShared Spatial (V)")
        ax.invert_yaxis()

        # Row 1: Per-subject weights
        ax = axes[1, r]
        x_labels = subject_ids if subject_ids else range(weights.shape[0])
        ax.bar(range(weights.shape[0]), weights[:, r],
               color='steelblue', alpha=0.8)
        ax.set_xlabel("Subject")
        ax.set_ylabel("Weight")
        ax.set_title(f"Subject Weights (S)")

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)

    return fig, axes

# =============================================================================
# 8. PARAFAC2 VS CP COMPARISON
# =============================================================================

def plot_parafac2_vs_cp_spatial(cp_factors, parafac2_V, channel_names=None,
                                title="Spatial Components: CP vs PARAFAC2",
                                save_path=None):
    """
    Compare CP spatial components with PARAFAC2 shared spatial components.
    
    Parameters
    ----------
    cp_factors : list of np.ndarray
        CP factors (assuming mode 1 is spatial/channels).
    parafac2_V : np.ndarray
        PARAFAC2 shared loading matrix V.
    channel_names : list of str or None
        Names of channels.
    title : str
        Plot title.
    save_path : str or None
        Path to save figure.
    """
    set_style()
    
    cp_spatial = cp_factors[1]  # Mode 1 is channels
    R = cp_spatial.shape[1]
    
    fig, axes = plt.subplots(R, 2, figsize=(10, 3 * R))
    if R == 1:
        axes = axes.reshape(1, -1)
        
    for r in range(R):
        # CP Spatial
        ax_cp = axes[r, 0]
        loadings = cp_spatial[:, r]
        n_ch = len(loadings)
        colors = ['red' if v > 0 else 'blue' for v in loadings]
        ax_cp.barh(range(n_ch), loadings, color=colors, alpha=0.7)
        if channel_names:
            ax_cp.set_yticks(range(0, n_ch, max(1, n_ch // 10)))
            ax_cp.set_yticklabels([channel_names[i] 
                                for i in range(0, n_ch, max(1, n_ch // 10))],
                                fontsize=7)
        ax_cp.set_title(f"CP Component {r+1}")
        ax_cp.invert_yaxis()
        
        # PARAFAC2 Spatial
        ax_pf = axes[r, 1]
        loadings = parafac2_V[:, r]
        colors = ['red' if v > 0 else 'blue' for v in loadings]
        ax_pf.barh(range(n_ch), loadings, color=colors, alpha=0.7)
        if channel_names:
            ax_pf.set_yticks(range(0, n_ch, max(1, n_ch // 10)))
            ax_pf.set_yticklabels([channel_names[i] 
                                for i in range(0, n_ch, max(1, n_ch // 10))],
                                fontsize=7)
        ax_pf.set_title(f"PARAFAC2 Component {r+1}")
        ax_pf.invert_yaxis()
        
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        
    return fig, axes
