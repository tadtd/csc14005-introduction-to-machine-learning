"""
Data Loading & Tensor Construction for EEG Experiments
=======================================================

This module provides functions to:
1. Download the PhysioNet EEG Motor Movement/Imagery Dataset
2. Load and preprocess raw EEG data
3. Construct tensors for CP decomposition

Dataset: PhysioNet EEG Motor Movement/Imagery Dataset
------------------------------------------------------
- Source: https://physionet.org/content/eegmmidb/1.0.0/
- 109 subjects, 64 EEG channels, ~160 Hz sampling rate
- 14 runs per subject:
    * Run 1: Baseline, eyes open
    * Run 2: Baseline, eyes closed
    * Runs 3, 7, 11: Motor execution (left fist vs right fist)
    * Runs 4, 8, 12: Motor imagery (left fist vs right fist)
    * Runs 5, 9, 13: Motor execution (both fists vs both feet)
    * Runs 6, 10, 14: Motor imagery (both fists vs both feet)

Tensor Construction
-------------------
We construct a 3-mode tensor:

    T ∈ ℝ^{Subjects × Channels × Time}

Mode 1 (Subjects): Each subject represents an observation.
    - Inter-subject variability captures individual differences
    - CP components along this mode reveal subject groupings

Mode 2 (Channels): 64 EEG electrodes in the 10-20 system.
    - Spatial distribution of brain activity
    - CP components reveal spatial patterns (e.g., motor cortex activation)

Mode 3 (Time): Temporal samples within each epoch.
    - Dynamics of brain activity over time
    - CP components reveal temporal patterns (e.g., event-related responses)

Preprocessing Pipeline
----------------------
1. Load raw EDF files via MNE
2. Standardize channel names (MNE convention)
3. Set montage (standard 10-20 system)
4. Bandpass filter (typically 8-30 Hz for motor imagery):
   - 8 Hz: lower bound of mu rhythm
   - 30 Hz: upper bound of beta rhythm
   - These bands are most relevant for motor imagery
5. Extract epochs around events
6. Average across trials per subject
7. Stack subjects into a 3D tensor

Dependencies
------------
- mne: EEG data handling
- numpy: Array operations
"""

import numpy as np
import warnings
import os


def check_mne_available():
    """Check if MNE-Python is installed and available."""
    try:
        import mne
        return True
    except ImportError:
        return False


def download_eeg_data(subjects=None, runs=None, path=None, verbose=False):
    """
    Download PhysioNet EEG Motor Movement/Imagery data via MNE.

    Tải dữ liệu EEG từ PhysioNet. MNE sẽ tự động cache dữ liệu
    vào thư mục ~/mne_data (hoặc path nếu được chỉ định).

    Lần đầu tải sẽ mất vài phút (tùy kết nối internet).
    Các lần sau sẽ sử dụng cache.

    Parameters
    ----------
    subjects : list of int or None
        Subject IDs (1-109). Default: [1, 2, ..., 20] (20 subjects).
    runs : list of int or None
        Run numbers. Default: [4, 8, 12] (motor imagery, left vs right).
        Run meanings:
        - 4, 8, 12: Motor imagery (left fist vs right fist)
        - 6, 10, 14: Motor imagery (both fists vs both feet)
    path : str or None
        Download directory. Default: MNE's default data path.
    verbose : bool
        Print download progress.

    Returns
    -------
    file_paths : dict
        Dictionary mapping subject_id -> list of local file paths.

    Examples
    --------
    >>> paths = download_eeg_data(subjects=[1, 2], runs=[4, 8, 12])
    >>> print(paths[1])  # Paths for subject 1
    """
    import mne
    from mne.datasets import eegbci

    if subjects is None:
        subjects = list(range(1, 21))  # 20 subjects mặc định
    if runs is None:
        runs = [4, 8, 12]  # Motor imagery: left vs right hand

    file_paths = {}
    for subj in subjects:
        if verbose:
            print(f"Downloading subject {subj}...")
        try:
            paths = eegbci.load_data(subj, runs, path=path,
                                     update_path=False, verbose=verbose)
            file_paths[subj] = paths
        except Exception as e:
            warnings.warn(f"Failed to download subject {subj}: {e}")
            file_paths[subj] = []

    if verbose:
        n_success = sum(1 for v in file_paths.values() if len(v) > 0)
        print(f"\nDownloaded: {n_success}/{len(subjects)} subjects")

    return file_paths


def load_eeg_raw(subjects=None, runs=None, path=None, verbose=False):
    """
    Load raw EEG data for specified subjects and runs.

    Tải và ghép nối dữ liệu EEG thô cho từng subject.
    Trả về danh sách các Raw objects.

    Preprocessing tối thiểu:
    - Standardize channel names (theo chuẩn MNE)
    - Set montage (vị trí điện cực theo 10-20 system)
    - Set EEG reference

    Parameters
    ----------
    subjects : list of int or None
        Subject IDs.
    runs : list of int or None
        Run numbers.
    path : str or None
        Data directory.
    verbose : bool
        Print progress.

    Returns
    -------
    raws : dict
        Dictionary mapping subject_id -> mne.io.Raw object.
    """
    import mne
    from mne.datasets import eegbci
    from mne.io import read_raw_edf, concatenate_raws

    if subjects is None:
        subjects = list(range(1, 21))
    if runs is None:
        runs = [4, 8, 12]

    raws = {}
    for subj in subjects:
        try:
            # Tải dữ liệu (hoặc sử dụng cache)
            fnames = eegbci.load_data(subj, runs, path=path,
                                      update_path=False, verbose=verbose)

            # Đọc và ghép nối các run
            raw_list = [read_raw_edf(f, preload=True, verbose=verbose)
                        for f in fnames]
            raw = concatenate_raws(raw_list)

            # Chuẩn hóa tên kênh (MNE convention)
            eegbci.standardize(raw)

            # Set montage (vị trí 3D của các điện cực)
            montage = mne.channels.make_standard_montage('standard_1005')
            raw.set_montage(montage, on_missing='warn')

            # Set reference
            raw.set_eeg_reference('average', projection=True, verbose=verbose)

            raws[subj] = raw

            if verbose:
                print(f"  Subject {subj}: {raw.info['nchan']} channels, "
                      f"{raw.n_times} samples, {raw.info['sfreq']} Hz")

        except Exception as e:
            warnings.warn(f"Failed to load subject {subj}: {e}")

    return raws


def preprocess_eeg(raw, l_freq=8.0, h_freq=30.0, tmin=-0.5, tmax=2.5,
                   event_id=None, verbose=False):
    """
    Preprocess raw EEG data for tensor construction.

    Quy trình tiền xử lý:

    1. Bandpass filter (8-30 Hz):
       - Loại bỏ nhiễu tần số thấp (< 8 Hz): drift, chuyển động
       - Loại bỏ nhiễu tần số cao (> 30 Hz): cơ, điện lưới
       - Giữ lại mu rhythm (8-12 Hz) và beta rhythm (13-30 Hz)
       - Đây là hai dải tần quan trọng nhất cho motor imagery

    2. Extract epochs:
       - Cắt dữ liệu theo các sự kiện (events) trong recording
       - tmin: thời gian trước event (baseline)
       - tmax: thời gian sau event

    3. Baseline correction:
       - Trừ trung bình của khoảng baseline khỏi toàn bộ epoch

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data (already loaded with preload=True).
    l_freq : float, default=8.0
        Low cutoff frequency (Hz) for bandpass filter.
    h_freq : float, default=30.0
        High cutoff frequency (Hz) for bandpass filter.
    tmin : float, default=-0.5
        Start time of epoch relative to event (seconds).
    tmax : float, default=2.5
        End time of epoch relative to event (seconds).
    event_id : dict or None
        Event labels. Default: {'T1': 1, 'T2': 2} for motor imagery.
    verbose : bool
        Print progress.

    Returns
    -------
    epochs : mne.Epochs
        Preprocessed epochs.
    """
    import mne

    # Bước 1: Bandpass filter
    raw_filtered = raw.copy().filter(l_freq, h_freq, verbose=verbose)

    # Apply projection (average reference)
    raw_filtered.apply_proj()

    # Bước 2: Tìm events
    events, event_dict = mne.events_from_annotations(raw_filtered, verbose=verbose)

    # Xác định event_id
    if event_id is None:
        # Mặc định cho motor imagery: T1 (left) và T2 (right)
        # Trong PhysioNet, annotations là 'T0' (rest), 'T1', 'T2'
        # Chọn T1 và T2
        event_id = {}
        for key, val in event_dict.items():
            if 'T1' in key:
                event_id['left'] = val
            elif 'T2' in key:
                event_id['right'] = val
            elif 'T0' in key:
                event_id['rest'] = val

    if verbose:
        print(f"  Events found: {event_dict}")
        print(f"  Using event_id: {event_id}")

    # Bước 3: Extract epochs
    picks = mne.pick_types(raw_filtered.info, eeg=True, exclude='bads')

    epochs = mne.Epochs(
        raw_filtered, events, event_id,
        tmin=tmin, tmax=tmax,
        picks=picks,
        baseline=(tmin, 0),  # Baseline correction: trừ trung bình [tmin, 0]
        preload=True,
        verbose=verbose
    )

    # Drop bad epochs (nếu có)
    epochs.drop_bad(verbose=verbose)

    if verbose:
        print(f"  Epochs: {len(epochs)} total, shape={epochs.get_data().shape}")

    return epochs


def build_tensor(raws, l_freq=8.0, h_freq=30.0, tmin=-0.5, tmax=2.5,
                 average_trials=True, verbose=False):
    """
    Build a 3D tensor from EEG data: Subjects × Channels × Time.

    Xây dựng tensor 3 mode từ dữ liệu EEG:

    Mode 1 (Subjects): Mỗi subject là một "observation"
        - Phản ánh sự khác biệt giữa các cá nhân
        - CP component dọc theo mode này → nhóm subjects tương tự

    Mode 2 (Channels): 64 kênh EEG (hệ thống 10-20)
        - Phân bố không gian của hoạt động não
        - CP component → mẫu không gian (vd: vùng motor cortex)

    Mode 3 (Time): Thời gian trong mỗi epoch
        - Động học của hoạt động não theo thời gian
        - CP component → mẫu thời gian (vd: ERD/ERS)

    Process
    -------
    Với mỗi subject:
    1. Preprocess (filter, epoch)
    2. Average across trials → ma trận (Channels × Time)
    3. Stack tất cả subjects → tensor (Subjects × Channels × Time)

    Parameters
    ----------
    raws : dict
        Dictionary mapping subject_id -> mne.io.Raw.
    l_freq : float, default=8.0
        Low cutoff for bandpass filter.
    h_freq : float, default=30.0
        High cutoff for bandpass filter.
    tmin : float, default=-0.5
        Epoch start time.
    tmax : float, default=2.5
        Epoch end time.
    average_trials : bool, default=True
        If True, average across trials for each subject.
        If False, return epochs_list (for PARAFAC2).
    verbose : bool
        Print progress.

    Returns
    -------
    tensor : np.ndarray
        If average_trials=True: shape (n_subjects, n_channels, n_times).
    subject_ids : list
        Subject IDs in the order they appear in the tensor.
    info : dict
        Metadata: channel names, sampling frequency, times, etc.
    """
    import mne

    subject_data = []
    subject_ids = []
    channel_names = None
    sfreq = None
    times = None

    sorted_subjects = sorted(raws.keys())

    for subj_id in sorted_subjects:
        raw = raws[subj_id]

        try:
            epochs = preprocess_eeg(
                raw, l_freq=l_freq, h_freq=h_freq,
                tmin=tmin, tmax=tmax, verbose=verbose
            )

            if len(epochs) == 0:
                if verbose:
                    print(f"  Subject {subj_id}: no valid epochs, skipping")
                continue

            # Lấy data: (n_epochs, n_channels, n_times)
            data = epochs.get_data()

            if average_trials:
                # Trung bình qua các trials → (n_channels, n_times)
                avg_data = np.mean(data, axis=0)
                subject_data.append(avg_data)
            else:
                subject_data.append(data)

            subject_ids.append(subj_id)

            # Lưu metadata (chỉ lần đầu)
            if channel_names is None:
                channel_names = epochs.ch_names
                sfreq = epochs.info['sfreq']
                times = epochs.times

            if verbose:
                print(f"  Subject {subj_id}: {data.shape[0]} trials, "
                      f"{data.shape[1]} channels, {data.shape[2]} time points")

        except Exception as e:
            warnings.warn(f"Failed to process subject {subj_id}: {e}")

    if len(subject_data) == 0:
        raise RuntimeError("No valid subjects could be processed")

    # Stack thành tensor
    if average_trials:
        tensor = np.stack(subject_data, axis=0)  # (Subjects, Channels, Time)
    else:
        tensor = subject_data  # List of arrays (for PARAFAC2)

    info = {
        'channel_names': channel_names,
        'sfreq': sfreq,
        'times': times,
        'n_subjects': len(subject_ids),
        'n_channels': len(channel_names) if channel_names else 0,
        'n_times': len(times) if times is not None else 0,
    }

    if verbose:
        if average_trials:
            print(f"\nTensor shape: {tensor.shape}")
            print(f"  Mode 0 (Subjects): {tensor.shape[0]}")
            print(f"  Mode 1 (Channels): {tensor.shape[1]}")
            print(f"  Mode 2 (Time):     {tensor.shape[2]}")
        else:
            print(f"\nBuilt {len(subject_data)} slices for PARAFAC2")

    return tensor, subject_ids, info


def build_parafac2_slices(raws, l_freq=8.0, h_freq=30.0, tmin=-0.5, tmax=2.5,
                          verbose=False):
    """
    Build PARAFAC2-compatible slices from EEG data.

    Cho PARAFAC2, mỗi subject là một "slice" X_k ∈ ℝ^{n_trials_k × n_channels}.
    Số trials có thể khác nhau giữa các subjects (I_k varies).

    Cấu trúc:
    - Mỗi slice X_k: (n_trials_k, n_channels)
    - Chiều chung: n_channels (cùng 64 kênh cho mọi subject)
    - Chiều khác nhau: n_trials_k (số trial hợp lệ của mỗi subject)

    Parameters
    ----------
    raws : dict
        Dictionary mapping subject_id -> mne.io.Raw.
    l_freq, h_freq : float
        Bandpass filter frequencies.
    tmin, tmax : float
        Epoch time window.
    verbose : bool

    Returns
    -------
    slices : list of np.ndarray
        K matrices, each (n_trials_k, n_channels).
    subject_ids : list
    info : dict
    """
    import mne

    slices = []
    subject_ids = []
    channel_names = None
    sfreq = None

    sorted_subjects = sorted(raws.keys())

    for subj_id in sorted_subjects:
        raw = raws[subj_id]

        try:
            epochs = preprocess_eeg(
                raw, l_freq=l_freq, h_freq=h_freq,
                tmin=tmin, tmax=tmax, verbose=verbose
            )

            if len(epochs) == 0:
                continue

            # Lấy data: (n_epochs, n_channels, n_times)
            data = epochs.get_data()

            # Cho PARAFAC2: average over time → (n_epochs, n_channels)
            # Hoặc: sử dụng power spectral density
            # Ở đây: lấy trung bình bình phương (power) theo thời gian
            slice_data = np.mean(data ** 2, axis=2)  # (n_trials, n_channels)

            slices.append(slice_data)
            subject_ids.append(subj_id)

            if channel_names is None:
                channel_names = epochs.ch_names
                sfreq = epochs.info['sfreq']

            if verbose:
                print(f"  Subject {subj_id}: slice shape {slice_data.shape}")

        except Exception as e:
            warnings.warn(f"Failed to process subject {subj_id}: {e}")

    info = {
        'channel_names': channel_names,
        'sfreq': sfreq,
        'n_subjects': len(subject_ids),
        'n_channels': len(channel_names) if channel_names else 0,
    }

    if verbose:
        sizes = [s.shape[0] for s in slices]
        print(f"\nPARAFAC2: {len(slices)} slices")
        print(f"  Trial counts: min={min(sizes)}, max={max(sizes)}, "
              f"mean={np.mean(sizes):.1f}")
        print(f"  Common dimension (channels): {slices[0].shape[1]}")

    return slices, subject_ids, info


def get_channel_positions(info_dict, raw=None):
    """
    Get 2D channel positions for topographic plotting.

    Parameters
    ----------
    info_dict : dict
        Info dictionary from build_tensor.
    raw : mne.io.Raw or None
        Raw object for getting positions. If None, uses standard 10-20.

    Returns
    -------
    positions : np.ndarray, shape (n_channels, 2)
        2D projected positions of channels.
    """
    import mne

    if raw is not None:
        # Lấy vị trí từ raw object
        pos = np.array([raw.info['chs'][i]['loc'][:3]
                        for i in range(len(raw.ch_names))])
        # Project 3D → 2D (simple projection: x, y)
        positions = pos[:, :2]
    else:
        # Sử dụng montage chuẩn
        montage = mne.channels.make_standard_montage('standard_1005')
        ch_names = info_dict['channel_names']

        positions = []
        for ch in ch_names:
            if ch in montage.ch_names:
                idx = montage.ch_names.index(ch)
                pos = montage.get_positions()['ch_pos'][ch]
                positions.append(pos[:2])
            else:
                positions.append([0.0, 0.0])

        positions = np.array(positions)

    return positions
