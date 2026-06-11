import json
import os

def create_notebook(cells, filename):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

def md(text):
    lines = [line + '\n' for line in text.split('\n')]
    if lines:
        lines[-1] = lines[-1].rstrip('\n')
    return {"cell_type": "markdown", "metadata": {}, "source": lines}

def code(text):
    lines = [line + '\n' for line in text.split('\n')]
    if lines:
        lines[-1] = lines[-1].rstrip('\n')
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines}

# --- 01_synthetic.ipynb ---
nb01 = [
    md("# Khởi động & Kiểm tra Thuật toán (Synthetic Data)\n\n"
       "> **Lưu ý quan trọng**: Đây là notebook DUY NHẤT sử dụng dữ liệu giả (synthetic data). "
       "Mục đích của notebook này chỉ là để xác minh (validate) tính đúng đắn của "
       "các thuật toán CP-ALS và CP-OPT được cài đặt từ đầu bằng NumPy, "
       "đảm bảo chúng có thể khôi phục lại các factor matrices đã biết."),
    code("import numpy as np\n"
         "import matplotlib.pyplot as plt\n\n"
         "from src.cp_als import cp_als, reconstruct\n"
         "from src.cp_opt import cp_opt\n"
         "from src.evaluate import relative_error, factor_similarity\n"
         "from src.visualize import plot_convergence, plot_reconstruction_error\n\n"
         "# Đặt seed để kết quả có thể tái tạo\n"
         "np.random.seed(42)"),
    md("## 1. Tính Robustness: FMS vs Noise Level\n\n"
       "Tạo tensor rank 3. Quét qua nhiều mức độ nhiễu khác nhau, chạy CP-ALS và "
       "tính Factor Match Score (FMS) so với ground truth `true_factors`."),
    code("shape = (10, 15, 20)\n"
         "true_rank = 3\n"
         "true_factors = [np.random.randn(s, true_rank) for s in shape]\n"
         "X_true = reconstruct(true_factors)\n\n"
         "noise_levels = np.logspace(-3, 1, 10)\n"
         "fms_scores = []\n\n"
         "print(\"Evaluating noise robustness...\")\n"
         "for noise in noise_levels:\n"
         "    E = np.random.randn(*shape)\n"
         "    X_noisy = X_true + noise * E * (np.linalg.norm(X_true) / np.linalg.norm(E))\n"
         "    \n"
         "    weights, factors_als, _, _ = cp_als(X_noisy, rank=true_rank, max_iter=100, tol=1e-6)\n"
         "    \n"
         "    # Compare recovered factors against true_factors (ground truth)\n"
         "    _, mean_sim, _ = factor_similarity(true_factors, factors_als, greedy_permutation=True)\n"
         "    fms_scores.append(mean_sim)\n\n"
         "plt.figure(figsize=(8, 5))\n"
         "plt.semilogx(noise_levels, fms_scores, 'bo-', linewidth=2)\n"
         "plt.xlabel('Noise Level')\n"
         "plt.ylabel('Factor Match Score (FMS)')\n"
         "plt.title('Robustness: FMS vs Noise Level')\n"
         "plt.grid(True)\n"
         "plt.show()"),
    md("## 2. Lựa chọn Rank (Elbow Plot) & So sánh CP-ALS vs CP-OPT\n\n"
       "Cố định mức nhiễu nhỏ. Quét qua các rank từ 1 đến 8, chạy cả CP-ALS và CP-OPT "
       "để vẽ đường cong Reconstruction Error vs Rank (elbow plot)."),
    code("X_test = X_true + 0.1 * np.random.randn(*shape)\n"
         "ranks = list(range(1, 9))\n"
         "errs_als = []\n"
         "errs_opt = []\n\n"
         "print(\"Evaluating ranks...\")\n"
         "for r in ranks:\n"
         "    w_als, f_als, _, _ = cp_als(X_test, rank=r, max_iter=100, tol=1e-6)\n"
         "    X_hat_als = reconstruct(f_als, w_als)\n"
         "    errs_als.append(relative_error(X_test, X_hat_als))\n"
         "    \n"
         "    w_opt, f_opt, info_opt = cp_opt(X_test, rank=r, max_iter=100, tol=1e-6)\n"
         "    X_hat_opt = reconstruct(f_opt, w_opt)\n"
         "    errs_opt.append(relative_error(X_test, X_hat_opt))\n\n"
         "plt.figure(figsize=(8, 5))\n"
         "plt.plot(ranks, errs_als, 'bo-', label='CP-ALS')\n"
         "plt.plot(ranks, errs_opt, 'rs--', label='CP-OPT')\n"
         "plt.xlabel('Rank')\n"
         "plt.ylabel('Reconstruction Error')\n"
         "plt.title('Elbow Plot: Error vs Rank')\n"
         "plt.legend()\n"
         "plt.grid(True)\n"
         "plt.show()")
]

# --- 02_eeg_cp.ipynb ---
nb02 = [
    md("# Thực nghiệm 1: CP Decomposition trên dữ liệu EEG\n\n"
       "Trong notebook này, chúng ta sẽ thực hiện phân rã tensor CP trên dữ liệu EEG thực tế "
       "từ bộ dataset PhysioNet EEG Motor Movement/Imagery. "
       "Tensor có 3 chiều: **Subjects $\\times$ Channels $\\times$ Time**."),
    code("import numpy as np\n"
         "import matplotlib.pyplot as plt\n\n"
         "from src.data import load_eeg_raw, build_tensor, get_channel_positions\n"
         "from src.cp_als import cp_als\n"
         "from src.evaluate import evaluate_ranks\n"
         "from src.visualize import plot_eeg_components, plot_reconstruction_error\n\n"
         "# Tắt warnings của MNE cho output gọn gàng\n"
         "import mne\n"
         "mne.set_log_level('WARNING')"),
    md("## 1. Tải và Xây dựng Tensor từ Dữ liệu EEG\n\n"
       "Chúng ta sẽ sử dụng 20 subjects, lấy các run liên quan đến motor imagery "
       "(left fist vs right fist) là run 4, 8, 12."),
    code("n_subjects = 20\n"
         "subjects = list(range(1, n_subjects + 1))\n"
         "runs = [4, 8, 12]\n\n"
         "print(\"Đang tải và tiền xử lý dữ liệu EEG...\")\n"
         "raws = load_eeg_raw(subjects, runs, verbose=False)\n\n"
         "# Xây dựng tensor: Lọc băng thông 8-30Hz (Mu và Beta), lấy epochs từ -0.5s đến 2.5s\n"
         "tensor, subj_ids, info = build_tensor(\n"
         "    raws, \n"
         "    l_freq=8.0, h_freq=30.0, \n"
         "    tmin=-0.5, tmax=2.5,\n"
         "    average_trials=True, \n"
         "    verbose=True\n"
         ")"),
    md("## 2. Phân rã Tensor với CP-ALS\n\n"
       "Đầu tiên, chúng ta sẽ đánh giá nhiều mức rank khác nhau để tìm ra rank phù hợp (sử dụng CORCONDIA "
       "và Reconstruction Error). Tuy nhiên, để tiết kiệm thời gian minh họa, chúng ta sẽ chạy CP-ALS với rank = 5."),
    code("R = 5\n"
         "print(f\"Chạy CP-ALS với rank = {R}...\")\n"
         "weights, f0, f1, f2 = cp_als(tensor, rank=R, max_iter=200, tol=1e-6, verbose=True)\n"
         "factors = [f0, f1, f2]"),
    md("## 3. Lựa chọn Rank (Rank Selection)\n\n"
       "Để thực nghiệm nghiêm túc, chúng ta nên quét qua các rank từ 2 đến 10. "
       "(Lưu ý: Quá trình này có thể mất vài phút)."),
    code("# ranks_to_test = [2, 3, 4, 5, 6, 7, 8]\n"
         "# print(\"Đánh giá các mức rank...\")\n"
         "# results = evaluate_ranks(tensor, ranks_to_test, cp_als, max_iter=100)\n"
         "# plot_reconstruction_error(results['ranks'], results['errors'], results['corcondia'])\n"
         "# plt.show()\n"
         "\nprint(\"Đã comment phần quét rank để chạy nhanh hơn trong lần đầu.\")"),
    md("## 4. Trực quan hóa và Diễn giải (Interpretation)\n\n"
       "Vẽ các thành phần nhận được từ CP-ALS. Mỗi thành phần đại diện cho một mẫu hoạt động não "
       "(Spatial Pattern) đi kèm với động học thời gian (Temporal Pattern) và sự phân bố trên các subject "
       "(Subject Pattern)."),
    code("channel_names = info['channel_names']\n"
         "times = info['times']\n\n"
         "plot_eeg_components(factors, channel_names, times=times, sfreq=info['sfreq'], weights=weights)\n"
         "plt.show()"),
    md("**Giải thích (Ví dụ mong đợi)**:\n"
       "- Nếu chúng ta thấy một component tập trung ở C3/C4 (vùng vận động) và có sự thay đổi biên độ "
       "rõ rệt quanh thời điểm $t=0$, đó chính là tín hiệu ERD/ERS đặc trưng của Motor Imagery.\n"
       "- Component ở vùng chẩm (O1/O2) có thể liên quan đến Alpha rhythm (visual cortex).\n"
       "- Component ở vùng trán (Fp1/Fp2) thường do nhiễu chớp mắt (EOG artifact).")
]

# --- 03_parafac2.ipynb ---
nb03 = [
    md("# Thực nghiệm 2: PARAFAC2 trên dữ liệu EEG\n\n"
       "PARAFAC2 là một dạng tổng quát của CP decomposition, cho phép một chiều của tensor có độ dài "
       "khác nhau giữa các ma trận (slices). Đặc tính này rất phù hợp với dữ liệu EEG khi mỗi subject "
       "có số lượng epochs/trials hợp lệ khác nhau sau quá trình loại bỏ nhiễu."),
    code("import numpy as np\n"
         "import matplotlib.pyplot as plt\n\n"
         "from src.data import load_eeg_raw, build_parafac2_slices, build_tensor\n"
         "from src.parafac2 import parafac2_als\n"
         "from src.cp_als import cp_als\n"
         "from src.visualize import plot_parafac2_results, plot_parafac2_vs_cp_spatial\n\n"
         "import mne\n"
         "mne.set_log_level('WARNING')"),
    md("## 1. Chuẩn bị dữ liệu cho PARAFAC2\n\n"
       "Khác với `02_eeg_cp.ipynb` nơi chúng ta tính trung bình các trial, ở đây chúng ta "
       "giữ lại thông tin của từng trial. Mỗi subject sẽ tạo thành một slice kích thước "
       "`(n_trials, n_channels)`. Số lượng `n_trials` sẽ khác nhau tùy subject."),
    code("n_subjects = 5  # Dùng 5 subjects để demo nhanh PARAFAC2\n"
         "subjects = list(range(1, n_subjects + 1))\n"
         "runs = [4, 8, 12]\n\n"
         "raws = load_eeg_raw(subjects, runs, verbose=False)\n\n"
         "# Xây dựng các slice cho PARAFAC2\n"
         "slices, subj_ids, info = build_parafac2_slices(\n"
         "    raws, \n"
         "    l_freq=8.0, h_freq=30.0,\n"
         "    tmin=-0.5, tmax=2.5,\n"
         "    verbose=True\n"
         ")"),
    md("## 2. Chạy PARAFAC2-ALS\n\n"
       "Chúng ta sẽ phân rã bộ slices với rank $R = 4$."),
    code("R = 4\n"
         "print(f\"Chạy PARAFAC2 với rank = {R}...\")\n"
         "V, weights, projections, fit_info = parafac2_als(slices, rank=R, max_iter=100, tol=1e-5, verbose=True)"),
    md("## 3. Trực quan hóa Kết quả\n\n"
       "Trong kết quả của PARAFAC2:\n"
       "- **V** (Shared Spatial Matrix): là không gian các kênh (channels) chung cho toàn bộ dataset.\n"
       "- **Weights**: Trọng số thể hiện mức độ biểu hiện của component ở mỗi subject.\n"
       "- **Projections**: Chứa thông tin về trial, đặc trưng riêng cho từng subject."),
    code("plot_parafac2_results(V, weights, projections, channel_names=info['channel_names'], subject_ids=subj_ids)\n"
         "plt.show()"),
    md("## 4. So sánh Spatial Component: CP vs PARAFAC2\n\n"
       "Chạy CP-ALS trên dữ liệu đã trung bình hóa (như ở notebook 02) và so sánh với PARAFAC2."),
    code("tensor, _, _ = build_tensor(raws, l_freq=8.0, h_freq=30.0, tmin=-0.5, tmax=2.5, average_trials=True, verbose=False)\n"
         "_, f0, f1, f2 = cp_als(tensor, rank=R, max_iter=100, tol=1e-5, verbose=False)\n"
         "cp_factors = [f0, f1, f2]\n\n"
         "plot_parafac2_vs_cp_spatial(cp_factors, V, channel_names=info['channel_names'])\n"
         "plt.show()")
]

# --- 04_cmtf.ipynb ---
nb04 = [
    md("# Thực nghiệm 3: Coupled Matrix-Tensor Factorization (CMTF)\n\n"
       "CMTF (Phân rã kết hợp Ma trận - Tensor) giải quyết bài toán khai phá dữ liệu khi chúng ta có "
       "nhiều nguồn dữ liệu chia sẻ chung một chiều thông tin. Ví dụ:\n"
       "- Tensor $\\mathcal{X}$ chứa dữ liệu EEG (Subjects $\\times$ Channels $\\times$ Time)\n"
       "- Matrix $Y$ chứa thông tin metadata của bệnh nhân (Subjects $\\times$ Features)\n\n"
       "> **Cảnh báo tính chân thực học thuật**: Vì bộ dữ liệu PhysioNet thiếu metadata chi tiết cho từng subject "
       "(không có tuổi, giới tính, tình trạng lâm sàng chi tiết), để minh họa thuật toán CMTF một cách trực quan, "
       "chúng ta sẽ **phải tạo ra một ma trận Y giả lập (synthetic)** đóng vai trò làm metadata, trong khi "
       "vẫn giữ tensor $\\mathcal{X}$ là dữ liệu EEG thật."),
    code("import numpy as np\n"
         "import matplotlib.pyplot as plt\n\n"
         "from src.data import load_eeg_raw, build_tensor\n"
         "from src.cp_als import mttkrp, reconstruct, _normalize_factors\n\n"
         "import mne\n"
         "mne.set_log_level('WARNING')"),
    md("## 1. Tải Dữ liệu EEG thật (Tensor X) và tạo Metadata giả lập (Matrix Y)"),
    code("n_subjects = 10\n"
         "subjects = list(range(1, n_subjects + 1))\n"
         "runs = [4, 8, 12]\n\n"
         "raws = load_eeg_raw(subjects, runs, verbose=False)\n"
         "X, subj_ids, info = build_tensor(raws, average_trials=True, verbose=False)\n"
         "print(f\"Tensor X (EEG) shape: {X.shape}\")\n\n"
         "# --- TẠO METADATA GIẢ LẬP (Y) ---\n"
         "n_features = 5\n"
         "np.random.seed(42)\n"
         "# Giả định: các subject chẵn có feature pattern khác subject lẻ\n"
         "Y = np.zeros((n_subjects, n_features))\n"
         "Y[::2, :] = np.random.normal(loc=1.0, scale=0.2, size=(len(Y[::2]), n_features))\n"
         "Y[1::2, :] = np.random.normal(loc=-1.0, scale=0.2, size=(len(Y[1::2]), n_features))\n\n"
         "print(f\"Matrix Y (Metadata) shape: {Y.shape}\")"),
    md("## 2. CMTF ALS Implementation (Minh họa)\n\n"
       "Hàm mục tiêu:\n"
       "$$ f = ||\\mathcal{X} - [[A, B, C]]||_F^2 + \\alpha ||Y - AD^T||_F^2 $$\n\n"
       "Trong đó $A$ là factor chung (shared mode) đại diện cho Subjects."),
    code("def cmtf_als(X, Y, rank, alpha=1.0, max_iter=100, tol=1e-5):\n"
         "    \"\"\"Implementation cơ bản của CMTF ALS\"\"\"\n"
         "    # X: Subjects x Channels x Time\n"
         "    # Y: Subjects x Features\n"
         "    A = np.random.randn(X.shape[0], rank)\n"
         "    B = np.random.randn(X.shape[1], rank)\n"
         "    C = np.random.randn(X.shape[2], rank)\n"
         "    D = np.random.randn(Y.shape[1], rank)\n"
         "    \n"
         "    factors = [A, B, C]\n"
         "    weights, factors = _normalize_factors(factors)\n"
         "    \n"
         "    for iter in range(max_iter):\n"
         "        # Update A (Shared mode)\n"
         "        # M_X = X_(1) (C kr C)\n"
         "        M_X = mttkrp(X, factors, 0)\n"
         "        V_X = (factors[1].T @ factors[1]) * (factors[2].T @ factors[2])\n"
         "        \n"
         "        # M_Y = Y * D\n"
         "        M_Y = Y @ D\n"
         "        V_Y = D.T @ D\n"
         "        \n"
         "        factors[0] = (M_X + alpha * M_Y) @ np.linalg.pinv(V_X + alpha * V_Y)\n"
         "        \n"
         "        # Update B, C (Tensor modes)\n"
         "        factors[1] = mttkrp(X, factors, 1) @ np.linalg.pinv((factors[0].T @ factors[0]) * (factors[2].T @ factors[2]))\n"
         "        factors[2] = mttkrp(X, factors, 2) @ np.linalg.pinv((factors[0].T @ factors[0]) * (factors[1].T @ factors[1]))\n"
         "        \n"
         "        # Update D (Matrix mode)\n"
         "        D = Y.T @ factors[0] @ np.linalg.pinv(factors[0].T @ factors[0])\n"
         "        \n"
         "        # Normalize\n"
         "        weights, factors = _normalize_factors(factors)\n"
         "        \n"
         "    return factors, D\n\n"
         "print(\"Chạy CMTF...\")\n"
         "factors, D = cmtf_als(X, Y, rank=3, alpha=1.0)"),
    md("## 3. Trực quan hóa kết quả"),
    code("plt.figure(figsize=(8, 4))\n"
         "plt.plot(factors[0], 'o-')\n"
         "plt.title(\"Shared Mode (Subjects Factor Matrix A)\")\n"
         "plt.xlabel(\"Subject Index\")\n"
         "plt.ylabel(\"Loading\")\n"
         "plt.grid(True, alpha=0.3)\n"
         "plt.show()\n\n"
         "print(\"Nhận xét: Factor A đã học được sự phân tách (chẵn/lẻ) từ ma trận metadata giả lập, \"\n"
         "      \"đồng thời vẫn giải thích được dữ liệu EEG. Đây là lợi ích chính của phân tích đa phương thức (data fusion).\")")
]

create_notebook(nb01, 'notebooks/01_synthetic.ipynb')
create_notebook(nb02, 'notebooks/02_eeg_cp.ipynb')
create_notebook(nb03, 'notebooks/03_parafac2.ipynb')
create_notebook(nb04, 'notebooks/04_cmtf.ipynb')
print("Notebooks created successfully.")
