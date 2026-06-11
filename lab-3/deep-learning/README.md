# Demo Deep Learning: LoRA Rank Sweep

Thư mục này chứa phần thực nghiệm deep-learning cho tutorial NeurIPS 2025
`Foundations of Tensor Computations for AI`.

Demo so sánh fine-tuning thông thường với Low-Rank Adaptation (LoRA) trên một
bài toán phân loại văn bản nhỏ. Mục tiêu là minh họa thông điệp chính của phần
deep learning trong tutorial: phân rã low-rank có thể giảm mạnh số tham số cần
huấn luyện trong khi vẫn giữ được hiệu năng hữu ích.

## Nội dung đã hiện thực

- Baseline frozen encoder.
- Baseline full fine-tuning.
- LoRA rank sweep với các rank 2, 4 và 8.
- Ghi log các metric: loss, accuracy, số tham số trainable, thời gian chạy,
  thiết bị và seed.
- Sinh biểu đồ từ các file metrics đã lưu.
- Fallback toy dataset nếu không tải được dataset từ xa.

## Cài đặt môi trường

Chạy từ thư mục này:

```bash
uv sync
```

## Chạy thí nghiệm

Nếu muốn kiểm tra nhanh trước, có thể giảm `train_subset` và `eval_subset` trong
file config, hoặc tạo một config tạm với số mẫu nhỏ hơn.

```bash
uv run python -m src.train --config configs/frozen.yaml
uv run python -m src.train --config configs/full_finetune.yaml
uv run python -m src.train --config configs/lora_r2.yaml
uv run python -m src.train --config configs/lora_r4.yaml
uv run python -m src.train --config configs/lora_r8.yaml
```

Mỗi lần chạy sẽ ghi kết quả vào:

```text
results/runs/<run_name>/
```

## Workflow bằng notebook

Các bước trên cũng có thể chạy bằng notebook tiếng Việt:

- `notebooks/01_baseline.ipynb`: chạy frozen và full fine-tuning baselines.
- `notebooks/02_lora_sweep.ipynb`: chạy LoRA với rank 2, 4 và 8.
- `notebooks/03_lora_diagnostics.ipynb`: phân tích các checkpoint LoRA đã lưu.
- `notebooks/04_visualize.ipynb`: tạo lại bảng tổng hợp và các biểu đồ.

Bản hiện thực trong submission này báo cáo LoRA rank sweep. Mô tả ban đầu có
nhắc đến LoRTA, nhưng LoRTA được xem là hướng mở rộng, không phải thí nghiệm đã
báo cáo chính thức.

## Đánh giá một checkpoint

```bash
uv run python -m src.evaluate --config configs/lora_r4.yaml --checkpoint results/runs/lora_r4/best_model.pt
```

## Sinh biểu đồ

```bash
uv run python -m src.visualize --results results/runs --out results/plots
```

Script trực quan hóa sẽ tạo:

- `accuracy_vs_params.png`
- `trainable_params.png`
- `loss_curves.png`
- `accuracy_vs_rank.png`
- `metrics.csv`

## Ghi chú

- Model mặc định là `prajjwal1/bert-tiny`, đủ nhỏ để chạy trên môi trường tài
  nguyên hạn chế.
- Dataset mặc định là GLUE SST-2. Lần chạy đầu tiên cần internet để tải model
  và dataset.
- Không nên ghi số liệu ước lượng vào báo cáo. Chỉ copy số liệu từ
  `metrics.json` và `metrics.csv` sau khi chạy thật.
