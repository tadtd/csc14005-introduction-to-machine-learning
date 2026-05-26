# Cài đặt

## 1. Môi trường

```powershell
cd lab-2
uv sync
```

`contrastive-ne` nằm trong `pyproject.toml` (bản vendored `external/contrastive-ne`, không kéo `annoy`). Trong code vẫn `from cne import CNE` — `cne` là tên module của package đó.

## 2. Kiểm tra

```powershell
$env:PYTHONPATH = "code"
uv run python -c "from cne import CNE; from models.neg_tsne import NegTSNE; print('OK')"
```

## 3. MNIST

File: `code/data/raw/mnist.hdf5` (xem `code/data/README.md`).

## 4. Tùy chọn: `scanpy` / `pillow`

```powershell
uv sync --extra data
```
