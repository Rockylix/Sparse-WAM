# Sparse-WAM

[Website](https://rockylix.github.io/Sparse-WAM.github.io/) · [Paper (arXiv)](https://arxiv.org/abs/2609.38984)

Action-guided sparse inference for World Action Models. The primary setup is
Cosmos3 Edge on a single NVIDIA RTX 4090 with Linux x86_64.

## Environment

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and use a
CUDA 13-compatible NVIDIA driver. Python 3.13 and the inference dependencies are
managed by the project lockfile.

```bash
git clone https://github.com/Rockylix/Sparse-WAM.git
cd Sparse-WAM
uv sync --locked --extra edge
source .venv/bin/activate
python -m sparse_wam config --backbone edge
```

Keep the complete source checkout; model weights are obtained separately.
Follow the [RTX 4090 guide](docs/rtx4090.md) to prepare assets and run offline inference.

[Python API](docs/api.md) · [Method](docs/method.md) · [Validation](docs/validation.md)

Nano requires a separate compatible environment; see the [backend notes](docs/backbone_differences.md).
