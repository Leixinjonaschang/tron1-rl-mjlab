# tron1-rl-mjlab

## Installation

```shell
uv sync
```

## Usage

### Training

```shell
uv run python scripts/rsl_rl/train.py Mjlab-WF-Tron
```

### Evaluation

```shell
uv run python scripts/rsl_rl/play.py Mjlab-WF-Tron --wandb-run-path <project>/mjlab_wf_tron # --viewer viser --export-onnx True
```
