# CableRouting (PyTorch)

This directory contains a PyTorch reimplementation of the CableRouting behaviour cloning project.

## Installation

```shell
git clone git@github.com:tan-liam/CableRouting.git
cd CableRouting/CalbleRouting_pytorch
pip install -r requirements.txt
```

The implementation automatically uses a CUDA device when available. You can force CPU execution by passing `--device=cpu` to the training scripts.

## Training scripts

Set your `WANDB_API_KEY` in the shell scripts under `local_scripts/` and update the dataset paths before launching experiments.

```shell
local_scripts/pretrain_resnet_embedding.sh
local_scripts/train_routing_bc.sh
local_scripts/train_highlevel.sh
local_scripts/finetune_highlevel.sh
```

### Behaviour cloning

`CalbleRouting_pytorch/src/bc_main.py` trains the low-level policy. Provide the route dataset via the `--dataset_path` flag. When `--save_model=True`, checkpoints contain PyTorch `state_dict`s for the model and optimizer.

### Primitive selection

`CalbleRouting_pytorch/src/primitive_selection_main.py` trains the high-level primitive selector. Use `--encoder_checkpoint_path` to load the pretrained features policy produced by `bc_main.py`.

## Checkpoints

Saved checkpoints are Python pickles containing:

* `variant`: configuration dictionary captured from flags.
* `model_state_dict`: PyTorch `state_dict` for the current model.
* `optimizer_state_dict`: optimizer state.
* `best_*_model_state_dict`: copies of the best performing model parameters according to validation metrics.

These files can be loaded with `cloudpickle.load` and fed back into the training scripts via the appropriate flags.
