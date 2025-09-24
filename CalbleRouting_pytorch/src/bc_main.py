import importlib
import pprint
from copy import deepcopy

import absl.app
import absl.flags
import numpy as np
import torch

from .data import (
    partition_batch_train_test,
    subsample_batch,
    preprocess_robot_dataset,
    augment_batch,
    get_data_augmentation,
    concatenate_batches,
)
from .utils import (
    define_flags_with_default,
    set_random_seed,
    get_user_flags,
    WandBLogger,
    average_metrics,
)
from .train_utils import (
    get_learning_rate,
    get_optimizer,
)
from .model import ResNetPolicy, TanhGaussianResNetPolicy, PretrainTanhGaussianResNetPolicy


FLAGS_DEF = define_flags_with_default(
    seed=42,
    dataset_path="",
    dataset_image_keys="side_image",
    image_augmentation="none",
    clip_action=0.99,
    train_ratio=0.9,
    batch_size=128,
    total_steps=10000,
    lr=1e-4,
    lr_warmup_steps=0,
    weight_decay=0.05,
    clip_gradient=1e9,
    log_freq=50,
    eval_freq=200,
    eval_batches=20,
    save_model=False,
    policy_class_name="TanhGaussianResNetPolicy",
    policy=TanhGaussianResNetPolicy.get_default_config(),
    logger=WandBLogger.get_default_config(),
    device="auto",
)

FLAGS = absl.flags.FLAGS


def _get_device(device_flag: str) -> torch.device:
    if device_flag.lower() == "cpu":
        return torch.device("cpu")
    if device_flag.lower() == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_flag.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_flag)


def _to_device(array, device):
    if isinstance(array, torch.Tensor):
        tensor = array
    else:
        tensor = torch.from_numpy(np.asarray(array))
    return tensor.to(device=device, dtype=torch.float32)


def _clone_state_dict(model: torch.nn.Module):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def main(argv):
    del argv
    assert FLAGS.dataset_path != ""
    policy_module = importlib.import_module("CalbleRouting_pytorch.src.model")
    policy_class = getattr(policy_module, FLAGS.policy_class_name)
    variant = get_user_flags(FLAGS, FLAGS_DEF)
    wandb_logger = WandBLogger(config=FLAGS.logger, variant=variant)
    set_random_seed(FLAGS.seed)

    image_keys = FLAGS.dataset_image_keys.split(":")
    dataset = np.load(FLAGS.dataset_path, allow_pickle=True)
    if isinstance(dataset, np.ndarray) and dataset.shape == ():
        dataset = dataset.item()
    if isinstance(dataset, np.ndarray):
        dataset = concatenate_batches(dataset)
    elif isinstance(dataset, dict):
        dataset = deepcopy(dataset)
    else:
        raise TypeError("Unsupported dataset format")
    dataset = preprocess_robot_dataset(dataset, FLAGS.clip_action)
    train_dataset, test_dataset = partition_batch_train_test(dataset, FLAGS.train_ratio)

    device = _get_device(FLAGS.device)
    policy = policy_class(
        output_dim=dataset["action"].shape[-1],
        config_updates=FLAGS.policy,
    ).to(device)

    learning_rate = get_learning_rate(FLAGS=FLAGS)

    if FLAGS.policy_class_name == "TanhGaussianResNetPolicy":
        optimizer = get_optimizer(
            model=policy,
            learning_rate=FLAGS.lr,
            weight_decay=FLAGS.weight_decay,
            exclusions=TanhGaussianResNetPolicy.get_weight_decay_exclusions(),
        )
    elif FLAGS.policy_class_name == "PretrainTanhGaussianResNetPolicy":
        optimizer = get_optimizer(
            model=policy,
            learning_rate=FLAGS.lr,
            weight_decay=FLAGS.weight_decay,
            exclusions=PretrainTanhGaussianResNetPolicy.get_weight_decay_exclusions(),
        )
    elif FLAGS.policy_class_name == "ResNetPolicy":
        optimizer = get_optimizer(
            model=policy,
            learning_rate=FLAGS.lr,
            weight_decay=FLAGS.weight_decay,
            exclusions=ResNetPolicy.get_weight_decay_exclusions(),
        )
    else:
        raise ValueError(f"{FLAGS.policy_class_name} is not a valid policy")

    augmentation = get_data_augmentation(FLAGS.image_augmentation)

    best_loss, best_mse = float("inf"), float("inf")
    best_loss_model, best_mse_model = None, None

    def train_step(step, batch):
        policy.train()
        state = _to_device(batch["robot_state"], device)
        action = _to_device(batch["action"], device)
        images = [_to_device(batch[key], device) for key in image_keys]
        optimizer.zero_grad(set_to_none=True)
        log_probs, mean = policy.log_prob(state, action, images, return_mean=True)
        loss = -log_probs.mean()
        mse = torch.mean(torch.sum((mean - action) ** 2, dim=-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), FLAGS.clip_gradient)
        current_lr = learning_rate(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        optimizer.step()
        metrics = dict(
            loss=loss.item(),
            mse=mse.item(),
            learning_rate=current_lr,
        )
        return metrics

    @torch.no_grad()
    def eval_step(batch):
        policy.eval()
        state = _to_device(batch["robot_state"], device)
        action = _to_device(batch["action"], device)
        images = [_to_device(batch[key], device) for key in image_keys]
        log_probs, mean = policy.log_prob(state, action, images, return_mean=True)
        loss = -log_probs.mean()
        mse = torch.mean(torch.sum((mean - action) ** 2, dim=-1))
        metrics = dict(
            eval_loss=loss.item(),
            eval_mse=mse.item(),
        )
        return metrics

    for step in range(FLAGS.total_steps):
        batch = subsample_batch(train_dataset, FLAGS.batch_size)
        batch = augment_batch(augmentation, batch)
        metrics = train_step(step, batch)
        metrics["step"] = step

        if step % FLAGS.log_freq == 0:
            wandb_logger.log(metrics)
            pprint.pprint(metrics)

        if step % FLAGS.eval_freq == 0:
            eval_metrics = []
            for _ in range(FLAGS.eval_batches):
                batch = subsample_batch(test_dataset, FLAGS.batch_size)
                eval_metrics.append(eval_step(batch))
            eval_metrics = average_metrics(eval_metrics)
            eval_metrics["step"] = step

            if eval_metrics["eval_loss"] < best_loss:
                best_loss = eval_metrics["eval_loss"]
                best_loss_model = _clone_state_dict(policy)

            if eval_metrics["eval_mse"] < best_mse:
                best_mse = eval_metrics["eval_mse"]
                best_mse_model = _clone_state_dict(policy)

            eval_metrics["best_loss"] = best_loss
            eval_metrics["best_mse"] = best_mse
            wandb_logger.log(eval_metrics)
            pprint.pprint(eval_metrics)

            if FLAGS.save_model:
                save_data = {
                    "variant": variant,
                    "step": step,
                    "model_state_dict": _clone_state_dict(policy),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_loss_model_state_dict": best_loss_model,
                    "best_mse_model_state_dict": best_mse_model,
                }
                wandb_logger.save_pickle(save_data, "model.pkl")

    if FLAGS.save_model:
        save_data = {
            "variant": variant,
            "step": FLAGS.total_steps,
            "model_state_dict": _clone_state_dict(policy),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_loss_model_state_dict": best_loss_model,
            "best_mse_model_state_dict": best_mse_model,
        }
        wandb_logger.save_pickle(save_data, "model.pkl")


if __name__ == "__main__":
    absl.app.run(main)
