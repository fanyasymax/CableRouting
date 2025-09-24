import pprint

import absl.app
import absl.flags
import numpy as np
import torch
import torch.nn.functional as F

from .data import (
    partition_batch_train_test,
    subsample_batch,
    preprocess_robot_dataset,
    augment_batch,
    get_data_augmentation,
    concatenate_batches,
)
from .model import PrimitiveSelectionPolicy, PretrainTanhGaussianResNetPolicy
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

import cloudpickle as pickle


FLAGS_DEF = define_flags_with_default(
    seed=42,
    dataset_path="",
    dataset_image_keys="side_image",
    image_augmentation="none",
    clip_action=0.99,
    train_ratio=0.9,
    batch_size=128,
    total_steps=10000,
    finetune_steps=500,
    lr=1e-4,
    lr_warmup_steps=0,
    weight_decay=0.05,
    clip_gradient=1e9,
    log_freq=50,
    eval_freq=200,
    eval_batches=20,
    save_model=False,
    policy=PrimitiveSelectionPolicy.get_default_config(),
    logger=WandBLogger.get_default_config(),
    gripper=False,
    encoder_checkpoint_path="",
    primitive_policy_checkpoint_path="",
    pretrained_model_key="model_state_dict",
    output_dim_gaussian_policy=0,
    finetune_policy=False,
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


def load_policy_and_params(ckpt_path, policy_config, model_key):
    assert ckpt_path != ""
    with open(ckpt_path, "rb") as fin:
        checkpoint_data = pickle.load(fin)
    checkpoint_policy_config = {
        k[7:]: v
        for k, v in checkpoint_data.get("variant", {}).items()
        if k.startswith("policy.")
    }
    if hasattr(policy_config, "update_from_flattened_dict"):
        policy_config.update_from_flattened_dict(checkpoint_policy_config)
    else:
        for key, value in checkpoint_policy_config.items():
            setattr(policy_config, key, value)
    if model_key not in checkpoint_data:
        raise KeyError(f"Key '{model_key}' not found in checkpoint")
    params = checkpoint_data[model_key]
    return policy_config, params


def main(argv):
    del argv
    assert FLAGS.dataset_path != ""
    variant = get_user_flags(FLAGS, FLAGS_DEF)
    wandb_logger = WandBLogger(config=FLAGS.logger, variant=variant)
    set_random_seed(FLAGS.seed)

    image_keys = FLAGS.dataset_image_keys.split(":")
    dataset_paths = FLAGS.dataset_path.split(":")
    dataset = []
    for dataset_path in dataset_paths:
        loaded = np.load(dataset_path, allow_pickle=True)
        if isinstance(loaded, np.ndarray) and loaded.shape == ():
            loaded = loaded.item()
        dataset.append(loaded)
    dataset = concatenate_batches(dataset)
    dataset = preprocess_robot_dataset(dataset, FLAGS.clip_action)
    train_dataset, test_dataset = partition_batch_train_test(dataset, FLAGS.train_ratio)

    pretrain_features_policy_config = PretrainTanhGaussianResNetPolicy.get_default_config()
    (
        pretrain_features_policy_config,
        pretrain_features_policy_state_dict,
    ) = load_policy_and_params(
        FLAGS.encoder_checkpoint_path,
        pretrain_features_policy_config,
        FLAGS.pretrained_model_key,
    )
    pretrain_features_policy = PretrainTanhGaussianResNetPolicy(
        output_dim=4,
        config_updates=pretrain_features_policy_config,
    )

    device = _get_device(FLAGS.device)
    pretrain_features_policy.load_state_dict(pretrain_features_policy_state_dict)
    pretrain_features_policy.to(device)
    pretrain_features_policy.eval()

    policy_config = PrimitiveSelectionPolicy.get_default_config(FLAGS.policy)
    policy_state_dict = None
    if FLAGS.finetune_policy:
        policy_config, policy_state_dict = load_policy_and_params(
            FLAGS.primitive_policy_checkpoint_path,
            policy_config,
            FLAGS.pretrained_model_key,
        )

    policy = PrimitiveSelectionPolicy(
        output_dim_gaussian_policy=FLAGS.output_dim_gaussian_policy,
        config_updates=policy_config,
    ).to(device)

    if policy_state_dict is not None:
        policy.load_state_dict(policy_state_dict)

    learning_rate = get_learning_rate(FLAGS=FLAGS)
    optimizer = get_optimizer(
        model=policy,
        learning_rate=FLAGS.lr,
        weight_decay=FLAGS.weight_decay,
        exclusions=PrimitiveSelectionPolicy.get_weight_decay_exclusions(),
    )

    augmentation = get_data_augmentation(FLAGS.image_augmentation)

    best_loss = float("inf")
    best_loss_model = None

    train_steps = FLAGS.finetune_steps if FLAGS.finetune_policy else FLAGS.total_steps

    def extract_features(state, images):
        with torch.no_grad():
            state_tensor = _to_device(state, device)
            image_tensors = [_to_device(image, device) for image in images]
            features, _, _ = pretrain_features_policy(
                state_tensor,
                image_tensors,
                deterministic=True,
                return_features=True,
            )
            return features.detach()

    def train_step(step, batch):
        policy.train()
        features = extract_features(batch["robot_state"], [batch[key] for key in image_keys])
        primitive_sequence = torch.from_numpy(batch["primitive_sequence"]).long().to(device)
        labels = torch.from_numpy(batch["labels"]).long().to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = policy(features, primitive_sequence)
        loss = F.cross_entropy(logits, labels)
        accuracy = (logits.argmax(dim=-1) == labels).float().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), FLAGS.clip_gradient)
        current_lr = learning_rate(step)
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        optimizer.step()
        metrics = dict(
            loss=loss.item(),
            train_accuracy=accuracy.item(),
            learning_rate=current_lr,
        )
        return metrics

    @torch.no_grad()
    def eval_step(batch):
        policy.eval()
        features = extract_features(batch["robot_state"], [batch[key] for key in image_keys])
        primitive_sequence = torch.from_numpy(batch["primitive_sequence"]).long().to(device)
        labels = torch.from_numpy(batch["labels"]).long().to(device)
        logits = policy(features, primitive_sequence)
        loss = F.cross_entropy(logits, labels)
        accuracy = (logits.argmax(dim=-1) == labels).float().mean()
        metrics = dict(
            eval_loss=loss.item(),
            eval_accuracy=accuracy.item(),
        )
        return metrics

    for step in range(train_steps):
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

            eval_metrics["best_loss"] = best_loss
            wandb_logger.log(eval_metrics)
            pprint.pprint(eval_metrics)

            if FLAGS.save_model:
                save_data = {
                    "variant": variant,
                    "step": step,
                    "model_state_dict": _clone_state_dict(policy),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_loss_model_state_dict": best_loss_model,
                }
                wandb_logger.save_pickle(save_data, f"model.pkl")

        if FLAGS.save_model and step in (0, train_steps - 1):
            save_data = {
                "variant": variant,
                "step": step,
                "model_state_dict": _clone_state_dict(policy),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_loss_model_state_dict": best_loss_model,
            }
            wandb_logger.save_pickle(save_data, f"model_{step}_steps.pkl")


if __name__ == "__main__":
    absl.app.run(main)
