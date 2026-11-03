"""
Train one run (one condition, one seed) and save everything about it to
<output_directory>/metrics.json: the settings, the loss curve, the final validation
loss, how long training took and a short text sample.

    python -m ablation.train --settings=configs/shakespeare_char.py --condition=baseline --seed=1

The training loop follows nanoGPT's train.py on a single GPU.
"""

import json
import math
import os
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict

import numpy as np
import torch

from ablation import data
from ablation.model import GPT, ModelConfig
from ablation.muon import Muon
from ablation.settings import parse_settings


def build_model(settings, vocabulary_size):
    model_config = ModelConfig(
        context_length=settings["context_length"],
        vocabulary_size=vocabulary_size,
        number_of_layers=settings["number_of_layers"],
        number_of_heads=settings["number_of_heads"],
        embedding_size=settings["embedding_size"],
        dropout=settings["dropout"],
        use_bias=settings["use_bias"],
        normalization=settings["normalization"],
        position_encoding=settings["position_encoding"],
    )
    return GPT(model_config), model_config


def build_optimizers(model, settings, device_type):
    """
    Returns a list of optimizers, so the training loop can treat one or two the same way.

    Baseline: AdamW trains everything.
    Muon condition: Muon takes over the hidden matrices, and everything else stays on
    exactly the same AdamW as the baseline. That way the optimizer for the hidden
    matrices is the only thing that changes.
    """
    hidden_matrices, other_matrices, no_decay = model.parameter_groups()
    if settings["optimizer"] == "adamw":
        adamw_matrices = hidden_matrices + other_matrices
    elif settings["optimizer"] == "muon":
        adamw_matrices = other_matrices
    else:
        raise ValueError(f"unknown optimizer: {settings['optimizer']}")

    adamw = torch.optim.AdamW(
        [
            {"params": adamw_matrices, "weight_decay": settings["weight_decay"]},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=settings["learning_rate"],
        betas=(settings["adam_beta1"], settings["adam_beta2"]),
        fused=(device_type == "cuda"),
    )
    optimizers = [adamw]

    if settings["optimizer"] == "muon":
        muon = Muon(
            hidden_matrices,
            lr=settings["muon_learning_rate"],
            momentum=settings["muon_momentum"],
            newton_schulz_steps=settings["newton_schulz_steps"],
            weight_decay=settings["weight_decay"],
        )
        optimizers.append(muon)

    # Remember each group's peak learning rate, because the schedule rewrites "lr" every step.
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            group["peak_learning_rate"] = group["lr"]
    return optimizers


def learning_rate_multiplier(step, settings):
    """nanoGPT's schedule as a fraction of the peak: linear warmup, then cosine decay to the minimum."""
    warmup_steps = settings["warmup_steps"]
    decay_steps = settings["decay_steps"]
    minimum_ratio = settings["minimum_learning_rate_ratio"]
    if step < warmup_steps:
        return (step + 1) / (warmup_steps + 1)
    if step > decay_steps:
        return minimum_ratio
    progress = (step - warmup_steps) / (decay_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum_ratio + cosine * (1.0 - minimum_ratio)


def set_learning_rates(optimizers, multiplier):
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            group["lr"] = group["peak_learning_rate"] * multiplier


def mixed_precision(device_type):
    """
    On a GPU, run the forward pass in bfloat16 (or float16 on older cards) like nanoGPT.
    On the CPU everything stays in float32.
    """
    if device_type != "cuda":
        return nullcontext(), torch.float32
    if torch.cuda.is_bf16_supported():
        precision = torch.bfloat16
    else:
        precision = torch.float16
    return torch.autocast(device_type="cuda", dtype=precision), precision


def wait_for_gpu(device_type):
    # GPU work runs asynchronously, so the clock is only honest after the GPU has caught up.
    if device_type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def estimate_losses(model, evaluation_sets, autocast_context, device):
    """Mean loss on the fixed training and validation batches, for the loss curves."""
    model.eval()
    losses = {}
    for split_name, batches in evaluation_sets.items():
        batch_losses = []
        for inputs, targets in batches:
            with autocast_context:
                _, loss = model(inputs.to(device), targets.to(device))
            batch_losses.append(loss.item())
        losses[split_name] = float(np.mean(batch_losses))
    model.train()
    return losses


@torch.no_grad()
def full_validation_loss(model, validation_ids, settings, autocast_context, device):
    """The main metric: mean loss per character over every window of the validation split."""
    model.eval()
    total_loss, total_characters = 0.0, 0
    for inputs, targets in data.every_window(validation_ids, settings["context_length"], settings["batch_size"]):
        with autocast_context:
            _, loss = model(inputs.to(device), targets.to(device))
        total_loss += loss.item() * targets.numel()
        total_characters += targets.numel()
    return total_loss / total_characters


def generate_sample(model, metadata, settings, autocast_context, device):
    """A short piece of generated text, just to eyeball that the model learned something."""
    torch.manual_seed(settings["seed"])
    newline_id = metadata["character_to_id"]["\n"]
    start = torch.tensor([[newline_id]], device=device)
    with autocast_context:
        token_ids = model.generate(start, settings["sample_length"], temperature=0.8, top_k=200)
    return "".join(metadata["id_to_character"][str(token_id)] for token_id in token_ids[0].tolist())


def train(settings):
    device = settings["device"]
    device_type = "cuda" if "cuda" in device else "cpu"
    os.makedirs(settings["output_directory"], exist_ok=True)

    # The seed controls both sources of run-to-run randomness: the starting weights
    # (and dropout masks) through torch, and the order of training batches through numpy.
    torch.manual_seed(settings["seed"])
    batch_order_generator = np.random.default_rng(settings["seed"])
    if device_type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    autocast_context, precision = mixed_precision(device_type)
    # float16 needs loss scaling to avoid gradients underflowing. bfloat16 and float32 don't.
    gradient_scaler = torch.amp.GradScaler("cuda", enabled=(precision == torch.float16))

    dataset_directory = os.path.join("data", settings["dataset"])
    train_ids = data.load_split(dataset_directory, "train")
    validation_ids = data.load_split(dataset_directory, "val")
    metadata = data.load_metadata(dataset_directory)

    model, model_config = build_model(settings, metadata["vocabulary_size"])
    model = model.to(device)
    parameter_count = model.count_parameters()
    print(f"[{settings['condition']} seed={settings['seed']}] {parameter_count / 1e6:.2f}M parameters on {device}")
    optimizers = build_optimizers(model, settings, device_type)
    # The compiled model shares its weights with `model`, so either can be used for evaluation.
    compiled_model = torch.compile(model) if settings["compile_model"] else model

    batch_size = settings["batch_size"]
    context_length = settings["context_length"]
    evaluation_seed = settings["evaluation_seed"]
    evaluation_batches = settings["evaluation_batches"]
    evaluation_sets = {
        "train": data.fixed_batches(train_ids, evaluation_seed, evaluation_batches, batch_size, context_length),
        "val": data.fixed_batches(validation_ids, evaluation_seed + 1, evaluation_batches, batch_size, context_length),
    }

    history = []
    training_seconds = 0.0  # time spent training only; evaluation is not counted
    diverged = False
    inputs, targets = data.random_batch(train_ids, batch_order_generator, batch_size, context_length, device)
    wait_for_gpu(device_type)
    clock_started = time.perf_counter()

    for step in range(settings["training_steps"] + 1):
        is_last_step = step == settings["training_steps"]
        if step % settings["evaluation_interval"] == 0 or is_last_step:
            # Stop the clock while evaluating, so only training time is measured.
            wait_for_gpu(device_type)
            training_seconds += time.perf_counter() - clock_started
            losses = estimate_losses(compiled_model, evaluation_sets, autocast_context, device)
            history.append(
                {"step": step, "train_loss": losses["train"], "val_loss": losses["val"], "time_s": training_seconds}
            )
            print(
                f"  step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} | {training_seconds:7.1f}s"
            )
            if not math.isfinite(losses["val"]):
                diverged = True
                break
            wait_for_gpu(device_type)
            clock_started = time.perf_counter()
        if is_last_step:
            break

        set_learning_rates(optimizers, learning_rate_multiplier(step, settings))

        for _ in range(settings["gradient_accumulation_steps"]):
            with autocast_context:
                _, loss = compiled_model(inputs, targets)
                loss = loss / settings["gradient_accumulation_steps"]
            # Fetch the next batch while the GPU is still busy with this one, as nanoGPT does.
            inputs, targets = data.random_batch(train_ids, batch_order_generator, batch_size, context_length, device)
            gradient_scaler.scale(loss).backward()

        for optimizer in optimizers:
            gradient_scaler.unscale_(optimizer)
        if settings["gradient_clip"] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"])
        for optimizer in optimizers:
            gradient_scaler.step(optimizer)
        gradient_scaler.update()
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)

    final_validation_loss = float("nan")
    sample = ""
    if not diverged:
        final_validation_loss = full_validation_loss(
            compiled_model, validation_ids, settings, autocast_context, device
        )
        print(
            f"  final validation loss (full split) {final_validation_loss:.4f}"
            f" | perplexity {math.exp(final_validation_loss):.2f}"
        )
        if settings["sample_length"]:
            sample = generate_sample(model, metadata, settings, autocast_context, device)

    steps_done = history[-1]["step"] if history else 0
    result = {
        "condition": settings["condition"],
        "seed": settings["seed"],
        "settings": settings,
        "model_config": asdict(model_config),
        "parameter_count": parameter_count,
        "device": torch.cuda.get_device_name() if device_type == "cuda" else device,
        "torch_version": torch.__version__,
        "steps": steps_done,
        "tokens_seen": steps_done * batch_size * context_length * settings["gradient_accumulation_steps"],
        "train_time_s": training_seconds,
        "ms_per_step": 1000 * training_seconds / max(steps_done, 1),
        "diverged": diverged,
        "final_val_loss": final_validation_loss,
        "final_val_perplexity": math.exp(final_validation_loss) if math.isfinite(final_validation_loss) else None,
        "history": history,
        "sample": sample,
    }
    with open(os.path.join(settings["output_directory"], "metrics.json"), "w") as metrics_file:
        json.dump(result, metrics_file, indent=1)
    return result


if __name__ == "__main__":
    train(parse_settings(sys.argv[1:]))
