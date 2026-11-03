"""
Settings for a training run, and how they are read from the command line.

The defaults are nanoGPT's Shakespeare-char settings (config/train_shakespeare_char.py).
A settings file can override them, and any --name=value on the command line overrides both:

    python -m ablation.train --settings=configs/shakespeare_char.py --condition=rope --seed=3
"""

import ast

import torch

# Each condition is the baseline plus exactly one change. Nothing else differs between them.
CONDITIONS = {
    "baseline": {},
}

DEFAULT_SETTINGS = dict(
    # which run this is
    condition="baseline",
    seed=0,
    output_directory="runs/default",
    dataset="shakespeare_char",
    device="cuda" if torch.cuda.is_available() else "cpu",
    compile_model=False,
    # model size (nanoGPT's Shakespeare-char model)
    number_of_layers=6,
    number_of_heads=6,
    embedding_size=384,
    context_length=256,
    dropout=0.2,
    use_bias=False,
    # training length
    batch_size=64,
    gradient_accumulation_steps=1,
    training_steps=5000,
    # AdamW, which trains every parameter in the baseline
    learning_rate=1e-3,
    minimum_learning_rate_ratio=0.1,  # nanoGPT decays to 1e-4, which is a tenth of the peak
    warmup_steps=100,
    decay_steps=None,  # None means "decay over the whole run", as in nanoGPT's config
    weight_decay=0.1,
    adam_beta1=0.9,
    adam_beta2=0.99,
    gradient_clip=1.0,
    # evaluation
    evaluation_interval=250,
    evaluation_batches=200,
    evaluation_seed=1234,  # fixed, so every run is scored on exactly the same batches
    sample_length=300,
)


def read_settings_file(path):
    """A settings file is plain Python assignments, the same idea as nanoGPT's config files."""
    file_settings = {}
    with open(path) as settings_file:
        exec(settings_file.read(), {}, file_settings)
    for name in file_settings:
        if not name.startswith("_") and name not in DEFAULT_SETTINGS:
            raise ValueError(f"unknown setting '{name}' in {path}")
    return {name: value for name, value in file_settings.items() if not name.startswith("_")}


def parse_value(text):
    """Turn '1e-3' into a float, 'True' into a bool and so on. Anything else stays a string."""
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def parse_settings(arguments):
    settings = dict(DEFAULT_SETTINGS)
    command_line_overrides = {}
    for argument in arguments:
        if not argument.startswith("--") or "=" not in argument:
            raise ValueError(f"expected --name=value, got {argument}")
        name, value = argument[2:].split("=", 1)
        if name == "settings":
            settings.update(read_settings_file(value))
        elif name in DEFAULT_SETTINGS:
            command_line_overrides[name] = parse_value(value)
        else:
            raise ValueError(f"unknown setting: {name}")

    # Command-line values win over the settings file, whatever order they were given in.
    settings.update(command_line_overrides)

    if settings["condition"] not in CONDITIONS:
        raise ValueError(f"unknown condition: {settings['condition']}")
    settings.update(CONDITIONS[settings["condition"]])

    if settings["decay_steps"] is None:
        settings["decay_steps"] = settings["training_steps"]
    return settings
