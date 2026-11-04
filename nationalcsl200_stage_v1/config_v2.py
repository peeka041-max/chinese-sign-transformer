from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping


@dataclass(frozen=True)
class V2Config:
    sequence_length: int = 30
    input_dim: int = 266
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.25

    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 1e-3
    max_epochs: int = 50
    early_stopping_patience: int = 5

    scheduler_factor: float = 0.5
    scheduler_patience: int = 2
    scheduler_min_lr: float = 1e-6
    validation_ratio: float = 0.2
    random_seed: int = 42
    num_workers: int = 0

    data_root: str = "data_v2"
    checkpoint_path: str = "checkpoints_v2/best_model_v2.pt"

    def validate(self):
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive.")
        if self.input_dim != 266:
            raise ValueError("Transformer V2 input_dim must be 266.")
        if self.d_model <= 0 or self.d_model % self.nhead != 0:
            raise ValueError("d_model must be positive and divisible by nhead.")
        if self.num_layers <= 0 or self.dim_feedforward <= 0:
            raise ValueError("Transformer layer sizes must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if self.batch_size <= 0 or self.max_epochs <= 0:
            raise ValueError("batch_size and max_epochs must be positive.")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("Invalid optimizer hyperparameters.")
        if self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive.")
        if not 0.0 < self.scheduler_factor < 1.0:
            raise ValueError("scheduler_factor must be in (0, 1).")
        if self.scheduler_patience < 0 or self.scheduler_min_lr < 0.0:
            raise ValueError("Invalid scheduler hyperparameters.")
        if not 0.0 < self.validation_ratio < 1.0:
            raise ValueError("validation_ratio must be in (0, 1).")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative.")

    def to_dict(self):
        return asdict(self)


def config_from_mapping(values: Mapping[str, Any]):
    """Build a config from checkpoint values while ignoring future extras."""
    allowed_names = {field.name for field in fields(V2Config)}
    filtered = {
        key: value
        for key, value in dict(values).items()
        if key in allowed_names
    }
    config = V2Config(**filtered)
    config.validate()
    return config


DEFAULT_CONFIG = V2Config()
DEFAULT_CONFIG.validate()
