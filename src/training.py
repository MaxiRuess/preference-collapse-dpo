"""DPO training with QLoRA for each experimental condition.

Step 5 of the pipeline. Trains DPO on Qwen 2.5 7B Instruct using:
  - 4-bit quantization (BitsAndBytes)
  - LoRA rank 16, alpha 32
  - DPO beta=0.1, lr=5e-6, 3 epochs

For the DPO-Multi condition, trains two separate LoRA adapters
(one on optimist data, one on skeptic data) that get routed at inference.
"""

from __future__ import annotations


def setup_model_and_tokenizer(config: dict) -> tuple:
    """Load the base model with QLoRA quantization and LoRA config.

    Args:
        config: Configuration dict (needs training section).

    Returns:
        Tuple of (model, tokenizer) ready for DPO training.
    """
    raise NotImplementedError


def train_dpo(
    dataset,
    config: dict,
    condition_name: str,
) -> str:
    """Train a single DPO condition and save the adapter.

    Uses TRL's DPOTrainer with the hyperparameters from config.

    Args:
        dataset: HuggingFace DatasetDict with train/eval splits.
        config: Configuration dict.
        condition_name: Name of the condition (for saving the adapter).

    Returns:
        Path to the saved LoRA adapter directory.
    """
    raise NotImplementedError


def train_all_conditions(
    datasets: dict,
    config: dict,
) -> dict[str, str]:
    """Train DPO for all conditions sequentially.

    Cleans up GPU memory between runs.

    Args:
        datasets: Dict mapping condition name -> DatasetDict.
        config: Configuration dict.

    Returns:
        Dict mapping condition name -> adapter path.
    """
    raise NotImplementedError


def train_multi_adapter(
    optimist_data,
    skeptic_data,
    config: dict,
) -> tuple[str, str]:
    """Train separate LoRA adapters for the DPO-Multi condition.

    Trains one adapter on optimist-preferred data and one on skeptic-preferred
    data. At inference time, the appropriate adapter is loaded based on which
    distribution we want to satisfy.

    Args:
        optimist_data: DatasetDict with optimist-preferred pairs.
        skeptic_data: DatasetDict with skeptic-preferred pairs.
        config: Configuration dict.

    Returns:
        Tuple of (optimist_adapter_path, skeptic_adapter_path).
    """
    raise NotImplementedError
