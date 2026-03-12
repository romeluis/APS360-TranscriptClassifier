"""
Model abstraction layer for NER evaluation.

Defines a common interface (NERModel) so the evaluator and visualizer
can work with any model — the rule-based baseline now, and a future
BERT-based model later.
"""

import sys
from abc import ABC, abstractmethod
from pathlib import Path

# Allow imports from sibling directories (baseline_model/, etc.)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "baseline_model"))


class NERModel(ABC):
    """Abstract base class for all NER models."""

    name: str = "unnamed"

    @abstractmethod
    def predict(self, tokens: list[str]) -> list[str]:
        """Predict BIO NER tags for a list of tokens.

        Args:
            tokens: whitespace-tokenized transcript text

        Returns:
            list of BIO tags, same length as tokens
        """
        ...


class BaselineModel(NERModel):
    """Wraps the rule-based baseline classifier from baseline_model/classifier.py."""

    name = "Rule-Based Baseline"

    def predict(self, tokens: list[str]) -> list[str]:
        from classifier import predict
        return predict(tokens)


class MLModel(NERModel):
    """BERT-based NER model for transcript entity extraction."""

    name = "BERT NER"

    def __init__(self, model_path: str | None = None):
        import sys
        sys.path.insert(0, str(_PROJECT_ROOT))
        from model.predict import BertNERPredictor
        from model.config import BEST_CHECKPOINT_DIR

        checkpoint = model_path or str(_PROJECT_ROOT / BEST_CHECKPOINT_DIR)
        self._predictor = BertNERPredictor(checkpoint_dir=checkpoint)

    def predict(self, tokens: list[str]) -> list[str]:
        return self._predictor.predict(tokens)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, type[NERModel]] = {
    "baseline": BaselineModel,
    "bert": MLModel,
}


def get_model(name: str = "baseline", **kwargs) -> NERModel:
    """Instantiate a model by registry name.

    Args:
        name: key in MODEL_REGISTRY (e.g. "baseline")
        **kwargs: forwarded to the model constructor

    Returns:
        an NERModel instance
    """
    if name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{name}'. Available: {available}")
    return MODEL_REGISTRY[name](**kwargs)
