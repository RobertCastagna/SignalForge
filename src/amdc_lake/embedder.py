"""BGE small embedding through Hugging Face Transformers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from transformers import AutoModel, AutoTokenizer

from amdc_lake.constants import MAX_LENGTH, MODEL_NAME


@dataclass
class BgeM3Embedder:
    model_name: str = MODEL_NAME
    device: str | None = None
    max_length: int = MAX_LENGTH

    def __post_init__(self) -> None:
        import torch
        import torch.nn.functional as functional

        self._torch = torch
        self._functional = functional
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.eval()
        self.model.to(self.device)

    def embed(self, texts: Iterable[str], *, batch_size: int = 8) -> list[list[float]]:
        items = [text or "" for text in texts]
        vectors: list[list[float]] = []
        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {name: tensor.to(self.device) for name, tensor in encoded.items()}
            with self._torch.no_grad():
                output = self.model(**encoded)
                pooled = _mean_pool(output.last_hidden_state, encoded["attention_mask"])
                normalized = self._functional.normalize(pooled, p=2, dim=1)
            vectors.extend(normalized.detach().cpu().to(self._torch.float32).tolist())
        return vectors


def _mean_pool(last_hidden_state, attention_mask):
    import torch

    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts
