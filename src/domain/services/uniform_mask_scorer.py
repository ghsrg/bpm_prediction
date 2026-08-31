from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
from typing import Sequence

import torch


@dataclass(frozen=True)
class UniformMaskScore:
    probabilities: torch.FloatTensor
    mask_cardinality: torch.LongTensor
    invalid_rows: torch.BoolTensor


class UniformMaskScorer:
    def __init__(self, *, empty_mask_policy: str = "raise") -> None:
        if empty_mask_policy not in {"raise", "record_invalid"}:
            raise ValueError("empty_mask_policy must be raise or record_invalid.")
        self.empty_mask_policy = empty_mask_policy

    def score(self, *, allowed_mask: torch.Tensor, candidate_keys: Sequence[str]) -> UniformMaskScore:
        mask = allowed_mask.to(dtype=torch.bool)
        if mask.dim() != 2:
            raise ValueError("allowed_mask must have shape [B, C].")
        if mask.size(1) != len(candidate_keys):
            raise ValueError("candidate_keys length must equal allowed_mask width.")
        cardinality = mask.sum(dim=1).to(dtype=torch.long)
        invalid_rows = cardinality.eq(0)
        if bool(invalid_rows.any()) and self.empty_mask_policy == "raise":
            raise ValueError("Uniform mask contains an empty candidate row.")
        probabilities = torch.zeros(mask.shape, dtype=torch.float32, device=mask.device)
        valid_rows = ~invalid_rows
        probabilities[valid_rows] = mask[valid_rows].float() / cardinality[valid_rows].unsqueeze(1)
        return UniformMaskScore(probabilities, cardinality, invalid_rows)

    def sample_prediction(
        self,
        *,
        allowed_mask: torch.Tensor,
        candidate_keys: Sequence[str],
        evaluation_seed: int,
        draw_index: int,
        sample_key: str,
    ) -> int:
        score = self.score(allowed_mask=allowed_mask, candidate_keys=candidate_keys)
        if score.probabilities.size(0) != 1:
            raise ValueError("sample_prediction requires exactly one mask row.")
        if bool(score.invalid_rows[0]):
            return -1
        keys = tuple(str(key) for key in candidate_keys)
        if len(set(keys)) != len(keys):
            raise ValueError("candidate_keys must be unique for seeded sampling.")
        allowed_indices = torch.where(allowed_mask.to(dtype=torch.bool)[0])[0].tolist()
        canonical_indices = [index for _key, index in sorted((keys[index], index) for index in allowed_indices)]
        seed_payload = f"{int(evaluation_seed)}|{int(draw_index)}|{sample_key}".encode("utf-8")
        stable_seed = int.from_bytes(hashlib.sha256(seed_payload).digest()[:8], byteorder="big")
        return int(canonical_indices[random.Random(stable_seed).randrange(len(canonical_indices))])
