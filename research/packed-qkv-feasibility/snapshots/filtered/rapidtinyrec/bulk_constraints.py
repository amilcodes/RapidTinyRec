"""Batch prefix transfer and mask writes while retaining the released score math."""
import warnings

import torch
from .source.LogitProcessor import ConstrainedLogitsProcessor


class BulkPrefixConstrainedLogitsProcessor(ConstrainedLogitsProcessor):
    def __call__(self, input_ids, scores):
        # Keep the released second log_softmax exactly as-is for this ablation.
        scores = torch.nn.functional.log_softmax(scores, dim=-1)
        mask = torch.full_like(scores, float('-inf'))
        suffix_length = self.prefix_index if self.count == 0 else self.count
        prefixes = input_ids[:, -suffix_length:].tolist()
        rows, columns = [], []
        for row, prefix in enumerate(prefixes):
            allowed = self._prefix_allowed_tokens_fn(row // self._num_beams, prefix)
            if not allowed:
                warnings.warn(f'No valid tokens found for hash_key {prefix} at step {self.count}. '
                              'This indicates the model generated an unexpected token. ')
                allowed = [] if self.eos_token_id is None else [self.eos_token_id]
            rows.extend([row] * len(allowed))
            columns.extend(allowed)
        if rows:
            row_ids = torch.tensor(rows, dtype=torch.long, device=scores.device)
            column_ids = torch.tensor(columns, dtype=torch.long, device=scores.device)
            mask[row_ids, column_ids] = 0
        self.count += 1
        return scores + mask
