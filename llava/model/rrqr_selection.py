from typing import Union

import torch
import torch.nn.functional as F


_RRQR_LOGGED = False


def _log_rrqr_once(token_budget: int, center: bool, normalize: bool, sort_indices: bool) -> None:
    global _RRQR_LOGGED
    if not _RRQR_LOGGED:
        print(
            f"[ApET] selection_method=rrqr token_budget={token_budget} "
            f"center={center} normalize={normalize} sort_indices={sort_indices}"
        )
        _RRQR_LOGGED = True


@torch.no_grad()
def _select_tokens_cpqr_indices_single(
    visual_tokens: torch.Tensor,
    token_budget: int,
    center: bool = True,
    normalize: bool = False,
    sort_indices: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    if visual_tokens.ndim != 2:
        raise ValueError(f"visual_tokens must be [N, D], got {tuple(visual_tokens.shape)}")

    n = visual_tokens.shape[0]
    k = min(max(int(token_budget), 0), n)
    if k == 0:
        return torch.empty(0, dtype=torch.long, device=visual_tokens.device)

    x_work = visual_tokens.float()
    if center:
        x_work = x_work - x_work.mean(dim=0, keepdim=True)
    if normalize:
        x_work = F.normalize(x_work, p=2, dim=-1, eps=eps)

    residual = x_work.transpose(0, 1).contiguous()
    selected = torch.empty(k, dtype=torch.long, device=visual_tokens.device)
    selected_mask = torch.zeros(n, dtype=torch.bool, device=visual_tokens.device)
    col_norm_sq = residual.pow(2).sum(dim=0)

    for step in range(k):
        scores = col_norm_sq.masked_fill(selected_mask, -float("inf"))
        idx = torch.argmax(scores)
        selected[step] = idx
        selected_mask[idx] = True

        q = residual[:, idx]
        q = q / (q.norm() + eps)
        residual = residual - q.unsqueeze(1) * torch.matmul(q.unsqueeze(0), residual)
        col_norm_sq = residual.pow(2).sum(dim=0)

    if sort_indices:
        selected = torch.sort(selected).values
    return selected


@torch.no_grad()
def select_tokens_cpqr_indices(
    visual_tokens: torch.Tensor,
    token_budget: int,
    center: bool = True,
    normalize: bool = False,
    sort_indices: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Greedy CPQR/RRQR-style retained-token index selection.

    The selection score is computed on a centered/normalized work copy, while
    callers should gather and merge from the original visual tokens.
    """
    _log_rrqr_once(token_budget, center, normalize, sort_indices)
    if visual_tokens.ndim == 2:
        return _select_tokens_cpqr_indices_single(
            visual_tokens, token_budget, center=center, normalize=normalize,
            sort_indices=sort_indices, eps=eps
        )
    if visual_tokens.ndim == 3:
        return torch.stack([
            _select_tokens_cpqr_indices_single(
                x, token_budget, center=center, normalize=normalize,
                sort_indices=sort_indices, eps=eps
            )
            for x in visual_tokens
        ], dim=0)
    raise ValueError(f"visual_tokens must be [N, D] or [B, N, D], got {tuple(visual_tokens.shape)}")


def _merge_tokens_to_selected_single(
    visual_tokens: torch.Tensor,
    selected_indices: torch.Tensor,
    scaling: Union[int, float] = 1,
) -> torch.Tensor:
    if visual_tokens.ndim != 2:
        raise ValueError(f"visual_tokens must be [N, D], got {tuple(visual_tokens.shape)}")
    if selected_indices.ndim != 1:
        raise ValueError(f"selected_indices must be [K], got {tuple(selected_indices.shape)}")
    if selected_indices.numel() == 0:
        raise ValueError("At least one selected token is required for merging.")

    selected_indices = selected_indices.to(device=visual_tokens.device, dtype=torch.long)
    n, d = visual_tokens.shape
    selected_mask = torch.zeros(n, dtype=torch.bool, device=visual_tokens.device)
    selected_mask.scatter_(0, selected_indices, True)

    retained_tokens = visual_tokens[selected_indices]
    non_retained_tokens = visual_tokens[~selected_mask]
    if non_retained_tokens.shape[0] == 0:
        return visual_tokens

    sim = F.cosine_similarity(
        non_retained_tokens.float().unsqueeze(1),
        retained_tokens.float().unsqueeze(0),
        dim=2,
    )
    nearest_token_indices = sim.argmax(dim=1)

    merged_features = retained_tokens.float() * float(scaling)
    merge_count = torch.zeros(selected_indices.numel(), device=visual_tokens.device, dtype=merged_features.dtype)
    merged_features.scatter_add_(
        0,
        nearest_token_indices.unsqueeze(-1).expand(-1, d),
        non_retained_tokens.float(),
    )
    merge_count.scatter_add_(
        0,
        nearest_token_indices,
        torch.ones_like(nearest_token_indices, dtype=merged_features.dtype),
    )
    merged_features = merged_features / (float(scaling) + merge_count.unsqueeze(1))
    visual_tokens[selected_indices] = merged_features.to(dtype=visual_tokens.dtype)
    return visual_tokens


def merge_tokens_to_selected(
    visual_tokens: torch.Tensor,
    selected_indices: torch.Tensor,
    scaling: Union[int, float] = 1,
) -> torch.Tensor:
    """Reuse ApET-style cosine assignment/average merge for selected anchors."""
    if visual_tokens.ndim == 2:
        return _merge_tokens_to_selected_single(visual_tokens, selected_indices, scaling=scaling)
    if visual_tokens.ndim == 3:
        if selected_indices.ndim != 2:
            raise ValueError(f"selected_indices must be [B, K], got {tuple(selected_indices.shape)}")
        for b in range(visual_tokens.shape[0]):
            _merge_tokens_to_selected_single(visual_tokens[b], selected_indices[b], scaling=scaling)
        return visual_tokens
    raise ValueError(f"visual_tokens must be [N, D] or [B, N, D], got {tuple(visual_tokens.shape)}")
