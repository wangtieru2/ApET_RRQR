import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

try:
    from .utils import fps
except ModuleNotFoundError:
    def fps(x, K):
        # Same farthest-point sampling logic as llava.model.utils.fps, kept as a
        # lightweight fallback for environments that have torch but not all LLaVA deps.
        bsz, n, _ = x.shape
        centroids = torch.zeros(bsz, K, dtype=torch.long, device=x.device)
        dist = torch.full((bsz, n), 1e10, device=x.device)
        farthest = torch.randint(0, n, (bsz,), device=x.device)
        for i in range(K):
            centroids[:, i] = farthest
            centroid = x[torch.arange(bsz, device=x.device), farthest].unsqueeze(1)
            dist_cur = ((x - centroid) ** 2).sum(-1)
            dist = torch.min(dist, dist_cur)
            farthest = dist.max(1).indices
        return centroids


TOKEN_STATE = {
    "drop": 0,
    "keep": 1,
    "merge": 2,
    "representative": 3,
}


def _normalize_scores(scores: torch.Tensor) -> torch.Tensor:
    min_v = scores.min()
    max_v = scores.max()
    return (scores - min_v) / (max_v - min_v + 1e-6)


def _topk_sorted(scores: torch.Tensor, k: int, largest: bool = True) -> torch.Tensor:
    if k <= 0:
        return torch.empty(0, dtype=torch.long, device=scores.device)
    k = min(k, scores.numel())
    indices = torch.topk(scores, k=k, largest=largest).indices
    return indices[torch.argsort(indices)]


def compute_approximation_error(
    visual_tokens: torch.Tensor,
    basis_token_num: int = 10,
    selection_method: str = "fps",
) -> torch.Tensor:
    """ApET-style linear reconstruction error for visual token importance."""
    if visual_tokens.ndim != 2:
        raise ValueError(f"visual_tokens must be [N, D], got {tuple(visual_tokens.shape)}")

    x = visual_tokens.float().unsqueeze(0)
    _, n, d = x.shape
    k = min(max(int(basis_token_num), 1), n)

    if selection_method == "fps":
        basis_idx = fps(x, k).unsqueeze(-1).expand(-1, -1, d)
        basis = x.gather(1, basis_idx)
    elif selection_method == "random":
        rand_idx = torch.randperm(n, device=x.device)[:k].unsqueeze(0)
        basis = x.gather(1, rand_idx.unsqueeze(-1).expand(-1, -1, d))
    else:
        raise ValueError(f"Unsupported basis selection method: {selection_method}")

    gram = torch.matmul(basis, basis.transpose(-1, -2))
    gram.diagonal(dim1=-2, dim2=-1).add_(1e-5)
    rhs = torch.matmul(basis, x.transpose(-1, -2))
    linear_group = torch.linalg.solve(gram, rhs).transpose(-1, -2)
    x_hat = torch.matmul(linear_group, basis)
    return _normalize_scores(torch.norm(x - x_hat, dim=-1).squeeze(0))


def compute_mergeability(visual_tokens: torch.Tensor) -> torch.Tensor:
    """M_i = max_{j != i} cosine_similarity(z_i, z_j)."""
    x = F.normalize(visual_tokens.float(), dim=-1)
    sim = torch.matmul(x, x.transpose(0, 1))
    sim.fill_diagonal_(-float("inf"))
    return _normalize_scores(sim.max(dim=-1).values)


def _merge_to_representatives(
    visual_tokens: torch.Tensor,
    representative_indices: torch.Tensor,
    source_indices: Optional[torch.Tensor] = None,
    weighted_scores: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    warnings: List[str] = []
    device = visual_tokens.device
    n = visual_tokens.shape[0]

    if source_indices is None:
        source_indices = torch.arange(n, device=device)
    if representative_indices.numel() == 0:
        raise ValueError("At least one representative is required for merging.")

    source_tokens = visual_tokens[source_indices]
    rep_tokens = visual_tokens[representative_indices]
    sim = torch.matmul(F.normalize(source_tokens.float(), dim=-1), F.normalize(rep_tokens.float(), dim=-1).t())
    nearest = sim.argmax(dim=1)

    merged = []
    for gid in range(representative_indices.numel()):
        member_mask = nearest == gid
        if not member_mask.any():
            warnings.append(f"empty merge group {gid}; using representative token only")
            group_tokens = visual_tokens[representative_indices[gid]].unsqueeze(0)
            weights = None
        else:
            group_tokens = source_tokens[member_mask]
            weights = weighted_scores[source_indices[member_mask]] if weighted_scores is not None else None

        if weights is None:
            merged.append(group_tokens.mean(dim=0))
        else:
            weights = weights.float().clamp_min(1e-6).to(group_tokens.device)
            merged.append((group_tokens.float() * weights[:, None]).sum(dim=0).to(group_tokens.dtype) / weights.sum())

    merge_group_id = torch.full((n,), -1, dtype=torch.long, device=device)
    merge_group_id[source_indices] = nearest
    return torch.stack(merged, dim=0).to(visual_tokens.dtype), merge_group_id, warnings


def _debug_dict(
    evidence_scores: torch.Tensor,
    mergeability_scores: Optional[torch.Tensor],
    token_state: torch.Tensor,
    merge_group_id: torch.Tensor,
    original_count: int,
    compressed_count: int,
    warnings: Optional[List[str]] = None,
) -> Dict:
    return {
        "evidence_scores": evidence_scores.detach().float().cpu(),
        "mergeability_scores": None if mergeability_scores is None else mergeability_scores.detach().float().cpu(),
        "token_state": token_state.detach().long().cpu(),
        "merge_group_id": merge_group_id.detach().long().cpu(),
        "original_token_count": int(original_count),
        "compressed_token_count": int(compressed_count),
        "token_state_legend": dict(TOKEN_STATE),
        "warnings": warnings or [],
    }


def compress_visual_tokens(
    visual_tokens: torch.Tensor,
    mode: str,
    budget: int,
    basis_token_num: int = 10,
    return_debug: bool = True,
    weighted_merge: bool = False,
) -> Tuple[torch.Tensor, Dict]:
    """Compress a single image's visual tokens with a fair fixed token budget."""
    if visual_tokens.ndim != 2:
        raise ValueError(f"compress_visual_tokens expects [N, D], got {tuple(visual_tokens.shape)}")
    if mode not in {"no_compression", "full_pruning", "full_merging", "selective_routing"}:
        raise ValueError(f"Unknown compression mode: {mode}")

    n = visual_tokens.shape[0]
    if mode == "no_compression":
        evidence = torch.empty(0, device=visual_tokens.device)
        token_state = torch.full((n,), TOKEN_STATE["keep"], dtype=torch.long, device=visual_tokens.device)
        merge_group_id = torch.full((n,), -1, dtype=torch.long, device=visual_tokens.device)
        return visual_tokens, _debug_dict(evidence, None, token_state, merge_group_id, n, n)

    if budget <= 0:
        raise ValueError(f"visual token budget must be positive for {mode}, got {budget}")
    if budget > n:
        raise ValueError(f"visual token budget {budget} exceeds available visual tokens {n}")

    evidence = compute_approximation_error(visual_tokens, basis_token_num=basis_token_num)
    warnings: List[str] = []

    if mode == "full_pruning":
        keep_idx = _topk_sorted(evidence, budget, largest=True)
        token_state = torch.full((n,), TOKEN_STATE["drop"], dtype=torch.long, device=visual_tokens.device)
        token_state[keep_idx] = TOKEN_STATE["keep"]
        merge_group_id = torch.full((n,), -1, dtype=torch.long, device=visual_tokens.device)
        compressed = visual_tokens[keep_idx]
        return compressed, _debug_dict(evidence, None, token_state, merge_group_id, n, compressed.shape[0], warnings)

    mergeability = compute_mergeability(visual_tokens)

    if mode == "full_merging":
        rep_idx = fps(visual_tokens.float().unsqueeze(0), budget).squeeze(0)
        compressed, merge_group_id, merge_warnings = _merge_to_representatives(
            visual_tokens,
            rep_idx,
            weighted_scores=evidence if weighted_merge else None,
        )
        warnings.extend(merge_warnings)
        token_state = torch.full((n,), TOKEN_STATE["merge"], dtype=torch.long, device=visual_tokens.device)
        token_state[rep_idx] = TOKEN_STATE["representative"]
        return compressed, _debug_dict(evidence, mergeability, token_state, merge_group_id, n, compressed.shape[0], warnings)

    keep_tokens = budget // 2
    merged_tokens = budget - keep_tokens
    keep_idx = _topk_sorted(evidence, keep_tokens, largest=True)
    keep_mask = torch.zeros(n, dtype=torch.bool, device=visual_tokens.device)
    keep_mask[keep_idx] = True

    remaining_idx = torch.where(~keep_mask)[0]
    if remaining_idx.numel() < merged_tokens:
        warnings.append("not enough non-keep tokens for requested merge groups; moving lowest-evidence keep tokens to merge pool")
        keep_tokens = max(0, budget - remaining_idx.numel())
        merged_tokens = budget - keep_tokens
        keep_idx = _topk_sorted(evidence, keep_tokens, largest=True)
        keep_mask.zero_()
        keep_mask[keep_idx] = True
        remaining_idx = torch.where(~keep_mask)[0]

    merge_source_count = min(remaining_idx.numel(), max(merged_tokens, merged_tokens * 2))
    ranked_remaining = remaining_idx[torch.topk(mergeability[remaining_idx], k=merge_source_count, largest=True).indices]
    merge_source_idx = ranked_remaining[torch.argsort(ranked_remaining)]

    if merge_source_idx.numel() == merged_tokens:
        rep_idx = merge_source_idx
    else:
        local_rep = fps(visual_tokens[merge_source_idx].float().unsqueeze(0), merged_tokens).squeeze(0)
        rep_idx = merge_source_idx[local_rep]

    merged, merge_group_id, merge_warnings = _merge_to_representatives(
        visual_tokens,
        rep_idx,
        source_indices=merge_source_idx,
        weighted_scores=evidence if weighted_merge else None,
    )
    warnings.extend(merge_warnings)

    token_state = torch.full((n,), TOKEN_STATE["drop"], dtype=torch.long, device=visual_tokens.device)
    token_state[keep_idx] = TOKEN_STATE["keep"]
    token_state[merge_source_idx] = TOKEN_STATE["merge"]
    token_state[rep_idx] = TOKEN_STATE["representative"]
    compressed = torch.cat([visual_tokens[keep_idx], merged], dim=0)

    if compressed.shape[0] != budget:
        warnings.append(f"budget mismatch after selective routing: got {compressed.shape[0]}, expected {budget}")
        if compressed.shape[0] > budget:
            compressed = compressed[:budget]
        else:
            pad_idx = _topk_sorted(evidence, budget - compressed.shape[0], largest=True)
            compressed = torch.cat([compressed, visual_tokens[pad_idx]], dim=0)

    return compressed, _debug_dict(evidence, mergeability, token_state, merge_group_id, n, compressed.shape[0], warnings)
