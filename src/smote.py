"""Within-ID SMOTE using only supplied, already-normalized source rows.

Save info['metadata'] as JSON and info['replay'] with the tensor pool. Replay
indices address the supplied x, so the caller must retain x's profiling row IDs.
"""
import math
import time

import torch


def _validate_x(x):
    if (not isinstance(x, torch.Tensor) or x.ndim != 2 or min(x.shape) < 1
            or x.dtype not in (torch.float32, torch.float64)):
        raise ValueError("source measurements must be nonempty floating N x D tensors")
    if not torch.isfinite(x).all():
        raise ValueError("source measurements must be finite")


def _validate_labels(labels, count, device):
    if (not isinstance(labels, torch.Tensor) or labels.shape != (count,)
            or labels.dtype != torch.long or labels.device != device):
        raise ValueError("labels must be matching int64 vectors on the measurement device")
    if ((labels < 0) | (labels > 255)).any():
        raise ValueError("labels must be in 0..255")


@torch.no_grad()
def replay_smote(x, info):
    """Reconstruct (measurements, labels) from unchanged x and saved replay data.

Exact tensor equality assumes the original source values, dtype, and device.
The caller's artifact hashes establish source identity; shape alone cannot.
"""
    _validate_x(x)
    metadata, replay = info["metadata"], info["replay"]
    if list(x.shape) != metadata["source_shape"] or str(x.dtype) != metadata["source_dtype"]:
        raise ValueError("replay requires the original source shape and dtype")
    a, b, lam, labels = (replay[name] for name in
                         ("anchor_indices", "neighbor_indices", "coefficients", "labels"))
    if not all(isinstance(t, torch.Tensor) and t.ndim == 1 for t in (a, b, lam, labels)):
        raise ValueError("replay entries must be vectors")
    if not (len(a) == len(b) == len(lam) == len(labels)):
        raise ValueError("replay entries must have equal lengths")
    if a.dtype != torch.long or b.dtype != torch.long or lam.dtype != x.dtype:
        raise ValueError("replay indices must be int64 and coefficients match source dtype")
    if (not torch.isfinite(lam).all() or ((lam < 0) | (lam > 1)).any()
            or ((a < 0) | (a >= len(x))).any() or ((b < 0) | (b >= len(x))).any()):
        raise ValueError("replay coefficients or source indices are out of range")
    _validate_labels(labels, len(a), labels.device)
    a, b, lam = a.to(x.device), b.to(x.device), lam.to(x.device)
    sx = (1 - lam[:, None]) * x[a] + lam[:, None] * x[b]
    return sx, labels.to(x.device).clone()


@torch.no_grad()
def make_smote(x, y, labels, seed, k=5):
    """Generate requested ID labels with uniform anchors/neighbors/coefficients.

Neighbors are the min(k,n_class-1) nearest same-class rows, excluding self.
Singleton classes duplicate their sole source row and are recorded explicitly.
Missing requested classes fail. No balancing or preprocessing is inferred here.
"""
    _validate_x(x)
    _validate_labels(y, len(x), x.device)
    if not isinstance(labels, torch.Tensor) or labels.ndim != 1 or len(labels) == 0:
        raise ValueError("requested labels must be a nonempty int64 vector")
    _validate_labels(labels, len(labels), x.device)
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2 ** 63:
        raise ValueError("seed must be an integer in [0,2**63)")
    source_y, requested = y.cpu(), labels.cpu()
    counts = torch.bincount(source_y, minlength=256)
    generated_counts = torch.bincount(requested, minlength=256)
    missing = torch.where((counts == 0) & (generated_counts > 0))[0].tolist()
    if missing:
        raise ValueError(f"source is missing requested classes {missing}")
    if x.is_cuda:
        torch.cuda.synchronize(x.device)
    started = time.perf_counter()
    rng = torch.Generator(device="cpu").manual_seed(seed)
    anchor_indices = torch.empty(len(labels), dtype=torch.long)
    neighbor_indices = torch.empty_like(anchor_indices)
    coefficients = torch.empty(len(labels), dtype=x.dtype)
    singleton_classes, effective_k = [], {}
    for label in requested.unique(sorted=True).tolist():
        source_indices = torch.where(source_y == label)[0]
        output_indices = torch.where(requested == label)[0]
        n, m = len(source_indices), len(output_indices)
        local_anchors = torch.randint(n, (m,), generator=rng)
        anchor_indices[output_indices] = source_indices[local_anchors]
        effective_k[str(label)] = min(k, n - 1)
        if n == 1:
            singleton_classes.append(label)
            neighbor_indices[output_indices] = source_indices[0]
            coefficients[output_indices] = 0
            continue
        rows = x[source_indices.to(x.device)]
        distances = torch.cdist(rows, rows)
        distances.fill_diagonal_(math.inf)
        neighbors = distances.topk(min(k, n - 1), dim=1, largest=False).indices.cpu()
        choices = torch.randint(min(k, n - 1), (m,), generator=rng)
        neighbor_indices[output_indices] = source_indices[neighbors[local_anchors, choices]]
        coefficients[output_indices] = torch.rand(m, dtype=x.dtype, generator=rng)
    info = {
        "metadata": {
            "method": "class-conditional SMOTE adaptation for balanced supplementation",
            "source_shape": list(x.shape), "source_dtype": str(x.dtype),
            "source_class_counts": counts.tolist(),
            "generated_count": len(labels), "generated_class_counts": generated_counts.tolist(),
            "seed": seed, "rng_device": "cpu", "k": k, "effective_k_by_class": effective_k,
            "distance": "Euclidean distance in supplied normalized training representation",
            "sampling": "uniform source anchor, uniform nearest-neighbor choice, Uniform[0,1) coefficient",
            "singleton_classes": singleton_classes,
            "singleton_generated_count": int(generated_counts[singleton_classes].sum()),
            "singleton_policy": "duplicate sole source row with coefficient zero",
            "index_scope": "row indices relative to supplied source x",
            "replay_formula": "(1-lambda)*x[anchor_indices]+lambda*x[neighbor_indices]",
        },
        "replay": {"anchor_indices": anchor_indices, "neighbor_indices": neighbor_indices,
                   "coefficients": coefficients, "labels": requested.clone()},
    }
    sx, sy = replay_smote(x, info)
    if x.is_cuda:
        torch.cuda.synchronize(x.device)
    info["metadata"]["generation_seconds"] = time.perf_counter() - started
    return sx, sy, info
