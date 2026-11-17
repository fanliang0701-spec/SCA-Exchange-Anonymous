"""Recorded fixed-endpoint generator training loop."""
import copy, math, time
from pathlib import Path
import torch

def train_measured(flow, model, x, y, vx, vy, seed, folder, steps=12000,
                   endpoints=(6000, 12000), fit_started=None):
    """Preserve legacy updates; fit_started uses perf_counter before initialization.

    Endpoint files contain the actual update state. As in the legacy trainer,
    the returned model is restored to its best validation network afterward.
    """
    if steps < 1 or not endpoints or any(s < 1 or s > steps for s in endpoints):
        raise ValueError("positive training steps and endpoints within the run are required")
    if model.kind not in ("fm", "ddpm"):
        raise ValueError("measured training supports fm and ddpm")
    endpoints = set(endpoints)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if fit_started is None:
        flow._synchronize(x.device)
        fit_started = time.perf_counter()
    batch, lr = 128, .001
    generator, validation_rng = flow._rng(x.device, seed), flow._rng(x.device, seed + 1)
    model.validation_noise = torch.randn(vx.shape, device=x.device, dtype=x.dtype, generator=validation_rng)
    if model.kind == "ddpm":
        model.validation_times = torch.randint(1000, (len(vx),), device=x.device, generator=validation_rng)
    else:
        model.validation_times = torch.rand(len(vx), device=x.device, dtype=x.dtype, generator=validation_rng)
    model.validation_labels = vy.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.0001)
    checkpoints = sorted({s for s in (1000, 3000, 6000, steps) if s <= steps} | endpoints)
    best_loss, best_step, best_state = float("inf"), None, None
    training_history, validation_history = [], []
    endpoint_fit_seconds, endpoint_rng_state_hex = {}, {}
    audit_overhead = 0.
    running = x.new_zeros(())
    initial_rng = generator.get_state().cpu().numpy().tobytes().hex()
    flow._synchronize(x.device)
    started = time.perf_counter()
    model.train()
    for step in range(1, steps + 1):
        indices = torch.randint(len(x), (batch,), device=x.device, generator=generator)
        real, labels = x[indices], y[indices]
        noise = torch.randn(real.shape, device=x.device, dtype=x.dtype, generator=generator)
        times = (torch.randint(1000, (batch,), device=x.device, generator=generator)
                 if model.kind == "ddpm" else torch.rand(batch, device=x.device, dtype=x.dtype, generator=generator))
        loss = flow._loss(model, real, labels, noise, times)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        running += loss.detach()
        if step % 100 == 0 or step == steps:
            count = 100 if step % 100 == 0 else step % 100
            training_history.append({"step": step, "mse": float(running / count)})
            running.zero_()
        if step in checkpoints:
            model.eval()
            with torch.no_grad():
                validation = float(flow._loss(model, vx, vy, model.validation_noise, model.validation_times))
            if not math.isfinite(validation):
                raise FloatingPointError(f"{model.kind} validation MSE became nonfinite at update {step}")
            validation_history.append({"step": step, "mse": validation})
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, folder / f"step{step}.pt")
            if validation < best_loss:
                best_loss, best_step = validation, step
                best_state = copy.deepcopy(model.net.state_dict())
            model.train()
        if step in endpoints:
            flow._synchronize(x.device)
            boundary = time.perf_counter()
            endpoint_fit_seconds[str(step)] = boundary - fit_started - audit_overhead
            endpoint_rng_state_hex[str(step)] = generator.get_state().cpu().numpy().tobytes().hex()
            # Infinite FM clipping buffers are intentional; check network parameters only.
            if not all(bool(torch.isfinite(p).all()) for p in model.net.parameters()):
                raise FloatingPointError(f"{model.kind} network parameters became nonfinite at update {step}")
            audit_overhead += time.perf_counter() - boundary
    flow._synchronize(x.device)
    elapsed = time.perf_counter() - started - audit_overhead
    model.net.load_state_dict(best_state)
    model.eval()
    return {"selected_step": best_step, "selected_validation_mse": best_loss,
            "validation_scope": "budgeted real profiling validation",
            "training_history": training_history, "validation_history": validation_history,
            "training_steps": steps, "training_nfe": steps,
            "validation_nfe": len(checkpoints), "training_example_evaluations": steps * batch,
            "validation_example_evaluations": len(vx) * len(checkpoints), "training_seconds": elapsed,
            "rng_seed": seed, "validation_seed": seed + 1, "rng_initial_state_hex": initial_rng,
            "rng_final_state_hex": generator.get_state().cpu().numpy().tobytes().hex(),
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "optimizer": "AdamW", "lr": lr, "weight_decay": .0001, "batch": batch,
            "endpoint_fit_seconds": endpoint_fit_seconds,
            "endpoint_rng_state_hex": endpoint_rng_state_hex,
            "audit_overhead_seconds": audit_overhead}

