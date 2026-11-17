"""Recorded five-fold HW Gaussian selection."""
import math
from types import SimpleNamespace
import torch
from . import gaussian, classifier, aes
CV_FOLDS=5
RIDGES=(.001,.01,.1)
SHRINKAGES=(.01,.1,.5,.9,1.)
def require(ok,msg):
    if not ok: raise ValueError(msg)

@torch.no_grad()
def tune_cv(common, x, y):
    mods = common.modules(); models, core, stats = mods['models'], mods['core'], mods['budget'].stats
    fold = torch.arange(len(x), device=x.device) % CV_FOLDS
    reps = torch.tensor([(1 << h) - 1 for h in range(9)], device=x.device)
    hw = core.hw_table(x.device)
    eye = torch.eye(x.shape[1], device=x.device, dtype=torch.float64)
    total = torch.zeros((len(RIDGES), len(SHRINKAGES)), device=x.device, dtype=torch.float64)
    for k in range(CV_FOLDS):
        train, held = fold != k, fold == k
        xt, yt, xv, yv = x[train], y[train], x[held], y[held]
        _, hw_prior = core.priors(yt)
        target = hw[yv]
        for i, ridge in enumerate(RIDGES):
            state = models.fit_gaussian(xt, yt, 'HW', ridge)
            residual = xt.double() - state['features'][yt] @ state['coefficients']
            residual -= residual.mean(0)
            sample_cov = residual.T @ residual / (len(xt) - 1)
            tau = sample_cov.trace().div(x.shape[1]).clamp_min(1e-6)
            means = state['means'][reps]
            for j, s in enumerate(SHRINKAGES):
                cov = state['cov'] if s == .1 else (1 - s) * sample_cov + s * tau * eye + 1e-6 * tau * eye
                logits = stats.gaussian_scores(xv.double(), means, cov) + hw_prior.log()
                total[i, j] += -logits.log_softmax(1).gather(1, target[:, None]).sum()
    nll = (total / len(x)).cpu().tolist()
    best, choice = float('inf'), None
    for i, ridge in enumerate(RIDGES):
        for j, s in enumerate(SHRINKAGES):
            require(math.isfinite(nll[i][j]), 'nonfinite CV NLL')
            if nll[i][j] < best:
                best, choice = nll[i][j], (ridge, s)
    selected = rq.refit_template(common, x, y, choice[0], choice[1])
    return selected, {'folds': CV_FOLDS, 'fold_rule': 'arange(n)%5 within fitting rows',
                      'ridge': choice[0], 'shrinkage': choice[1], 'cv_hw_nll': best,
                      'grid': [{'ridge': r, 'shrinkage': s, 'cv_hw_nll': nll[i][j]}
                               for i, r in enumerate(RIDGES) for j, s in enumerate(SHRINKAGES)],
                      'validation_rows_used': 0}



@torch.no_grad()
def refit_template(common, x, y, ridge, shrinkage):
    state = common.modules()['models'].fit_gaussian(x, y, 'HW', ridge)
    if shrinkage != .1:
        residual = x.double() - state['features'][y] @ state['coefficients']
        residual -= residual.mean(0)
        cov = residual.T @ residual / (len(x) - 1)
        tau = cov.trace().div(x.shape[1]).clamp_min(1e-6)
        eye = torch.eye(x.shape[1], device=x.device, dtype=torch.float64)
        state['cov'] = (1 - shrinkage) * cov + shrinkage * tau * eye + 1e-6 * tau * eye
    return {**state, 'shrinkage': shrinkage}


rq=SimpleNamespace(refit_template=refit_template)
common=SimpleNamespace(modules=lambda: {"models":gaussian,"core":classifier,"budget":SimpleNamespace(stats=aes)})
