"""PyTorch implementation of the cited Zaid CNN and Keras-style Adam."""
import math
import numpy as np
import torch
from torch import nn

class KerasBatchNorm1d(nn.Module):
    """Keras defaults, including population running variance (unlike PyTorch BN)."""

    def __init__(self, channels: int):
        super().__init__()
        self.eps = .001
        self.momentum = .99
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.register_buffer('running_mean', torch.zeros(channels))
        self.register_buffer('running_var', torch.ones(channels))

    def forward(self, x):
        if self.training:
            variance, mean = torch.var_mean(x, dim=(0, 2), correction=0)
            with torch.no_grad():
                self.running_mean.mul_(self.momentum).add_(mean.detach(), alpha=1-self.momentum)
                self.running_var.mul_(self.momentum).add_(variance.detach(), alpha=1-self.momentum)
        else:
            mean, variance = self.running_mean, self.running_var
        normalized = (x - mean[None, :, None]) * torch.rsqrt(variance[None, :, None] + self.eps)
        return normalized * self.weight[None, :, None] + self.bias[None, :, None]


class ZaidCNN(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        if input_dim < 2:
            raise ValueError('input_dim must be at least 2 for AveragePooling1D(2)')
        self.input_dim = int(input_dim)
        self.conv = nn.Conv1d(1, 4, 1)
        self.bn = KerasBatchNorm1d(4)
        self.pool = nn.AvgPool1d(2, stride=2)
        self.fc1 = nn.Linear(4 * (input_dim // 2), 10)
        self.fc2 = nn.Linear(10, 10)
        self.output = nn.Linear(10, 256)
        for layer in (self.conv, self.fc1, self.fc2):
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        x = self.pool(self.bn(F.selu(self.conv(x.unsqueeze(1)))))
        # Keras Flatten sees [batch, time, channels], not PyTorch's channel-major layout.
        x = x.transpose(1, 2).flatten(1)
        x = F.selu(self.fc1(x))
        return self.output(F.selu(self.fc2(x)))


class KerasAdam(torch.optim.Optimizer):
    """Legacy Keras Adam places epsilon before variance bias correction."""

    def __init__(self, params, lr=.005, betas=(.9, .999), eps=1e-7):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for parameter in group['params']:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.is_sparse:
                    raise RuntimeError('KerasAdam requires dense gradients')
                state = self.state[parameter]
                if not state:
                    state.update(step=0, exp_avg=torch.zeros_like(parameter),
                                 exp_avg_sq=torch.zeros_like(parameter))
                state['step'] += 1
                first, second = state['exp_avg'], state['exp_avg_sq']
                first.mul_(beta1).add_(gradient, alpha=1-beta1)
                second.mul_(beta2).addcmul_(gradient, gradient, value=1-beta2)
                alpha = group['lr'] * math.sqrt(1-beta2**state['step']) / (1-beta1**state['step'])
                parameter.addcdiv_(first, second.sqrt().add_(group['eps']), value=-alpha)
        return loss


def one_cycle_lr(update_index: int, total_updates: int) -> float:
    """Author callback schedule; the terminal value at total_updates is not applied."""
    if total_updates < 1 or not 0 <= update_index <= total_updates:
        raise ValueError('update_index must lie between zero and positive total_updates')
    # Author callback uses int(0.4*N); use fractional boundary for tiny smoke runs.
    midpoint = max(1, int(.4 * total_updates)) if total_updates >= 3 else .4 * total_updates
    if update_index <= midpoint:
        return .0005 + .0045 * update_index / midpoint
    if update_index <= 2 * midpoint:
        return .005 - .0045 * (update_index-midpoint) / midpoint
    return .0005 - .000495 * (update_index-2*midpoint) / (total_updates-2*midpoint)

from torch.nn import functional as F
from . import classifier
HW=np.array([i.bit_count() for i in range(256)])
def mods():return {"core":classifier}
def require(ok,msg):
    if not ok:raise ValueError(msg)

def build_zaid(seed, device):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed); model = Z.ZaidCNN(32)
        torch.manual_seed(seed + 9001); out = nn.Linear(10, 9)
        nn.init.xavier_uniform_(out.weight); nn.init.zeros_(out.bias)
    model.output = out
    return model.to(device)


def train_zaid(px, py, seed, out, name, device):
    model = build_zaid(seed, device)
    opt = Z.KerasAdam(model.parameters()); rng = np.random.default_rng(seed)
    target = torch.as_tensor(HW, device=device)[py]; total, step = 20000, 0
    for _ in range(50):
        slots = torch.as_tensor(rng.permutation(len(px)), device=device)
        model.train()
        for b in range(0, len(px), 50):
            lr = Z.one_cycle_lr(step, total)
            for g in opt.param_groups:
                g['lr'] = lr
            sel = slots[b:b + 50]
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(px[sel]), target[sel]); loss.backward(); opt.step(); step += 1
    model.eval()
    require(step == total and math.isfinite(float(loss)), 'zaid training')
    _, q = mods()['core'].priors(py)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({'state': {k: v.detach().cpu() for k, v in model.state_dict().items()}, 'HW_prior': q.cpu(),
                'seed': seed, 'kind': 'zaid9', 'updates': step, 'final_loss': float(loss)}, out / (name + '.pt'))


import sys
Z=sys.modules[__name__]
