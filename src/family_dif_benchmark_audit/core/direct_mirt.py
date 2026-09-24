"""Direct penalized joint logistic MIRT used in the estimator comparison.

The objective and optimizer match the recorded comparison: mean Bernoulli
loss, factor ridge 0.01, intercept penalty 0.00005, Adam at 0.03, float32,
and objective-plateau stopping. This is joint estimation, not marginal or
Bayesian estimation. PyTorch is an optional dependency.
"""
from __future__ import annotations

import time
import numpy as np


def fit(y, k, device, seed, epochs, ridge=0.01, fixed=None):
    import torch

    torch.manual_seed(seed)
    truth = torch.tensor(y, dtype=torch.float32, device=device)
    n, m = y.shape
    rng = np.random.default_rng(seed)
    if fixed is None:
        b = torch.nn.Parameter(torch.logit((truth.mean(1) * m + .5) / (m + 1)))
        a = torch.nn.Parameter(torch.tensor(
            rng.normal(size=(n, k)).astype('float32') * .1 / k**.25, device=device))
        theta = torch.nn.Parameter(torch.tensor(
            rng.normal(size=(m, k)).astype('float32') * .1 / k**.25, device=device))
        parameters = [a, theta, b]
    else:
        a, b = [torch.tensor(x, dtype=torch.float32, device=device) for x in fixed]
        theta = torch.nn.Parameter(torch.zeros(m, k, device=device))
        parameters = [theta]
    optimizer = torch.optim.Adam(parameters, lr=.03)
    history = []
    stable = 0
    start = time.perf_counter()
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = a @ theta.T + b[:, None]
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, truth)
        penalty = .5 * ridge * theta.square().sum(1).mean()
        if fixed is None:
            penalty = penalty + .5 * ridge * a.square().sum(1).mean() + .00005 * b.square().mean()
        loss = bce + penalty
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite training objective')
        loss.backward()
        grad = float(torch.nn.utils.clip_grad_norm_(parameters, 10).detach().cpu())
        optimizer.step()
        if epoch % 20 == 0 or epoch == epochs - 1:
            value = float(loss.detach().cpu())
            history.append(dict(epoch=epoch, objective=value,
                                bce=float(bce.detach().cpu()), grad_norm=grad))
            stable = stable + 1 if len(history) > 1 and abs(value - history[-2]['objective']) < 1e-6 else 0
            if stable >= 3:
                break
    if device == 'mps':
        torch.mps.synchronize()
    return (a.detach().cpu().numpy(), b.detach().cpu().numpy(),
            theta.detach().cpu().numpy(),
            dict(seconds=time.perf_counter() - start, epochs=epoch + 1,
                 converged=stable >= 3, history=history))
