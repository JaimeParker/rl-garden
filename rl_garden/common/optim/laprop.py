"""LaProp optimizer.

MIT License
Source: https://github.com/Z-T-WANG/LaProp-Optimizer
Copyright (c) 2020 Wang, T. Zhikang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Ported from ``3rd_party/r2dreamer/optim/laprop.py`` (plan model-based-base
Part 2 / decision 1: DreamerV3's single unified optimizer over world model +
encoder/decoder + actor + critic, combined here with ``agc.py``'s per-
parameter clipping in place of a global grad-norm clip). LaProp differs from
Adam by normalizing the gradient by its RMS *before* accumulating the
momentum buffer (``exp_avg``) rather than after -- the learning rate is
baked into the momentum buffer itself (see ``step_of_this_grad``/
``exp_avg`` below), which is what makes the very first update step have the
simple closed form ``tests/test_laprop.py`` checks: with all state at zero,
``exp_avg_sq`` after one step is exactly ``(1-beta2) * grad**2``, so
``denom == |grad|`` (its own bias-correction cancels the ``1-beta2``
factor), ``step_of_this_grad == sign(grad)``, ``exp_avg == (1-beta1)*lr*
sign(grad)``, and ``step_size == 1/(1-beta1)`` -- the parameter therefore
moves by exactly ``lr * sign(grad)`` on step 1, independent of ``beta1``.

Rewritten from the upstream file's removed-in-current-PyTorch positional-
scalar overloads (``addcmul_(value, t1, t2)``, ``add_(value, tensor)``) to
the current ``value=``/``alpha=`` keyword forms -- same arithmetic, no
numeric change.
"""
from __future__ import annotations

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class LaProp(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 4e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-15,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
        centered: bool = False,
    ) -> None:
        self.steps_before_using_centered = 10

        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad, centered=centered
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("LaProp does not support sparse gradients")
                amsgrad = group["amsgrad"]
                centered = group["centered"]

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_lr_1"] = 0.0
                    state["exp_avg_lr_2"] = 0.0
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    if centered:
                        state["exp_mean_avg_beta2"] = torch.zeros_like(p)
                    if amsgrad:
                        state["max_exp_avg_sq"] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                if centered:
                    exp_mean_avg_beta2 = state["exp_mean_avg_beta2"]
                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient.
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                state["exp_avg_lr_1"] = state["exp_avg_lr_1"] * beta1 + (1 - beta1) * group["lr"]
                state["exp_avg_lr_2"] = state["exp_avg_lr_2"] * beta2 + (1 - beta2)

                bias_correction1 = (
                    state["exp_avg_lr_1"] / group["lr"] if group["lr"] != 0.0 else 1.0
                )  # 1 - beta1 ** state['step']
                step_size = 1 / bias_correction1

                bias_correction2 = state["exp_avg_lr_2"]
                denom = exp_avg_sq
                if centered:
                    exp_mean_avg_beta2.mul_(beta2).add_(grad, alpha=1 - beta2)
                    if state["step"] > self.steps_before_using_centered:
                        mean = exp_mean_avg_beta2**2
                        denom = denom - mean

                if amsgrad and not (centered and state["step"] <= self.steps_before_using_centered):
                    # Maintains the maximum of all (centered) 2nd moment running avg. till now.
                    torch.max(max_exp_avg_sq, denom, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq

                denom = denom.div(bias_correction2).sqrt_().add_(group["eps"])
                step_of_this_grad = grad / denom
                exp_avg.mul_(beta1).add_(step_of_this_grad, alpha=(1 - beta1) * group["lr"])

                p.add_(exp_avg, alpha=-step_size)
                if group["weight_decay"] != 0:
                    p.add_(p, alpha=-group["weight_decay"])

        return loss


def linear_warmup(optimizer: torch.optim.Optimizer, warmup_steps: int) -> LambdaLR:
    """``LambdaLR`` linearly ramping the LR multiplier from ``0`` to ``1``
    over ``warmup_steps`` optimizer steps, then holding at ``1`` -- r2dreamer
    ``dreamer.py``'s inline ``lr_lambda`` closure (identical in the official
    JAX ``optax.linear_schedule``), factored out here since it's used
    standalone on top of ``LaProp`` rather than through
    ``rl_garden.common.optim.make_lr_scheduler``'s ``"linear_warmup"`` mode
    (that factory's ``warmup_steps=0`` means "no warmup" via early return
    from ``schedule_type="constant"``; this one -- matching r2dreamer's own
    ``if config.warmup: ...`` guard -- treats ``warmup_steps<=0`` as
    "already fully warmed up", the natural default for a call site that
    always wants a scheduler object back, never ``None``).
    """
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")

    def lr_lambda(step: int) -> float:
        if warmup_steps <= 0:
            return 1.0
        return min(1.0, (step + 1) / warmup_steps)

    return LambdaLR(optimizer, lr_lambda=lr_lambda)
