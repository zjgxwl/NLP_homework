import math
import torch
from torch.optim.optimizer import Optimizer
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union
from torch import Tensor

Params = Union[Iterable[Tensor], Iterable[Dict[str, Any]]] #Union type; Union[X, Y] means either X or Y
LossClosure = Callable[[], float]
OptLossClosure = Optional[LossClosure]
Betas2 = Tuple[float, float]
State = Dict[str, Any]
OptFloat = Optional[float]
Nus2 = Tuple[float, float]
Eps2 = Tuple[float, float]
ParamGroup = Dict[str, Any]

__all__ = ('QHM', 'DiffGrad', 'QHAdam', 'Adafactor', 'Adamax',)


# QHM+NAG, QHM+NAdam
class QHM(Optimizer):
    GRAD = 'grad'
    DIRECT = 'direct'

    def __init__(
            self,
            params: Params,
            lr: float = 1e-3,
            momentum: float = 0.0,
            nu: float = 0.7,
            weight_decay: float = 0.0,
            weight_decay_type: str = 'grad',
            # 添加
            nesterov=False,
            NAdam=False,
    ) -> None:
        if lr <= 0.0:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if momentum < 0.0:
            raise ValueError('Invalid momentum value: {}'.format(momentum))
        if weight_decay < 0.0:
            raise ValueError(
                'Invalid weight_decay value: {}'.format(weight_decay)
            )
        if weight_decay_type not in (self.GRAD, self.DIRECT):
            _type = weight_decay_type
            msg = 'Invalid weight_decay_type value: {}'.format(_type)
            raise ValueError(msg)

        defaults = {
            'lr': lr,
            'momentum': momentum,
            'nu': nu,
            'weight_decay': weight_decay,
            'weight_decay_type': weight_decay_type,
            # 添加
            "nesterov": nesterov,
            "NAdam": NAdam,
        }
        super(QHM, self).__init__(params, defaults)

    def step(self, closure: OptLossClosure = None) -> OptFloat:
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr, nu, momentum = group['lr'], group['nu'], group['momentum']
            weight_decay, weight_decay_type = (
                group['weight_decay'],
                group['weight_decay_type'],
            )
            # 添加
            nesterov = group["nesterov"]
            NAdam = group["NAdam"]

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state = self.state[p]

                if weight_decay != 0:
                    if weight_decay_type == self.GRAD:
                        d_p.add_(p.data, alpha=weight_decay)
                    else:
                        p.data.mul_(1.0 - lr * weight_decay)

                if len(param_state) == 0:
                    param_state['momentum_buffer'] = torch.zeros_like(
                        p.data, memory_format=torch.preserve_format
                    )

                if nesterov:
                    momentum_buffer = param_state["momentum_buffer"]
                    momentum_buffer.mul_(momentum)
                    temp0 = momentum_buffer.clone()
                    temp1 = d_p.clone().mul_(1.0 - momentum)
                    momentum_buffer.add_(temp1)

                    temp2 = d_p.clone().mul_(1.0 - nu)
                    temp3 = momentum_buffer.clone().mul_(momentum + nu)
                    temp3.add_(temp0, alpha=-1).add_(temp2)

                    p.data.add_(temp3, alpha=-lr)

                elif NAdam:
                    momentum_buffer = param_state["momentum_buffer"]
                    momentum_buffer.mul_(momentum).add_(d_p, alpha=1.0 - momentum)

                    p.data.add_(momentum_buffer, alpha=-lr * nu * momentum)
                    p.data.add_(d_p, alpha=-lr * (1.0 - nu * momentum))

                else:
                    momentum_buffer = param_state['momentum_buffer']
                    momentum_buffer.mul_(momentum).add_(d_p, alpha=1.0 - momentum)

                    p.data.add_(momentum_buffer, alpha=-lr * nu)
                    p.data.add_(d_p, alpha=-lr * (1.0 - nu))

        return loss


# DiffGrad+regress, DiffGrad+Adamax
class DiffGrad(Optimizer):
    def __init__(
            self,
            params: Params,
            lr: float = 1e-3,
            betas: Betas2 = (0.9, 0.999),
            eps: float = 1e-8,
            weight_decay: float = 0.0,
            regress: bool = False,
            Adamax: bool = False,
    ) -> None:
        if lr <= 0.0:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if eps < 0.0:
            raise ValueError('Invalid epsilon value: {}'.format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                'Invalid beta parameter at index 0: {}'.format(betas[0])
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                'Invalid beta parameter at index 1: {}'.format(betas[1])
            )
        if weight_decay < 0.0:
            raise ValueError(
                'Invalid weight_decay value: {}'.format(weight_decay)
            )

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, regress=regress, Adamax=Adamax)
        super(DiffGrad, self).__init__(params, defaults)

    def step(self, closure: OptLossClosure = None) -> OptFloat:
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']

            if group['regress']:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    grad = p.grad.data
                    if grad.is_sparse:
                        msg = (
                            'DiffGrad does not support sparse gradients, '
                            'please consider SparseAdam instead'
                        )
                        raise RuntimeError(msg)

                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        state['step'] = 0
                        state['exp_avg_sq'] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                        # Previous gradient
                        state['previous_grad'] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )

                    exp_avg_sq, previous_grad = (
                        state['exp_avg_sq'],
                        state['previous_grad'],
                    )

                    state['step'] += 1

                    if group['weight_decay'] != 0:
                        grad.add_(p.data, alpha=group['weight_decay'])

                    # Decay the first and second moment running average coefficient
                    # exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                    # bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']

                    # compute diffgrad coefficient (dfc)
                    diff = torch.abs(previous_grad - grad)
                    dfc = torch.div(1.0, (1.0 + torch.exp(-diff)))
                    state['previous_grad'] = grad.clone()

                    # update momentum with dfc
                    # exp_avg1 = exp_avg * dfc
                    exp_avg1 = grad * dfc

                    step_size = (
                            group['lr']
                            * math.sqrt(bias_correction2)
                        # / bias_correction1
                    )

                    p.data.addcdiv_(exp_avg1, denom, value=-step_size)

            elif group['Adamax']:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    grad = p.grad.data
                    if grad.is_sparse:
                        msg = (
                            'DiffGrad does not support sparse gradients, '
                            'please consider SparseAdam instead'
                        )
                        raise RuntimeError(msg)

                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                        # Previous gradient
                        state['previous_grad'] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )

                    exp_avg, exp_avg_sq, previous_grad = (
                        state['exp_avg'],
                        state['exp_avg_sq'],
                        state['previous_grad'],
                    )

                    state['step'] += 1

                    if group['weight_decay'] != 0:
                        grad.add_(p.data, alpha=group['weight_decay'])

                    # Decay the first and second moment running average coefficient
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    # exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    norm_buf = torch.cat([
                        exp_avg_sq.mul_(beta2).unsqueeze(0),
                        grad.abs().add_(group['eps']).unsqueeze_(0)
                    ], 0)
                    torch.amax(norm_buf, 0, keepdim=False, out=exp_avg_sq)
                    # denom = exp_avg_sq.sqrt().add_(group['eps'])

                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']

                    # compute diffgrad coefficient (dfc)
                    diff = torch.abs(previous_grad - grad)
                    dfc = torch.div(1.0, (1.0 + torch.exp(-diff)))
                    state['previous_grad'] = grad.clone()

                    # update momentum with dfc
                    exp_avg1 = exp_avg * dfc

                    step_size = (
                            group['lr']
                            # * math.sqrt(bias_correction2)
                            / bias_correction1
                    )

                    p.data.addcdiv_(exp_avg1, exp_avg_sq, value=-step_size)

            else:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    grad = p.grad.data
                    if grad.is_sparse:
                        msg = (
                            'DiffGrad does not support sparse gradients, '
                            'please consider SparseAdam instead'
                        )
                        raise RuntimeError(msg)

                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                        # Previous gradient
                        state['previous_grad'] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )

                    exp_avg, exp_avg_sq, previous_grad = (
                        state['exp_avg'],
                        state['exp_avg_sq'],
                        state['previous_grad'],
                    )

                    state['step'] += 1

                    if group['weight_decay'] != 0:
                        grad.add_(p.data, alpha=group['weight_decay'])

                    # Decay the first and second moment running average coefficient
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']

                    # compute diffgrad coefficient (dfc)
                    diff = torch.abs(previous_grad - grad)
                    dfc = torch.div(1.0, (1.0 + torch.exp(-diff)))
                    state['previous_grad'] = grad.clone()

                    # update momentum with dfc
                    exp_avg1 = exp_avg * dfc

                    step_size = (
                            group['lr']
                            * math.sqrt(bias_correction2)
                            / bias_correction1
                    )

                    p.data.addcdiv_(exp_avg1, denom, value=-step_size)

        return loss


# QHAdam+NAG, QHAdam+NAdam
class QHAdam(Optimizer):
    def __init__(
            self,
            params: Params,
            lr: float = 1e-3,
            betas: Betas2 = (0.9, 0.999),
            nus: Nus2 = (1.0, 1.0),
            weight_decay: float = 0.0,
            decouple_weight_decay: bool = False,
            eps: float = 1e-8,
            # 添加
            nesterov=False,
            NAdam=False,
            exp_avg=None,
            exp_avg_sq=None,
            params1=None,
            exp_avgs=None,
    ):
        if lr <= 0.0:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if eps < 0.0:
            raise ValueError('Invalid epsilon value: {}'.format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                'Invalid beta parameter at index 0: {}'.format(betas[0])
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                'Invalid beta parameter at index 1: {}'.format(betas[1])
            )
        if weight_decay < 0:
            raise ValueError(
                'Invalid weight_decay value: {}'.format(weight_decay)
            )

        defaults = {
            'lr': lr,
            'betas': betas,
            'nus': nus,
            'weight_decay': weight_decay,
            'decouple_weight_decay': decouple_weight_decay,
            'eps': eps,
            # 添加
            "nesterov": nesterov,
            "NAdam": NAdam,
            "exp_avg": exp_avg,
            "exp_avg_sq": exp_avg_sq,
            "params1": params1,
            "exp_avgs": exp_avgs,
        }
        super(QHAdam, self).__init__(params, defaults)

    def step(self, closure: OptLossClosure = None) -> OptFloat:
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            nu1, nu2 = group['nus']
            weight_decay = group['weight_decay']
            decouple_weight_decay = group['decouple_weight_decay']
            eps = group['eps']
            # 添加
            nesterov = group["nesterov"]
            NAdam = group["NAdam"]
            exp_avg = group["exp_avg"]
            exp_avg_sq = group["exp_avg_sq"]
            exp_avgs = group["exp_avgs"] = []

            if nesterov:
                for p, p1 in zip(group["params"], group["params1"]):
                    if p.grad is None:
                        continue

                    d_p = p.grad.data
                    # 添加
                    if p1.grad is None:
                        d_p1 = p.grad.data
                    else:
                        d_p1 = p1.grad.data
                    if d_p.is_sparse:
                        raise RuntimeError(
                            'QHAdam does not support sparse gradients, '
                            'please consider SparseAdam instead'
                        )

                    state = self.state[p]

                    if weight_decay != 0:
                        if decouple_weight_decay:
                            p.data.mul_(1 - lr * weight_decay)
                            # 添加
                            p1.data.mul_(1 - lr * weight_decay)
                        else:
                            d_p.add_(p.data, alpha=weight_decay)
                            # 添加
                            d_p1.add_(p1.data, alpha=weight_decay)

                    d_p_sq = d_p.mul(d_p)

                    if len(state) == 0:
                        state['beta1_weight'] = 0.0
                        state['beta2_weight'] = 0.0
                        state['exp_avg'] = torch.zeros_like(
                            p.data, memory_format=torch.preserve_format
                        )
                        state['exp_avg_sq'] = torch.zeros_like(
                            p.data, memory_format=torch.preserve_format
                        )

                    # 添加
                    exp_avgs.append(state['exp_avg'])

                    state['beta1_weight'] = 1.0 + beta1 * state['beta1_weight']
                    state['beta2_weight'] = 1.0 + beta2 * state['beta2_weight']

                    beta1_weight = state['beta1_weight']
                    beta2_weight = state['beta2_weight']
                    exp_avg = state['exp_avg']
                    exp_avg_sq = state['exp_avg_sq']

                    beta1_adj = 1.0 - (1.0 / beta1_weight)
                    beta2_adj = 1.0 - (1.0 / beta2_weight)
                    # 修改
                    exp_avg.mul_(beta1_adj).add_(d_p1, alpha=1.0 - beta1_adj)
                    exp_avg_sq.mul_(beta2_adj).add_(d_p_sq, alpha=1.0 - beta2_adj)

                    avg_grad = exp_avg.mul(nu1)
                    if nu1 != 1.0:
                        # 修改
                        avg_grad.add_(d_p1, alpha=1.0 - nu1)

                    avg_grad_rms = exp_avg_sq.mul(nu2)
                    if nu2 != 1.0:
                        avg_grad_rms.add_(d_p_sq, alpha=1.0 - nu2)
                    avg_grad_rms.sqrt_()
                    if eps != 0.0:
                        avg_grad_rms.add_(eps)

                    p.data.addcdiv_(avg_grad, avg_grad_rms, value=-lr)

            elif NAdam:
                for p in group['params']:
                    if p.grad is None:
                        continue

                    d_p = p.grad.data
                    if d_p.is_sparse:
                        raise RuntimeError(
                            'QHAdam does not support sparse gradients, '
                            'please consider SparseAdam instead'
                        )

                    state = self.state[p]

                    if weight_decay != 0:
                        if decouple_weight_decay:
                            p.data.mul_(1 - lr * weight_decay)
                        else:
                            d_p.add_(p.data, alpha=weight_decay)

                    d_p_sq = d_p.mul(d_p)

                    if len(state) == 0:
                        state['beta1_weight'] = 0.0
                        state['beta2_weight'] = 0.0
                        state['exp_avg'] = torch.zeros_like(
                            p.data, memory_format=torch.preserve_format
                        )
                        state['exp_avg_sq'] = torch.zeros_like(
                            p.data, memory_format=torch.preserve_format
                        )

                    state['beta1_weight'] = 1.0 + beta1 * state['beta1_weight']
                    state['beta2_weight'] = 1.0 + beta2 * state['beta2_weight']

                    beta1_weight = state['beta1_weight']
                    beta2_weight = state['beta2_weight']
                    exp_avg = state['exp_avg']
                    exp_avg_sq = state['exp_avg_sq']

                    beta1_adj = 1.0 - (1.0 / beta1_weight)
                    beta2_adj = 1.0 - (1.0 / beta2_weight)
                    exp_avg.mul_(beta1_adj).add_(d_p, alpha=1.0 - beta1_adj)
                    exp_avg_sq.mul_(beta2_adj).add_(d_p_sq, alpha=1.0 - beta2_adj)

                    avg_grad = exp_avg.mul(nu1)
                    if nu1 != 1.0:
                        # 修改
                        avg_grad.add_(d_p, alpha=1.0 - nu1 * beta1_adj)

                    # 修改
                    avg_grad_rms = exp_avg_sq.clone()
                    # 删除
                    avg_grad_rms.sqrt_()
                    if eps != 0.0:
                        avg_grad_rms.add_(eps)

                    p.data.addcdiv_(avg_grad, avg_grad_rms, value=-lr)

            else:
                for p in group['params']:
                    if p.grad is None:
                        continue

                    d_p = p.grad.data
                    if d_p.is_sparse:
                        raise RuntimeError(
                            'QHAdam does not support sparse gradients, '
                            'please consider SparseAdam instead'
                        )

                    state = self.state[p]

                    if weight_decay != 0:
                        if decouple_weight_decay:
                            p.data.mul_(1 - lr * weight_decay)
                        else:
                            d_p.add_(p.data, alpha=weight_decay)

                    d_p_sq = d_p.mul(d_p)

                    if len(state) == 0:
                        state['beta1_weight'] = 0.0
                        state['beta2_weight'] = 0.0
                        state['exp_avg'] = torch.zeros_like(
                            p.data, memory_format=torch.preserve_format
                        )
                        state['exp_avg_sq'] = torch.zeros_like(
                            p.data, memory_format=torch.preserve_format
                        )

                    state['beta1_weight'] = 1.0 + beta1 * state['beta1_weight']
                    state['beta2_weight'] = 1.0 + beta2 * state['beta2_weight']

                    beta1_weight = state['beta1_weight']
                    beta2_weight = state['beta2_weight']
                    exp_avg = state['exp_avg']
                    exp_avg_sq = state['exp_avg_sq']

                    beta1_adj = 1.0 - (1.0 / beta1_weight)
                    beta2_adj = 1.0 - (1.0 / beta2_weight)
                    exp_avg.mul_(beta1_adj).add_(d_p, alpha=1.0 - beta1_adj)
                    exp_avg_sq.mul_(beta2_adj).add_(d_p_sq, alpha=1.0 - beta2_adj)

                    avg_grad = exp_avg.mul(nu1)
                    if nu1 != 1.0:
                        avg_grad.add_(d_p, alpha=1.0 - nu1)

                    avg_grad_rms = exp_avg_sq.mul(nu2)
                    if nu2 != 1.0:
                        avg_grad_rms.add_(d_p_sq, alpha=1.0 - nu2)
                    avg_grad_rms.sqrt_()
                    if eps != 0.0:
                        avg_grad_rms.add_(eps)

                    p.data.addcdiv_(avg_grad, avg_grad_rms, value=-lr)

            group["params1"] = None
            group["exp_avg"] = None
            group["exp_avg_sq"] = None
            group["exp_avgs"] = None

        return loss


# Adafactor+DiffGrad1, Adafactor+DiffGrad2
class Adafactor(Optimizer):
    def __init__(
            self,
            params: Params,
            lr: OptFloat = None,
            eps2: Eps2 = (1e-30, 1e-3),
            clip_threshold: float = 1.0,
            decay_rate: float = -0.8,
            beta1: OptFloat = None,
            weight_decay: float = 0.0,
            scale_parameter: bool = True,
            relative_step: bool = True,
            warmup_init: bool = False,
            diff1: bool = False,
            diff2: bool = False,
            beta2: float = 0.9
    ):
        if lr is not None and lr <= 0.0:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if weight_decay < 0.0:
            raise ValueError(
                'Invalid weight_decay value: {}'.format(weight_decay)
            )

        defaults = dict(
            lr=lr,
            eps2=eps2,
            clip_threshold=clip_threshold,
            decay_rate=decay_rate,
            beta1=beta1,
            weight_decay=weight_decay,
            scale_parameter=scale_parameter,
            relative_step=relative_step,
            warmup_init=warmup_init,
            # 添加
            diff1=diff1,
            diff2=diff2,
            beta2=beta2
        )
        super(Adafactor, self).__init__(params, defaults)

    def _get_lr(self, param_group: ParamGroup, param_state: State) -> float:
        rel_step_sz = param_group['lr']
        if param_group['relative_step']:
            min_step = (
                1e-6 * param_state['step']
                if param_group['warmup_init']
                else 1e-2
            )
            rel_step_sz = min(min_step, 1.0 / math.sqrt(param_state['step']))
        param_scale = 1.0
        if param_group['scale_parameter']:
            param_scale = max(param_group['eps2'][1], param_state['RMS'])
        return param_scale * rel_step_sz

    def _get_options(
            self, param_group: ParamGroup, param_shape: Tuple[int, ...]
    ) -> Tuple[bool, bool]:
        factored = len(param_shape) >= 2
        use_first_moment = param_group['beta1'] is not None
        return factored, use_first_moment

    def _rms(self, tensor: torch.Tensor) -> float:
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    def _approx_sq_grad(
            self,
            exp_avg_sq_row: torch.Tensor,
            exp_avg_sq_col: torch.Tensor,
            output: torch.Tensor,
    ) -> None:
        r_factor = (
            (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1))
            .rsqrt_()
            .unsqueeze(-1)
        )
        c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        torch.mul(r_factor, c_factor, out=output)

    def step(self, closure: OptLossClosure = None) -> OptFloat:
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            if group['diff1']:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    grad = p.grad.data
                    if grad.is_sparse:
                        raise RuntimeError(
                            'Adafactor does not support sparse gradients.'
                        )

                    state = self.state[p]
                    grad_shape = grad.shape

                    factored, use_first_moment = self._get_options(
                        group, grad_shape
                    )
                    # State Initialization
                    if len(state) == 0:
                        state['step'] = 0

                        if use_first_moment:
                            # Exponential moving average of gradient values
                            state['exp_avg'] = torch.zeros_like(
                                grad, memory_format=torch.preserve_format
                            )
                        if factored:
                            state['exp_avg_sq_row'] = torch.zeros(
                                grad_shape[:-1]
                            ).type_as(grad)
                            state['exp_avg_sq_col'] = torch.zeros(
                                grad_shape[:-2] + grad_shape[-1:]
                            ).type_as(grad)
                        else:
                            state['exp_avg_sq'] = torch.zeros_like(
                                grad, memory_format=torch.preserve_format
                            )
                        # Previous gradient
                        state['previous_grad'] = torch.zeros_like(
                            grad, memory_format=torch.preserve_format
                        )
                        state['g_diff'] = torch.zeros_like(
                            grad, memory_format=torch.preserve_format
                        )

                        state['RMS'] = 0

                    state['step'] += 1
                    state['RMS'] = self._rms(p.data)
                    lr = self._get_lr(group, state)

                    previous_grad = state['previous_grad']
                    diff = torch.abs(previous_grad - grad)
                    dfc = torch.div(1.0, (1.0 + torch.exp(-diff)))
                    state['previous_grad'] = grad.clone()
                    state['g_diff'].mul_(group['beta2']).add_(grad, alpha=1 - group['beta2'])
                    bias_correction2 = 1 - group['beta2'] ** state['step']

                    beta2t = 1.0 - math.pow(state['step'], group['decay_rate'])
                    update = (grad ** 2) + group['eps2'][0]
                    if factored:
                        exp_avg_sq_row = state['exp_avg_sq_row']
                        exp_avg_sq_col = state['exp_avg_sq_col']

                        exp_avg_sq_row.mul_(beta2t).add_(
                            update.mean(dim=-1), alpha=1.0 - beta2t
                        )
                        exp_avg_sq_col.mul_(beta2t).add_(
                            update.mean(dim=-2), alpha=1.0 - beta2t
                        )

                        # Approximation of exponential moving average of square
                        # of gradient
                        self._approx_sq_grad(
                            exp_avg_sq_row, exp_avg_sq_col, update
                        )
                        update.mul_(dfc).mul_(state['g_diff'])
                    else:
                        exp_avg_sq = state['exp_avg_sq']

                        exp_avg_sq.mul_(beta2t).add_(update, alpha=1.0 - beta2t)
                        torch.rsqrt(exp_avg_sq, out=update).mul_(dfc).mul_(state['g_diff'])

                    update.div_(
                        max(1.0, self._rms(update) / group['clip_threshold'])
                    )
                    update.mul_(lr)
                    update.div_(bias_correction2)

                    if use_first_moment:
                        exp_avg = state['exp_avg']
                        exp_avg.mul_(group['beta1']).add_(
                            update, alpha=1 - group['beta1']
                        )
                        update = exp_avg

                    if group['weight_decay'] != 0:
                        p.data.add_(p.data, alpha=-group['weight_decay'] * lr)

                    p.data.add_(-update)

            elif group['diff2']:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    grad = p.grad.data
                    if grad.is_sparse:
                        raise RuntimeError(
                            'Adafactor does not support sparse gradients.'
                        )

                    state = self.state[p]
                    grad_shape = grad.shape

                    factored, use_first_moment = self._get_options(
                        group, grad_shape
                    )
                    # State Initialization
                    if len(state) == 0:
                        state['step'] = 0

                        if use_first_moment:
                            # Exponential moving average of gradient values
                            state['exp_avg'] = torch.zeros_like(
                                grad, memory_format=torch.preserve_format
                            )
                        if factored:
                            state['exp_avg_sq_row'] = torch.zeros(
                                grad_shape[:-1]
                            ).type_as(grad)
                            state['exp_avg_sq_col'] = torch.zeros(
                                grad_shape[:-2] + grad_shape[-1:]
                            ).type_as(grad)
                        else:
                            state['exp_avg_sq'] = torch.zeros_like(
                                grad, memory_format=torch.preserve_format
                            )
                        # Previous gradient
                        state['previous_grad'] = torch.zeros_like(
                            grad, memory_format=torch.preserve_format
                        )

                        state['RMS'] = 0

                    state['step'] += 1
                    state['RMS'] = self._rms(p.data)
                    lr = self._get_lr(group, state)

                    previous_grad = state['previous_grad']
                    diff = torch.abs(previous_grad - grad)
                    dfc = torch.div(1.0, (1.0 + torch.exp(-diff)))
                    state['previous_grad'] = grad.clone()

                    beta2t = 1.0 - math.pow(state['step'], group['decay_rate'])
                    update = (grad ** 2) + group['eps2'][0]
                    if factored:
                        exp_avg_sq_row = state['exp_avg_sq_row']
                        exp_avg_sq_col = state['exp_avg_sq_col']

                        exp_avg_sq_row.mul_(beta2t).add_(
                            update.mean(dim=-1), alpha=1.0 - beta2t
                        )
                        exp_avg_sq_col.mul_(beta2t).add_(
                            update.mean(dim=-2), alpha=1.0 - beta2t
                        )

                        # Approximation of exponential moving average of square
                        # of gradient
                        self._approx_sq_grad(
                            exp_avg_sq_row, exp_avg_sq_col, update
                        )
                        update.mul_(grad)
                    else:
                        exp_avg_sq = state['exp_avg_sq']

                        exp_avg_sq.mul_(beta2t).add_(update, alpha=1.0 - beta2t)
                        torch.rsqrt(exp_avg_sq, out=update).mul_(grad)

                    update.div_(
                        max(1.0, self._rms(update) / group['clip_threshold'])
                    )
                    update.mul_(lr)

                    if use_first_moment:
                        exp_avg = state['exp_avg']
                        exp_avg.mul_(group['beta1']).add_(
                            update, alpha=1 - group['beta1']
                        )
                        update = exp_avg

                    if group['weight_decay'] != 0:
                        p.data.add_(p.data, alpha=-group['weight_decay'] * lr)

                    update.mul_(dfc)

                    p.data.add_(-update)
            else:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    grad = p.grad.data
                    if grad.is_sparse:
                        raise RuntimeError(
                            'Adafactor does not support sparse gradients.'
                        )

                    state = self.state[p]
                    grad_shape = grad.shape

                    factored, use_first_moment = self._get_options(
                        group, grad_shape
                    )
                    # State Initialization
                    if len(state) == 0:
                        state['step'] = 0

                        if use_first_moment:
                            # Exponential moving average of gradient values
                            state['exp_avg'] = torch.zeros_like(
                                grad, memory_format=torch.preserve_format
                            )
                        if factored:
                            state['exp_avg_sq_row'] = torch.zeros(
                                grad_shape[:-1]
                            ).type_as(grad)
                            state['exp_avg_sq_col'] = torch.zeros(
                                grad_shape[:-2] + grad_shape[-1:]
                            ).type_as(grad)
                        else:
                            state['exp_avg_sq'] = torch.zeros_like(
                                grad, memory_format=torch.preserve_format
                            )

                        state['RMS'] = 0

                    state['step'] += 1
                    state['RMS'] = self._rms(p.data)
                    lr = self._get_lr(group, state)

                    beta2t = 1.0 - math.pow(state['step'], group['decay_rate'])
                    update = (grad ** 2) + group['eps2'][0]
                    if factored:
                        exp_avg_sq_row = state['exp_avg_sq_row']
                        exp_avg_sq_col = state['exp_avg_sq_col']

                        exp_avg_sq_row.mul_(beta2t).add_(
                            update.mean(dim=-1), alpha=1.0 - beta2t
                        )
                        exp_avg_sq_col.mul_(beta2t).add_(
                            update.mean(dim=-2), alpha=1.0 - beta2t
                        )

                        # Approximation of exponential moving average of square
                        # of gradient
                        self._approx_sq_grad(
                            exp_avg_sq_row, exp_avg_sq_col, update
                        )
                        update.mul_(grad)
                    else:
                        exp_avg_sq = state['exp_avg_sq']

                        exp_avg_sq.mul_(beta2t).add_(update, alpha=1.0 - beta2t)
                        torch.rsqrt(exp_avg_sq, out=update).mul_(grad)

                    update.div_(
                        max(1.0, self._rms(update) / group['clip_threshold'])
                    )
                    update.mul_(lr)

                    if use_first_moment:
                        exp_avg = state['exp_avg']
                        exp_avg.mul_(group['beta1']).add_(
                            update, alpha=1 - group['beta1']
                        )
                        update = exp_avg

                    if group['weight_decay'] != 0:
                        p.data.add_(p.data, alpha=-group['weight_decay'] * lr)

                    p.data.add_(-update)

        return loss


# Adamax+NAG
class Adamax(Optimizer):
    def __init__(self, params, lr=2e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, foreach: Optional[bool] = None, *, maximize: bool = False,
                 # 添加
                 exp_avgs=None, params1=None, nesterov=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        foreach=foreach, maximize=maximize,
                        # 添加
                        exp_avgs=exp_avgs, params1=params1, nesterov=nesterov, Adamax=Adamax)
        super(Adamax, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('foreach', None)
            group.setdefault('maximize', False)
            # 添加
            group.setdefault('nesterov', False)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(state_values[0]['step'])
        if not step_is_tensor:
            for s in state_values:
                s['step'] = torch.tensor(float(s['step']))

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = group["exp_avgs"] = []
            exp_infs = []
            state_steps = []

            beta1, beta2 = group['betas']
            eps = group['eps']
            lr = group['lr']
            weight_decay = group['weight_decay']
            foreach = group['foreach']
            maximize = group['maximize']
            # 添加
            params1_with_grad = []
            nesterov = group['nesterov']
            grads1 = []

            if nesterov:
                for p, p1 in zip(group['params'], group['params1']):
                    if p.grad is None:
                        continue
                    params_with_grad.append(p)
                    params1_with_grad.append(p1)

                    if p.grad.is_sparse:
                        raise RuntimeError('Adamax does not support sparse gradients')
                    grads.append(p.grad)
                    grads1.append(p1.grad)

                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        state['step'] = torch.tensor(0.)
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['exp_inf'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_infs.append(state['exp_inf'])
                    state_steps.append(state['step'])

            else:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('Adamax does not support sparse gradients')
                    grads.append(p.grad)

                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        state['step'] = torch.tensor(0.)
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['exp_inf'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_infs.append(state['exp_inf'])
                    state_steps.append(state['step'])

            if not all(isinstance(t, torch.Tensor) for t in state_steps):
                raise RuntimeError("API has changed, `state_steps` argument must contain a list of singleton tensors")

            if foreach is None:
                # Placeholder for more complex foreach logic to be added when value is not set
                foreach = False

            if foreach and torch.jit.is_scripting():
                raise RuntimeError('torch.jit.script not supported with foreach optimizers')

            if nesterov:
                i = 0
                for param, param1 in zip(params_with_grad, params1_with_grad):
                    grad = grads[i]
                    grad = grad if not maximize else -grad
                    exp_avg = exp_avgs[i]
                    exp_inf = exp_infs[i]
                    step_t = state_steps[i]
                    # update step
                    step_t += 1
                    step = step_t.item()
                    # 添加
                    grad1 = grads1[i] if not maximize else -grads1[i]
                    if grad1 is None:
                        grad1 = grad

                    if weight_decay != 0:
                        grad = grad.add(param, alpha=weight_decay)

                    if torch.is_complex(param):
                        param = torch.view_as_real(param)
                        grad = torch.view_as_real(grad)
                        exp_avg = torch.view_as_real(exp_avg)
                        exp_inf = torch.view_as_real(exp_inf)

                    # Update biased first moment estimate.
                    exp_avg.mul_(beta1).add_(grad1, alpha=1 - beta1)
                    # Update the exponentially weighted infinity norm.
                    norm_buf = torch.cat([
                        exp_inf.mul_(beta2).unsqueeze(0),
                        grad.abs().add_(eps).unsqueeze_(0)
                    ], 0)
                    torch.amax(norm_buf, 0, keepdim=False, out=exp_inf)

                    bias_correction = 1 - beta1 ** step
                    clr = lr / bias_correction

                    param.addcdiv_(exp_avg, exp_inf, value=-clr)

                    i += 1

            else:
                for i, param in enumerate(params_with_grad):
                    grad = grads[i]
                    grad = grad if not maximize else -grad
                    exp_avg = exp_avgs[i]
                    exp_inf = exp_infs[i]
                    step_t = state_steps[i]
                    # update step
                    step_t += 1
                    step = step_t.item()

                    if weight_decay != 0:
                        grad = grad.add(param, alpha=weight_decay)

                    if torch.is_complex(param):
                        param = torch.view_as_real(param)
                        grad = torch.view_as_real(grad)
                        exp_avg = torch.view_as_real(exp_avg)
                        exp_inf = torch.view_as_real(exp_inf)

                    # Update biased first moment estimate.
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    # Update the exponentially weighted infinity norm.
                    norm_buf = torch.cat([
                        exp_inf.mul_(beta2).unsqueeze(0),
                        grad.abs().add_(eps).unsqueeze_(0)
                    ], 0)
                    torch.amax(norm_buf, 0, keepdim=False, out=exp_inf)

                    bias_correction = 1 - beta1 ** step
                    clr = lr / bias_correction

                    param.addcdiv_(exp_avg, exp_inf, value=-clr)

            group["exp_avgs"] = None

        return loss


# AdamW+NAG, AdamW+Adamax, AdamW+Adamax+NAG
class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False, *, maximize: bool = False,
                 foreach: Optional[bool] = None,
                 capturable: bool = False,
                 # 添加
                 exp_avgs=None, params1=None, nesterov=False, Adamax=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad,
                        foreach=foreach, maximize=maximize, capturable=capturable,
                        # 添加
                        exp_avgs=exp_avgs, params1=params1, nesterov=nesterov, Adamax=Adamax)
        super(AdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('maximize', False)
            group.setdefault('foreach', None)
            group.setdefault('capturable', False)
            # 添加
            group.setdefault('nesterov', False)
            group.setdefault('Adamax', False)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(state_values[0]['step'])
        if not step_is_tensor:
            for s in state_values:
                s['step'] = torch.tensor(float(s['step']))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = group["exp_avgs"] = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            amsgrad = group['amsgrad']
            beta1, beta2 = group['betas']
            # 添加
            params1_with_grad = []
            nesterov = group['nesterov']
            Adamax = group['Adamax']
            grads1 = []
            maximize = group['maximize']
            foreach = group['foreach']
            capturable = group['capturable']
            eps = group['eps']
            lr = group['lr']
            weight_decay = group['weight_decay']

            if nesterov:
                for p, p1 in zip(group['params'], group['params1']):
                    if p.grad is None:
                        continue
                    params_with_grad.append(p)
                    params1_with_grad.append(p1)

                    if p.grad.is_sparse:
                        raise RuntimeError('AdamW does not support sparse gradients')
                    grads.append(p.grad)
                    grads1.append(p1.grad)

                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        state['step'] = torch.zeros((1,), dtype=torch.float, device=p.device) \
                            if self.defaults['capturable'] else torch.tensor(0.)
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if amsgrad:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])

                    if amsgrad:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    state_steps.append(state['step'])

            else:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    params_with_grad.append(p)

                    if p.grad.is_sparse:
                        raise RuntimeError('AdamW does not support sparse gradients')
                    grads.append(p.grad)

                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        state['step'] = torch.zeros((1,), dtype=torch.float, device=p.device) \
                            if self.defaults['capturable'] else torch.tensor(0.)
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if amsgrad:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])

                    if amsgrad:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    state_steps.append(state['step'])

            if not all(isinstance(t, torch.Tensor) for t in state_steps):
                raise RuntimeError("API has changed, `state_steps` argument must contain a list of singleton tensors")

            if foreach is None:
                # Placeholder for more complex foreach logic to be added when value is not set
                foreach = False

            if foreach and torch.jit.is_scripting():
                raise RuntimeError('torch.jit.script not supported with foreach optimizers')

            if nesterov:
                i = 0
                for param, param1 in zip(params_with_grad, params1_with_grad):
                    grad = grads[i] if not maximize else -grads[i]
                    exp_avg = exp_avgs[i]
                    exp_avg_sq = exp_avg_sqs[i]
                    step_t = state_steps[i]
                    # 添加
                    grad1 = grads1[i] if not maximize else -grads1[i]
                    if grad1 is None:
                        grad1 = grad

                    if capturable:
                        assert param.is_cuda and step_t.is_cuda, "If capturable=True, params and state_steps must be CUDA tensors."

                    if torch.is_complex(param):
                        grad = torch.view_as_real(grad)
                        exp_avg = torch.view_as_real(exp_avg)
                        exp_avg_sq = torch.view_as_real(exp_avg_sq)
                        param = torch.view_as_real(param)
                        # 添加
                        grad1 = torch.view_as_real(grad1)
                        param1 = torch.view_as_real(param1)

                    # update step
                    step_t += 1

                    # Perform stepweight decay
                    param.mul_(1 - lr * weight_decay)

                    # Decay the first and second moment running average coefficient
                    exp_avg.mul_(beta1).add_(grad1, alpha=1 - beta1)
                    if Adamax:
                        norm_buf = torch.cat([
                            exp_avg_sq.mul_(beta2).unsqueeze(0),
                            grad.abs().add_(eps).unsqueeze_(0)
                        ], 0)
                        torch.amax(norm_buf, 0, keepdim=False, out=exp_avg_sq)
                    else:
                        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                    if capturable:
                        step = step_t

                        # 1 - beta1 ** step can't be captured in a CUDA graph, even if step is a CUDA tensor
                        # (incurs "RuntimeError: CUDA error: operation not permitted when stream is capturing")
                        bias_correction1 = 1 - torch.pow(beta1, step)
                        bias_correction2 = 1 - torch.pow(beta2, step)

                        step_size = lr / bias_correction1
                        step_size_neg = step_size.neg()

                        bias_correction2_sqrt = bias_correction2.sqrt()

                        if amsgrad:
                            # Maintains the maximum of all 2nd moment running avg. till now
                            torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
                            # Uses the max. for normalizing running avg. of gradient
                            # Folds in (admittedly ugly) 1-elem step_size math here to avoid extra param-set-sized read+write
                            # (can't fold it into addcdiv_ below because addcdiv_ requires value is a Number, not a Tensor)
                            if Adamax:
                                denom = (max_exp_avg_sqs[i] / (bias_correction2 * step_size_neg)).add_(
                                    eps / step_size_neg)
                            else:
                                denom = (max_exp_avg_sqs[i].sqrt() / (bias_correction2_sqrt * step_size_neg)).add_(
                                    eps / step_size_neg)

                        else:
                            if Adamax:
                                denom = (exp_avg_sq / (bias_correction2 * step_size_neg)).add_(eps / step_size_neg)
                            else:
                                denom = (exp_avg_sq.sqrt() / (bias_correction2_sqrt * step_size_neg)).add_(
                                    eps / step_size_neg)

                        param.addcdiv_(exp_avg, denom)
                    else:
                        step = step_t.item()

                        bias_correction1 = 1 - beta1 ** step
                        bias_correction2 = 1 - beta2 ** step

                        step_size = lr / bias_correction1

                        bias_correction2_sqrt = math.sqrt(bias_correction2)

                        if amsgrad:
                            # Maintains the maximum of all 2nd moment running avg. till now
                            torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
                            # Use the max. for normalizing running avg. of gradient
                            if Adamax:
                                denom = (max_exp_avg_sqs[i] / bias_correction2).add_(eps)
                            else:
                                denom = (max_exp_avg_sqs[i].sqrt() / bias_correction2_sqrt).add_(eps)

                        else:
                            if Adamax:
                                denom = (exp_avg_sq / bias_correction2).add_(eps)
                            else:
                                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

                        # temp = exp_avg.clone()
                        # temp.div_(denom).add_(param, alpha=5e-4)
                        # param.add_(temp, alpha=-step_size)
                        param.addcdiv_(exp_avg, denom, value=-step_size)

                    i += 1

            else:
                for i, param in enumerate(params_with_grad):
                    grad = grads[i] if not maximize else -grads[i]
                    exp_avg = exp_avgs[i]
                    exp_avg_sq = exp_avg_sqs[i]
                    step_t = state_steps[i]

                    if capturable:
                        assert param.is_cuda and step_t.is_cuda, "If capturable=True, params and state_steps must be CUDA tensors."

                    if torch.is_complex(param):
                        grad = torch.view_as_real(grad)
                        exp_avg = torch.view_as_real(exp_avg)
                        exp_avg_sq = torch.view_as_real(exp_avg_sq)
                        param = torch.view_as_real(param)

                    # update step
                    step_t += 1

                    # Perform stepweight decay
                    param.mul_(1 - lr * weight_decay)

                    # Decay the first and second moment running average coefficient
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    if Adamax:
                        norm_buf = torch.cat([
                            exp_avg_sq.mul_(beta2).unsqueeze(0),
                            grad.abs().add_(eps).unsqueeze_(0)
                        ], 0)
                        torch.amax(norm_buf, 0, keepdim=False, out=exp_avg_sq)
                    else:
                        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                    if capturable:
                        step = step_t

                        # 1 - beta1 ** step can't be captured in a CUDA graph, even if step is a CUDA tensor
                        # (incurs "RuntimeError: CUDA error: operation not permitted when stream is capturing")
                        bias_correction1 = 1 - torch.pow(beta1, step)
                        bias_correction2 = 1 - torch.pow(beta2, step)

                        step_size = lr / bias_correction1
                        step_size_neg = step_size.neg()

                        bias_correction2_sqrt = bias_correction2.sqrt()

                        if amsgrad:
                            # Maintains the maximum of all 2nd moment running avg. till now
                            torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
                            # Uses the max. for normalizing running avg. of gradient
                            # Folds in (admittedly ugly) 1-elem step_size math here to avoid extra param-set-sized read+write
                            # (can't fold it into addcdiv_ below because addcdiv_ requires value is a Number, not a Tensor)
                            if Adamax:
                                denom = (max_exp_avg_sqs[i] / (bias_correction2 * step_size_neg)).add_(
                                    eps / step_size_neg)
                            else:
                                denom = (max_exp_avg_sqs[i].sqrt() / (bias_correction2_sqrt * step_size_neg)).add_(
                                    eps / step_size_neg)
                        else:
                            if Adamax:
                                denom = (exp_avg_sq / (bias_correction2 * step_size_neg)).add_(eps / step_size_neg)
                            else:
                                denom = (exp_avg_sq.sqrt() / (bias_correction2_sqrt * step_size_neg)).add_(
                                    eps / step_size_neg)

                        param.addcdiv_(exp_avg, denom)
                    else:
                        step = step_t.item()

                        bias_correction1 = 1 - beta1 ** step
                        bias_correction2 = 1 - beta2 ** step

                        step_size = lr / bias_correction1

                        bias_correction2_sqrt = math.sqrt(bias_correction2)

                        if amsgrad:
                            # Maintains the maximum of all 2nd moment running avg. till now
                            torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
                            # Use the max. for normalizing running avg. of gradient
                            if Adamax:
                                denom = (max_exp_avg_sqs[i] / bias_correction2).add_(eps)
                            else:
                                denom = (max_exp_avg_sqs[i].sqrt() / bias_correction2_sqrt).add_(eps)
                        else:
                            if Adamax:
                                denom = (exp_avg_sq / bias_correction2).add_(eps)
                            else:
                                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

                        # temp = exp_avg.clone()
                        # temp.div_(denom).add_(param, alpha=5e-4)
                        # param.add_(temp, alpha=-step_size)
                        param.addcdiv_(exp_avg, denom, value=-step_size)

            group["params1"] = None
            group["exp_avgs"] = None

        return loss

