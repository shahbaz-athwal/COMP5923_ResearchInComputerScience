"""
The Muon optimizer (Keller Jordan et al., 2024), single-GPU version.

Adapted from the reference implementation at github.com/KellerJordan/Muon.

The idea, as I understand it: for each weight matrix, Muon keeps a momentum buffer just
like SGD with momentum. But before applying it, it replaces the momentum matrix with the
closest "orthogonal" matrix, meaning every direction in the update gets the same size.
With plain gradients, a few directions usually dominate the update and the rest barely
move. Orthogonalising evens that out.

Muon is only meant for the 2D weight matrices inside the transformer blocks. Embeddings,
the output head, norm gains and biases stay on AdamW (see build_optimizers in train.py).
"""

import torch

# Coefficients of the quintic Newton-Schulz iteration, taken from the reference implementation.
# They are tuned so that singular values move towards 1 as fast as possible in 5 steps,
# ending up roughly between 0.7 and 1.2 rather than exactly at 1, which is good enough.
NEWTON_SCHULZ_COEFFICIENTS = (3.4445, -4.7750, 2.0315)


def orthogonalize(matrix, steps=5):
    """
    Approximately replace a matrix U S V^T by U V^T, i.e. set all its singular values to 1.

    Doing this exactly would need an SVD, which is slow on a GPU. Newton-Schulz gets close
    using only matrix multiplications, which is why it can run every step.
    """
    a, b, c = NEWTON_SCHULZ_COEFFICIENTS
    result = matrix.bfloat16()
    # Work with the wide orientation so that result @ result.T is the smaller square matrix.
    is_tall = matrix.size(-2) > matrix.size(-1)
    if is_tall:
        result = result.mT
    # The iteration only converges if every singular value starts at 1 or below.
    # Dividing by the Frobenius norm guarantees that.
    result = result / (result.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        gram = result @ result.mT
        polynomial = b * gram + c * gram @ gram
        result = a * result + polynomial @ result
    if is_tall:
        result = result.mT
    return result


class Muon(torch.optim.Optimizer):
    def __init__(self, parameters, lr=0.02, momentum=0.95, nesterov=True, newton_schulz_steps=5, weight_decay=0.0):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            newton_schulz_steps=newton_schulz_steps,
            weight_decay=weight_decay,
        )
        super().__init__(parameters, defaults)
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.dim() != 2:
                    raise ValueError("Muon only works on 2D weight matrices")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            momentum = group["momentum"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = self.state[parameter]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(parameter)
                momentum_buffer = state["momentum_buffer"]

                # Running average of gradients: buffer = momentum * buffer + (1 - momentum) * gradient
                momentum_buffer.lerp_(parameter.grad, 1 - momentum)
                if group["nesterov"]:
                    # Nesterov: look a bit further ahead by mixing the gradient with the updated buffer.
                    direction = parameter.grad.lerp(momentum_buffer, momentum)
                else:
                    direction = momentum_buffer

                update = orthogonalize(direction, steps=group["newton_schulz_steps"])
                # An orthogonal update has entries of a different typical size for tall and
                # wide matrices. This factor from the reference code evens that out.
                rows, columns = parameter.shape
                update = update * max(1, rows / columns) ** 0.5

                # Decoupled weight decay, the same form AdamW uses.
                parameter.mul_(1 - group["lr"] * group["weight_decay"])
                parameter.add_(update.type_as(parameter), alpha=-group["lr"])
        return loss
