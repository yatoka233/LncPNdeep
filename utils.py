import random
import numpy as np
import torch
import torch.nn.functional as F

def sequence_split(document, s_l=1000, s_u=2000, repeat=10):
    """
    split a document into s * repeat sequences
    args:
        s_l: lower bound of s
        s_u: upper bound of s
        repeat: sample times
    """
    l = len(document)
    data = []
    if l <= s_u:
        return [document]
    i = 0
    data_len = 0
    while i < repeat:
        q = random.randint(0, 100)
        while q < l:
            s = random.randint(s_l, s_u)
            if q+s < l:
                data.append(document[q:(q+s)])
            elif l-q > s_l:
                data.append(document[q:l])
            q = q + s
        if len(data)>data_len:
            i += 1
            data_len = len(data)
    return data





def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.7,
    gamma: float = 2,
    reduction: str = "sum",
):
    """
    Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py .
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default = 0.25
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss