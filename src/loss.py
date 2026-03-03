import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    """
    Focal Loss module that down-weights well-classified examples to focus training on hard negatives.

    Attributes:
        gamma (float): Focusing parameter that adjusts the rate at which easy examples are down-weighted.
        alpha (Tensor or None): Weighting factor for classes.
        size_average (bool): Determines if the loss is averaged over the batch.
    """
    def __init__(self, gamma=0, alpha=None, size_average=True):
        """
        Initialize the FocalLoss.

        Args:
            gamma (float): Focusing parameter gamma.
            alpha (Tensor, list, or None): Weighting factors for classes.
            size_average (bool): If True, average the loss over all samples.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        """
        Compute the focal loss between the input and target.

        Args:
            input (Tensor): Predicted logits with shape [N, C, ...].
            target (Tensor): Ground truth labels with shape [N, ...].

        Returns:
            Tensor: Computed focal loss.
        """
        # If input has more than 2 dimensions, reshape it to (N*H*W, C)
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # (N, C, H*W)
            input = input.transpose(1, 2)  # (N, H*W, C)
            input = input.contiguous().view(-1, input.size(2))  # (N*H*W, C)
        target = target.view(-1, 1)

        logpt = F.log_softmax(input)  # Compute log-softmax
        logpt = logpt.gather(1, target)  # Gather log-probabilities corresponding to target labels
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())  # Convert log-probabilities to probabilities

        if self.alpha is not None:
            # Ensure alpha is of same type as input data
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        # Compute the focal loss
        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()