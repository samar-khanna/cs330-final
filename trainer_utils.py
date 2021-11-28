"""Utilities for scoring the model."""
import torch


def score(logits, labels, label_mask):
    """Returns the mean accuracy of a model's predictions on a set of examples.

    Args:
        logits (torch.Tensor): model predicted logits
            shape (examples, classes)
        labels (torch.Tensor): classification labels from 0 to num_classes - 1
            shape (examples, classes)
    """

    assert logits.dim() == 2
    assert labels.dim() == 2
    assert logits.shape[0] == labels.shape[0]
    labels[~label_mask] = 0.
    preds = logits >= 0.
    tp = (labels == preds)[label_mask.type(torch.bool)].type(torch.float)
    return torch.mean(tp).item()
