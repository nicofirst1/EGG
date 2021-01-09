from dataclasses import dataclass

import torch
import torch.nn.functional as F

from egg.zoo.coco_game.utils import get_labels


def loss_init(
    cross_lambda: float,
    batch_size: int,
):
    """
    Init loss and return function
    """
    losses = Losses(
        cross_lambda=cross_lambda,
        batch_size=batch_size,
    )

    return losses.final_loss


@dataclass
class Losses:
    """
    Stores losses functions together with weights
    """

    cross_lambda: float
    batch_size: int

    def final_loss(
        self,
        _sender_input,
        _message,
        _receiver_input,
        receiver_output,
        labels,
    ):
        """
        Estimate l1 and cross entropy loss. Adds it to final loss
        """

        metrics = {}

        label_class, _ = get_labels(labels)

        x_loss = get_cross_entropy(receiver_output, label_class)
        metrics["x_loss"] = x_loss

        acc = get_accuracy(receiver_output, label_class)
        acc = acc.unsqueeze(dim=-1)
        metrics["class_accuracy"] = acc

        loss = x_loss * self.cross_lambda

        metrics["custom_loss"] = loss
        return loss, metrics


def get_accuracy(pred_class: torch.Tensor, true_class: torch.Tensor) -> torch.Tensor:
    acc = (torch.argmax(pred_class, dim=1) == true_class).sum()
    acc = torch.div(acc, pred_class.shape[0])

    return acc


def get_cross_entropy(pred_classes: torch.Tensor, targets):
    targets = targets.long()
    targets = targets.to(pred_classes.device)
    # pytorch does softmax inside cross entropy
    return F.cross_entropy(pred_classes, targets, reduction="none")

