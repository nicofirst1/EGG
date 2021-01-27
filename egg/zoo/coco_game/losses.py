from dataclasses import dataclass

import torch
import torch.nn.functional as F

from egg.zoo.coco_game.utils.utils import get_labels


def loss_init(
        cross_lambda: float,
        kl_lambda: float,
        batch_size: int,
        class_weights: torch.Tensor,
):
    """
    Init loss and return function
    """

    losses = Losses(
        cross_lambda=cross_lambda,
        kl_lambda=kl_lambda,
        batch_size=batch_size,
        class_weights=class_weights,
    )

    return losses.final_loss


@dataclass
class Losses:
    """
    Stores losses functions together with weights
    """

    cross_lambda: float
    kl_lambda: float
    batch_size: int
    class_weights: torch.Tensor

    def final_loss(
            self,
            sender_input,
            message,
            receiver_input,
            receiver_output,
            labels,
    ):
        """
        Estimate l1 and cross entropy loss. Adds it to final loss
        """

        metrics = {}

        res_dict = get_labels(labels)
        label_class = res_dict['class_id']

        x_loss = get_cross_entropy(receiver_output, label_class, weights=self.class_weights)
        metrics["x_loss"] = x_loss

        kl_loss = get_kl(receiver_output, label_class, weights=self.class_weights)
        metrics["kl_loss"] = kl_loss

        acc = get_accuracy(receiver_output, label_class)
        acc = acc.unsqueeze(dim=-1)
        metrics["class_accuracy"] = acc

        loss = x_loss * self.cross_lambda + kl_loss * self.kl_lambda

        metrics["custom_loss"] = loss
        return loss, metrics




def get_accuracy(pred_class: torch.Tensor, true_class: torch.Tensor) -> torch.Tensor:
    acc = (torch.argmax(pred_class, dim=1) == true_class).sum()
    acc = torch.div(acc, pred_class.shape[0])

    return acc


def get_cross_entropy(pred_classes: torch.Tensor, targets, weights=None):
    targets = targets.long()
    targets = targets.to(pred_classes.device)
    # pytorch does softmax inside cross entropy
    return F.cross_entropy(pred_classes, targets, weight=weights, reduction="none")


def get_kl(pred_classes: torch.Tensor, targets, weights=None):
    """
    Return kl divergence loss
    """
    pred_classes = F.log_softmax(pred_classes, dim=1)

    targets = targets.unsqueeze(dim=1)
    target_dist = torch.zeros(pred_classes.shape).to(targets)
    target_dist.scatter_(1, targets, 1)
    target_dist = target_dist.float()

    kl = F.kl_div(pred_classes, target_dist, reduction="none", log_target=True)
    if weights is not None:
        kl *= weights

    kl = kl.mean(dim=1)
    return kl
