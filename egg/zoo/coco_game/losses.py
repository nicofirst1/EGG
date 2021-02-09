from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from egg.zoo.coco_game.utils.utils import get_labels


def loss_init(
    lambda_cross: float,
    lambda_kl: float,
    lambda_f: float,
    batch_size: int,
    class_weights: torch.Tensor,
):
    """
    Init loss and return function
    """

    focal_loss = FocalLoss(gamma=2)

    losses = Losses(
        lambda_cross=lambda_cross,
        lambda_kl=lambda_kl,
        lambda_f=lambda_f,
        batch_size=batch_size,
        class_weights=class_weights,
        focal_loss=focal_loss,
    )

    return losses.final_loss


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=0, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        return loss


@dataclass
class Losses:
    """
    Stores losses functions together with weights
    """

    lambda_cross: float
    lambda_kl: float
    lambda_f: float
    batch_size: int
    class_weights: torch.Tensor
    focal_loss: FocalLoss

    def final_loss(
        self,
        sender_input,
        message,
        receiver_input,
        receiver_output,
        labels,
        sender_output=None,
    ):
        """
        Estimate complete loss, logs accuracy
        """

        metrics = {}

        res_dict = get_labels(labels)
        # label_class = res_dict["class_id"]
        label_class = res_dict["true_segment"]

        # assert that we have an output to work on
        if receiver_output is None and sender_output is None:
            raise RuntimeError("Both sender and receiver output are None")
        # either sender
        elif receiver_output is None:
            output = sender_output
        # or receiver or both
        else:
            output = receiver_output

        f_loss = self.focal_loss(output, label_class)
        metrics["f_loss"] = f_loss

        x_loss = get_cross_entropy(output, label_class)  # , weights=self.class_weights)
        metrics["x_loss"] = x_loss

        kl_loss = get_kl(output, label_class)  # , weights=self.class_weights)
        metrics["kl_loss"] = kl_loss

        if receiver_output is not None:
            rec_acc = get_accuracy(receiver_output, label_class)
            rec_acc = rec_acc.unsqueeze(dim=-1)
            metrics["accuracy_receiver"] = rec_acc

        if sender_output is not None:
            send_acc = get_accuracy(sender_output, label_class)
            send_acc = send_acc.unsqueeze(dim=-1)
            metrics["accuracy_sender"] = send_acc

        loss = (
            x_loss * self.lambda_cross
            + kl_loss * self.lambda_kl
            + f_loss * self.lambda_f
        )

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
    assert all(kl >= 0), "Some parametrs of the KL are <0"
    return kl
