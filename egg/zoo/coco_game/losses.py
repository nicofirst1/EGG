import torch
import torch.nn.functional as F

from egg.zoo.coco_game.utils.utils import get_labels


def final_loss(
    sender_input,
    message,
    receiver_input,
    receiver_output,
    labels,
):
    """
    Estimate complete loss, logs accuracy
    """

    metrics = {}

    res_dict = get_labels(labels)
    # label_class = res_dict["class_id"]
    label_discr = res_dict["true_segment"]

    x_loss = get_cross_entropy(
        receiver_output, label_discr
    )  # , weights=self.class_weights)
    metrics["x_loss"] = x_loss

    rec_acc = get_accuracy(receiver_output, label_discr)
    rec_acc = rec_acc.unsqueeze(dim=-1)
    metrics["accuracy_receiver"] = rec_acc

    return x_loss, metrics


def get_accuracy(pred_class: torch.Tensor, true_class: torch.Tensor) -> torch.Tensor:
    acc = (torch.argmax(pred_class, dim=1) == true_class).sum()
    acc = torch.div(acc, pred_class.shape[0])

    return acc


def get_cross_entropy(pred_classes: torch.Tensor, targets, weights=None):
    targets = targets.long()
    targets = targets.to(pred_classes.device)
    # pytorch does softmax inside cross entropy
    return F.cross_entropy(pred_classes, targets, weight=weights, reduction="none")
