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
    target_position = res_dict["target_position"]

    x_loss = get_cross_entropy(
        receiver_output, target_position
    )
    metrics["x_loss"] = x_loss

    rec_acc = get_accuracy(receiver_output, target_position)
    rec_acc = rec_acc.unsqueeze(dim=-1)
    metrics["accuracy"] = rec_acc

    return x_loss, metrics


def get_accuracy(pred_class: torch.Tensor, true_class: torch.Tensor) -> torch.Tensor:
    acc = torch.argmax(pred_class, dim=1) == true_class
    acc = acc.double()
    # acc = torch.div(acc.sum(), pred_class.shape[0])

    return acc


def get_cross_entropy(pred_classes: torch.Tensor, targets):
    targets = targets.long()
    targets = targets.to(pred_classes.device)
    # pytorch does softmax inside cross entropy
    return F.cross_entropy(pred_classes, targets, reduction="none")
