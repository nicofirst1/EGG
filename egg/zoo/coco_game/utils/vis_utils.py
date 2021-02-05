import cv2
import numpy as np
import PIL
import torch
from PIL import Image


def denormalize_imgs(imgs, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    mean = torch.as_tensor(mean, device=imgs.device)
    std = torch.as_tensor(std, device=imgs.device)

    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)

    imgs.mul_(std).add_(mean)
    return imgs


def normalize_imgs(imgs, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    mean = torch.as_tensor(mean, device=imgs.device)
    std = torch.as_tensor(std, device=imgs.device)

    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)

    imgs.sub_(mean).div_(std)
    return imgs


def numpy2pil(img):
    """
    Convert numpy array to PIL image
    """
    img = img.copy()
    img = np.uint8(img)
    return Image.fromarray(img)


def torch2pil(img, denormalize=False):
    """
    Convert torch tensor to PIL image
    """
    img = img.cpu()
    img = np.transpose(img)

    if denormalize:
        img *= np.array([0.229, 0.224, 0.225])
        img += np.array([0.485, 0.456, 0.406])
        img *= 255
    img = np.uint8(img)
    return Image.fromarray(img)


def show_img(img):
    if isinstance(img, np.ndarray):
        img = numpy2pil(img)
    elif isinstance(img, torch.Tensor):
        img = torch2pil(img)
    elif isinstance(img, PIL.Image.Image):
        pass
    else:
        raise TypeError(f"Type '{type(img)}' of image not recognized")
    img.show()


def visualize_bbox(img, bbox, class_name, color, thickness=2):
    """Visualizes a single bounding box on the image"""

    if not isinstance(img, np.ndarray):
        img = np.array(img)

    TEXT_COLOR = (255, 255, 255)  # White
    bbox = [
        bbox[0],
        bbox[1],
        bbox[0] + bbox[2],
        bbox[1] + bbox[3],
    ]
    x_min, y_min, x_max, y_max = [int(x) for x in bbox]

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, thickness)

    ((text_width, text_height), _) = cv2.getTextSize(
        class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1
    )
    cv2.rectangle(
        img,
        (x_min, y_min - int(1.3 * text_height)),
        (x_min + text_width, y_min),
        color,
        -1,
    )
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=TEXT_COLOR,
        lineType=cv2.LINE_AA,
    )
    return img
