import albumentations as A
REGISTER = {}


def register_transform(func):
    REGISTER[func.__name__] = func
    return func


def get_transform(transform_name):
    if transform_name == "none":
        return None
    else:
        return A.Compose(REGISTER[transform_name](),
                         bbox_params=A.BboxParams(
                             format='yolo', label_fields=['class_labels']))


@register_transform
def rotate():
    return [
        A.Rotate((90, 90), p=1),
    ]


@register_transform
def hflip():
    return [
        A.HorizontalFlip(p=1),
    ]
