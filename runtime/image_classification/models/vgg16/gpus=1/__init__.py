from .vgg16 import VGG16Partitioned

def arch():
    return "vgg16"

def model(criterion):
    return [
        (vgg16.VGG16Partitioned(), ["input0"], ["output"]),
        (criterion, ["output"], ["loss"])
    ]

def full_model():
    return VGG16Partitioned()
