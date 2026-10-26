"""The two architectures (PLAN.md: Models), each with a new, untrained 10-class output layer."""
import timm

# timm names for the ImageNet-1k weights. "resnet18.tv_in1k" is torchvision's original ResNet-18
# checkpoint, mirrored on the Hugging Face hub.
TIMM_NAMES = {
    "vit": "deit_tiny_patch16_224.fb_in1k",
    "cnn": "resnet18.tv_in1k",
}

# The first and last layers are never pruned (PLAN.md: Pruning): the ViT's patch embedding and the
# ResNet's first conv, and the 10-class output layer of each.
SKIP = {
    "vit": ["patch_embed.proj", "head"],
    "cnn": ["conv1", "fc"],
}

DISPLAY = {"vit": "DeiT-Tiny", "cnn": "ResNet-18"}


def new_model(arch, pretrained, device):
    """`num_classes=10` swaps the 1,000-class ImageNet output layer for a new 10-class one."""
    return timm.create_model(TIMM_NAMES[arch], pretrained=pretrained, num_classes=10).to(device)
