import torch.nn as nn


class MLP(nn.Module):
    """Fully connected network: flattens each 28x28 image into 784 pixels."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def conv_block(in_ch, out_ch):
    """Two 3x3 convs, then halve the spatial size with max pooling."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Dropout(0.25),
    )


class CNN(nn.Module):
    """Convolutional network: learns local patterns (edges, textures) before classifying."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(1, 32),   # 1x28x28 -> 32x14x14
            conv_block(32, 64),  # 32x14x14 -> 64x7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


MODELS = {"mlp": MLP, "cnn": CNN}
