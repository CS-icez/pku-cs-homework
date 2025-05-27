import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self, num_class=10):
        super(ConvNet, self).__init__()
        pass
        # ----------TODO------------
        # define a network 
        wraped_conv_tuple = lambda in_channels, out_channels: (
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # Simplified vgg11_bn.
        self.features = nn.Sequential(
            *wraped_conv_tuple(3, 64),
            nn.MaxPool2d(kernel_size=2),
            *wraped_conv_tuple(64, 128),
            nn.MaxPool2d(kernel_size=2),
            *wraped_conv_tuple(128, 256),
            nn.MaxPool2d(kernel_size=2),
            *wraped_conv_tuple(256, 512),
            nn.MaxPool2d(kernel_size=2),
            *wraped_conv_tuple(512, 512)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(128, num_class),
            nn.Softmax()
        )
        # ----------TODO------------

    def forward(self, x):

        # ----------TODO------------
        # network forwarding 
        # ----------TODO------------
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    import torch
    from torch.utils.tensorboard  import SummaryWriter
    from dataset import CIFAR10
    writer = SummaryWriter(log_dir='../experiments/network_structure')
    net = ConvNet()
    train_dataset = CIFAR10()
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=2, shuffle=False, num_workers=2)
    # Write a CNN graph. 
    # Please save a figure/screenshot to '../results' for submission.
    for imgs, labels in train_loader:
        writer.add_graph(net, imgs)
        writer.close()
        break 
