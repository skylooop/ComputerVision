import torch.nn as nn
import torch


class res_block(nn.Module):
    def __init__(self, inp, out, down=None, stride=1):
        super(res_block, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Sequential(nn.Conv2d(inp, out, kernel_size=1), nn.BatchNorm2d(out))
        self.conv2 = nn.Sequential(nn.Conv2d(out, out, kernel_size=3, stride=stride, padding=1), nn.BatchNorm2d(out))
        self.conv3 = nn.Sequential(nn.Conv2d(out, out*self.expansion, kernel_size=1, stride=1), nn.BatchNorm2d(out*self.expansion))
        self.relu = nn.ReLU()
        self.down = down

    def forward(self, x):
        id = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.down is not None:
            id = self.down(id)
        x += id
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, img_chan, num_clas):
        super(ResNet, self).__init__()
        self.inp = 64
        self.conv1 = nn.Sequential(nn.Conv2d(img_chan, 64, kernel_size=7,stride=2,padding=3),
                                   nn.BatchNorm2d(64), nn.ReLU())
        self.maxp1 = nn.MaxPool2d(kernel_size=3, stride=2,padding=1)
        self.layer1 = self.make_layer(block, layers[0], out=64, stride=1)
        self.layer2 = self.make_layer(block, layers[1], out=128, stride=2)
        self.layer3 = self.make_layer(block, layers[2], out=256, stride=2)
        self.layer4 = self.make_layer(block, layers[3], out=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*4, num_clas)
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxp1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def make_layer(self, block, num_res_bl, out, stride):
        id = None
        layer = []
        if stride != 1 or self.inp != out * 4:
            id = nn.Sequential(nn.Conv2d(self.inp, out*4, kernel_size=1, stride=stride),
                               nn.BatchNorm2d(out*4)
                               )
        layer.append(block(self.inp, out, id, stride))
        self.inp = out * 4
        for i in range(num_res_bl - 1):
            layer.append(block(self.inp, out))
        return nn.Sequential(*layer)


def Resnet(img_chan=3, num_clas=1000):
    return ResNet(res_block, [3, 4, 6, 3], img_chan, num_clas)

def test():
    net = Resnet()
    x = torch.randn(2,3,224,224)
    y = net(x).to("cuda")
    print(y.shape)
test()