import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torchvision.transforms import transforms
import torch.nn as nn
import torchvision.models as models
from torchvision.utils import save_image

model = models.vgg19(pretrained=True).features


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()

        self.chosen_feat = ['0', '5', '10', '19', '28']
        self.model = models.vgg19(pretrained=True).features[:29]

    def forward(self, x):
        features = []
        for idx, layer in enumerate(self.model):
            x = layer(x)
            if str(idx) in self.chosen_feat:
                features.append(x)
        return features


def load_image(image_name):
    img = Image.open(image_name)
    img = loader(img).unsqueeze(0)
    return img.to(device)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_size = 350
loader = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor()
    #transforms.Normalize(mean=[], std = [])
])
orig = load_image("scarlett.jpg")
style = load_image('picasso.jpg')
generated = orig.clone().requires_grad_(True)
model = VGG().to(device).eval()

#hyper
steps = 6000
rate = 0.001
alpha = 1
beta = 0.01
optim = optim.Adam([generated], lr=rate)

for step in range(steps):
    generated_feat = model(generated)
    orig_img_features = model(orig)
    style_features = model(style)

    style_loss = orig_loss = 0

    for gen_feat, orig_feat, style_feat in zip(generated_feat, orig_img_features, style_features):
        batch_size, channel, height, width = gen_feat.shape
        orig_loss += torch.mean((gen_feat - orig_feat))**2
        Gram = gen_feat.view(channel, height*width).mm(gen_feat.view(channel, height*width).t())
        A = style_feat.view(channel, height*width).mm(style_feat.view(channel, height*width).t())

        style_loss +=torch.mean((Gram - A)**2)

    total_loss = alpha*orig_loss + beta*style_loss
    optim.zero_grad()
    total_loss.backward()
    optim.step()
    if step % 200 == 0:
        print(total_loss)
        save_image(generated, "gen.jpg")










