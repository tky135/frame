import torch
from model import*
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image as Image
import torchvision.models as models
def deep_dream(model, saved_model_path, img_size, prior="given", n_epochs=500, lr=1):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = models.resnet18(True)
    model = torch.nn.DataParallel(model)
    # state_dict = torch.load(saved_model_path)
    # model.load_state_dict(state_dict)
    model = model.to(device)
    if prior == "normal":
        dream_img = 0.5 * torch.randn((1, 3, img_size, img_size), requires_grad=True, device=device)
    elif prior == "blank":
        dream_img = 0.5 * torch.ones((1, 3, img_size, img_size), requires_grad=True, device=device)
    elif prior == "given":
        dream_img = Image.open("cloud.jpg")
        transforms = T.Compose([T.Resize(224), T.ToTensor()])
        dream_img = transforms(dream_img)
        dream_img = dream_img.unsqueeze(0)
        print(dream_img.shape)
    elif prior == "blue":
        dream_img = torch.zeros((1, 3, img_size, img_size), device=device)
        dream_img[0, 2, :, :] = 1

    
    # tensor.detach() creates a tensor that shares storage with tensor that does not require grad.
    for epoch in range(n_epochs):
        # with torch.no_grad():
        #     dream_img[0,1] = dream_img[0, 0]
        #     dream_img[0, 2]  = dream_img[0, 0]
        dream_img = dream_img.detach()
        dream_img.requires_grad_()

        # resnet18
        # layer = model.module.net.layer4[0].conv2
        layer = model.module.fc

        # mobilenet
        # layer = model.module.net.features[6].conv[1][0]

        # VGG
        # layer = model.module.net.features[18]

        hook = Hook(layer)
        y = model(dream_img)
        loss = hook.output[0,20].norm()
        
        # print(y.shape)
        
        # # print(y.grad)
        # # print(dream_img)
        # y = y.flatten()
        # y = torch.softmax(y, 0)
        # loss = y[0] - 1e-7 * torch.norm(dream_img)
        # loss = y#  - 1e-7 * torch.norm(dream_img)
        # loss = torch.sum(y[0, 20, 60, 60])

        # if loss > 0.995:
        #     break
        print(loss)
        loss.backward()
        with torch.no_grad():
            dream_img += lr * dream_img.grad
            dream_img.grad.zero_()
            dream_img[dream_img > 1] = 1
            dream_img[dream_img < 0] = 0
        # for param in model.parameters():
        #     param.grad.zero_()
    dream_img = dream_img.detach().to("cpu")
    # dream_img[0,1] = dream_img[0, 0]
    # dream_img[0, 2]  = dream_img[0, 0]
    # dream_img /= torch.mean(dream_img.flatten())
    print(dream_img)
    plt.imshow(dream_img.squeeze(0).permute(1, 2, 0))
    plt.show()
    print(model)

class Hook():
    def __init__(self, module, backward=False):
        if backward==False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output
    def close(self):
        self.hook.remove()

if __name__ == "__main__":
    model = ResNet18(3, 4, True)
    print(model)
    deep_dream(model, "outputs/lung_mobilenet/model.t7", 224)