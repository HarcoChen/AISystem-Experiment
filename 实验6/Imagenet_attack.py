import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from fgsm import fgsm_attack

def test(model, img, epsilon):

    # Accuracy counter
    adv_examples = []

    # Set requires_grad attribute of tensor. Important for Attack
    img.requires_grad = True
    # Forward pass the data through the model
    output = model(img)

    _, target = torch.max(output.data, 1)

    # Calculate the loss
    CEloss = nn.CrossEntropyLoss()
    loss = CEloss(output, target)#用交叉熵

    # Zero all existing gradients
    model.zero_grad()

    # Calculate gradients of model in backward pass
    loss.backward()

    # Collect datagrad
    data_grad = img.grad.data

    # Call FGSM Attack
    perturbed_data = fgsm_attack(img, epsilon, data_grad)

    # Re-classify the perturbed image
    output = model(perturbed_data)

    # Check for success
    _, final_pred = torch.max(output.data, 1)

    # Save some adv examples for visualization later
    adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
    adv_examples.append((target.data, final_pred.data, adv_ex))

    # Return an adversarial example
    return adv_examples

def load_image(shape=(224, 224), bounds=(0, 1), dtype=np.float32,
               data_format='channels_last', fname='example.png', abs_path=False, fpath=None):
    """ Returns a resized image of target fname"""
    if abs_path == True:
        assert fpath is not None, "fpath has not to be None when abs_path is True."
    assert len(shape) == 2
    assert data_format in ['channels_first', 'channels_last']
    if not abs_path:
        path = os.path.join(os.path.dirname(__file__), 'images/%s' % fname)
    else:
        path = fpath
    image = Image.open(path)
    image = image.resize(shape)
    image = image.convert('RGB')
    image = np.asarray(image, dtype=dtype)
    image = image[:, :, :3]
    if data_format == 'channels_first':
        image = np.transpose(image, (2, 0, 1))
    if bounds != (0, 255):
        image /= 255.
    return image

def main():
    epsilons = [.05, .1, .12, .15]
    pretrained_model = "weights/vgg16.pth"
    use_cuda=True

    # Define what device we are using
    print("CUDA Available: ",torch.cuda.is_available())
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    model = torchvision.models.vgg16().to(device)

    # Load the pretrained model
    model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))

    model.eval()

    image_path = './data/imagenet/cat.jpg'
    image_np = load_image(shape=(224, 224), data_format='channels_first', abs_path=True, fpath=image_path)
    images_var = Variable(torch.unsqueeze(torch.tensor(image_np), dim=0).float()).to(device)
    images_var = images_var.to(device)

    examples = []

    # Run test for each epsilon
    for eps in epsilons:
        ex = test(model, images_var, eps)
        examples.append(ex)

    with open('./data/imagenet_classes.txt') as f:
        classes = [line.strip().split(',')[1] for line in f.readlines()]

    # Plot several examples of adversarial samples at each epsilon
    cnt = 0
    plt.figure(figsize=(7,7))
    for i in range(len(epsilons)):
        cnt += 1
        plt.subplot(2,2,cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
        orig,adv,ex = examples[i][0]
        plt.title("{} -> {}".format(classes[orig], classes[adv]), fontsize=12)
        ex = np.transpose((ex*255).astype(np.uint8), (1, 2, 0))
        plt.imshow(ex)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()