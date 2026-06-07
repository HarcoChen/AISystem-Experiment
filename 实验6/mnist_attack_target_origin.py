import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import Net
from fgsm import fgsm_attack

def test_target(model, device, test_loader, epsilon):

    # Accuracy counter
    suc = 0
    tar = 0
    adv_examples = []

    # Loop over all examples in test set
    for data, target in tqdm(test_loader):

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        # target-2
        adv_label = 2 * torch.ones_like(target)

        # TODO: Calculate the loss (F.nll_loss)


        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data

        # TODO: Call FGSM Attack


        # Re-classify the perturbed image
        output = model(perturbed_data)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        if target.item() == adv_label.item():
            tar += 1
        elif (final_pred.item() == adv_label.item()) and (target.item() != adv_label.item()):
            suc +=1
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))

    # Calculate final accuracy for this epsilon
    suc_rate = suc/float(len(test_loader) - tar)
    print("Epsilon: {}\tattack success rate = {:.2%}".format(epsilon, suc_rate))

    # Return the accuracy and an adversarial example
    return suc_rate, adv_examples

def main():
    epsilons = [0, .05, .1, .15, .2, .25, .3]
    pretrained_model = "weights/lenet_mnist_model.pth"
    use_cuda=True

    # MNIST Test dataset and dataloader declaration
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
        ])),
        batch_size=1, shuffle=False)

    # Define what device we are using
    print("CUDA Available: ",torch.cuda.is_available())
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    # Initialize the network
    model = Net().to(device)

    # Load the pretrained model
    model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))

    # Set the model in evaluation mode. In this case this is for the Dropout layers
    model.eval()

    accuracies = []
    examples = []

    # Run test for each epsilon
    for eps in epsilons:
        acc, ex = test_target(model, device, test_loader, eps)
        accuracies.append(acc)
        examples.append(ex)

    plt.figure(figsize=(5,5))
    plt.plot(epsilons, accuracies, "*-")
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xticks(np.arange(0, .35, step=0.05))
    plt.title("Targeted attack")
    plt.xlabel("Epsilon")
    plt.ylabel("Success Rate")
    plt.show()

    # Plot several examples of adversarial samples at each epsilon
    cnt = 0
    max_cols = max(len(row) for row in examples)
    plt.figure(figsize=(8,10))
    for i in range(len(epsilons)):
        for j in range(len(examples[i])):
            cnt += 1
            plt.subplot(len(epsilons), max_cols, cnt)
            plt.xticks([], [])
            plt.yticks([], [])
            if j == 0:
                plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
            orig,adv,ex = examples[i][j]
            plt.title("{} -> {}".format(orig, adv))
            plt.imshow(ex, cmap="gray")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
