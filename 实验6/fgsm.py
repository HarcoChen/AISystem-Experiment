import torch

def fgsm_attack(image, epsilon, data_grad):
    """FGSM attack code"""
    sign_data_grad = data_grad.sign()

    perturbed_image = image + epsilon * sign_data_grad

    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image