#!/usr/bin/env python3
import os
import torch
import torchvision.models as models
import eagerpy as ep
from foolbox import PyTorchModel, accuracy, samples
import foolbox.attacks as fa
import cv2
import numpy as np


def main() -> None:
    # instantiate a model
    pretrained_model = "./weights/vgg16.pth"
    model = models.vgg16()
    # Load the pretrained model
    model.load_state_dict(torch.load(pretrained_model, map_location="cpu"))
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    fmodel = PyTorchModel(
        model, bounds=(0, 1), preprocessing=preprocessing, device=device
    )

    # get data and test the model
    # wrapping the tensors with ep.astensors is optional, but it allows
    # us to work with EagerPy tensors in the following
    images, labels = ep.astensors(*samples(fmodel, dataset="imagenet", batchsize=16))
    clean_acc = accuracy(fmodel, images, labels)
    print(f"原始精度:  {clean_acc * 100:.1f} %")

    # apply the attack
    attack = fa.LinfDeepFoolAttack()  # DeepFool攻击

    epsilons = [
        0.0008,
        0.001,
        0.0015,
        0.002,
        0.003,
        0.01,
    ]
    raw_advs, clipped_advs, success = attack(fmodel, images, labels, epsilons=epsilons)

    # calculate and report the robust accuracy (the accuracy of the model when
    # it is attacked)
    robust_accuracy = []
    print("攻击后的精度：")
    for eps, acc in enumerate(epsilons):
        acc = 1 - success[eps].float32().mean()
        robust_accuracy.append(acc)
        print(f"  Linf norm ≤ {eps:<6}: {acc.item() * 100:4.1f} %")

    # defense method
    print("防御后的精度：")
    for eps, adv in zip(epsilons, clipped_advs):
        defense_imgs = []
        adv = adv.numpy()
        for i in range(adv.shape[0]):
            img = np.transpose(adv[i], (1, 2, 0))
            
            noise = np.random.normal(loc=0, scale=0.01, size=img.shape)
            img = img + noise

            img = np.clip(img, 0, 1)

            defense_img = torch.tensor(np.transpose(img, (2, 0, 1)), device=device)

            defense_imgs.append(defense_img)
        defense_imgs = torch.stack(defense_imgs)
        # calculate and report the defense accuracy (the accuracy of the model when
        # it is defensed)
        def_accuracy = accuracy(fmodel, defense_imgs, labels)
        print(f"  Linf norm ≤ {eps:<6}: {def_accuracy * 100:4.1f} %")


if __name__ == "__main__":
    main()
