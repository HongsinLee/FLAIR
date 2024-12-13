import torchattacks
import pdb

import math
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


def FGSM_with_normalized_input(images, labels, mean, std, num_class, model,
        logit_needed=False, eps=8/255, random_start=False):
    model.train()
    for _, m in model.named_modules():
        if 'BatchNorm' in m.__class__.__name__:
            m = m.eval()
        if 'Dropout' in m.__class__.__name__:
            m = m.eval()

    mean = torch.tensor(mean).reshape(1, len(mean), 1, 1).cuda()
    std = torch.tensor(std).reshape(1, len(std), 1, 1).cuda()  

    images = (images * std + mean).clone().detach().cuda()
    labels = labels.clone().detach().cuda()

    loss = nn.CrossEntropyLoss()
    adv_images = images.clone().detach()

    if random_start:
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    adv_images.requires_grad = True
    if logit_needed:
        outputs = model.get_logit((adv_images - mean)/std)[:, :num_class]
    else:
        outputs = model((adv_images - mean)/std)[:, :num_class]

    cost = loss(outputs, labels)

    grad = torch.autograd.grad(cost, adv_images,
                                retain_graph=False, create_graph=False)[0]

    adv_images = adv_images.detach() + eps*grad.sign()
    delta = torch.clamp(adv_images - images, min= -eps, max= eps)
    adv_images = torch.clamp(images + delta, min=0, max=1).detach()

    model.train()

    return (adv_images-mean)/std


def softXEnt (input, target):
    logprobs = nn.functional.log_softmax (input, dim = 1)
    return  -(target * logprobs).sum() / input.shape[0]


def PGD_with_normalized_input(images, labels, mean, std, num_class, model,
        logit_needed=False, eps=8/255, alpha=2/225, steps=10, random_start=True):
    model.train()
    for _, m in model.named_modules():
        if 'BatchNorm' in m.__class__.__name__:
            m = m.eval()
        if 'Dropout' in m.__class__.__name__:
            m = m.eval()

    mean = torch.tensor(mean).reshape(1, len(mean), 1, 1).cuda()
    std = torch.tensor(std).reshape(1, len(std), 1, 1).cuda()  

    images = (images * std + mean).clone().detach().cuda()
    labels = labels.clone().detach().cuda()

    loss = nn.CrossEntropyLoss()
    adv_images = images.clone().detach()

    if random_start:
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    for _ in range(steps):
        adv_images.requires_grad = True
        if logit_needed:
            outputs = model.get_logit((adv_images - mean)/std)[:, :num_class]
        else:
            outputs = model((adv_images - mean)/std)[:, :num_class]
    
        cost = loss(outputs, labels)

        grad = torch.autograd.grad(cost, adv_images,
                                    retain_graph=False, create_graph=False)[0]

        adv_images = adv_images.detach() + alpha*grad.sign()
        delta = torch.clamp(adv_images - images, min= -eps, max= eps)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()

    model.train()

    return (adv_images-mean)/std


def PGD_targetattack_with_normalized_input(images, labels, mean, std, num_class, model,
        logit_needed=False, eps=8/255, alpha=2/225, steps=10, random_start=True):
    model.train()
    for _, m in model.named_modules():
        if 'BatchNorm' in m.__class__.__name__:
            m = m.eval()
        if 'Dropout' in m.__class__.__name__:
            m = m.eval()

    mean = torch.tensor(mean).reshape(1, len(mean), 1, 1).cuda()
    std = torch.tensor(std).reshape(1, len(std), 1, 1).cuda()  

    images = (images * std + mean).clone().detach().cuda()
    labels = labels.clone().detach().cuda()

    loss = nn.CrossEntropyLoss()
    adv_images = images.clone().detach()

    if random_start:
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    for _ in range(steps):
        adv_images.requires_grad = True
        if logit_needed:
            outputs = model.get_logit((adv_images - mean)/std)[:, :num_class]
        else:
            outputs = model((adv_images - mean)/std)[:, :num_class]
    
        cost = -loss(outputs, labels)

        grad = torch.autograd.grad(cost, adv_images,
                                    retain_graph=False, create_graph=False)[0]

        adv_images = adv_images.detach() + alpha*grad.sign()
        delta = torch.clamp(adv_images - images, min= -eps, max= eps)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()

    model.train()

    return (adv_images-mean)/std




def PGD_softlabels_with_normalized_input(images, labels, mean, std, num_class, model,
        logit_needed=False, eps=8/255, alpha=2/225, steps=10, random_start=True):
    model.train()
    for _, m in model.named_modules():
        if 'BatchNorm' in m.__class__.__name__:
            m = m.eval()
        if 'Dropout' in m.__class__.__name__:
            m = m.eval()

    mean = torch.tensor(mean).reshape(1, len(mean), 1, 1).cuda()
    std = torch.tensor(std).reshape(1, len(std), 1, 1).cuda()  

    images = (images * std + mean).clone().detach().cuda()
    labels = labels.clone().detach().cuda()

    adv_images = images.clone().detach()

    if random_start:
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    for _ in range(steps):
        adv_images.requires_grad = True
        if logit_needed:
            outputs = model.get_logit((adv_images - mean)/std)[:, :num_class]
        else:
            outputs = model((adv_images - mean)/std)[:, :num_class]
    
        cost = softXEnt(outputs, labels)

        grad = torch.autograd.grad(cost, adv_images,
                                    retain_graph=False, create_graph=False)[0]

        adv_images = adv_images.detach() + alpha*grad.sign()
        delta = torch.clamp(adv_images - images, min= -eps, max= eps)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()

    model.train()

    return (adv_images-mean)/std



def PGD_regulization_with_normalized_input(images, labels, mean, std, num_class, net, old_net,
        gamma=1, eps=8/255, alpha=2/225, steps=10, random_start=True):
    if old_net == None:
        raise NotImplementedError('PGD regulization with no old_net error')

    net.train()
    for _, m in net.named_modules():
        if 'BatchNorm' in m.__class__.__name__:
            m = m.eval()
        if 'Dropout' in m.__class__.__name__:
            m = m.eval()

    mean = torch.tensor(mean).reshape(1, len(mean), 1, 1).cuda()
    std = torch.tensor(std).reshape(1, len(std), 1, 1).cuda()  

    images = (images * std + mean).clone().detach().cuda()
    labels = labels.clone().detach().cuda()

    loss = nn.CrossEntropyLoss()
    adv_images = images.clone().detach()

    if random_start:
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    for _ in range(steps):
        adv_images.requires_grad = True
        outputs, feats = net((adv_images - mean)/std, returnt='all')
        outputs = outputs[:, :num_class]
        with torch.no_grad():
            _, feats_old = old_net((adv_images - mean)/std, returnt='all')

        cost = loss(outputs, labels) + gamma * F.mse_loss(feats, feats_old)

        grad = torch.autograd.grad(cost, adv_images,
                                    retain_graph=False, create_graph=False)[0]

        adv_images = adv_images.detach() + alpha*grad.sign()
        delta = torch.clamp(adv_images - images, min= -eps, max= eps)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()

    net.train()

    return (adv_images-mean)/std


def PGD_var_eps_with_normalized_input(images, labels, mean, std, num_class, model, eps, alpha,
        logit_needed=False, steps=10, random_start=True):
    
    if eps.shape[0] != images.shape[0]:
        raise NotImplementedError('PGD var eps : eps tensor shape error')
    if alpha.shape[0] != images.shape[0]:
        raise NotImplementedError('PGD var eps : alpha tensor shape error')

    model.train()
    for _, m in model.named_modules():
        if 'BatchNorm' in m.__class__.__name__:
            m = m.eval()
        if 'Dropout' in m.__class__.__name__:
            m = m.eval()

    mean = torch.tensor(mean).reshape(1, len(mean), 1, 1).cuda()
    std = torch.tensor(std).reshape(1, len(std), 1, 1).cuda()  

    images = (images * std + mean).clone().detach().cuda()
    labels = labels.clone().detach().cuda()

    loss = nn.CrossEntropyLoss()
    adv_images = images.clone().detach()
    view_tuple = (-1,) + (1,) * (images.dim()-1) #(-1, 1, 1, 1)
    if random_start:
        adv_images = adv_images + (2 * eps.view(view_tuple) * torch.rand_like(adv_images) - eps.view(view_tuple)) 
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    for _ in range(steps):
        adv_images.requires_grad = True
        if logit_needed:
            outputs = model.get_logit((adv_images - mean)/std)[:, :num_class]
        else:
            outputs = model((adv_images - mean)/std)[:, :num_class]
    
        cost = loss(outputs, labels)

        grad = torch.autograd.grad(cost, adv_images,
                                    retain_graph=False, create_graph=False)[0]

        adv_images = adv_images.detach() + alpha.view(view_tuple) *grad.sign()

        delta = torch.max(torch.min(adv_images - images, eps.view(view_tuple)), -eps.view(view_tuple))
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()

    model.train()

    return (adv_images-mean)/std