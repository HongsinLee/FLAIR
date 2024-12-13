import torchattacks
import pdb

import math
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

def PGD_mixed(images, labels, model, lam ,eps=8/255, alpha=2/225, steps=10, random_start=True, idx = 0 ):
    model.train()
    for _, m in model.named_modules():
        if 'BatchNorm' in m.__class__.__name__:
            m = m.eval()
        if 'Dropout' in m.__class__.__name__:
            m = m.eval()

    labels1 = labels[0]
    labels2 = labels[1]



    images = images.clone().detach().cuda()
    labels1 = labels1.clone().detach().cuda()
    labels2 = labels2.clone().detach().cuda()

    loss = nn.CrossEntropyLoss()
    adv_images = images.clone().detach()

    if random_start:
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    for _ in range(steps):
        adv_images.requires_grad = True
        outputs = model(adv_images)
        cost = lam * loss(outputs, labels1) + (1 -lam) * loss(outputs, labels2)

        try :
            grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]
        except :
            pdb.set_trace()
            return adv_images


        adv_images = adv_images.detach() + alpha*grad.sign()
        delta = torch.clamp(adv_images - images, min= -eps, max= eps)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()

    model.train()

    return adv_images

def Gaussian(images, eps=8/255):


    images = images.clone().detach().cuda()


    adv_images = images + eps*torch.randn_like(images)
    adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    return adv_images

def Random(images, eps=8/255):


    images = images.clone().detach().cuda()


    adv_images = images + eps*torch.rand_like(images)
    adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    return adv_images

def FGSM(images, labels, model, eps=8/255, random_start=False):
    model.train()
    for _, m in model.named_modules():
        if 'BatchNorm' in m.__class__.__name__:
            m = m.eval()
        if 'Dropout' in m.__class__.__name__:
            m = m.eval()


    images = images.clone().detach().cuda()
    labels = labels.clone().detach().cuda()

    loss = nn.CrossEntropyLoss()
    adv_images = images.clone().detach()

    if random_start:
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    adv_images.requires_grad = True
    outputs = model(adv_images)

    cost = loss(outputs, labels)

    grad = torch.autograd.grad(cost, adv_images,
                                retain_graph=False, create_graph=False)[0]

    adv_images = adv_images.detach() + eps*grad.sign()
    delta = torch.clamp(adv_images - images, min= -eps, max= eps)
    adv_images = torch.clamp(images + delta, min=0, max=1).detach()

    model.train()

    return adv_images


def softXEnt (input, target):
    logprobs = nn.functional.log_softmax (input, dim = 1)
    return  -(target * logprobs).sum() / input.shape[0]


def PGD(images, labels, model, eps=8/255, alpha=2/225, steps=10, random_start=True):
    model.train()
    for _, m in model.named_modules():
        if 'BatchNorm' in m.__class__.__name__:
            m = m.eval()
        if 'Dropout' in m.__class__.__name__:
            m = m.eval()


    images = images.clone().detach().cuda()
    labels = labels.clone().detach().cuda()

    loss = nn.CrossEntropyLoss()
    adv_images = images.clone().detach()

    if random_start:
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    for _ in range(steps):
        adv_images.requires_grad = True
        outputs = model(adv_images)
        cost = loss(outputs, labels)

        grad = torch.autograd.grad(cost, adv_images,
                                    retain_graph=False, create_graph=False)[0]

        adv_images = adv_images.detach() + alpha*grad.sign()
        delta = torch.clamp(adv_images - images, min= -eps, max= eps)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()

    model.train()

    return adv_images




def PGD_multiout(images, labels, model, eps=8/255, alpha=2/225, steps=2, random_start=True):
    model.train()
    for _, m in model.named_modules():
        if 'BatchNorm' in m.__class__.__name__:
            m = m.eval()
        if 'Dropout' in m.__class__.__name__:
            m = m.eval()


    images = images.clone().detach().cuda()
    labels = labels.clone().detach().cuda()
    list_adv_images = []
    loss = nn.CrossEntropyLoss()
    adv_images = images.clone().detach()
    if random_start:
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    for _ in range(steps):
        adv_images.requires_grad = True
        outputs = model(adv_images)
        cost = loss(outputs, labels)

        grad = torch.autograd.grad(cost, adv_images,
                                    retain_graph=False, create_graph=False)[0]

        adv_images = adv_images.detach() + alpha*grad.sign()
        delta = torch.clamp(adv_images - images, min= -eps, max= eps)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()
        list_adv_images.append(adv_images)
    model.train()

    return list_adv_images



def TRADES(images, labels, model, eps=8/255, alpha=2/225, steps=10):
    model.train()
    for _, m in model.named_modules():
        if 'BatchNorm' in m.__class__.__name__:
            m = m.eval()
        if 'Dropout' in m.__class__.__name__:
            m = m.eval()


    images = images.clone().detach().cuda()
    labels = labels.clone().detach().cuda()
    logit_ori = model(images).detach()

    loss = nn.KLDivLoss(reduction='sum')

    adv_images = images.clone().detach()
    adv_images = adv_images + 0.001*torch.randn_like(adv_images)
    adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    for _ in range(steps):
        adv_images.requires_grad = True
        logit_adv = model(adv_images)
        cost = loss(F.log_softmax(logit_adv, dim=1), F.softmax(logit_ori, dim=1))

        grad = torch.autograd.grad(cost, adv_images,
                                    retain_graph=False, create_graph=False)[0]

        adv_images = adv_images.detach() + alpha*grad.sign()
        delta = torch.clamp(adv_images - images, min= -eps, max= eps)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()

    model.train()

    return adv_images




############ for batch augmentation #######################################################################################
def PGD_targetattack(images, labels, num_class, model, eps=8/255, alpha=2/225, steps=10, random_start=True):
    model.train()
    for _, m in model.named_modules():
        if 'BatchNorm' in m.__class__.__name__:
            m = m.eval()
        if 'Dropout' in m.__class__.__name__:
            m = m.eval()

    images = images.clone().detach().cuda()
    labels = labels.clone().detach().cuda()

    loss = nn.CrossEntropyLoss()
    adv_images = images.clone().detach()

    if random_start:
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    for _ in range(steps):
        if steps % 2 == 0:
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()


        adv_images.requires_grad = True
        outputs = model(adv_images)[:,:num_class]
    
        cost = -loss(outputs, labels)

        grad = torch.autograd.grad(cost, adv_images,
                                    retain_graph=False, create_graph=False)[0]

        adv_images = adv_images.detach() + alpha*grad.sign()
        delta = torch.clamp(adv_images - images, min= -eps, max= eps)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()

    model.train()

    return adv_images
###################################### batch augmentation end ###########################################################



def PGD_softlabels(images, labels, model, eps=8/255, alpha=2/225, steps=10, random_start=True):
    model.train()
    for _, m in model.named_modules():
        if 'BatchNorm' in m.__class__.__name__:
            m = m.eval()
        if 'Dropout' in m.__class__.__name__:
            m = m.eval()


    images = images.clone().detach().cuda()
    labels = labels.clone().detach().cuda()

    adv_images = images.clone().detach()

    if random_start:
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    for _ in range(steps):
        adv_images.requires_grad = True
        outputs = model(adv_images)
    
        cost = softXEnt(outputs, labels)

        grad = torch.autograd.grad(cost, adv_images,
                                    retain_graph=False, create_graph=False)[0]

        adv_images = adv_images.detach() + alpha*grad.sign()
        delta = torch.clamp(adv_images - images, min= -eps, max= eps)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()

    model.train()

    return adv_images



def PGD_var_eps(images, labels, model, eps, alpha, steps=10, random_start=True):
    
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


    images = images.clone().detach().cuda()
    labels = labels.clone().detach().cuda()

    loss = nn.CrossEntropyLoss()
    adv_images = images.clone().detach()
    view_tuple = (-1,) + (1,) * (images.dim()-1) #(-1, 1, 1, 1)
    if random_start:
        adv_images = adv_images + (2 * eps.view(view_tuple) * torch.rand_like(adv_images) - eps.view(view_tuple)) 
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    for _ in range(steps):
        adv_images.requires_grad = True
        outputs = model(adv_images)
    
        cost = loss(outputs, labels)

        grad = torch.autograd.grad(cost, adv_images,
                                    retain_graph=False, create_graph=False)[0]

        adv_images = adv_images.detach() + alpha.view(view_tuple) *grad.sign()

        delta = torch.max(torch.min(adv_images - images, eps.view(view_tuple)), -eps.view(view_tuple))
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()

    model.train()

    return adv_images