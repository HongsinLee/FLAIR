import torch.nn as nn
import torch
from torchvision import transforms
import numpy as np
from torch.nn import functional as F
from PIL import Image
import torch.optim as optim
import pdb, os, copy
import torchattacks

def robust_finetune_2(model, epochs, train_loader, args):
        # Todo
        origin_model = copy.deepcopy(model)
        origin_model.eval()
        model.train()
        lr = 0.001
        optimizer = optim.Adam(model.parameters(), lr = lr)
        pgd_attack = torchattacks.PGD(model, eps=8/255, alpha=2/225, steps=args.robust_steps, random_start=True)
        criterion_kl = nn.KLDivLoss(size_average=False)
        XENT_loss = nn.CrossEntropyLoss()

        adv_batch_set = []

        for epoch in range(epochs):
            for step, (indexs, x, y) in enumerate(train_loader):
                x, y = x.cuda(), y.cuda()
                N = x.shape[0]
                x_adv = pgd_attack(x,y)

                adv_batch_set.append((x_adv, y))
                #output = self.model(images)
                adv_out = model(x_adv)

                feat_clean = origin_model.feature_extractor(x)
                feat_adv = origin_model.feature_extractor(x_adv)

                score = torch.exp(-args.robust_beta * ((feat_clean - feat_adv)**2)) 
                score = score.reshape(N,-1)
                teacher_out = origin_model.forward_with_score(feat_clean, score)
     
                # kl_loss = criterion_kl(F.log_softmax(adv_out, dim=1), F.softmax(teacher_out.detach(), dim=1))
                kl_loss = ((adv_out - teacher_out)**2).mean()
                loss = XENT_loss(adv_out,y) + args.robust_alpha * kl_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
               
            adjust_learning_rate(lr, optimizer, epoch, epochs)
        return adv_batch_set

def robust_finetune(model, train_loader, args, logits=False):
    # Todo
    origin_model = copy.deepcopy(model)
    origin_model.eval()
    model.train()
    lr = 0.001
    optimizer = optim.Adam(model.parameters(), lr = lr)
    pgd_attack = torchattacks.PGD(model, eps=8/255, alpha=2/225, steps=args.robust_steps, random_start=True)
    criterion_kl = nn.KLDivLoss(size_average=False)
    XENT_loss = nn.CrossEntropyLoss()

    adv_batch_set = []

    for epoch in range(args.robust_epochs):

        for i, data in enumerate(train_loader):
            if args.debug_mode and i > 3:
                break
            if logits:
                inputs, labels, not_aug_inputs, logits = data
                inputs = inputs.cuda()
                labels = labels.cuda()
                not_aug_inputs = not_aug_inputs.cuda()
                logits = logits.cuda()
                loss = model.meta_observe(inputs, labels, not_aug_inputs, logits)
            else:
                inputs, labels, not_aug_inputs = data
                inputs, labels = inputs.cuda(), labels.cuda()

                N = inputs.shape[0]
                inputs_adv = pgd_attack(inputs,labels)
                
                adv_out = model(inputs_adv)

                
                feat_clean = origin_model(inputs, returnt='features')
                feat_adv = origin_model(inputs_adv, returnt='features')

                score = torch.exp(-args.robust_beta * ((feat_clean - feat_adv)**2)) 
                score = score.reshape(N,-1)
                teacher_out = origin_model.forward_with_score(feat_clean, score)
    
                kl_loss = criterion_kl(F.log_softmax(adv_out, dim=1), F.softmax(teacher_out.detach(), dim=1))
                
                loss = XENT_loss(adv_out,labels) + args.robust_alpha * (1.0 / N) *  kl_loss 

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        adjust_learning_rate(lr, optimizer, epoch, args.robust_epochs)
    return 


def robust_finetune_naive(model, train_loader, args, logits=False):
    origin_model = copy.deepcopy(model)
    origin_model.eval()
    model.train()
    lr = 0.001
    optimizer = optim.Adam(model.parameters(), lr = lr)
    pgd_attack = torchattacks.PGD(model, eps=8/255, alpha=2/225, steps=args.robust_steps, random_start=True)
    criterion_kl = nn.KLDivLoss(size_average=False)
    XENT_loss = nn.CrossEntropyLoss()

    for epoch in range(args.robust_epochs):
        for i, data in enumerate(train_loader):
            if args.debug_mode and i > 3:
                break
            inputs, labels, not_aug_inputs = data
            inputs, labels = inputs.cuda(), labels.cuda()

            inputs_adv = pgd_attack(inputs,labels)

            adv_out = model(inputs_adv)

            loss = XENT_loss(adv_out,labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        adjust_learning_rate(lr, optimizer, epoch, args.robust_epochs)
    return



def adjust_learning_rate(lr, optimizer, epoch, epochs):
    criteria = epochs // 2
    max_lr = lr
    if epoch + 1 <= criteria :
        lr = max_lr/criteria * (epoch + 1)
    else  :
        # lr = max_lr/criteria * (10 - epoch)
        lr = (1/(2**(epoch + 1 - criteria)))*max_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr