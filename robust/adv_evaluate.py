

import torchattacks
import pdb, os
import numpy as np
from pathlib import Path


import math
import sys
from argparse import Namespace
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import get_dataset
from datasets.utils.continual_dataset import ContinualDataset
from models.utils.continual_model import ContinualModel

from robust.attacks import *
from robust.autoattack import AutoAttack
#from hessian_eigenthings import compute_hessian_eigenthings

def mask_classes(outputs: torch.Tensor, dataset: ContinualDataset, k: int) -> None:
    """
    Given the output tensor, the dataset at hand and the current task,
    masks the former by setting the responses for the other tasks at -inf.
    It is used to obtain the results for the task-il setting.
    :param outputs: the output tensor
    :param dataset: the continual dataset
    :param k: the task index
    """
    outputs[:, 0:k * dataset.N_CLASSES_PER_TASK] = -float('inf')
    outputs[:, (k + 1) * dataset.N_CLASSES_PER_TASK:
               dataset.N_TASKS * dataset.N_CLASSES_PER_TASK] = -float('inf')



def evaluate_PGD(model: ContinualModel, dataset: ContinualDataset) :
    status = model.net.training
    model.net.eval()
    accs, accs_adv = [], []

    num_total_class = dataset.N_TASKS * dataset.N_CLASSES_PER_TASK
    per_class_output = np.zeros((num_total_class, num_total_class))
    per_class_output_adv = np.zeros((num_total_class, num_total_class))

    feat_distance = np.zeros((num_total_class))



    for k, test_loader in enumerate(dataset.test_loaders):
        correct, total = 0.0, 0.0
        correct_adv = 0.0
        for data in test_loader:

            inputs, labels = data
            inputs, labels = inputs.to(model.device), labels.to(model.device)

            inputs_adv = PGD(inputs, labels, model, steps=20)
            
            model.eval()

            with torch.no_grad():
                if 'class-il' not in model.COMPATIBILITY:
                    outputs = model(inputs, k)
                    outputs_adv = model(inputs_adv, k)
                else:
                    outputs = model(inputs)
                    outputs_adv = model(inputs_adv)
                _, feat = model.net(inputs, returnt='all')
                _, feat_adv = model.net(inputs_adv, returnt='all')

                _, pred = torch.max(outputs.data, 1)
                correct += torch.sum(pred == labels).item()

                _, pred_adv = torch.max(outputs_adv.data, 1)
                correct_adv += torch.sum(pred_adv == labels).item()

                total += labels.shape[0]

                for i in range(num_total_class):
                    if len(feat[labels==i])>0:
                        feat_distance[i] += torch.norm((feat[labels==i]-feat_adv[labels==i]), dim=1).sum().item()
                    for k in range(num_total_class):
                        per_class_output[i][k] += ((labels==i)&(pred==k)).sum().item()
                        per_class_output_adv[i][k] += ((labels==i)&(pred_adv==k)).sum().item()
    

        accs.append(correct/total * 100
                    if 'class-il' in model.COMPATIBILITY else 0)

        accs_adv.append(correct_adv/total *100.0
                    if 'class-il' in model.COMPATIBILITY else 0)

    per_class_output = per_class_output/10
    per_class_output_adv = per_class_output_adv/10

    feat_distance = feat_distance/1000
    model.net.train(status)

    return accs, accs_adv, per_class_output, per_class_output_adv, feat_distance



def evaluate_FGSM(model: ContinualModel, dataset: ContinualDataset, eps=8/255) :
    status = model.net.training
    model.net.eval()
    accs, accs_adv = [], []

    num_total_class = dataset.N_TASKS * dataset.N_CLASSES_PER_TASK
    per_class_output = np.zeros((num_total_class, num_total_class))
    per_class_output_adv = np.zeros((num_total_class, num_total_class))

    feat_distance = np.zeros((num_total_class))


    for k, test_loader in enumerate(dataset.test_loaders):
        correct, total = 0.0, 0.0
        correct_adv = 0.0
        for data in test_loader:

            inputs, labels = data
            inputs, labels = inputs.to(model.device), labels.to(model.device)

            inputs_adv = FGSM(inputs, labels, model, eps=eps)
            
            model.eval()

            with torch.no_grad():
                if 'class-il' not in model.COMPATIBILITY:
                    outputs = model(inputs, k)
                    outputs_adv = model(inputs_adv, k)
                else:
                    outputs = model(inputs)
                    outputs_adv = model(inputs_adv)

                _, feat = model.net(inputs, returnt='all')
                _, feat_adv = model.net(inputs_adv, returnt='all')

                _, pred = torch.max(outputs.data, 1)
                correct += torch.sum(pred == labels).item()

                _, pred_adv = torch.max(outputs_adv.data, 1)
                correct_adv += torch.sum(pred_adv == labels).item()

                total += labels.shape[0]

                for i in range(num_total_class):
                    if len(feat[labels==i])>0:
                        feat_distance[i] += torch.norm((feat[labels==i]-feat_adv[labels==i]), dim=1).sum().item()
                    for k in range(num_total_class):
                        per_class_output[i][k] += ((labels==i)&(pred==k)).sum().item()
                        per_class_output_adv[i][k] += ((labels==i)&(pred_adv==k)).sum().item()
    

        accs.append(correct/total * 100
                    if 'class-il' in model.COMPATIBILITY else 0)

        accs_adv.append(correct_adv/total *100.0
                    if 'class-il' in model.COMPATIBILITY else 0)

    per_class_output = per_class_output/10
    per_class_output_adv = per_class_output_adv/10

    feat_distance = feat_distance/1000
    model.net.train(status)
    return accs, accs_adv, per_class_output, per_class_output_adv, feat_distance


def evaluate_AA(model: ContinualModel, dataset: ContinualDataset) :
    status = model.net.training
    model.net.eval()

    autoattack = AutoAttack(model, norm='Linf', eps=8/255.0, version='standard')

    x_total = None
    y_total = None
    for k, test_loader in enumerate(dataset.test_loaders):
        x_task = [x for (x, y) in test_loader]
        y_task = [y for (x, y) in test_loader]

        if x_total is None:
            x_total = torch.cat(x_task, 0)
            y_total = torch.cat(y_task, 0)
        else : 
            x_total = torch.cat((x_total, torch.cat(x_task, 0)),0)
            y_total = torch.cat((y_total, torch.cat(y_task, 0)),0)

    adv_complete, robust_acc = autoattack.run_standard_evaluation(x_total, y_total)
    model.eval()
    model.net.train(status)
    return  robust_acc * 100



def evaluate_curvature(model: ContinualModel, dataset: ContinualDataset, mean, std, last=False) :
    '''
    23-05-18 seungju
    need to install pytorch-hessian-eigenthings
    pip install --upgrade git+https://github.com/noahgolmant/pytorch-hessian-eigenthings.git@master#egg=hessian-eigenthings
    'https://github.com/noahgolmant/pytorch-hessian-eigenthings'
    '''
    
    num_eigenthings = 20
    loss = nn.CrossEntropyLoss()
    status = model.net.training
    model.net.eval()
    eigenval_norms = []



    for k, test_loader in enumerate(dataset.test_loaders):
        if last and k < len(dataset.test_loaders) - 1:
            continue


        eigenvals, eigenvecs = compute_hessian_eigenthings(model.net, test_loader, loss, num_eigenthings)
        eigenval_norms.append(np.linalg.norm(eigenvals))
        
           
        # accs.append(correct/total * 100
        #             if 'class-il' in model.COMPATIBILITY else 0)

    model.net.train(status)
    return eigenval_norms



def evaluate_curvature_input(model: ContinualModel, dataset: ContinualDataset, last=False, args = None) :
    
    num_eigenthings = 20
    loss = nn.CrossEntropyLoss()
    status = model.training
    model.eval()
    eigenval_norms = []
    XENT_loss = nn.CrossEntropyLoss()


    curvatures = [] 
    gradient_norms = []
    for k, test_loader in enumerate(dataset.test_loaders):
        curvature = 0 
        gradient_norm = 0
        for x,y in test_loader : 
            x, y = x.cuda(), y.cuda()
            N,_,_,_ = x.shape
            
            x.requires_grad = True
            h = 0.01 # 0.01
            
            out = model(x)
            loss = XENT_loss(out,y)
            grad_1 = torch.autograd.grad(loss, [x], retain_graph=True, create_graph=True)[0]
            z_ = grad_1/grad_1.reshape(N,-1).norm(dim = 1).reshape(-1,1,1,1)
            
            x_hat = x.detach() + h * z_.detach()
            x_hat.requires_grad = True

            
            out_hat = model(x_hat)
            loss = XENT_loss(out_hat,y)
            grad_2 = torch.autograd.grad(loss, [x_hat], retain_graph=True, create_graph=True)[0]
            
            curvature += (((grad_1 - grad_2)**2)/h**2).detach().sum().item()
            gradient_norm += grad_1.norm().detach().sum().item()
            
        curvatures.append(curvature)
        gradient_norms.append(gradient_norm)
        


    model.train(status)
    return gradient_norms, curvatures

def evaluate_gf_cf(model: ContinualModel, dataset: ContinualDataset, last=False, args = None, MODELS = []) :

    num_eigenthings = 20
    loss = nn.CrossEntropyLoss()
    status = model.training
    model.eval()
    eigenval_norms = []
    XENT_loss = nn.CrossEntropyLoss()
    path = Path(os.path.realpath(__file__))


    GradForgetting = []
    CurvatureForgetting = []
    num_test_examples = 0

    # model_load_path = os.path.join(str(path.parent.absolute().parent.absolute()),'checkpoint/')+ args.model + '/' + args.dataset  + '/' + args.aug +'/' + '/%.1f_%.1f_%.1f_%.1f_task_%d'%(args.alpha, args.beta, args.gamma, args.threshold, 4) + '.pkl'

    # checkpoint = torch.load(model_load_path)
    # model.classes_so_far = torch.arange(0,(4+1) *dataset.N_CLASSES_PER_TASK ) 
    # model.load_state_dict(checkpoint)


    for t, test_loader in enumerate(dataset.test_loaders):
        curvature = 0
        gradient_norm = 0
        # load k th model

        # model_load_path = os.path.join(str(path.parent.absolute().parent.absolute()),'checkpoint/')+ args.model + '/' + args.dataset  + '/' + args.aug +'/' + '/%.1f_%.1f_%.1f_%.1f_task_%d'%(args.alpha, args.beta, args.gamma, args.threshold, t) + '.pkl'

        # past_model = copy.deepcopy(model)
        # checkpoint = torch.load(model_load_path)
        # past_model.classes_so_far = torch.arange(0,(t+1) *dataset.N_CLASSES_PER_TASK ) 
        # past_model.load_state_dict(checkpoint)
        past_model = MODELS[t]


        gf = 0
        gc = 0
        past_model.eval()
        model.eval()

        for idx,(x,y) in enumerate(test_loader) :
            x, y = x.cuda(), y.cuda()

            x.requires_grad = True
            h = 1 # 0.01

            # calculate for correct example
            out = past_model(x)
            idx = out.argmax(dim = 1) == y

            x,y = x[idx],y[idx]
            N,_,_,_ = x.shape

            if N == 0:
                continue
            out = model(x)
            loss = XENT_loss(out,y)
            grad_1_cur = torch.autograd.grad(loss, [x], retain_graph=True, create_graph=True)[0]
            #Normalize
            grad_1_cur_norm = grad_1_cur.reshape(N,-1).norm(dim = 1).reshape(-1,1,1,1)
            grad_1_cur_norm[grad_1_cur_norm == 0] = 1
            grad_1_cur = grad_1_cur/grad_1_cur_norm

            out = past_model(x)
            loss = XENT_loss(out,y)
            grad_1_past = torch.autograd.grad(loss, [x], retain_graph=True, create_graph=True)[0]
            #Normalize
            grad_1_past_norm = grad_1_past.reshape(N,-1).norm(dim = 1).reshape(-1,1,1,1)
            grad_1_past_norm[grad_1_past_norm == 0] = 1
            grad_1_past = grad_1_past/grad_1_past_norm

            gf += ((grad_1_past - grad_1_cur)**2).detach().cpu().sum().item()
            # print (gf)
            # if torch.isnan(torch.tensor(gf)).item() == True:
            #     pdb.set_trace()

            z_past = grad_1_past#/grad_1_past.reshape(N,-1).norm(dim = 1).reshape(-1,1,1,1)
            x_hat_past = x.detach() + h * z_past.detach()
            x_hat_past.requires_grad = True


            out_hat_past = model(x_hat_past)
            loss = XENT_loss(out_hat_past,y)
            grad_2_past = torch.autograd.grad(loss, [x_hat_past], retain_graph=True, create_graph=True)[0]
            #Normalize
            grad_2_past_norm = grad_2_past.reshape(N,-1).norm(dim = 1).reshape(-1,1,1,1)
            grad_2_past_norm[grad_2_past_norm == 0] = 1
            grad_2_past = grad_2_past/grad_2_past_norm


            curvature_past = (grad_1_past - grad_2_past)/h

            z_cur = grad_1_cur
            x_hat_cur = x.detach() + h * z_cur.detach()
            x_hat_cur.requires_grad = True


            out_hat_cur = model(x_hat_cur)
            loss = XENT_loss(out_hat_cur,y)
            grad_2_cur = torch.autograd.grad(loss, [x_hat_cur], retain_graph=True, create_graph=True)[0]
            #Normalize
            grad_2_cur_norm = grad_2_cur.reshape(N,-1).norm(dim = 1).reshape(-1,1,1,1)
            grad_2_cur_norm[grad_2_cur_norm == 0] = 1
            grad_2_cur = grad_2_cur/grad_2_cur_norm



            curvature_cur = (grad_1_cur - grad_2_cur)/h

            gc += ((curvature_past - curvature_cur)**2).detach().cpu().sum().item()

            # if torch.isnan(torch.tensor(gc)).item() == True:
            #     pdb.set_trace()

            num_test_examples += N
            # print (gf,gc)

        GradForgetting.append(gf)
        CurvatureForgetting.append(gc)


    GradForgetting = (sum(GradForgetting)/num_test_examples)
    CurvatureForgetting = (sum(CurvatureForgetting)/num_test_examples)
    model.train(status)
    print (GradForgetting, CurvatureForgetting)

    return GradForgetting, CurvatureForgetting