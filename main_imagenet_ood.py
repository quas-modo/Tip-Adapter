import os
import random
import argparse
import yaml
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn

from datasets import build_ood_dataset
from datasets.imagenet import ImageNet
import clip
from datasets.utils import build_data_loader
from utils import *
from log_utils import *

from datetime import datetime


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id_config', dest='id_config', help='in domain dataset settings in yaml format')
    parser.add_argument('--ood_config', dest="ood_config", help="out of domain dataset settings in yaml format")
    args = parser.parse_args()

    return args


def run_tip_adapter(cfg, cache_keys, cache_values, test_features, test_labels, clip_weights):
    # Zero-shot CLIP
    clip_logits = 100. * test_features @ clip_weights
    acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(acc))

    # Tip-Adapter
    beta, alpha = cfg['init_beta'], cfg['init_alpha']

    affinity = test_features @ cache_keys
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values

    tip_logits = clip_logits + cache_logits * alpha
    acc = cls_acc(tip_logits, test_labels)
    print("**** Tip-Adapter's test accuracy: {:.2f}. ****\n".format(acc))

    # Search Hyperparameters
    _ = search_hp(cfg, cache_keys, cache_values, test_features, test_labels, clip_weights)


def run_tip_adapter_F_ood(log, cfg, cache_keys, cache_values, test_features, test_labels, clip_weights, clip_model,
                      train_loader_F, open_features, open_labels):
    # Enable the cached keys to be learnable
    adapter = nn.Linear(cache_keys.shape[0], cache_keys.shape[1], bias=False).to(clip_model.dtype).cuda()
    adapter.weight = nn.Parameter(cache_keys.t())

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=cfg['lr'], eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['train_epoch'] * len(train_loader_F))

    beta, alpha = cfg['init_beta'], cfg['init_alpha']
    best_acc, best_epoch = 0.0, 0

    for train_idx in range(cfg['train_epoch']):
        # Train
        adapter.train()
        correct_samples, all_samples = 0, 0
        loss_list = []
        log.debug('Train Epoch: {:} / {:}'.format(train_idx, cfg['train_epoch']))

        for i, (images, target) in enumerate(tqdm(train_loader_F)):
            images, target = images.cuda(), target.cuda()
            with torch.no_grad():
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)

            affinity = adapter(image_features)
            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
            clip_logits = 100. * image_features @ clip_weights
            tip_logits = clip_logits + cache_logits * alpha

            loss = F.cross_entropy(tip_logits, target)

            # todo: add auroc loss

            acc = cls_acc(tip_logits, target)
            correct_samples += acc / 100 * len(tip_logits)
            all_samples += len(tip_logits)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        log.debug('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples,
                                                                       correct_samples, all_samples,
                                                                       sum(loss_list) / len(loss_list)))

        # Eval
        adapter.eval()

        affinity = adapter(test_features)
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        clip_logits = 100. * test_features @ clip_weights
        tip_logits = clip_logits + cache_logits * alpha
        acc = cls_acc(tip_logits, test_labels)

        log.debug("**** Tip-Adapter-F's test accuracy: {:.2f}. ****\n".format(acc))

        open_affinity = adapter(open_features)
        open_cache_logits = ((-1) * (beta - beta * open_affinity)).exp() @ cache_values
        open_logits = 100. * open_features @ clip_weights
        open_tip_logits = open_logits + open_cache_logits * alpha

        auroc, aupr, fpr = cls_auroc_mcm(tip_logits, open_tip_logits, 1)
        log.debug("**** Tip-Adapter's test auroc, aupr, fpr: {:.2f}, {:.2f}, {:.2f}. ****\n".format(auroc, aupr, fpr))

        if acc > best_acc:
            best_acc = acc
            best_epoch = train_idx
            torch.save(adapter.weight, cfg['cache_dir'] + "/best_F_" + str(cfg['shots']) + "shots.pt")

    adapter.weight = torch.load(cfg['cache_dir'] + "/best_F_" + str(cfg['shots']) + "shots.pt")
    log.debug(f"**** After fine-tuning, Tip-Adapter-F's best test accuracy: {best_acc:.2f}, at epoch: {best_epoch}. ****\n")

    # Search Hyperparameters
    # todo: cache_keys? affinity?
    _ = search_hp_ood(log, cfg, cache_keys, cache_values, test_features, test_labels, open_features, open_labels, clip_weights, adapter=adapter)


def run_tip_adapter_ood(log, cfg, cache_keys, cache_values, test_features, test_labels, clip_weights, open_features,
                        open_labels):
    # Zero-shot CLIP
    clip_logits = 100. * test_features @ clip_weights
    acc = cls_acc(clip_logits, test_labels)
    log.debug("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(acc))

    # Tip-Adapter
    beta, alpha = cfg['init_beta'], cfg['init_alpha']

    affinity = test_features @ cache_keys
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values

    tip_logits = clip_logits + cache_logits * alpha
    acc = cls_acc(tip_logits, test_labels)
    log.debug("**** Tip-Adapter's test accuracy: {:.2f}. ****\n".format(acc))

    # Tip-Adapter-Auroc
    open_logits = 100. * open_features @ clip_weights
    open_affinity = open_features @ cache_keys
    open_cache_logits = ((-1) * (beta - beta * open_affinity)).exp() @ cache_values
    open_tip_logits = open_logits + open_cache_logits * alpha

    auroc, aupr, fpr = cls_auroc_mcm(tip_logits, open_tip_logits, 1)
    log.debug("**** Tip-Adapter's val auroc, aupr, fpr: {:.2f}, {:.2f}, {:.2f}. ****\n".format(auroc, aupr, fpr))

    # Search Hyperparameters
    #_ = search_hp_ood(log, cfg, cache_keys, cache_values, test_features, test_labels, open_features, open_labels,
    #                  clip_weights)


def cal_dual_logits(features, pos_clip_weights, neg_clip_weights, cache_keys, cache_values, 
                    top_indices, ood_indices, cfg):
    beta, alpha = cfg['init_beta'], cfg['init_alpha']
    pos_cache_keys = cache_keys[top_indices, :]
    neg_cache_keys = cache_keys[ood_indices, :]

    # pos_features = features[:, top_indices]
    # neg_features = features[:, ood_indices]

    max_pool = nn.MaxPool1d(2, stride=2)
    pos_features = max_pool(features)
    pos_features = pos_features.view(-1, 256)
    pos_features /= pos_features.norm(p=2, dim=1, keepdim=True)

    # tip pos adapter
    pos_zero_logits = 100. * features @ pos_clip_weights
    pos_affinity = pos_features @ pos_cache_keys
    pos_tip_logits = ((-1) * (beta - beta * pos_affinity)).exp() @ cache_values
    pos_logits = pos_zero_logits + pos_tip_logits  * alpha

    # tip neg adapter
    # neg_zero_logits = 100. * (1 - features @ neg_clip_weights)
    # neg_affinity = neg_features @ neg_cache_keys
    # neg_tip_logits = ((-1) * beta * neg_affinity).exp() @ cache_values
    # neg_logits = neg_zero_logits + neg_tip_logits * alpha

    neg_zero_logits = 100. * features @ neg_clip_weights
    neg_affinity = pos_features @ neg_cache_keys
    neg_tip_logits = ((-1) * (beta - beta * neg_affinity)).exp() @ cache_values
    neg_logits = neg_zero_logits + neg_tip_logits * alpha

    return pos_logits, neg_logits

def cal_dual_logits_ab(features, pos_clip_weights, neg_clip_weights, cache_keys, cache_values, 
                    top_indices, ood_indices):
    pos_cache_keys = cache_keys[top_indices, :]
    neg_cache_keys = cache_keys[ood_indices, :]

    # pos_features = features[:, top_indices]
    # neg_features = features[:, ood_indices]

    max_pool = nn.MaxPool1d(2, stride=2)
    pos_features = max_pool(features)
    pos_features = pos_features.view(-1, 256)
    pos_features /= pos_features.norm(p=2, dim=1, keepdim=True)

    # tip pos adapter
    pos_zero_logits = 100. * features @ pos_clip_weights
    pos_affinity = pos_features @ pos_cache_keys
    # pos_tip_logits = ((-1) * (beta - beta * pos_affinity)).exp() @ cache_values
    # pos_logits = pos_zero_logits + pos_tip_logits  * alpha

    # neg_zero_logits = 100. * (1 - features @ neg_clip_weights)
    # zero_scale = pos_zero_logits.mean() / neg_zero_logits.mean()
    # neg_zero_logits = neg_zero_logits * zero_scale
    # neg_affinity = neg_features @ neg_cache_keys
    # neg_tip_logits = ((-1) * beta * neg_affinity).exp() @ cache_values
    # tip_scale = pos_tip_logits.mean() / neg_tip_logits
    # neg_tip_logits = neg_tip_logits * tip_scale
    # neg_logits = neg_zero_logits + neg_tip_logits * alpha

    neg_zero_logits = 100. * features @ neg_clip_weights
    # zero_scale = pos_zero_logits.mean() / neg_zero_logits.mean()
    # neg_zero_logits = neg_zero_logits * zero_scale
    neg_affinity = pos_features @ neg_cache_keys
    # aff_scale = pos_affinity.mean() / neg_affinity.mean()
    # neg_affinity = neg_affinity * aff_scale
    # neg_tip_logits = ((-1) * (beta - beta * neg_affinity)).exp() @ cache_values
    # neg_logits = neg_zero_logits + neg_tip_logits * alpha

    return pos_zero_logits, pos_affinity, neg_zero_logits, neg_affinity

def cal_loss_auroc(logits, pos_cate_num):
    to_np = lambda x: x.data.cpu().numpy()

    logits /= 100.0
    logits = to_np(F.softmax(logits, dim=1))
    pos_half = np.max(logits[:, :pos_cate_num], axis=1)
    neg_half = np.max(logits[:, pos_cate_num:], axis=1)

    condition = pos_half < neg_half
    indices = np.where(condition)[0]
    # print("*** pos_half < neg_half indices")
    # print(indices.shape)

    # tmp_condition = pos_half > neg_half
    # tmp_indices = np.where(tmp_condition)[0]
    # print("*** pos_half < neg_half  indices")
    # print(tmp_indices.shape)

    p = torch.tensor(neg_half[indices], dtype=torch.float32)
    # print("*** torch.tensor(neg_half[indices]): *** ")
    # print(p.shape)
    # print(p)
    if p.shape[0] == 0:
        return torch.tensor([0]).cuda()
    return -torch.mean(torch.sum(p * torch.log(p + 1e-5)), 0)


def APE(log, cfg, cache_keys, cache_values,  test_features, test_labels, pos_clip_weights, neg_clip_weights,
        open_features, open_labels):

    cfg['w'] = cfg['w_training_free']
    top_indices, ood_indices = cal_criterion(cfg, pos_clip_weights, cache_keys, only_use_txt=True)

    id_pos_logits, id_neg_logits = cal_dual_logits(test_features, pos_clip_weights, neg_clip_weights, cache_keys, cache_values,
                                top_indices, ood_indices, cfg)
    
    ood_pos_logits, ood_neg_logits = cal_dual_logits(open_features, pos_clip_weights, neg_clip_weights, cache_keys, cache_values,
                                top_indices, ood_indices, cfg)
    
    
    
    pos_acc = cls_acc(id_pos_logits, test_labels)
    log.debug("**** pos test accuracy: {:.2f}. ****\n".format(pos_acc))
    neg_acc = cls_acc(- id_neg_logits, test_labels)
    log.debug("**** neg test(-) accuracy: {:.2f}. ****\n".format(neg_acc))
    neg_acc = cls_acc(id_neg_logits, test_labels)
    log.debug("**** neg test(+) accuracy: {:.2f}. ****\n".format(neg_acc))
    dual_acc = cls_acc(id_pos_logits - id_neg_logits, test_labels)
    log.debug("**** dual test accuracy: {:.2f}. ****\n".format(dual_acc))
    
    auroc, _ , fpr = cls_auroc_mcm(id_pos_logits, ood_pos_logits)
    log.debug("**** Our's test pos auroc: {:.2f}, fpr: {:.2f}. ****\n".format(auroc, fpr))

    auroc, _ , fpr = cls_auroc_mcm(- id_neg_logits, - ood_neg_logits)
    log.debug("**** Our's test neg(-) auroc: {:.2f}, fpr: {:.2f}. ****\n".format(auroc, fpr))

    auroc, _ , fpr = cls_auroc_mcm(id_neg_logits, ood_neg_logits)
    log.debug("**** Our's test neg(+) auroc: {:.2f}, fpr: {:.2f}. ****\n".format(auroc, fpr))

    auroc, _, fpr = cls_auroc_mcm(id_pos_logits - id_neg_logits, ood_pos_logits - ood_neg_logits)
    log.debug("**** Our's test dual auroc: {:.2f}, fpr: {:.2f}. ****\n".format(auroc, fpr))

    id_logits = torch.cat([id_pos_logits, id_neg_logits], dim=1)
    ood_logits = torch.cat([ood_pos_logits, ood_neg_logits], dim=1)

    # id_logits /= 100
    # id_logits = F.softmax(id_logits, dim=1)
    # our_acc = cls_acc(id_logits, test_labels)
    # log.debug("**** our dual test accuracy: {:.2f}. ****\n".format(our_acc))

    id_logits = torch.cat([id_pos_logits, id_neg_logits], dim=1)
    ood_logits = torch.cat([ood_pos_logits, ood_neg_logits], dim=1)

    auroc, fpr = cls_auroc_ours(id_logits, ood_logits)
    log.debug("**** Our's test dual-ours auroc : {:.2f}, fpr: {:.2f}. ****\n".format(auroc, fpr))

    beta_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in
                     range(cfg['search_step'][0])]
    alpha_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in
                    range(cfg['search_step'][1])]
    
    id_pzero, id_paff, id_nzero, id_naff = cal_dual_logits_ab(test_features, pos_clip_weights, neg_clip_weights, cache_keys, cache_values,
                    top_indices, ood_indices)
    ood_pzero, ood_paff, ood_nzero, ood_naff = cal_dual_logits_ab(open_features, pos_clip_weights, neg_clip_weights, cache_keys, cache_values,
                        top_indices, ood_indices)

    # for beta in beta_list:
    #     for alpha in alpha_list:

    #         id_pos_logits = id_pzero + alpha * (((-1) * (beta - beta * id_paff)).exp() @ cache_values)
    #         id_neg_logits = id_nzero + alpha * (((-1) * (beta - beta * id_naff)).exp() @ cache_values)
    #         ood_pos_logits = ood_pzero + alpha * (((-1) * (beta - beta * ood_paff)).exp() @ cache_values)
    #         ood_neg_logits = ood_nzero + alpha *  (((-1) * (beta - beta * ood_naff)).exp() @ cache_values)

    #         # print(id_pos_logits[:2, :100])
    #         # print(id_neg_logits[:2, :100])
    #         # print(ood_pos_logits[:2, :100])
    #         # print(ood_neg_logits[:2, :100])

    #         id_logits = torch.cat([id_pos_logits, id_neg_logits], dim=1)
    #         ood_logits = torch.cat([ood_pos_logits, ood_neg_logits], dim=1)

    #         t_list = [0.03, 0.05, 0.07, 0.1, 0.3, 0.5, 1, 2, 3, 5, 10]
    #         for i in t_list:
    #             pos_auroc, _ , fpr = cls_auroc_mcm(id_pos_logits, ood_pos_logits, t = i)
    #             neg_auroc, _ , fpr = cls_auroc_mcm(120 - id_neg_logits, 120 - ood_neg_logits, t = i)
    #             dual_auroc, _, fpr = cls_auroc_mcm(120 + id_pos_logits - id_neg_logits, 120 + ood_pos_logits - ood_neg_logits, t = i)
    #             dual_auroc_ours, fpr = cls_auroc_ours(id_logits, ood_logits, t = i)
    #             log.debug("temperature: {:.2f}, alpha: {:.2f}, beta: {:.2f}, pos_auroc: {:.2f}, neg_auroc: {:.2f}, dual_auroc: {:.2f}, dual_auroc_ours:{:.2f}, auroc_fpr:{:.2f}".format(i, alpha, beta, pos_auroc, 
    #                                                 neg_auroc, dual_auroc, dual_auroc_ours, fpr))


    # for i in range(100):
    #     id_logits = id_pos_logits - 0.1 * i * id_neg_logits
    #     ood_logits = ood_pos_logits - 0.1 * i * ood_neg_logits
    #     i_acc = cls_acc(id_logits, test_labels)
    #     log.debug("**** dual acc {:.2f}: {:.2f}".format(i * 0.1, i_acc))
    #     auroc, _, fpr = cls_auroc_mcm(id_logits, ood_logits)
    #     log.debug("**** Our's test dual auroc {:.2f}: {:.2f}, fpr: {:.2f}. ****\n".format(i * 0.1, auroc, fpr))


    #  clipn trial

    # idex = torch.argmax(id_pos_logits, -1).unsqueeze(-1)
    # yesno = torch.cat([id_pos_logits.unsqueeze(-1), id_neg_logits.unsqueeze(-1)], -1)
    # yesno = torch.softmax(yesno, dim=-1)[:,:,0]
    # yesno_s = torch.gather(yesno, dim=1, index=idex)
    # ind_ctw = list(yesno_s.detach().cpu().numpy())
    # ind_atd = list((yesno * torch.softmax(id_pos_logits, -1)).sum(1).detach().cpu().numpy())
    
    # o_idex = torch.argmax(ood_pos_logits, -1).unsqueeze(-1)
    # o_yesno = torch.cat([ood_pos_logits.unsqueeze(-1), ood_neg_logits.unsqueeze(-1)], -1)
    # o_yesno = torch.softmax(o_yesno, dim=-1)[:,:,0]
    # o_yesno_s = torch.gather(o_yesno, dim=1, index=o_idex)

    # ood_ctw = list(o_yesno_s.detach().cpu().numpy())
    # ood_atd = list((o_yesno * torch.softmax(ood_pos_logits, -1) ).sum(1).detach().cpu().numpy())

    # ctw_auroc, fpr = cal_auc_fpr(ind_ctw, ood_ctw)
    # log.debug("**** Our's test ctw auroc: {:.2f}, fpr: {:.2f}. ****\n".format(ctw_auroc, fpr))

    # atd_auroc, fpr = cal_auc_fpr(ind_atd, ood_atd)
    # log.debug("**** Our's test atd auroc: {:.2f}, fpr: {:.2f}. ****\n".format(atd_auroc, fpr))



    # softmax trial

    # print(id_pos_logits)
    # print(id_neg_logits)
    # print(ood_pos_logits)
    # print(ood_neg_logits)

    # id_pos_logits = F.softmax(id_pos_logits, dim=1)
    # # print(id_pos_logits)
    # id_neg_logits = F.softmax(id_neg_logits, dim=1)
    # # print(id_neg_logits)
    # ood_pos_logits = F.softmax(ood_pos_logits, dim=1)
    # # print(ood_pos_logits)
    # ood_neg_logits = F.softmax(ood_neg_logits, dim=1)
    # # print(ood_neg_logits)
    # id_logits = 1000.0 * (id_pos_logits + id_neg_logits)
    # ood_logits = 1000.0 * (ood_pos_logits + ood_neg_logits) 
    # print(id_logits)
    # print(ood_logits)

    # id_pos_logits = F.normalize(id_pos_logits, dim=-1)
    # id_neg_logits = F.normalize(id_neg_logits, dim=-1)
    # id_logits = id_pos_logits + id_neg_logits
    # print(id_pos_logits)
    # print(id_neg_logits)
    # print(id_logits)
    # ood_pos_logits = F.normalize(ood_pos_logits, dim=-1)
    # ood_neg_logits = F.normalize(ood_neg_logits, dim=-1)
    # ood_logits = ood_pos_logits + ood_neg_logits
    # print(ood_pos_logits)
    # print(id_neg_logits)
    # print(ood_logits)

    # auroc, _ , fpr = cls_auroc_mcm(id_logits, ood_logits)
    # log.debug("**** Our's test dual auroc: {:.2f}, fpr: {:.2f}. ****\n".format(auroc, fpr))

    # best_beta, best_alpha = search_hp_ape(log, cfg, new_cache_keys, new_cache_values, test_features, new_test_features, test_labels,
    #                                       open_features, new_open_features, open_labels, zero_clip_weights)


def APE_ood(log, cfg, cache_keys, cache_values, test_features, test_labels, pos_clip_weights, neg_clip_weights, 
            clip_model, train_loader_F, open_features, open_labels):
    cfg['w'] = cfg['w_training_free']
    top_indices, ood_indices = cal_criterion(cfg, pos_clip_weights, cache_keys, only_use_txt=True)

    pos_cache_keys = cache_keys[top_indices,:]
    neg_cache_keys = cache_keys[ood_indices,:]

    # pos_test_features = test_features[:, top_indices]
    # neg_test_features = test_features[:, ood_indices]

    # pos_open_features = open_features[:, top_indices]
    # neg_open_features = open_features[:, ood_indices]

    max_pool = nn.MaxPool1d(2, stride=2)
    pos_test_features = max_pool(test_features)
    pos_test_features = pos_test_features.view(-1, 256)
    pos_test_features /= pos_test_features.norm(p=2, dim=1, keepdim=True)

    pos_open_features = max_pool(open_features)
    pos_open_features = pos_open_features.view(-1, 256)
    pos_open_features /= pos_open_features.norm(p=2, dim=1, keepdim=True)

    pos_adapter = nn.Linear(pos_cache_keys.shape[0], pos_cache_keys.shape[1], bias=False).to(clip_model.dtype).cuda()
    pos_adapter.weight = nn.Parameter(pos_cache_keys.t())

    neg_adapter = nn.Linear(neg_cache_keys.shape[0], neg_cache_keys.shape[1], bias=False).to(clip_model.dtype).cuda()
    neg_adapter.weight = nn.Parameter(neg_cache_keys.t())

    pos_optimizer = torch.optim.AdamW(pos_adapter.parameters(), lr=cfg['lr'], eps=1e-4)
    pos_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(pos_optimizer, cfg['train_epoch'] * len(train_loader_F))

    neg_optimizer = torch.optim.AdamW(neg_adapter.parameters(), lr=cfg['lr'], eps=1e-4)
    neg_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(neg_optimizer, cfg['train_epoch'] * len(train_loader_F))

    beta, alpha = cfg['init_beta'], cfg['init_alpha']
    best_score, best_acc, best_auroc, best_fpr, best_epoch = 0.0, 0.0, 0.0, 0.0, 0

    for train_idx in range(cfg['train_epoch']):
        # Train
        neg_adapter.train()

        correct_samples, all_samples = 0, 0
        loss_list = []
        log.debug('Train Epoch: {:} / {:}'.format(train_idx, cfg['train_epoch']))

        for i, (images, target) in enumerate(tqdm(train_loader_F)):
            images, target = images.cuda(), target.cuda()
            with torch.no_grad():
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)

            pos_image_features = max_pool(image_features)
            pos_image_features = pos_image_features.view(-1, 256)
            pos_image_features /= pos_image_features.norm(p=2, dim=1, keepdim=True)

            # neg_zero_logits = 100. * (1 - image_features @ neg_clip_weights)
            # neg_affinity = neg_adapter(neg_image_features)
            # neg_tip_logits =  ((-1) * beta * neg_affinity).exp() @ cache_values
            # neg_logits = neg_zero_logits + neg_tip_logits * alpha
            # neg_logits_re = 150 - neg_logits 

            neg_zero_logits = 100. * image_features @ neg_clip_weights
            neg_affinity = neg_adapter(pos_image_features)
            neg_tip_logits = ((-1) * (beta - beta * neg_affinity)).exp() @ cache_values
            neg_logits = neg_zero_logits + neg_tip_logits * alpha

            loss_neg_acc = F.cross_entropy(neg_logits, target)
            neg_acc = cls_acc(neg_logits, target)

            neg_optimizer.zero_grad()
            loss_neg_acc.backward()
            neg_optimizer.step()
            neg_scheduler.step()

            correct_samples += neg_acc / 100 * len(neg_logits)
            all_samples += len(neg_logits)
            loss_list.append(loss_neg_acc.item())

        current_lr = neg_scheduler.get_last_lr()[0]
        log.debug('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples,
                                                                       correct_samples, all_samples,
                                                                       sum(loss_list) / len(loss_list)))

        # Eval
        neg_adapter.eval()
        # neg_zero_logits = 100. * (1 - test_features @ neg_clip_weights)
        # neg_affinity = neg_adapter(neg_test_features)
        # neg_tip_logits = ((-1) * beta * neg_affinity).exp() @ cache_values
        # neg_logits = neg_zero_logits + neg_tip_logits * alpha
        # neg_logits_re = 120 - neg_logits 
        neg_zero_logits = 100. * test_features @ neg_clip_weights
        neg_affinity = neg_adapter(pos_test_features)
        neg_tip_logits = ((-1) * (beta - beta * neg_affinity)).exp() @ cache_values
        neg_logits = neg_zero_logits + neg_tip_logits * alpha
        neg_acc = cls_acc(neg_logits, test_labels)
        log.debug("**** Dual-F's neg test accuracy: {:.2f}. ****\n".format(neg_acc))

    for train_idx in range(cfg['train_epoch']):
        # Train
        pos_adapter.train()

        correct_samples, all_samples = 0, 0
        loss_list = []
        log.debug('Train Epoch: {:} / {:}'.format(train_idx, cfg['train_epoch']))

        for i, (images, target) in enumerate(tqdm(train_loader_F)):
            images, target = images.cuda(), target.cuda()
            with torch.no_grad():
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)

            # pos_image_features = image_features[:, top_indices]
            pos_image_features = max_pool(image_features)
            pos_image_features = pos_image_features.view(-1, 256)
            pos_image_features /= pos_image_features.norm(p=2, dim=1, keepdim=True)

            pos_zero_logits = 100. * image_features @ pos_clip_weights
            pos_affinity = pos_adapter(pos_image_features)
            pos_tip_logits = ((-1) * (beta - beta * pos_affinity)).exp() @ cache_values
            pos_logits = pos_zero_logits + pos_tip_logits * alpha

            loss_pos_acc = F.cross_entropy(pos_logits, target)
            pos_acc = cls_acc(pos_logits, target)

            neg_zero_logits = 100. * image_features @ neg_clip_weights
            neg_affinity = neg_adapter(pos_image_features)
            neg_tip_logits = ((-1) * (beta - beta * neg_affinity)).exp() @ cache_values
            neg_logits = neg_zero_logits + neg_tip_logits * alpha

            logits = torch.cat([pos_logits, neg_logits], dim=1)
            pos_cate = pos_logits.shape[1]

            # loss_auroc = cal_loss_auroc(logits, pos_cate)
            # loss = loss_pos_acc + loss_auroc

            pos_optimizer.zero_grad()
            loss_pos_acc.backward()
            pos_optimizer.step()
            pos_scheduler.step()

            correct_samples += pos_acc / 100 * len(pos_logits)
            all_samples += len(pos_logits)
            loss_list.append(loss_pos_acc.item())

        current_lr = pos_scheduler.get_last_lr()[0]
        log.debug('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples,
                                                                       correct_samples, all_samples,
                                                                       sum(loss_list) / len(loss_list)))

        # Eval
        pos_adapter.eval()
        pos_zero_logits = 100. * test_features @ pos_clip_weights
        pos_affinity = pos_adapter(pos_test_features)
        pos_tip_logits = ((-1) * (beta - beta * pos_affinity)).exp() @ cache_values
        pos_logits = pos_zero_logits + pos_tip_logits * alpha
        pos_acc = cls_acc(pos_logits, test_labels)
        log.debug("**** Dual-F's pos test accuracy: {:.2f}. ****\n".format(pos_acc))

    pos_zero_logits = 100. * test_features @ pos_clip_weights
    pos_affinity = pos_adapter(pos_test_features)
    pos_tip_logits = ((-1) * (beta - beta * pos_affinity)).exp() @ cache_values
    pos_logits = pos_zero_logits + pos_tip_logits * alpha

    neg_zero_logits = 100. * test_features @ neg_clip_weights
    neg_affinity = neg_adapter(pos_test_features)
    neg_tip_logits = ((-1) * (beta - beta * neg_affinity)).exp() @ cache_values
    neg_logits = neg_zero_logits + neg_tip_logits * alpha

    pos_acc = cls_acc(pos_logits, test_labels)
    log.debug("**** Dual's pos test accuracy: {:.2f}. ****\n".format(pos_acc))

    neg_acc = cls_acc(neg_logits, test_labels)
    log.debug("**** Dual's neg test accuracy: {:.2f}. ****\n".format(neg_acc))

    dual_acc = cls_acc(pos_logits + neg_logits, test_labels)
    log.debug("**** Dual's dual test accuracy: {:.2f}. ****\n".format(dual_acc))

    open_pos_zero_logits = 100. * open_features @ pos_clip_weights
    open_pos_affinity = pos_adapter(pos_open_features)
    open_pos_tip_logits = ((-1) * (beta - beta * open_pos_affinity)).exp() @ cache_values
    open_pos_logits = open_pos_zero_logits + open_pos_tip_logits * alpha

    open_neg_zero_logits = 100. * open_features @ neg_clip_weights
    open_neg_affinity = neg_adapter(pos_open_features)
    open_neg_tip_logits = ((-1) * (beta - beta * open_neg_affinity)).exp() @ cache_values
    open_neg_logits = open_neg_zero_logits + open_neg_tip_logits * alpha

    auroc, _ , fpr = cls_auroc_mcm(pos_logits, open_pos_logits)
    log.debug("**** Our's test pos auroc: {:.2f}, fpr: {:.2f}. ****\n".format(auroc, fpr))

    auroc, _ , fpr = cls_auroc_mcm(neg_logits, open_neg_logits)
    log.debug("**** Our's test neg auroc: {:.2f}, fpr: {:.2f}. ****\n".format(auroc, fpr))

    auroc, _, fpr = cls_auroc_mcm(pos_logits + neg_logits, open_pos_logits + open_neg_logits)
    log.debug("**** Our's test dual auroc: {:.2f}, fpr: {:.2f}. ****\n".format(auroc, fpr))

    id_logits = torch.cat([pos_logits, neg_logits], dim=1)
    ood_logits = torch.cat([open_pos_logits, open_neg_logits], dim=1)

    auroc, fpr = cls_auroc_ours(id_logits, ood_logits)
    log.debug("**** Our's test dual-ours auroc : {:.2f}, fpr: {:.2f}. ****\n".format(auroc, fpr))

    beta_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in
                range(cfg['search_step'][0])]
    alpha_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in
                    range(cfg['search_step'][1])]
    
    id_pzero = 100. * test_features @ pos_clip_weights
    id_paff = pos_adapter(pos_test_features)
    id_nzero = 100. * test_features @ neg_clip_weights
    id_naff = neg_adapter(pos_test_features)

    ood_pzero = 100. * open_features @ pos_clip_weights
    ood_paff = pos_adapter(pos_open_features)
    ood_nzero = 100. * open_features @ neg_clip_weights
    ood_naff = neg_adapter(pos_open_features)

    for beta in beta_list:
            for alpha in alpha_list:
                id_pos_logits = id_pzero + alpha * (((-1) * (beta - beta * id_paff)).exp() @ cache_values)
                id_neg_logits = id_nzero + alpha * (((-1) * (beta - beta * id_naff)).exp() @ cache_values)
                ood_pos_logits = ood_pzero + alpha * (((-1) * (beta - beta * ood_paff)).exp() @ cache_values)
                ood_neg_logits = ood_nzero + alpha *  (((-1) * (beta - beta * ood_naff)).exp() @ cache_values)

                id_logits = torch.cat([id_pos_logits, id_neg_logits], dim=1)
                ood_logits = torch.cat([ood_pos_logits, ood_neg_logits], dim=1)

                t_list = [0.03, 0.05, 0.07, 0.1, 0.3, 0.5, 1, 2, 3, 5, 10]
                for i in t_list:
                    pos_auroc, _ , pos_fpr = cls_auroc_mcm(id_pos_logits, ood_pos_logits, t = i)
                    neg_auroc, _ , neg_fpr = cls_auroc_mcm(id_neg_logits, ood_neg_logits, t = i)
                    dual_auroc, _, dual_fpr = cls_auroc_mcm(id_pos_logits + id_neg_logits,ood_pos_logits + ood_neg_logits, t = i)
                    dual_auroc_ours, dual_ours_fpr = cls_auroc_ours(id_logits, ood_logits, t = i)
                    log.debug("temperature: {:.2f}, alpha: {:.2f}, beta: {:.2f}, pos_auroc: {:.2f}, pos_fpr: {:.2f}, neg_auroc: {:.2f}, neg_fpr: {:.2f}, dual_auroc: {:.2f}, dual_fpr: {:.2f}, dual_auroc_ours: {:.2f}, auroc_fpr: {:.2f}".format(i, alpha, beta, pos_auroc, 
                                                        pos_fpr, neg_auroc, neg_fpr, dual_auroc, dual_fpr, dual_auroc_ours, dual_ours_fpr))
        

    

def main():
    # Load config file
    args = get_arguments()
    assert (os.path.exists(args.id_config))
    assert (os.path.exists(args.ood_config))

    # Load configuration
    id_cfg = yaml.load(open(args.id_config, 'r'), Loader=yaml.Loader)
    ood_cfg = yaml.load(open(args.ood_config, 'r'), Loader=yaml.Loader)

    # Set logging
    current = datetime.now()
    formatted_time = current.strftime("%Y_%m_%d_%H_%M_%S")
    args.log_directory = f"logs/{id_cfg['dataset']}/{id_cfg['backbone']}/{ood_cfg['dataset']}/{id_cfg['shots']}/{str(formatted_time)}"
    args.name = "TRAIN_EVAL_INFO"
    os.makedirs(args.log_directory, exist_ok=True)
    log = setup_log(args)

    # Set cache
    id_cache_dir = os.path.join('/home/nfs03/zengtc/tip/caches', id_cfg['dataset'])
    os.makedirs(id_cache_dir, exist_ok=True)
    id_cfg['cache_dir'] = id_cache_dir

    log.debug("\nRunning in-domain dataset configs.")
    log.debug(id_cfg)

    # CLIP
    clip_model, preprocess = clip.load(id_cfg['backbone'])
    clip_model.eval()

    # ImageNet dataset
    random.seed(id_cfg['seed'])
    torch.manual_seed(id_cfg['seed'])

    log.debug("Preparing ImageNet dataset.")
    imagenet = ImageNet(id_cfg['root_path'], id_cfg['shots'], preprocess)

    test_loader = torch.utils.data.DataLoader(imagenet.test, batch_size=64, num_workers=8, shuffle=False)

    train_loader_cache = torch.utils.data.DataLoader(imagenet.train, batch_size=256, num_workers=8, shuffle=False)
    train_loader_F = torch.utils.data.DataLoader(imagenet.train, batch_size=256, num_workers=8, shuffle=True)

    # Textual features
    log.debug("Getting textual features as CLIP's classifier.")
    clip_weights = clip_classifier(imagenet.classnames, imagenet.template, clip_model)
    neg_clip_weights = clip_classifier(imagenet.classnames, imagenet.neg_template, clip_model)

    # Construct the cache model by few-shot training set
    log.debug("\nConstructing cache model by few-shot visual features and labels.")
    cache_keys, cache_values = build_cache_model(log, id_cfg, clip_model, train_loader_cache)

    # Pre-load test features
    log.debug("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(id_cfg, "test", clip_model, test_loader)

    # Load open-set dataset
    log.debug("\nRunning out-domain dataset configs.")
    log.debug(ood_cfg)
    ood_cache_dir = os.path.join('/home/nfs03/zengtc/tip/caches', ood_cfg['dataset'])
    os.makedirs(ood_cache_dir, exist_ok=True)
    ood_cfg['cache_dir'] = ood_cache_dir
    ood_dataset = build_ood_dataset(ood_cfg['dataset'], ood_cfg['root_path'], log)
    ood_loader = build_data_loader(data_source=ood_dataset.all, batch_size=64, is_train=False, tfm=preprocess,
                                   shuffle=False)
    ood_features, ood_labels = pre_load_features(ood_cfg, "ood", clip_model, ood_loader)

    APE(log, id_cfg, cache_keys, cache_values, test_features, test_labels, clip_weights, neg_clip_weights,
        ood_features, ood_labels)

    APE_ood(log, id_cfg, cache_keys, cache_values, test_features, test_labels, clip_weights, neg_clip_weights, clip_model,
            train_loader_F, ood_features, ood_labels)

    # ------------------------------------------ Tip-Adapter ------------------------------------------
    # run_tip_adapter_ood(log, id_cfg, cache_keys, cache_values, test_features, test_labels, clip_weights, ood_features,
    #                     ood_labels)

    # ------------------------------------------ Tip-Adapter-F ------------------------------------------
    # run_tip_adapter_F_ood(log, id_cfg, cache_keys, cache_values, test_features, test_labels, clip_weights, clip_model,
    #                       train_loader_F, ood_features, ood_labels)


if __name__ == '__main__':
    main()