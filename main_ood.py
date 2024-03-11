#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/5 23:16
# @Author  : quasdo
# @Site    : 
# @File    : main_ood.py
# @Software: PyCharm

import os
import random
import argparse

import numpy as np
import yaml
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms

from datasets import build_dataset, build_ood_dataset
from datasets.utils import build_data_loader
import clip
from utils import *
from log_utils import *

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id_config', dest='id_config', help='in domain dataset settings in yaml format')
    parser.add_argument('--ood_config', dest="ood_config", help="out of domain dataset settings in yaml format")
    args = parser.parse_args()
    return args


def run_tip_adapter_F_ood(log, cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights,
                      clip_model, train_loader_F, open_features, open_labels):
    # Enable the cached keys to be learnable
    adapter = nn.Linear(cache_keys.shape[0], cache_keys.shape[1], bias=False).to(clip_model.dtype).cuda()
    adapter.weight = nn.Parameter(cache_keys.t())

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=cfg['lr'], eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['train_epoch'] * len(train_loader_F))

    beta, alpha = cfg['init_beta'], cfg['init_alpha']
    best_acc, best_epoch, best_auroc = 0.0, 0, 0.0

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

    log.debug("\n-------- Searching hyperparameters on the val set. --------")

    # Search Hyperparameters
    best_beta, best_alpha = search_hp_ood(log, cfg, cache_keys, cache_values, val_features, val_labels,
                                          open_features, open_labels, clip_weights, adapter=adapter)

    log.debug("\n-------- Evaluating on the test set. --------")

    affinity = adapter(test_features)
    cache_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values
    clip_logits = 100. * test_features @ clip_weights
    tip_logits = clip_logits + cache_logits * best_alpha
    acc = cls_acc(tip_logits, test_labels)
    log.debug("**** Tip-Adapter-F's test accuracy: {:.2f}. ****\n".format(max(best_acc, acc)))

    open_affinity = adapter(open_features)
    open_cache_logits = ((-1) * (beta - beta * open_affinity)).exp() @ cache_values
    open_logits = 100. * open_features @ clip_weights
    open_tip_logits = open_logits + open_cache_logits * alpha

    auroc, aupr, fpr = cls_auroc_mcm(tip_logits, open_tip_logits, 1)
    log.debug("**** Tip-Adapter's test auroc, aupr, fpr: {:.2f}, {:.2f}, {:.2f}. ****\n".format(auroc, aupr, fpr))


def run_tip_adapter_ood(log, cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels,
                        clip_weights, open_features, open_labels):
    log.debug("\n-------- Searching hyperparameters on the val set. --------")

    # Zero-shot CLIP
    clip_logits = 100. * val_features @ clip_weights
    # clip_logtis [1633, 102] val_labels (1633,)
    acc = cls_acc(clip_logits, val_labels)
    log.debug("\n**** Zero-shot CLIP's val accuracy: {:.2f}. ****\n".format(acc))

    # Tip-Adapter
    beta, alpha = cfg['init_beta'], cfg['init_alpha']

    affinity = val_features @ cache_keys
    # affinity [1633, 1632]
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
    # cache_logits [1633, 102]
    tip_logits = clip_logits + cache_logits * alpha
    acc = cls_acc(tip_logits, val_labels)
    log.debug("**** Tip-Adapter's val accuracy: {:.2f}. ****\n".format(acc))

    # Tip-Adapter-Auroc
    open_logits = 100. * open_features @ clip_weights
    open_affinity = open_features @ cache_keys
    open_cache_logits = ((-1) * (beta - beta * open_affinity)).exp() @ cache_values
    open_tip_logits = open_logits + open_cache_logits * alpha

    auroc, aupr, fpr = cls_auroc_mcm(tip_logits, open_tip_logits, 1)
    log.debug("**** Tip-Adapter's val auroc, aupr, fpr: {:.2f}, {:.2f}, {:.2f}. ****\n".format(auroc, aupr, fpr))

    # Search Hyperparameters
    best_beta, best_alpha = search_hp_ood(log, cfg, cache_keys, cache_values, val_features, val_labels, open_features, open_labels, clip_weights)

    log.debug("\n-------- Evaluating on the test set. --------")

    # Zero-shot CLIP
    clip_logits = 100. * test_features @ clip_weights
    acc = cls_acc(clip_logits, test_labels)
    log.debug("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(acc))

    # Tip-Adapter acc
    affinity = test_features @ cache_keys
    cache_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values

    tip_logits = clip_logits + cache_logits * best_alpha
    acc = cls_acc(tip_logits, test_labels)
    log.debug("**** Tip-Adapter's test accuracy: {:.2f}. ****\n".format(acc))

    # Tip-Adapter auroc
    open_affinity = open_features @ cache_keys
    open_logits = 100. * open_features @ clip_weights

    open_cache_logits = ((-1) * (best_beta - best_beta * open_affinity)).exp() @ cache_values
    open_tip_logits = open_logits + open_cache_logits * best_alpha

    auroc, aupr, fpr = cls_auroc_mcm(tip_logits, open_tip_logits, 1)
    log.debug("**** Tip-Adapter's test auroc, aupr, fpr: {:.2f}, {:.2f}, {:.2f}. ****\n".format(auroc, aupr, fpr))


def main():
    # Load config file
    args = get_arguments()
    assert (os.path.exists(args.id_config))
    assert (os.path.exists(args.ood_config))

    # Load configuration
    id_cfg = yaml.load(open(args.id_config, 'r'), Loader=yaml.Loader)
    ood_cfg = yaml.load(open(args.ood_config, 'r'), Loader=yaml.Loader)

    # Set logging
    args.log_directory = f"logs/{id_cfg['dataset']}/{id_cfg['backbone']}/{ood_cfg['dataset']}"
    args.name = "TRAIN_EVAL_INFO"
    os.makedirs(args.log_directory, exist_ok=True)
    log = setup_log(args)

    # Set cache
    id_cache_dir = os.path.join('./caches', id_cfg['dataset'])
    os.makedirs(id_cache_dir, exist_ok=True)
    id_cfg['cache_dir'] = id_cache_dir

    log.debug("\nRunning in-domain dataset configs.")
    log.debug(id_cfg)

    # CLIP
    clip_model, preprocess = clip.load(id_cfg['backbone'])
    clip_model.eval()

    # Prepare dataset
    random.seed(1)
    torch.manual_seed(1)

    log.debug("Preparing dataset.")
    # construct dataset and train set
    id_dataset = build_dataset(id_cfg['dataset'], id_cfg['root_path'], id_cfg['shots'])
    id_val_loader = build_data_loader(data_source=id_dataset.val, batch_size=64, is_train=False, tfm=preprocess,
                                      shuffle=False)
    id_test_loader = build_data_loader(data_source=id_dataset.test, batch_size=64, is_train=False, tfm=preprocess,
                                       shuffle=False)

    train_tranform = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),  # 50 percent flip the image
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])

    train_loader_cache = build_data_loader(data_source=id_dataset.train_x, batch_size=256, tfm=train_tranform,
                                           is_train=True, shuffle=False)
    train_loader_F = build_data_loader(data_source=id_dataset.train_x, batch_size=256, tfm=train_tranform,
                                       is_train=True,
                                       shuffle=True)

    # Textual features
    log.debug("\nGetting textual features as CLIP's classifier.")
    clip_weights = clip_classifier(id_dataset.classnames, id_dataset.template, clip_model)

    # Construct the cache model by few-shot training set
    log.debug("\nConstructing cache model by few-shot visual features and labels.")
    cache_keys, cache_values = build_cache_model(log, id_cfg, clip_model, train_loader_cache)

    # Pre-load val features
    log.debug("\nLoading visual features and labels from val set.")
    val_features, val_labels = pre_load_features(id_cfg, "val", clip_model, id_val_loader)

    # Pre-load test features
    log.debug("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(id_cfg, "test", clip_model, id_test_loader)

    # Load open-set dataset
    log.debug("\nRunning out-domain dataset configs.")
    log.debug(ood_cfg)
    ood_cache_dir = os.path.join('./caches', ood_cfg['dataset'])
    os.makedirs(ood_cache_dir, exist_ok=True)
    ood_cfg['cache_dir'] = ood_cache_dir
    ood_dataset = build_ood_dataset(ood_cfg['dataset'], ood_cfg['root_path'], log)
    ood_loader = build_data_loader(data_source=ood_dataset.all, batch_size=64, is_train=False, tfm=preprocess,
                                   shuffle=False)
    ood_features, ood_labels = pre_load_features(ood_cfg, "ood", clip_model, ood_loader)
    # ood_features [5714, 1024]
    # ood_labels (5714,)
    # ------------------------------------------ Tip-Adapter ------------------------------------------
    run_tip_adapter_ood(log, id_cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels,
                        clip_weights, ood_features, ood_labels)



    # ------------------------------------------ Tip-Adapter-F ------------------------------------------
    run_tip_adapter_F_ood(log, id_cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels,
                      clip_weights, clip_model, train_loader_F, ood_features, ood_labels)


if __name__ == '__main__':
    main()
