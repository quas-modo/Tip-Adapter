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

    pos_features = features[:, top_indices]
    neg_features = features[:, ood_indices]

    # tip pos adapter
    pos_zero_logits = 100. * features @ pos_clip_weights
    # print('pos_zero_logits')
    # print(pos_zero_logits)
    pos_affinity = pos_features @ pos_cache_keys
    # print('pos_affinity')
    # print(pos_affinity)
    pos_tip_logits = ((-1) * (beta - beta * pos_affinity)).exp() @ cache_values
    # print('pos_tip_logits')
    # print(pos_tip_logits)
    pos_logits = pos_zero_logits + pos_tip_logits  * alpha
    # print('pos_logits')
    # print(pos_logits)

    # tip neg adapter
    # todo: mean value scaling parameter
    neg_zero_logits = 100. * (1 - features @ neg_clip_weights)
    zero_scale = pos_zero_logits.mean() / neg_zero_logits.mean()
    neg_zero_logits = neg_zero_logits * zero_scale
    # print('neg_zero_logits')
    # print(neg_zero_logits)
    neg_affinity = neg_features @ neg_cache_keys
    # print('neg_affinity')
    # print(neg_affinity)
    neg_tip_logits = ((-1) * beta * neg_affinity).exp() @ cache_values
    tip_scale = pos_tip_logits.mean() / neg_tip_logits
    neg_tip_logits = neg_tip_logits * tip_scale
    
    # print('neg_tip_logits')
    # print(neg_tip_logits)
    neg_logits = neg_zero_logits + neg_tip_logits * alpha
    # print('neg_logits')
    # print(neg_logits)

    # scaling
    # mean_pos = pos_logits.mean()
    # std_pos = pos_logits.std()
    # mean_neg = neg_logits.mean()
    # std_neg = neg_logits.std()

    # scale_factor = std_pos / std_neg
    # offset = mean_pos - (mean_neg * scale_factor)

    # neg_logits = (neg_logits * scale_factor) + offset

    # logits = F.softmax(pos_logits) + F.softmax(neg_logits)

    return pos_logits, neg_logits

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
    log.debug("**** neg test accuracy: {:.2f}. ****\n".format(neg_acc))
    dual_acc = cls_acc(id_pos_logits - id_neg_logits, test_labels)
    log.debug("**** dual test accuracy: {:.2f}. ****\n".format(dual_acc))
    
    auroc, _ , fpr = cls_auroc_mcm(id_pos_logits, ood_pos_logits)
    log.debug("**** Our's test pos auroc: {:.2f}, fpr: {:.2f}. ****\n".format(auroc, fpr))

    auroc, _ , fpr = cls_auroc_mcm(- id_neg_logits, - ood_neg_logits)
    log.debug("**** Our's test neg auroc: {:.2f}, fpr: {:.2f}. ****\n".format(auroc, fpr))

    auroc, _, fpr = cls_auroc_mcm(id_pos_logits - id_neg_logits, ood_pos_logits - ood_neg_logits)
    log.debug("**** Our's test dual auroc: {:.2f}, fpr: {:.2f}. ****\n".format(auroc, fpr))
    
    for i in range(100):
        id_logits = id_pos_logits - 0.1 * i * id_neg_logits
        ood_logits = ood_pos_logits - 0.1 * i * ood_neg_logits
        i_acc = cls_acc(id_logits, test_labels)
        log.debug("**** dual acc {:.2f}: {:.2f}".format(i * 0.1, i_acc))
        auroc, _, fpr = cls_auroc_mcm(id_logits, ood_logits)
        log.debug("**** Our's test dual auroc {:.2f}: {:.2f}, fpr: {:.2f}. ****\n".format(i * 0.1, auroc, fpr))

    idex = torch.argmax(id_pos_logits, -1).unsqueeze(-1)
    yesno = torch.cat([id_pos_logits.unsqueeze(-1), id_neg_logits.unsqueeze(-1)], -1)
    yesno = torch.softmax(yesno, dim=-1)[:,:,0]
    yesno_s = torch.gather(yesno, dim=1, index=idex)
    ind_ctw = list(yesno_s.detach().cpu().numpy())
    ind_atd = list((yesno * torch.softmax(id_pos_logits, -1)).sum(1).detach().cpu().numpy())
    
    o_idex = torch.argmax(ood_pos_logits, -1).unsqueeze(-1)
    o_yesno = torch.cat([ood_pos_logits.unsqueeze(-1), ood_neg_logits.unsqueeze(-1)], -1)
    o_yesno = torch.softmax(o_yesno, dim=-1)[:,:,0]
    o_yesno_s = torch.gather(o_yesno, dim=1, index=o_idex)

    ood_ctw = list(o_yesno_s.detach().cpu().numpy())
    ood_atd = list((o_yesno * torch.softmax(ood_pos_logits, -1) ).sum(1).detach().cpu().numpy())

    ctw_auroc, _, fpr = cal_auc_fpr(ind_ctw, ood_ctw)
    log.debug("**** Our's test ctw auroc: {:.2f}, fpr: {:.2f}. ****\n".format(ctw_auroc, fpr))

    atd_auroc, _, fpr = cal_auc_fpr(ind_atd, ood_atd)
    log.debug("**** Our's test atd auroc: {:.2f}, fpr: {:.2f}. ****\n".format(atd_auroc, fpr))

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

def APE_ood(log, cfg, cache_keys, cache_values, test_features, test_labels, clip_weights, neg_clip_weights, clip_model,
                      train_loader_F, open_features, open_labels):
    cfg['w'] = cfg['w_training_free']
    top_indices, ood_indices = cal_criterion(cfg, clip_weights, cache_keys, only_use_txt=False)

    top_cache_keys = cache_keys[top_indices, :]
    ood_cache_keys = cache_keys[ood_indices, :]
    new_cache_keys = torch.cat((top_cache_keys, ood_cache_keys), dim=1)

    zero_clip_weights = torch.cat((clip_weights, neg_clip_weights), dim=1)
    top_clip_weights = clip_weights[top_indices, :]
    ood_clip_weights = neg_clip_weights[top_indices, :]

    new_test_features = test_features[:, top_indices]
    new_open_features = open_features[:, top_indices]

    new_cache_values = cache_values

    # Enable the cached keys to be learnable
    adapter = nn.Linear(new_cache_keys.shape[0], new_cache_keys.shape[1], bias=False).to(clip_model.dtype).cuda()
    adapter.weight = nn.Parameter(new_cache_keys.t())

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=cfg['lr'], eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['train_epoch'] * len(train_loader_F))

    beta, alpha = cfg['init_beta'], cfg['init_alpha']
    best_score, best_acc, best_auroc, best_fpr, best_epoch = 0.0, 0.0, 0.0, 0.0, 0

    for train_idx in range(cfg['train_epoch']):
        # Train
        adapter.train()
        correct_samples, all_samples = 0, 0
        loss_list = []
        auroc_list = []
        log.debug('Train Epoch: {:} / {:}'.format(train_idx, cfg['train_epoch']))

        for i, (images, target) in enumerate(tqdm(train_loader_F)):
            images, target = images.cuda(), target.cuda()
            with torch.no_grad():
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)

            # acc loss
            new_image_features = image_features[:, top_indices]
            affinity = adapter(new_image_features)
            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ new_cache_values
            clip_logits = 100. * image_features @ zero_clip_weights
            tip_logits = clip_logits + cache_logits * alpha

            sample_num, cate_num = tip_logits.shape   
            pos_cate_num = cate_num // 2
            
            loss_acc = F.cross_entropy(tip_logits[:, :pos_cate_num], target)
            # loss_auroc = cal_loss_auroc(tip_logits, pos_cate_num)
            loss_auroc = 0
            loss = loss_acc + loss_auroc
            # print("***loss_acc: {:.2f}, loss_auroc: {:.2f}, loss: {:.2f}".format(loss_acc, loss_auroc, loss))

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

        affinity = adapter(new_test_features)
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ new_cache_values
        clip_logits = 100. * test_features @ zero_clip_weights
        tip_logits = clip_logits + cache_logits * alpha
        acc = cls_acc(tip_logits, test_labels)

        log.debug("**** Tip-Adapter-F's test accuracy: {:.2f}. ****\n".format(acc))

        open_affinity = adapter(new_open_features)
        open_cache_logits = ((-1) * (beta - beta * open_affinity)).exp() @ new_cache_values
        open_logits = 100. * open_features @ zero_clip_weights
        open_tip_logits = open_logits + open_cache_logits * alpha

        auroc, fpr = cls_auroc_ours(tip_logits, open_tip_logits)
        log.debug("**** Tip-Adapter's test auroc, fpr: {:.2f}, {:.2f}. ****\n".format(auroc, fpr))

        score = 0.4 * acc + 0.6 * auroc

        if auroc > best_auroc:
            best_score = score
            best_acc = acc
            best_auroc = auroc
            best_fpr = fpr
            best_epoch = train_idx
            torch.save(adapter.weight, cfg['cache_dir'] + "/best_F_" + str(cfg['shots']) + "shots.pt")

    adapter.weight = torch.load(cfg['cache_dir'] + "/best_F_" + str(cfg['shots']) + "shots.pt")
    log.debug(f"**** After fine-tuning, Tip-Adapter-F's best test score: {best_score:.2f},  "
              f"acc: {best_acc:.2f}, auroc: {best_auroc:.2f}, fpr: {best_fpr:.2f}"
              f"at epoch: {best_epoch}. ****\n")

    # Search Hyperparameters
    # todo: cache_keys? affinity?
    best_beta, best_alpha = search_hp_ape(log, cfg, new_cache_keys, new_cache_values, test_features, new_test_features, test_labels,
                                          open_features, new_open_features, open_labels, zero_clip_weights, adapter=adapter)


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
    args.log_directory = f"logs/{id_cfg['dataset']}/{id_cfg['backbone']}/{ood_cfg['dataset']}/{str(formatted_time)}"
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

    # APE_ood(log, id_cfg, cache_keys, cache_values, test_features, test_labels, clip_weights, neg_clip_weights, clip_model,
    #         train_loader_F, ood_features, ood_labels)

    # ------------------------------------------ Tip-Adapter ------------------------------------------
    # run_tip_adapter_ood(log, id_cfg, cache_keys, cache_values, test_features, test_labels, clip_weights, ood_features,
    #                     ood_labels)

    # ------------------------------------------ Tip-Adapter-F ------------------------------------------
    # run_tip_adapter_F_ood(log, id_cfg, cache_keys, cache_values, test_features, test_labels, clip_weights, clip_model,
    #                       train_loader_F, ood_features, ood_labels)


if __name__ == '__main__':
    main()