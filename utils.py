import numpy as np
import sklearn.metrics
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn

import clip

from sklearn import metrics
import statistics


def cls_acc(output, target, topk=1):
    # top-1 只有当模型的最高得分与真实标签匹配时，预测才被认为是正确的
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred)) # -1自动计算匹配新值, topk不一定为1所以要用expand_as
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc

def cls_auroc_mcm(closed_logits, open_logits, t=1):
    to_np = lambda x: x.data.cpu().numpy()
    concat = lambda x: np.concatenate(x, axis=0)

    closed_logits /= 100.0
    smax_closed = to_np(F.softmax(closed_logits/t, dim=1))
    mcm_closed = np.max(smax_closed, axis=1)

    open_logits /= 100.0
    smax_open = to_np(F.softmax(open_logits/t, dim=-1))
    mcm_open = np.max(smax_open, axis=1)

    auroc, aupr, fpr = get_measure(mcm_closed, mcm_open)
    return auroc, aupr, fpr

def get_measure(_pos, _neg, recall_level=0.95):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1
    auroc = metrics.roc_auc_score(labels, examples)
    aupr = metrics.average_precision_score(labels, examples)
    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)

    return auroc, aupr, fpr

def fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))   # , fps[cutoff]/(fps[cutoff] + tps[cutoff])

def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out


def cls_auroc(closed_logits, open_logits, target, topk=1):
    return cls_auroc_softmax(closed_logits, open_logits, target, topk)

def cls_auroc_origin(closed_logits, open_logtis, target, topk=1):
    """
    用原本的logtis计算阈值
    """
    flat_closed_logits = closed_logits.reshape(-1).tolist()
    mean = statistics.mean(flat_closed_logits)
    std = statistics.stdev(flat_closed_logits)
    threshold = mean + 3 * std
    closed_pred = [1 if max(logit) > threshold else 0 for logit in closed_logits]
    open_pred = [1 if max(logit) > threshold else 0 for logit in open_logtis]
    auroc = metrics.roc_auc_score(target, closed_pred + open_pred)

    return auroc


def cls_auroc_softmax(closed_logits, open_logits, target, topk=1):
    """
    用softmax函数处理logits之后，计算阈值
    """
    softmax = nn.Softmax(dim=1)
    closed_softmax_logits = softmax(closed_logits)
    open_softmax_logits = softmax(open_logits)
    # 设置固定阈值
    threshold = 0.8
    closed_pred = [1 if max(logit) > threshold else 0 for logit in closed_softmax_logits]
    open_pred = [1 if max(logit) > threshold else 0 for logit in open_softmax_logits]
    auroc = metrics.roc_auc_score(target, closed_pred + open_pred)
    return auroc


def clip_classifier(classnames, template, clip_model):
    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    return clip_weights


def build_cache_model(log, cfg, clip_model, train_loader_cache):

    if cfg['load_cache'] == False:    
        cache_keys = []
        cache_values = []

        with torch.no_grad():
            # Data augmentation for the cache model
            for augment_idx in range(cfg['augment_epoch']):
                train_features = []

                log.debug('Augment Epoch: {:} / {:}'.format(augment_idx, cfg['augment_epoch']))
                for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                    images = images.cuda()
                    image_features = clip_model.encode_image(images)
                    train_features.append(image_features)
                    if augment_idx == 0:
                        target = target.cuda()
                        cache_values.append(target)
                cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))
            
        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        cache_keys = cache_keys.permute(1, 0)
        cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()

        torch.save(cache_keys, cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
        torch.save(cache_values, cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")

    else:
        cache_keys = torch.load(cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
        cache_values = torch.load(cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")

    return cache_keys, cache_values


def pre_load_features(cfg, split, clip_model, loader):

    if cfg['load_pre_feat'] == False:
        features, labels = [], []

        with torch.no_grad():
            for i, (images, target) in enumerate(tqdm(loader)):
                images, target = images.cuda(), target.cuda()
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                features.append(image_features)
                labels.append(target)

        features, labels = torch.cat(features), torch.cat(labels)

        torch.save(features, cfg['cache_dir'] + "/" + split + "_f.pt")
        torch.save(labels, cfg['cache_dir'] + "/" + split + "_l.pt")
   
    else:
        features = torch.load(cfg['cache_dir'] + "/" + split + "_f.pt")
        labels = torch.load(cfg['cache_dir'] + "/" + split + "_l.pt")
    
    return features, labels


def search_hp(cfg, cache_keys, cache_values, features, labels, clip_weights, adapter=None):

    if cfg['search_hp'] == True:
    
        beta_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in range(cfg['search_step'][0])]
        alpha_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in range(cfg['search_step'][1])]

        best_acc = 0
        best_beta, best_alpha = 0, 0

        for beta in beta_list:
            for alpha in alpha_list:
                if adapter:
                    affinity = adapter(features)
                else:
                    affinity = features @ cache_keys

                cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
                clip_logits = 100. * features @ clip_weights
                tip_logits = clip_logits + cache_logits * alpha
                acc = cls_acc(tip_logits, labels)
            
                if acc > best_acc:
                    print("New best setting, beta: {:.2f}, alpha: {:.2f}; accuracy: {:.2f}".format(beta, alpha, acc))
                    best_acc = acc
                    best_beta = beta
                    best_alpha = alpha

        print("\nAfter searching, the best accuarcy: {:.2f}.\n".format(best_acc))

    return best_beta, best_alpha


def search_hp_ood(log, cfg, cache_keys, cache_values, id_features, id_labels, ood_features, ood_labels, clip_weights, adapter=None):
    if cfg['search_hp'] == True:

        beta_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in
                     range(cfg['search_step'][0])]
        alpha_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in
                      range(cfg['search_step'][1])]

        best_score = 0
        best_acc = 0
        best_auroc = 0
        best_beta, best_alpha = 0, 0

        for beta in beta_list:
            for alpha in alpha_list:
                if adapter:
                    id_affinity = adapter(id_features)
                    ood_affinity = adapter(ood_features)
                else:
                    id_affinity = id_features @ cache_keys
                    ood_affinity = ood_features @ cache_keys

                # calculate acc
                id_cache_logits = ((-1) * (beta - beta * id_affinity)).exp() @ cache_values
                id_clip_logits = 100. * id_features @ clip_weights
                id_tip_logits = id_clip_logits + id_cache_logits * alpha
                acc = cls_acc(id_tip_logits, id_labels)

                # calculate auroc
                ood_cache_logits = ((-1) * (beta - beta * ood_affinity)).exp() @ cache_values
                ood_clip_logits = 100. * ood_features @ clip_weights
                ood_tip_logits = ood_clip_logits + ood_cache_logits * alpha
                auroc, aupr, fpr = cls_auroc_mcm(id_tip_logits, ood_tip_logits, 1)
                # todo: 目前暂时未简单地相加
                score = acc + auroc

                if acc > best_score:
                    log.debug("New best setting, beta: {:.2f}, alpha: {:.2f}; accuracy: {:.2f}, auroc: {:.2f}".format(beta, alpha, acc, auroc))
                    best_score = score
                    best_acc = acc
                    best_auroc = auroc
                    best_beta = beta
                    best_alpha = alpha

        log.debug("\nAfter searching, the best score: {:.2f}, best acc: {:.2f}, best auroc: {:.2f}.\n".format(best_score, best_acc, best_auroc))

    return best_beta, best_alpha