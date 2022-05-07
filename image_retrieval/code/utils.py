import numpy as np
from sklearn import cluster
import torch
import logging
import losses
import json
from scipy.special import comb
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score

from sklearn.decomposition import PCA
import faiss
from tqdm import tqdm
from scipy.spatial.distance import squareform, pdist, cdist
import torch.nn.functional as F
import math

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)

    return output

def calc_recall_at_k(T, Y, k):
    """
    T : [nb_samples] (target labels)
    Y : [nb_samples x k] (k predicted labels/neighbours)
    """

    s = 0
    for t,y in zip(T,Y):
        if t in torch.Tensor(y).long()[:k]:
            s += 1
    return s / (1. * len(T))


def predict_batchwise(model, dataloader):
    device = "cuda"
    model_is_training = model.training
    model.eval()
    
    ds = dataloader.dataset
    A = [[] for i in range(len(ds[0]))]
    with torch.no_grad():
        # extract batches (A becomes list of samples)
        for batch in tqdm(dataloader):
            for i, J in enumerate(batch):
                # i = 0: sz_batch * images
                # i = 1: sz_batch * labels
                # i = 2: sz_batch * indices
                if i == 0:
                    # move images to device of model (approximate device)
                    J, _ = model(J.cuda())

                for j in J:
                    A[i].append(j)
    model.train()
    model.train(model_is_training) # revert to previous training state
    
    return [torch.stack(A[i]) for i in range(len(A))]

def proxy_init_calc(model, dataloader):
    nb_classes = dataloader.dataset.nb_classes()
    X, T, *_ = predict_batchwise(model, dataloader)

    proxy_mean = torch.stack([X[T==class_idx].mean(0) for class_idx in range(nb_classes)])

    return proxy_mean

def f1_score(query_labels, cluster_labels):
    # compute tp_plus_fp
    qlabels_set, qlabels_counts = np.unique(query_labels, return_counts=True)
    tp_plut_fp = sum([comb(item, 2) for item in qlabels_counts if item > 1])

    # compute tp
    tp = sum([sum([comb(item, 2) for item in np.unique(cluster_labels[query_labels==query_label], return_counts=True)[1] if item > 1]) for query_label in qlabels_set])

    # compute fp
    fp = tp_plut_fp - tp

    # compute fn
    fn = sum([comb(item, 2) for item in np.unique(cluster_labels, return_counts=True)[1] if item > 1]) - tp

    # compute F1
    P, R = tp / (tp+fp), tp / (tp+fn)
    F1 = 2*P*R / (P+R)
    return F1

def get_relevance_mask(shape, gt_labels, embeds_same_source, label_counts):
    relevance_mask = np.zeros(shape=shape, dtype=np.int)
    for k, v in label_counts.items():
        matching_rows = np.where(gt_labels==k)[0]
        max_column = v-1 if embeds_same_source else v
        relevance_mask[matching_rows, :max_column] = 1
    return relevance_mask

def get_label_counts(ref_labels):
    unique_labels, label_counts = np.unique(ref_labels, return_counts=True)
    num_k = min(1023, int(np.max(label_counts)))
    return {k:v for k, v in zip(unique_labels, label_counts)}, num_k

def r_precision(knn_labels, gt_labels, embeds_same_source, label_counts):
    relevance_mask = get_relevance_mask(knn_labels.shape, gt_labels, embeds_same_source, label_counts)
    matches_per_row = np.sum((knn_labels == gt_labels) * relevance_mask.astype(bool), axis=1)
    max_possible_matches_per_row = np.sum(relevance_mask, axis=1)
    accuracy_per_sample = matches_per_row / max_possible_matches_per_row
    return np.mean(accuracy_per_sample)

def mean_average_precision_at_r(knn_labels, gt_labels, embeds_same_source, label_counts):
    relevance_mask = get_relevance_mask(knn_labels.shape, gt_labels, embeds_same_source, label_counts)
    num_samples, num_k = knn_labels.shape
    equality = (knn_labels == gt_labels) * relevance_mask.astype(bool)
    cumulative_correct = np.cumsum(equality, axis=1)
    k_idx = np.tile(np.arange(1, num_k+1), (num_samples, 1))
    precision_at_ks = (cumulative_correct * equality) / k_idx
    summed_precision_pre_row = np.sum(precision_at_ks * relevance_mask, axis=1)
    max_possible_matches_per_row = np.sum(relevance_mask, axis=1)
    accuracy_per_sample = summed_precision_pre_row / max_possible_matches_per_row
    return np.mean(accuracy_per_sample)

def get_lone_query_labels(query_labels, ref_labels, ref_label_counts, embeds_same_source):
    if embeds_same_source:
        return np.array([k for k, v in ref_label_counts.items() if v <= 1])
    else:
        return np.setdiff1d(query_labels, ref_labels)

def get_knn(ref_embeds, embeds, k, embeds_same_source=False, device_ids=None):
    d = ref_embeds.shape[1]
    if device_ids is not None:
        index = faiss.IndexFlatL2(d)
        index = utils.index_cpu_to_gpu_multiple(index, gpu_ids=device_ids)
        index.add(ref_embeds)
        distances, indices = index.search(embeds, k+1)
        if embeds_same_source:
            return indices[:, 1:], distances[:, 1:]
        else:
            return indices[:, :k], distances[:, :k]
    else:
        neigh = NearestNeighbors(n_neighbors=k)
        neigh.fit(ref_embeds)
        distances, indices = neigh.kneighbors(embeds, k + 1)
        if embeds_same_source:
            return indices[:, 1:], distances[:, 1:]
        else:
            return indices[:, :k], distances[:, :k]

def run_kmeans(x, num_clusters, device_ids=None):
    _, d = x.shape
    if device_ids is not None:
        # faiss implementation of k-means
        clus = faiss.Clustering(d, num_clusters)
        clus.niter = 20
        clus.max_points_per_centroid = 10000000
        index = faiss.IndexFlatL2(d)
        index = utils.index_cpu_to_gpu_multiple(index, gpu_ids=device_ids)
        # perform the training
        clus.train(x, index)
        _, idxs = index.search(x, 1)
        return np.array([int(n[0]) for n in idxs], dtype=np.int64)
    else:
        # k-means
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(x)
        return kmeans.labels_ 

def calculate_mean_average_precision_at_r(knn_labels, query_labels, not_lone_query_mask, embeds_same_source, label_counts):
    if not any(not_lone_query_mask):
        return 0
    knn_labels, query_labels = knn_labels[not_lone_query_mask], query_labels[not_lone_query_mask]
    return mean_average_precision_at_r(knn_labels, query_labels[:, None], embeds_same_source, label_counts)
    
def calculate_r_precision(knn_labels, query_labels, not_lone_query_mask, embeds_same_source, label_counts):
    if not any(not_lone_query_mask):
        return 0
    knn_labels, query_labels = knn_labels[not_lone_query_mask], query_labels[not_lone_query_mask]
    return r_precision(knn_labels, query_labels[:, None], embeds_same_source, label_counts)

def recall_at_k(knn_labels, gt_labels, k):
    accuracy_per_sample = np.array([float(gt_label in recalled_predictions[:k]) for gt_label, recalled_predictions in zip(gt_labels, knn_labels)])
    return np.mean(accuracy_per_sample)

def evaluate_cos(model, dataloader):
    torch.cuda.empty_cache()

    _ = model.eval()
    n_classes = dataloader.dataset.nb_classes()

    with torch.no_grad():
        ### For all test images, extract features
        X, T = predict_batchwise(model, dataloader)
        X = l2_norm(X)
        target_labels = T.cpu().detach().numpy()
        feature_coll = X.cpu().detach().numpy()
        feature_coll = feature_coll.astype('float32')

        torch.cuda.empty_cache()
        label_counts, num_k = get_label_counts(target_labels)
        knn_indices, knn_distances = get_knn(feature_coll, feature_coll, num_k, True, None)
        knn_labels = target_labels[knn_indices]
        lone_query_labels = get_lone_query_labels(target_labels, target_labels, label_counts, True)
        not_lone_query_mask = ~np.isin(target_labels, lone_query_labels)

        cluster_labels = run_kmeans(feature_coll, n_classes, None)
        NMI = normalized_mutual_info_score(target_labels, cluster_labels)
        F1 = f1_score(target_labels, cluster_labels)
        MAP = calculate_mean_average_precision_at_r(knn_labels, target_labels, not_lone_query_mask, True, label_counts)
        RP = calculate_r_precision(knn_labels, target_labels, not_lone_query_mask, True, label_counts)
        recall_all_k = []
        for k in [1,2,4,8]:
            recall = recall_at_k(knn_labels, target_labels, k)
            recall_all_k.append(recall)
        
        print('F1:',F1)
        print('NMI:',NMI)
        print('recall@1:',recall_all_k[0])
        print('recall@2:',recall_all_k[1])
        print('recall@4:',recall_all_k[2])
        print('recall@8:',recall_all_k[3])
        print('MAP@R:',MAP)
        print('RP:',RP)

    return F1, NMI, recall_all_k, MAP, RP

def evaluate_cos_SOP(model, dataloader):
    torch.cuda.empty_cache()

    _ = model.eval()
    n_classes = dataloader.dataset.nb_classes()

    with torch.no_grad():
        ### For all test images, extract features
        X, T = predict_batchwise(model, dataloader)
        X = l2_norm(X)
        target_labels = T.cpu().detach().numpy()
        feature_coll = X.cpu().detach().numpy()
        feature_coll = feature_coll.astype('float32')

        torch.cuda.empty_cache()
        label_counts, num_k = get_label_counts(target_labels)
        knn_indices, knn_distances = get_knn(feature_coll, feature_coll, num_k, True, None)
        knn_labels = target_labels[knn_indices]
        lone_query_labels = get_lone_query_labels(target_labels, target_labels, label_counts, True)
        not_lone_query_mask = ~np.isin(target_labels, lone_query_labels)

        cluster_labels = run_kmeans(feature_coll, n_classes, None)
        NMI = normalized_mutual_info_score(target_labels, cluster_labels)
        F1 = f1_score(target_labels, cluster_labels)
        MAP = calculate_mean_average_precision_at_r(knn_labels, target_labels, not_lone_query_mask, True, label_counts)
        RP = calculate_r_precision(knn_labels, target_labels, not_lone_query_mask, True, label_counts)
        recall_all_k = []
        for k in [1,10,100]:
            recall = recall_at_k(knn_labels, target_labels, k)
            recall_all_k.append(recall)
        
        print('F1:',F1)
        print('NMI:',NMI)
        print('recall@1:',recall_all_k[0])
        print('recall@10:',recall_all_k[1])
        print('recall@100:',recall_all_k[2])
        print('MAP@R:',MAP)
        print('RP:',RP)

    return F1, NMI, recall_all_k, MAP, RP
