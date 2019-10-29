import torch
import numpy as np

from src.utils.tools import Tools


class Loss:
    @staticmethod
    def triplet_semihard_loss(embeddings,
                              labels,
                              perspectives,
                              margin=1.0,
                              device=None):
        """Computes triplet loss with semi-hard negative mining.

        Encourages the distance between positive pairs to be smaller than the
        minimum distance between the anchor and all negative embeddings from
        the same perspective as the anchor which are further than the positive
        distance plus the margin. If all negative embeddings are closer to the
        anchor than that, the maximum distance between the anchor and negative
        embeddings is used instead.

        Args:
            embeddings:   (N,D) FloatTensor - TCN embeddings of inputs
            labels:       (N,) FloatTensor - labels indicating positive pairs
            perspectives: (N,) FloatTensor - labels indicating video
                               perspective
            margin:       float
            device:       torch.cuda.device
        Returns:
            triplet loss Tensor(float)
        """

        # compute pairwise distance matrix
        pdist_matrix = Tools.pdist(embeddings)

        # print("\nPDIST MATRIX")
        # print(pdist_matrix)

        # mask labels equal
        leq_mask = Tools.pequal(labels).float()
        # mask perspectives equal
        peq_mask = Tools.pequal(perspectives).float()

        # mask positive pairs: same label, different perspective
        pos_mask = leq_mask * (1 - peq_mask)
        # mask negative pairs: different label, same perspective
        neg_mask = (1 - leq_mask) * peq_mask

        # print("\nMASKS")
        # print(pos_mask)
        # print(neg_mask)

        # compute max distance between positive pairs for each entry
        pos_dists, _ = torch.max(
            pdist_matrix * pos_mask, dim=1, keepdim=True)

        # print("\nPOS_DISTS")
        # print(pos_dists)

        # mask semi-hard negative pairs: all negative pairs of which the
        # distance to the anchor is greater than the positive distance and
        # smaller than the positive distance plus the margin
        neg_ou_mask = torch.ge(
            pdist_matrix,
            pos_dists.expand_as(pdist_matrix)).float() * \
            (1 - torch.ge(
                pdist_matrix,
                pos_dists.expand_as(pdist_matrix) + margin).float()) * \
            neg_mask

        # mask hard negative pairs: all negative pairs of which the distance
        # to the anchor is smaller than the positive distance
        neg_in_mask = neg_mask * (1 - neg_ou_mask) * torch.ge(
            pos_dists.expand_as(pdist_matrix) + margin,
            pdist_matrix).float()

        # compute the minimum distance to the anchor of all negative pairs
        # which are outside pos dist + margin for each entry
        neg_ou_dists = torch.where(
            neg_ou_mask > 0,
            pdist_matrix,
            torch.full(neg_ou_mask.size(), float("inf")).to(device))
        neg_ou_dists = torch.min(
            neg_ou_dists, dim=1, keepdim=True)[0]

        # print("\nNEG MASKS")
        # print(neg_ou_mask * pdist_matrix)
        # print(neg_in_mask)

        # compute the maximum distance to the anchor of all negative pairs
        # which are inside pos_dist + margin for each entry
        neg_in_dists = torch.max(
            pdist_matrix * neg_in_mask, dim=1, keepdim=True)[0] * \
            (1 - torch.sum(neg_ou_mask, 1, keepdim=True).clamp(0., 1.))

        # print("\nNEG_DISTS")
        # print(neg_ou_dists)
        # print(neg_in_dists)

        # compute semi-hard sampled negative distance by mergin outside and
        # inside negative pairs for each entry: use min outside distance if
        # available, o/w use max inside distance
        neg_dists = torch.where(
            torch.sum(neg_ou_mask, 1, keepdim=True) > 0,
            neg_ou_dists,
            neg_in_dists)

        # element-wise triplet loss
        # 0 if no elements in semi-hard or hard range
        L_element = (pos_dists + margin - neg_dists).clamp(min=0.) * \
            (torch.sum(neg_ou_mask, dim=1, keepdim=True) +
             torch.sum(neg_in_mask, dim=1, keepdim=True)).clamp(0., 1.)

        # print("\nELEMENT WISE LOSS")
        # print(L_element)

        # reduce by mean for batch loss
        loss = torch.mean(L_element)

        return loss

    @staticmethod
    def embedding_accuracy(embeddings, labels, perspectives, device=None):
        """Computes the ratio of positive embeddings in the batch which are
        closer together than any negative pair.

        Args:
            embeddings:   (N,D) FloatTensor - TCN embeddings of inputs
            labels:       (N,) FloatTensor - labels indicating positive pairs
            perspectives: (N,) FloatTensor - labels indicating video
                               perspective
            device:       torch.cuda.device
        Returns:
            accuracy Tensor(float)
        """
        # Tools.pyout(embeddings)

        # compute pairwise distance matrix
        pdist_matrix = Tools.pdist(embeddings)

        # Tools.pyout(pdist_matrix)

        # mask labels equal
        leq_mask = Tools.pequal(labels).float()
        # mask perspectives equal
        peq_mask = Tools.pequal(perspectives).float()

        # mask positive pairs: same label, different perspective
        pos_mask = leq_mask * (1 - peq_mask)
        # mask negative pairs: different label, same perspective
        neg_mask = (1 - leq_mask) * peq_mask

        # compute max distance between positive pairs for each entry
        pos_dists, _ = torch.max(
            pdist_matrix * pos_mask, dim=1, keepdim=True)

        # compute min distance between negative pairs for each entry
        neg_dists = torch.where(
            neg_mask > 0,
            pdist_matrix,
            torch.full(neg_mask.size(), float("inf")).to(device))
        neg_dists, _ = torch.min(neg_dists, dim=1, keepdim=True)

        # compute ratio where positive distance is smaller than all
        # negative distance element wise
        dist_diff = (pos_dists < neg_dists).float()

        # reduce mean
        accuracy = sum(dist_diff) / dist_diff.size()[0]

        return accuracy.squeeze()

    @staticmethod
    def embedding_accuracy_ratio(embeddings, labels, perspectives):
        """Computes the ratio of positive embeddings in the batch which are
        closer together than any negative pair.

        Args:
            embeddings:   (N,D) FloatTensor - TCN embeddings of inputs
            labels:       (N,) FloatTensor - labels indicating positive pairs
            perspectives: (N,) FloatTensor - labels indicating video
                               perspective
            device:       torch.cuda.device
        Returns:
            accuracy Tensor(float)
        """

        # compute pairwise distance matrix
        pdist_matrix = Tools.pdist(embeddings)

        # mask labels equal
        leq_mask = Tools.pequal(labels).float()
        # mask perspectives equal
        peq_mask = Tools.pequal(perspectives).float()

        # mask positive pairs: same label, different perspective
        pos_mask = leq_mask * (1 - peq_mask)
        # mask negative pairs: different label, same perspective
        neg_mask = (1 - leq_mask) * peq_mask

        # compute max distance between positive pairs for each entry
        pos_dists, _ = torch.max(
            pdist_matrix * pos_mask, dim=1, keepdim=True)

        neg_se_pos = (
            pdist_matrix <= pos_dists.expand_as(pdist_matrix)).float()
        neg_se_pos = torch.sum(neg_se_pos * neg_mask, dim=1)
        neg_se_pos /= torch.sum(neg_mask, dim=1)
        neg_se_pos[neg_se_pos != neg_se_pos] = 0.  # replace nan by 0.

        # reduce mean
        accuracy_ratio = torch.mean(neg_se_pos)

        return accuracy_ratio

    @staticmethod
    def labeled_accuracy(Y, V, R, L):
        # compute prediction for each frame-v2 pair
        correct = 0
        N = 0
        nearest_neighbors = {}
        for idx in range(len(V)):  # loop over all v1
            nearest_neighbors[idx] = {}
            for jdx in range(len(V)):  # loop over all v2
                # only compare comparable videos
                if R[idx] == R[jdx] and not idx == jdx:
                    try:
                        if np.linalg.norm(Y[idx] - Y[jdx]) < \
                                nearest_neighbors[idx][V[jdx]]['dist']:
                            nearest_neighbors[idx][V[jdx]]['dist'] = \
                                np.linalg.norm(Y[idx] - Y[jdx])
                            nearest_neighbors[idx][V[jdx]]['idx'] = jdx
                    except KeyError:
                        nearest_neighbors[idx][V[jdx]] = {
                            'idx': jdx,
                            'dist': np.linalg.norm(Y[idx] - Y[jdx])
                        }

        # compute accuracy on predicted neighbors
        for k_i in nearest_neighbors:
            for k_j in nearest_neighbors[k_i]:
                N += 1
                correct += sum(
                    1 / 6 if L[k_i][k] ==
                    L[nearest_neighbors[k_i][k_j]['idx']][k]
                    else 0
                    for k in L[idx])

        accuracy = correct / N
        return accuracy, correct, N
