"""
Code modified from Kaolin.
Cite: LoopReg: Self-supervised Learning of Implicit Surface Correspondences, Pose and Shape for 3D Human Mesh Registration, NeurIPS' 20.
Author: Bharat
"""
import torch


def batch_gather(arr, ind):
    """
    :param arr: B x N x D
    :param ind: B x M
    :return: B x M x D
    """
    dummy = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), arr.size(2))
    out = torch.gather(arr, 1, dummy)
    return out

def batch_sparse_dense_matmul(S, D):
    """
    Batch sparse-dense matrix multiplication

    :param torch.SparseTensor S: a sparse tensor of size (batch_size, p, q)
    :param torch.Tensor D: a dense tensor of size (batch_size, q, r)
    :return: a dense tensor of size (batch_size, p, r)
    :rtype: torch.Tensor
    """

    num_b = D.shape[0]
    S_shape = S.shape
    if not S.is_coalesced():
        S = S.coalesce()

    indices = S.indices().view(3, num_b, -1)
    values = S.values().view(num_b, -1)
    ret = torch.stack([
        torch.sparse.mm(
            torch.sparse_coo_tensor(indices[1:, i], values[i], S_shape[1:], device=D.device),
            D[i]
        )
        for i in range(num_b)
    ])
    return ret

def chamfer_distance(s1, s2, w1=1., w2=1.):
    """
    :param s1: B x N x 3
    :param s2: B x M x 3
    :param w1: weight for distance from s1 to s2
    :param w2: weight for distance from s2 to s1
    """
    from kaolin.metrics.point import SidedDistance

    assert s1.is_cuda and s2.is_cuda
    sided_minimum_dist = SidedDistance()
    closest_index_in_s2 = sided_minimum_dist(s1, s2)
    closest_index_in_s1 = sided_minimum_dist(s2, s1)
    closest_s2 = batch_gather(s2, closest_index_in_s2)
    closest_s1 = batch_gather(s1, closest_index_in_s1)

    dist_to_s2 = (((s1 - closest_s2) ** 2).sum(dim=-1)).mean() * w1
    dist_to_s1 = (((s2 - closest_s1) ** 2).sum(dim=-1)).mean() * w2

    return dist_to_s2 + dist_to_s1