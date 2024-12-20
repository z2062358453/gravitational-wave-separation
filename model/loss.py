import torch
from  torch import nn
import torch.nn.functional as F
from itertools import permutations

def sisnr(x, s, eps=1e-8):
    """
    calculate training loss
    input:
          x: separated signal, N x S tensor
          s: reference signal, N x S tensor
    Return:
          sisnr: N tensor
    """

    def l2norm(mat, keepdim=False):
        return torch.norm(mat, dim=-1, keepdim=keepdim)

    if x.shape != s.shape:
        raise RuntimeError(
            "Dimention mismatch when calculate si-snr, {} vs {}".format(
                x.shape, s.shape))
    x_zm = x - torch.mean(x, dim=-1, keepdim=True)
    s_zm = s - torch.mean(s, dim=-1, keepdim=True)
    t = torch.sum(
        x_zm * s_zm, dim=-1,
        keepdim=True) * s_zm / (l2norm(s_zm, keepdim=True)**2 + eps)
    return 20 * torch.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps))


def Loss(ests, egs1,egs2):
    # spks x n x S
    refs = [egs1,egs2]
    egs=   [egs1,egs2]
    num_spks = len(refs)

    def sisnr_loss(permute):
        # for one permute
        return sum(
            [sisnr(ests[s], refs[t])
             for s, t in enumerate(permute)]) / len(permute)
             # average the value

    # P x N
    N = egs[0].size(0)
    sisnr_mat = torch.stack(
        [sisnr_loss(p) for p in permutations(range(num_spks))])
    max_perutt, _ = torch.max(sisnr_mat, dim=0)
    # si-snr
    return -torch.sum(max_perutt) / N
# def cosine_similarity(y_true1, y_true2,y_pred1,y_pred2):
#     y_true1 = F.normalize(y_true1,p=2,dim=-1)
#     y_true2 = F.normalize(y_true2, p=2, dim=-1)
#     y_pred1 = F.normalize(y_pred1,p=2,dim=-1)
#     y_pred2 = F.normalize(y_pred2, p=2, dim=-1)
#     cos_sim1=-torch.mean(y_true1 * y_pred1, dim=-1)
#     cos_sim2 = -torch.mean(y_true2 * y_pred2, dim=-1)
#     return cos_sim1,cos_sim2
#
# 构建新的损失函数
def combined_loss(y_true1, y_true2,y_pred1,y_pred2):
    # mse_loss1 = torch.mean((y_true1 - y_pred1)**2)
    # mse_loss2 = torch.mean((y_true2 - y_pred2) ** 2)
    mse_loss1 = nn.MSELoss()(y_true1, y_pred1)
    mse_loss2= nn.MSELoss()(y_true2, y_pred2)

    return mse_loss1+mse_loss2


# def Loss1(y_true,y_pred):
#     # 结合了均方误差（Mean Squared Error, MSE）和余弦相似度（Cosine Similarity）两种损失的结果。
#     mse_loss1=F.mse_loss(y_pred,y_true)
#     # mse_loss2=F.mse_loss(y_pred2,y_true2)
#     # cos_sim1 = torch.mean(1-F.cosine_similarity(y_pred,y_true))
#     # cos_sim2 = torch.mean(1-F.cosine_similarity(y_pred2,y_true2))
#     # return mse_loss1 + 0.5 * cos_sim1
#     return mse_loss1
#     # return mse_loss1+mse_loss2
def cosine_similarity(y_true, y_pred):
    y_true = F.normalize(y_true, p=2, dim=-1)
    y_pred = F.normalize(y_pred,p=2,dim=-1)
    return -torch.mean(y_true * y_pred, dim=-1)


# 构建新的损失函数
# def Loss1(y_pred, y_true):
#     mse_loss = F.mse_loss(y_pred,y_true)
#     cosine_loss = nn.CosineSimilarity(dim=1)(y_true, y_pred)
#     return mse_loss + 0.5 * cosine_loss



def Loss1(y_true, y_pred):
    # 结合了均方误差（Mean Squared Error, MSE）和余弦相似度（Cosine Similarity）两种损失的结果。
    mse_loss = nn.MSELoss()(y_true, y_pred)
    # cosine_similarity = nn.CosineSimilarity(dim=1)(y_true, y_pre)
    return mse_loss
    # return mse_loss + cosine_similarity
