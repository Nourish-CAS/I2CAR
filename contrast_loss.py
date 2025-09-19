import torch
import torch.nn.functional as F

def inter_time_contrastive_loss(pos, neg, temperature: float = 0.1, eps: float = 1e-8):
    """
    多正样本 InfoNCE：按变量通道独立，在时间轴上做对比。
    - 正样本：同 (b, v) 内的任意两个不同时间步 (t != t')
    - 负样本：同 (b, v) 内来自 neg 的任意时间步

    Args:
        pos: Tensor, shape (B, T, V) 或 (B, T, V, D)
        neg: Tensor, shape 同 pos
        temperature: float, softmax 温度
        eps: float, 数值稳定项

    Returns:
        loss: 标量张量
    """
    assert pos.shape[:3] == neg.shape[:3], "pos/neg 的 (B,T,V) 需一致"
    B, T, V = pos.shape[0], pos.shape[1], pos.shape[2]
    has_D = (pos.dim() == 4)
    if not has_D:
        pos = pos.unsqueeze(-1)  # (B,T,V,1)
        neg = neg.unsqueeze(-1)  # (B,T,V,1)
    D = pos.shape[-1]

    # 合并 batch 与变量通道，按通道独立：N = B*V
    # x_pos: (N, T, D), x_neg: (N, T, D)
    x_pos = pos.permute(0, 2, 1, 3).reshape(B * V, T, D)
    x_neg = neg.permute(0, 2, 1, 3).reshape(B * V, T, D)

    # 归一化到单位球面做余弦相似度
    x_pos = F.normalize(x_pos, dim=-1, eps=eps)
    x_neg = F.normalize(x_neg, dim=-1, eps=eps)

    # 相似度矩阵
    # S_pp: (N, T, T)  —— 正集内部（多正样本，含自对角，后续会mask）
    # S_pn: (N, T, T)  —— 正 vs 负
    S_pp = torch.bmm(x_pos, x_pos.transpose(1, 2)) / max(temperature, eps)
    S_pn = torch.bmm(x_pos, x_neg.transpose(1, 2)) / max(temperature, eps)

    # mask 自身（同一时间步 t 与自己不算正对）
    eye = torch.eye(T, device=pos.device, dtype=torch.bool).unsqueeze(0)  # (1,T,T)
    S_pp = S_pp.masked_fill(eye, float("-inf"))  # 去掉对角正对

    # 多正样本 InfoNCE：
    # 对每个 query（每一行），分子是所有正对（同通道内其余时间步）的 logsumexp，
    # 分母是 [正对 + 负对] 的 logsumexp。
    # loss = - mean( logsumexp(正) - logsumexp(正∪负) )
    log_pos = torch.logsumexp(S_pp, dim=-1)              # (N, T)
    log_all = torch.logsumexp(torch.cat([S_pp, S_pn], dim=-1), dim=-1)  # (N, T)
    # 处理无正对的极端情形（T=1）：令该位置不计入
    valid = torch.isfinite(log_pos)  # (N, T)
    loss_terms = -(log_pos[valid] - log_all[valid])
    loss_infonce = loss_terms.mean() if loss_terms.numel() > 0 else x_pos.new_tensor(0.0)

    # （可选）加入一个轻微的“同 t 分离”margin（欧氏距离），强化 pos_t vs neg_t 的分离
    # margin = 0.2
    # diff = x_pos - x_neg  # (N,T,D)，注意这里默认按索引 t 对齐
    # dist2 = (diff * diff).sum(-1)  # (N,T)
    # loss_margin = F.relu(margin - dist2).mean()
    # return loss_infonce + 0.1 * loss_margin

    return loss_infonce


def inter_variable_contrastive_loss(pos, neg, temperature: float = 0.1, eps: float = 1e-8):
    """
    Inter-Variable 多正样本 InfoNCE：对每个 (b,t) 的变量维进行对比。
    - 正样本：同一 (b, t) 内任意两个不同变量 v != v'
    - 负样本：同一 (b, t) 内，来自 neg 的任意变量

    Args:
        pos: Tensor, 形状 (B, T, V) 或 (B, T, V, D)
        neg: Tensor, 形状同 pos
        temperature: softmax 温度
        eps: 数值稳定项

    Returns:
        标量 loss
    """
    assert pos.shape[:3] == neg.shape[:3], "pos/neg 的 (B,T,V) 必须一致"
    B, T, V = pos.shape[0], pos.shape[1], pos.shape[2]
    if pos.dim() == 3:
        pos = pos.unsqueeze(-1)  # (B,T,V,1)
        neg = neg.unsqueeze(-1)  # (B,T,V,1)
    D = pos.shape[-1]

    # 将 (B,T) 合并：N = B*T；在每个 (b,t) 上做变量维对比
    # x_pos: (N, V, D), x_neg: (N, V, D)
    x_pos = pos.reshape(B * T, V, D)
    x_neg = neg.reshape(B * T, V, D)

    # 归一化到单位球面做余弦相似度
    x_pos = F.normalize(x_pos, dim=-1, eps=eps)
    x_neg = F.normalize(x_neg, dim=-1, eps=eps)

    # 相似度矩阵（按变量维）
    # S_pp: (N, V, V) —— 正集内部（同 (b,t)，变量-变量）
    # S_pn: (N, V, V) —— 正 vs 负（同 (b,t)）
    inv_temp = 1.0 / max(temperature, eps)
    S_pp = torch.bmm(x_pos, x_pos.transpose(1, 2)) * inv_temp
    S_pn = torch.bmm(x_pos, x_neg.transpose(1, 2)) * inv_temp

    # 去掉自对角（变量自身不作为正对）
    if V == 1:
        # 没有可用正对，直接退化为 0（或只做 pos-vs-neg 分离也可扩展）
        return x_pos.new_tensor(0.0)

    eye = torch.eye(V, device=pos.device, dtype=torch.bool).unsqueeze(0)  # (1,V,V)
    S_pp = S_pp.masked_fill(eye, float("-inf"))

    # 多正样本 InfoNCE
    # 对每个 query（每个变量 v 作为一行）：
    #   log_pos = logsumexp(同一 (b,t) 内其他变量的相似度)
    #   log_all = logsumexp(正样本 ∪ 负样本)
    log_pos = torch.logsumexp(S_pp, dim=-1)                           # (N, V)
    log_all = torch.logsumexp(torch.cat([S_pp, S_pn], dim=-1), dim=-1)  # (N, V)

    valid = torch.isfinite(log_pos)  # (N, V)；一般全 True，防 T/V 边界
    loss_terms = -(log_pos[valid] - log_all[valid])
    loss = loss_terms.mean() if loss_terms.numel() > 0 else x_pos.new_tensor(0.0)
    return loss