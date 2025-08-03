import numpy as np
import torch
from collections import defaultdict

import verl.utils.torch_functional as verl_F

def compute_sft_pure_loss(log_prob, eos_mask):
    sft_losses = -log_prob
    sft_loss = verl_F.masked_mean(sft_losses, eos_mask)
    return sft_loss

def compute_grpo_outcome_advantage_split(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   on_policy_mask: torch.Tensor,
                                   epsilon: float = 1e-6,
                                   use_std: bool = True):
    """
    Compute advantage for GRPO, operating only on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    # 获取响应序列的长度
    response_length = token_level_rewards.shape[-1]
    non_zero_mask = (token_level_rewards != 0)
    scores = (token_level_rewards * non_zero_mask).sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            # only include on-policy samples for mean and std calculation
            if on_policy_mask[i].item() is True:
                id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        # process std
        # 所有样本分数相同时设置为1，避免除0
        for idx in id2std:
            if id2std[idx].item() == 0:
                id2std[idx] = torch.tensor(1.0)
        for i in range(bsz):
            if use_std:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            # 也可以不除标准差
            else:
                scores[i] = (scores[i] - id2mean[index[i]])
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores

def compute_token_on_off_policy_loss(
    old_log_prob,                    # 旧策略的对数概率 [bs, seq_len]
    log_prob,                        # 当前策略的对数概率 [bs, seq_len]
    advantages,                      # 优势函数值 [bs, seq_len]
    eos_mask,                        # 结束符掩码，标记有效token [bs, seq_len]
    cliprange,                       # PPO裁剪范围 (通常0.2)
    clip_upper_bound,                # 裁剪上界
    prefix_mask,                     # 前缀掩码：True=离线策略，False=在线策略
    off_cliprange,                   # 离线策略裁剪范围
    off_normalize=False,             # 是否对离线策略进行归一化
    off_abs_cliprange=None, 
    off_max_clip=None, 
    off_min_clip=None,
    all_max_clip=None, 
    off_policy_reshape="no_reshape", 
    off_policy_reshape_weight=1.0, 
    off_policy_reshape_pow_exp=0.5,
    on_policy_reshape="no_reshape", 
    on_policy_reshape_weight=1.0,
    on_policy_reshape_pow_exp=0.5,
    target_probs=None,
    loss_remove_token_mean=False,
    loss_remove_clip=False,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347
        prefix_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            a float number indicating the fraction of policy gradient loss being clipped

    """
    # off-policy loss
    # compute off-policy probability
    
    negative_approx_kl = log_prob - old_log_prob  # π_new(a|s) / π_old(a|s) 的对数
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)  # 平均KL散度

    # 第116-138行：在线策略的比率计算
    if on_policy_reshape == "no_reshape":
        ratio = torch.exp(negative_approx_kl)  # 标准重要性采样比率
    elif on_policy_reshape == "logp":
        ratio = log_prob - old_log_prob        # 直接使用对数差
    elif on_policy_reshape == "p_logp":
        ratio = torch.exp(negative_approx_kl) + on_policy_reshape_weight * negative_approx_kl
        # 结合概率比率和对数差
    elif on_policy_reshape == "square_root":
        ratio = torch.exp(negative_approx_kl)
        ratio = torch.sqrt(ratio)              # 平方根变换，缓解极值
    elif on_policy_reshape == "pow":
        ratio = torch.exp(negative_approx_kl)
        ratio = torch.pow(ratio, on_policy_reshape_pow_exp)  # 幂函数变换
    elif on_policy_reshape == "p_div_p_0.1":
        prob = torch.exp(log_prob)             # 当前策略概率
        old_prob = torch.exp(old_log_prob)     # 旧策略概率
        f_prob = prob / (prob + 0.1)           # 重塑函数 f(p) = p/(p+0.1)
        f_old_prob = old_prob / (old_prob + 0.1)
        ratio = f_prob / f_old_prob            # 重塑后的比率
    elif on_policy_reshape == "p_div_p_0.5":
        prob = torch.exp(log_prob)
        old_prob = torch.exp(old_log_prob)
        f_prob = prob / (prob + 0.5)
        f_old_prob = old_prob / (old_prob + 0.5)
        ratio = f_prob / f_old_prob
    else:
        raise ValueError(f"Invalid on_policy_reshape: {on_policy_reshape}")

    # 第140-151行：计算在线策略损失
    on_pg_losses = -advantages * ratio        # 策略梯度损失：-A(s,a) * π(a|s)/π_old(a|s)

    # 确定重要性采样比率的上界
    upper_bound = max(clip_upper_bound, 1.0 + cliprange)  # 确定裁剪上界

    if loss_remove_clip is False:
        # 标准PPO裁剪
        # ratio限制在[1.0 - cliprange, upper_bound]之间
        on_pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, upper_bound)
        # 计算裁剪比例
        on_pg_clipfrac = verl_F.masked_mean(torch.gt(on_pg_losses2, on_pg_losses).float(), eos_mask)
        # 取悲观估计，防止过大的梯度
        on_pg_losses = torch.max(on_pg_losses, on_pg_losses2)  # 取悲观估计
        # 关键：只在非前缀部分计算在线策略损失
        on_pg_loss = verl_F.masked_mean(on_pg_losses, (~prefix_mask) * eos_mask)
    else:
        # 移除裁剪的版本
        on_pg_loss = verl_F.masked_mean(on_pg_losses, (~prefix_mask) * eos_mask)
        on_pg_clipfrac = torch.tensor(0.0)
    
    # compute off-policy loss
    if target_probs is None:
        # 也可以不提供，不提供的话就无法使用重要性采样
        # 第153-175行：离线策略处理
        off_ratio = torch.exp(log_prob)        # 直接使用当前策略概率（不是比率！）
        
        if off_policy_reshape == "no_reshape":
            pass                               # 不进行重塑
        elif off_policy_reshape == "logp":
            off_ratio = log_prob * off_policy_reshape_weight
        elif off_policy_reshape == "p_logp":
            off_ratio = log_prob * off_policy_reshape_weight + off_ratio
        elif off_policy_reshape == "square_root":
            off_ratio = torch.sqrt(off_ratio)
        elif off_policy_reshape == "p_div_p_0.1":
            off_ratio = off_ratio / (off_ratio + 0.1)  # 论文核心公式：f(p) = p/(p+γ)
        elif off_policy_reshape == "p_div_p_0.5":
            off_ratio = off_ratio / (off_ratio + 0.5)
        elif off_policy_reshape == "p_div_p_0.3":
            off_ratio = off_ratio / (off_ratio + 0.3)
        elif off_policy_reshape == "pow":
            off_ratio = torch.pow(off_ratio, off_policy_reshape_pow_exp)
        else:
            raise ValueError(f"Invalid off_policy_reshape: {off_policy_reshape}")
    else:
        # 如果提供了目标概率，使用重要性采样
        # 如果有目标概率，就不再使用policy reshape，看样子是这样。
        assert target_probs.shape == log_prob.shape
        off_ratio = torch.exp(log_prob) / (target_probs + 1e-6)
        off_ratio = off_ratio * prefix_mask    # 只在前缀部分有效
        # assert ((target_probs > 0) == prefix_mask).all()
        
    # clip off-policy ratio
    if off_max_clip is not None: # 检查是否设置了最大裁剪值
        off_ratio = torch.clamp(off_ratio, max=off_max_clip) # 将 off_ratio 裁剪到不超过 off_max_clip
        # 计算被最大值裁剪的token比例，作为监控
        off_ratio_max_clip_frac = verl_F.masked_mean((off_ratio == off_max_clip).float(), prefix_mask * eos_mask)
    else:
        off_ratio_max_clip_frac = torch.tensor(0.0)
        
    if off_min_clip is not None: # 检查是否设置了最小裁剪值
        off_ratio = torch.clamp(off_ratio, min=off_min_clip) # 将 off_ratio 裁剪到不低于 off_min_clip
        off_ratio_min_clip_frac = verl_F.masked_mean((off_ratio == off_min_clip).float(), prefix_mask * eos_mask)
    else:
        off_ratio_min_clip_frac = torch.tensor(0.0) # 如果没有设置最小裁剪值，则设置为0

    # 离线策略比率平均值，只在有效token上计算，前缀部分不计算。
    off_ratio_mean = verl_F.masked_mean(off_ratio, prefix_mask * eos_mask)
    if off_ratio_mean.isnan().any().item():
        off_ratio_mean = torch.tensor(0.0)  # 处理NaN情况

    # 第192-197行：计算离线策略损失
    off_pg_losses = -advantages * off_ratio
    off_pg_loss = verl_F.masked_mean(off_pg_losses, prefix_mask * eos_mask)  # 只在前缀部分
    # off_pg_loss是单纯用来记录的
    # 这里记得仔细看一看。
    if off_pg_loss.isnan().item() is True:
        off_pg_loss = torch.tensor(0.0)
    off_pg_clipfrac = torch.tensor(0.0)   # 离线策略不使用裁剪
    
    # 第199-201行：混合两种损失
    prefix_mask = prefix_mask.float()
    pg_losses = off_pg_losses * prefix_mask + on_pg_losses * (1 - prefix_mask)
    
    # 第203-215行：计算概率统计
    off_policy_probs = torch.exp(log_prob)
    off_policy_prob = verl_F.masked_mean(off_policy_probs, prefix_mask * eos_mask)
    if off_policy_prob.isnan().item() is True:
        off_policy_prob = torch.tensor(0.0)
    
    on_policy_probs = torch.exp(old_log_prob)  
    on_policy_prob = verl_F.masked_mean(on_policy_probs, (1.0-prefix_mask) * eos_mask)
    if on_policy_prob.isnan().item() is True:
        on_policy_prob = torch.tensor(0.0)
            
    # 第217-222行：额外的概率裁剪（可选）
    # 高概率token算出来的梯度比较大
    # 概率过高的token的损失被设置为0，不参与反向传播
    # 这确保了高概率token不会影响梯度更新
    if all_max_clip is not None:
        p_on = torch.exp(log_prob)
        p_on_mask = (p_on <= all_max_clip).float()
        eos_mask = eos_mask * p_on_mask
        pg_losses = pg_losses * p_on_mask
        
    # 第224-230行：最终损失计算
    if loss_remove_token_mean is True:
        pg_loss = (pg_losses * eos_mask).sum() / eos_mask.shape[-1]  # 简单平均
        print(f'no token mean: mean normalization {eos_mask.shape[-1]}')
    else:
        pg_loss = verl_F.masked_mean(pg_losses, eos_mask)  # 掩码平均

    return {
        "pg_loss": pg_loss,                    # 总损失
        "off_pg_loss": off_pg_loss,            # 离线策略损失
        "on_pg_loss": on_pg_loss,              # 在线策略损失
        "off_pg_clipfrac": off_pg_clipfrac,    # 离线策略裁剪比例
        "on_pg_clipfrac": on_pg_clipfrac,      # 在线策略裁剪比例
        "ppo_kl": ppo_kl,                      # KL散度
        "off_policy_prob": off_policy_prob,    # 离线策略平均概率
        "on_policy_prob": on_policy_prob,      # 在线策略平均概率
        "off_ratio_mean": off_ratio_mean,      # 离线策略比率均值
        "off_ratio_max_clip_frac": off_ratio_max_clip_frac,  # 最大裁剪比例
        "off_ratio_min_clip_frac": off_ratio_min_clip_frac,  # 最小裁剪比例
    }