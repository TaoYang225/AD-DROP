import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def compute_actual_lens(args, attn_mask, device):
    max_len = attn_mask.size(-1)
    actual_len = torch.sum(attn_mask, dim=-1)
    drop_nums = (args.g_dropout * actual_len).long()  # [N,]  ---> [N, 12, L, L]
    drop_nums_onehot = torch.eye(max_len).to(device)[drop_nums]  # [N, L]
    drop_nums_onehot = drop_nums_onehot.unsqueeze(1).unsqueeze(2).expand([-1, 12, max_len, -1])
    return drop_nums_onehot

def generate_mask(args, grad_t, extend_attn_mask, attn_mask, device):
    max_len = attn_mask.size(-1)
    # 计算要mask的数量
    drop_nums_onehot = compute_actual_lens(args, attn_mask, device)
    # 排序
    grad_p = grad_t + (1 - extend_attn_mask) * 100 # [B, H, L, L]
    sorted_grad = torch.sort(grad_p, dim=-1)[0] # [B, H, L, L]
    st_grad = torch.sum(drop_nums_onehot * sorted_grad, dim=-1) # [B, H, L]
    st_grad = st_grad.unsqueeze(-1).expand([-1, -1, -1, max_len]) # [B, H, L, L]
    grad_masks = (torch.ge(grad_p, st_grad).long() * extend_attn_mask) # [B, H, L, L]
    # random select
    sampler_rate = args.keep_rate * torch.ones_like(grad_masks).to(device)
    sampler_masks = torch.bernoulli(sampler_rate)

    total_masks = ((grad_masks + sampler_masks) >= 1).long() * extend_attn_mask

    return total_masks

def compute_actual_lens_rev(args, attn_mask, device, num_heads):
    max_len = attn_mask.size(-1)
    actual_len = torch.sum(attn_mask, dim=-1)
    drop_nums = ((1- args.g_dropout) * actual_len).long()  # [N,]  ---> [N, 12, L, L]
    drop_nums_onehot = torch.eye(max_len).to(device)[drop_nums]  # [N, L]
    drop_nums_onehot = drop_nums_onehot.unsqueeze(1).unsqueeze(2).expand([-1, num_heads, max_len, -1])
    return drop_nums_onehot

def generate_mask_rev(args, grad_t, extend_attn_mask, attn_mask, device, num_heads):
    # 计算要mask的数量
    max_len = attn_mask.size(-1)
    drop_nums_onehot = compute_actual_lens_rev(args, attn_mask, device, num_heads)
    # 排序
    grad_p = grad_t + (1 - extend_attn_mask) * 100 # [B, H, L, L]
    sorted_grad = torch.sort(grad_p, dim=-1)[0] # [B, H, L, L]
    st_grad = torch.sum(drop_nums_onehot * sorted_grad, dim=-1) # [B, H, L]
    st_grad = st_grad.unsqueeze(-1).expand([-1, -1, -1, max_len]) # [B, H, L, L]
    grad_masks = (1 - torch.ge(grad_p, st_grad).long())  # 反向mask
    # random select
    sampler_rate = args.keep_rate * torch.ones_like(grad_masks).to(device)
    sampler_masks = torch.bernoulli(sampler_rate)

    total_masks = ((grad_masks + sampler_masks) >= 1).long() * extend_attn_mask

    return total_masks

def generate_mask_rev_random(args, grad_t, extend_attn_mask, attn_mask, device):
    # 计算要mask的数量
    max_len = attn_mask.size(-1)
    # random select
    sampler_rate1 = (1- args.g_dropout) * torch.ones_like(extend_attn_mask).to(device)
    sampler_rate2 = args.keep_rate * torch.ones_like(extend_attn_mask).to(device)

    sampler_masks1 = torch.bernoulli(sampler_rate1)
    sampler_masks2 = torch.bernoulli(sampler_rate2)

    # first_masks = (sampler_masks1 * extend_attn_mask).long()
    total_masks = ((sampler_masks1 + sampler_masks2) >= 1).long() * extend_attn_mask


    return total_masks

def compute_integrated_gradient(model, input_ids, labels, extend_attn_mask_list_A, layer):
    model.eval()

    grad_tmp = []
    for step in range(1, 21):
        alpha = step / 20
        alpha_hook = model.roberta.encoder.layer[layer].attention.self.register_forward_hook(
            lambda module, input_data, output_data: (output_data[0] * alpha, output_data[1] ))

        step_out = model(input_ids,
                         attention_mask=extend_attn_mask_list_A,
                         # token_type_ids=segment_ids,
                         labels=labels)  # 得到模型输出
        step_loss = step_out[0]
        step_outputs = step_out[1]
        step_attention = step_out[2]

        step_attention[layer].retain_grad()

        # step_true = step_outputs.gather(1, labels.view(-1, 1)).sum()
        (step_grad,) = torch.autograd.grad(step_loss, step_attention[layer],
                                           retain_graph=True)  # 梯度归因

        grad_tmp.append(step_grad)
        alpha_hook.remove()

    att_scores = step_attention[layer]
    mean_grad = torch.mean(torch.stack(grad_tmp), dim=0)

    grad = (att_scores * mean_grad)  # 积分梯度
    return grad