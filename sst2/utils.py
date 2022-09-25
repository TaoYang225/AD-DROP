import torch

def compute_actual_lens(args, attn_mask, device): # compute num of positions need to drop, and tag the position
    actual_len = torch.sum(attn_mask, dim=-1)
    drop_nums = (args.p_rate * actual_len).long()  # [N,]  ---> [N, 12, L, L]
    drop_nums_onehot = torch.eye(args.max_len).to(device)[drop_nums]  # [N, L]
    drop_nums_onehot = drop_nums_onehot.unsqueeze(1).unsqueeze(2).expand([-1, args.layers, args.max_len, -1])
    return drop_nums_onehot

def generate_mask_low(args, grad_t, extend_attn_mask, attn_mask, device):

    drop_nums_onehot = compute_actual_lens(args, attn_mask, device)
    # sort
    grad_p = grad_t + (1 - extend_attn_mask) * 100 # [B, H, L, L]   keep pad positions unchange
    sorted_grad = torch.sort(grad_p, dim=-1)[0] # [B, H, L, L]  sort values by row from small to large
    st_grad = torch.sum(drop_nums_onehot * sorted_grad, dim=-1) # [B, H, L]  obtain standard value for comparison
    st_grad = st_grad.unsqueeze(-1).expand([-1, -1, -1, args.max_len]) # [B, H, L, L]
    grad_masks = (torch.ge(grad_p, st_grad).long() * extend_attn_mask) # [B, H, L, L] element-wise comparison

    # random select from above candidate regions
    sampler_rate = (1 - args.q_rate) * torch.ones_like(grad_masks).to(device)
    sampler_masks = torch.bernoulli(sampler_rate)

    total_masks = ((grad_masks + sampler_masks) >= 1).long() * extend_attn_mask

    return total_masks

def compute_actual_lens_rev(args, attn_mask, device): # compute num of positions need to drop, and tag the position
    actual_len = torch.sum(attn_mask, dim=-1)
    drop_nums = ((1- args.p_rate) * actual_len).long()  # [N,]  ---> [N, 12, L, L] dropping 1- args.p_rate
    drop_nums_onehot = torch.eye(args.max_len).to(device)[drop_nums]  # [N, L]
    drop_nums_onehot = drop_nums_onehot.unsqueeze(1).unsqueeze(2).expand([-1, args.layers, args.max_len, -1])
    return drop_nums_onehot

def generate_mask_high(args, grad_t, extend_attn_mask, attn_mask, device):
    drop_nums_onehot = compute_actual_lens_rev(args, attn_mask, device)
    # sort
    grad_p = grad_t + (1 - extend_attn_mask) * 100 # [B, H, L, L] keep pad positions unchange
    sorted_grad = torch.sort(grad_p, dim=-1)[0] # [B, H, L, L] sort values by row from small to large
    st_grad = torch.sum(drop_nums_onehot * sorted_grad, dim=-1) # [B, H, L]  obtain standard value for comparison
    st_grad = st_grad.unsqueeze(-1).expand([-1, -1, -1, args.max_len]) # [B, H, L, L]
    grad_masks = (1 - torch.ge(grad_p, st_grad).long()) * extend_attn_mask  # element-wise comparison, keep (1- args.p_rate) with low = drop args.p_rate with high

    # random select from above candidate regions
    sampler_rate = (1 - args.q_rate) * torch.ones_like(grad_masks).to(device)
    sampler_masks = torch.bernoulli(sampler_rate)

    total_masks = ((grad_masks + sampler_masks) >= 1).long() * extend_attn_mask
    return total_masks

def generate_mask_random(args, extend_attn_mask, device): # vanilla dropout
    # random select
    sampler_rate1 = (1- args.p_rate) * torch.ones_like(extend_attn_mask).to(device)
    sampler_rate2 = (1 - args.q_rate) * torch.ones_like(extend_attn_mask).to(device)

    sampler_masks1 = torch.bernoulli(sampler_rate1)
    sampler_masks2 = torch.bernoulli(sampler_rate2)

    total_masks = ((sampler_masks1 + sampler_masks2) >= 1).long() * extend_attn_mask

    return total_masks

def compute_integrated_gradient(args, model, input_ids, labels, extend_attn_mask_list_A, layer):
    model.eval()

    grad_tmp = []
    for step in range(1, 21): # m steps
        alpha = step / 20 # k / m

        if args.model == 'RoBERTa':
            alpha_hook = model.roberta.encoder.layer[layer].attention.self.register_forward_hook(
                lambda module, input_data, output_data: (output_data[0] * alpha, output_data[1] ))
        elif args.model == 'BERT':
            alpha_hook = model.bert.encoder.layer[layer].attention.self.register_forward_hook(
                lambda module, input_data, output_data: (output_data[0] * alpha, output_data[1] ))
        else:
            raise NotImplementedError

        # obtain gradient in each step
        step_out = model(input_ids,
                         attention_mask=extend_attn_mask_list_A,
                         # token_type_ids=segment_ids,
                         labels=labels)

        step_outputs = step_out[1]
        step_attention = step_out[2]

        step_attention[layer].retain_grad()

        p_label = step_outputs.argmax(dim=-1)  # pseudo label
        step_true = step_outputs.gather(1, p_label.view(-1, 1)).sum()

        (step_grad,) = torch.autograd.grad(step_true.view(-1), step_attention[layer],
                                      retain_graph=True)  # gradient in each step

        grad_tmp.append(step_grad) # save step_grad
        alpha_hook.remove()

    att_scores = step_attention[layer] # A_h
    mean_grad = torch.mean(torch.stack(grad_tmp), dim=0)

    grad = (att_scores * mean_grad)  # integrated gradient
    return grad