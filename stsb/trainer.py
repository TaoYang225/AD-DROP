import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
import logging
import os
from tensorboardX import SummaryWriter
from dataset import My_dataset
from scipy.stats import pearsonr
from utils import generate_mask_low, generate_mask_high, generate_mask_random, compute_integrated_gradient

logger = logging.getLogger(__name__)

def pearson_cc(out, labels):
    r = pearsonr(labels.reshape(-1), out.reshape(-1))[0]
    if np.isnan(r):
        r = 0.0
    return float(r)

def train_process(args, model, optimizer, train_dataloader, eval_dataloader, test_dataloader, device):
    writer = SummaryWriter(os.path.join('runs', args.output_dir))
    best_eval_pcc = 0.
    best_test_pcc = 0.
    best_epoch = 0
    early_stop_count = 0
    epoch_steps = len(train_dataloader)

    for epoch in range(args.epoch):
        model.train()
        preds = None
        annos = None
        train_loss = 0.

        for step, datas in enumerate(train_dataloader):
            input_ids, attn_mask, segment_ids, labels, _ = [data.to(device) for data in datas]

            # creat initial attention mask for each layer
            extend_attn_mask = attn_mask.unsqueeze(1).unsqueeze(2).expand([-1, args.heads, args.max_len, -1])
            extend_attn_mask_list = [extend_attn_mask for i in range(args.layers)]
            extend_attn_mask_list_A = torch.stack(extend_attn_mask_list, dim=0)

            # the first forward computation without masking
            out = model(input_ids,
                        attention_mask=extend_attn_mask_list_A,
                        token_type_ids=segment_ids,  # used for pair-wise input
                        labels=labels)
            loss_ori = out[0]
            outputs_ori = out[1]
            attention_ori = out[2]

            if args.do_mask and (epoch % 2) != 0: # do mask + cross-tuning
                # obtain new mask matrices for each layer
                for l, layer in enumerate(args.mask_layer):
                    attention_ori[layer].retain_grad()
                    if args.attribution == 'AA' or args.attribution == 'RD':
                        grad = attention_ori[layer]  # directly use attention weights for attribution
                    elif args.attribution == 'GA':
                        (grad,) = torch.autograd.grad(loss_ori, attention_ori[layer], retain_graph=True)
                    elif args.attribution == 'IGA':  # compute integrated gradient for attribution
                        grad = compute_integrated_gradient(args, model, input_ids, labels, extend_attn_mask_list_A, layer, segment_ids)
                    else:
                        raise NotImplementedError

                    # masking low or high
                    if args.attribution == 'RD':
                        grad_masks = generate_mask_random(args, extend_attn_mask, device)
                    elif args.dropping_method == 'low':
                        grad_masks = generate_mask_low(args, -grad, extend_attn_mask, attn_mask,
                                                       device)  # [B, H, L, L]
                    elif args.dropping_method == 'high':
                        grad_masks = generate_mask_high(args, -grad, extend_attn_mask, attn_mask,
                                                        device)  # [B, H, L, L]
                    else:
                        raise NotImplementedError

                    extend_attn_mask_list[layer] = grad_masks.long() # replace the initial attention mask
                extend_attn_mask_list_B = torch.stack(extend_attn_mask_list, dim=0)

                # the second forward computation by feeding the new mask
                out_B = model(input_ids,
                              attention_mask=extend_attn_mask_list_B,
                              token_type_ids=segment_ids,
                              labels=labels)

                loss_B = out_B[0]
                outputs = out_B[1]

            if preds is None:
                if args.do_mask and (epoch % 2) != 0: # store new prediction
                    preds = outputs.detach().cpu().numpy()
                else: # store original predict
                    preds = outputs_ori.detach().cpu().numpy()
                annos = labels.detach().cpu().numpy()
            else:
                if args.do_mask and (epoch % 2) != 0:
                    preds = np.append(preds, outputs.detach().cpu().numpy(), axis=0)
                else:
                    preds = np.append(preds, outputs_ori.detach().cpu().numpy(), axis=0)
                annos = np.append(annos, labels.detach().cpu().numpy(), axis=0)

            optimizer.zero_grad()
            if args.do_mask and (epoch % 2) != 0: # new loss
                loss_B.backward()
            else: # original loss
                loss_ori.backward()
            optimizer.step()
            # scheduler.step()

            if args.do_mask and (epoch % 2) != 0:
                train_loss += loss_B.item()
            else:
                train_loss += loss_ori.item()

        train_pcc = pearson_cc(preds, annos)  # compute pcc
        # recode
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_pcc', train_pcc, epoch)

        #record training-set result
        logger.info('epoch = {}, train_result:'.format(epoch))
        logger.info('train_loss = {:,.4f}, train_pcc = {:,.4f}'.format(train_loss / epoch_steps, train_pcc))

        logger.info('epoch = {}, eval_result:'.format(epoch))

        with torch.no_grad():  # validation
            eval_loss, eval_pcc, _, _ = test_process(args, model, eval_dataloader, device)
            writer.add_scalar('eval_loss', eval_loss, epoch)
            writer.add_scalar('eval_pcc', eval_pcc, epoch)

            logger.info('eval_loss = {:,.4f}, eval_pcc = {:,.4f}'.format(eval_loss, eval_pcc))

            if eval_pcc > best_eval_pcc:
                best_eval_pcc = eval_pcc
                early_stop_count = 0
                best_epoch = epoch

                if args.do_mask:  # save AD-DROP finetuned model
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(args.output_dir, 'finetuned_{}_AD.pth'.format(args.model))
                    torch.save(model_to_save.state_dict(), output_model_file)
                else: # save vanilla finetuned model
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(args.output_dir, 'finetuned_{}_org.pth'.format(args.model))
                    torch.save(model_to_save.state_dict(), output_model_file)

            else:
                early_stop_count += 1
                logger.info('No improvement: early_stop_count = {}'.format(early_stop_count))
                if early_stop_count == args.patient:
                    logger.info('Early Stop!')
                    logger.info(
                        'best_epoch = {}, best_eval_pcc = {:,.4f} \n'.format(
                            best_epoch, best_eval_pcc))
                    break
        logger.info('best_epoch = {}, best_eval_pcc = {:,.4f} \n'.format(best_epoch, best_eval_pcc))
    writer.close()

def test_process(args, model, test_dataloader, device):
    model.eval()

    test_loss = 0.
    preds = None
    annos = None
    epoch_steps = len(test_dataloader)
    for step, datas in enumerate(test_dataloader):
        input_ids, attn_mask, segment_ids, labels, indexs = [data.to(device) for data in datas]

        extend_attn_mask = attn_mask.unsqueeze(1).unsqueeze(2).expand([-1, args.heads, args.max_len, -1])

        extend_attn_mask_list = [extend_attn_mask for i in range(args.layers)]

        extend_attn_mask_list = torch.stack(extend_attn_mask_list, dim=0)

        out = model(input_ids,
                    attention_mask=extend_attn_mask_list,
                    token_type_ids=segment_ids,
                    labels=labels)
        loss = out[0]
        outputs = out[1]

        if preds is None:
            preds = outputs.detach().cpu().numpy()
            annos = labels.detach().cpu().numpy()
        else:
            preds = np.append(preds, outputs.detach().cpu().numpy(), axis=0)
            annos = np.append(annos, labels.detach().cpu().numpy(), axis=0)

        test_loss += loss.item()

    test_pcc = pearson_cc(preds, annos)

    return test_loss / epoch_steps, test_pcc, preds, annos

