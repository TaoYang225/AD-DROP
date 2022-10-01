import torch
import numpy as np
from sklearn.metrics import accuracy_score
import logging
import os
from tensorboardX import SummaryWriter
from utils import generate_mask_low, generate_mask_high, generate_mask_random, compute_integrated_gradient

logger = logging.getLogger(__name__)

def train_process(args, model, optimizer,scheduler, train_dataloader, eval_dataloader, device):
    writer = SummaryWriter(os.path.join('runs', args.output_dir))
    best_eval_acc = 0.
    best_test_acc = 0.
    best_epoch = 0
    early_stop_count = 0
    epoch_steps = len(train_dataloader)

    for epoch in range(args.epoch):
        model.train()
        preds = None
        annos = None
        train_loss = 0.

        for step, datas in enumerate(train_dataloader):
            input_ids, attn_mask, labels, _ = [data.to(device) for data in datas]

            # creat initial attention mask for each layer
            extend_attn_mask = attn_mask.unsqueeze(1).unsqueeze(2).expand([-1, args.heads, args.max_len, -1])
            extend_attn_mask_list = [extend_attn_mask for i in range(args.layers)]
            extend_attn_mask_list_A = torch.stack(extend_attn_mask_list, dim=0)

            # the first forward computation without masking
            out = model(input_ids,
                        attention_mask=extend_attn_mask_list_A,
                        # token_type_ids=segment_ids,
                        labels=labels)
            loss_ori = out[0]
            outputs_ori = out[1]
            attention_ori = out[2]

            # p_label = outputs_ori.argmax(dim=-1)  # pseudo label
            outputs_true = outputs_ori.gather(1, labels.view(-1, 1)).sum() # logit output of gold label

            if args.do_mask:
                # obtain new mask matrices for each layer
                for l, layer in enumerate(args.mask_layer):
                    attention_ori[layer].retain_grad()
                    if args.attribution == 'AA' or args.attribution == 'RD':
                        grad = attention_ori[layer]  # directly use attention weights for attribution
                    elif args.attribution == 'GA':
                        (grad,) = torch.autograd.grad(outputs_true.view(-1), attention_ori[layer], retain_graph=True)
                    elif args.attribution == 'IGA':  # compute integrated gradient for attribution
                        grad = compute_integrated_gradient(args, model, input_ids, labels, extend_attn_mask_list_A, layer)
                    else:
                        raise NotImplementedError
                    # masking low or high

                    if args.attribution == 'RD':
                        grad_masks = generate_mask_random(args, extend_attn_mask, device)
                    elif args.dropping_method == 'low':
                        grad_masks = generate_mask_low(args, grad, extend_attn_mask, attn_mask, device)  # [B, H, L, L]
                    elif args.dropping_method == 'high':
                        grad_masks = generate_mask_high(args, grad, extend_attn_mask, attn_mask, device)  # [B, H, L, L]
                    else:
                        raise NotImplementedError

                    extend_attn_mask_list[layer] = grad_masks.long() # replace the initial attention mask
                extend_attn_mask_list_B = torch.stack(extend_attn_mask_list, dim=0)

                # the second forward computation by feeding the new mask
                out_B = model(input_ids,
                              attention_mask=extend_attn_mask_list_B,
                              # token_type_ids=segment_ids,
                              labels=labels)

                loss_B = out_B[0]
                outputs = out_B[1]
            if preds is None:
                if args.do_mask: # store new prediction
                    preds = outputs.detach().cpu().numpy()
                else: # store original predict
                    preds = outputs_ori.detach().cpu().numpy()
                annos = labels.detach().cpu().numpy()
            else:
                if args.do_mask:
                    preds = np.append(preds, outputs.detach().cpu().numpy(), axis=0)
                else:
                    preds = np.append(preds, outputs_ori.detach().cpu().numpy(), axis=0)
                annos = np.append(annos, labels.detach().cpu().numpy(), axis=0)

            optimizer.zero_grad()
            if args.do_mask: # new loss
                loss_B.backward()
            else: # original loss
                loss_ori.backward()
            optimizer.step()
            scheduler.step()

            if args.do_mask:
                train_loss += loss_B.item()
            else:
                train_loss += loss_ori.item()

        preds_id = np.argmax(preds, axis=1)  # compute predict label
        train_acc = accuracy_score(preds_id, annos) # compute accuracy
        # recode
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)

        #record training-set result
        logger.info('epoch = {}, train_result:'.format(epoch))
        logger.info('train_loss = {:,.4f}, train_acc = {:,.4f}'.format(train_loss/epoch_steps, train_acc))

        logger.info('epoch = {}, eval_result:'.format(epoch))

        with torch.no_grad():  # validation
            eval_loss, eval_acc,_,_ = test_process(args, model, eval_dataloader, device)
            writer.add_scalar('eval_loss', eval_loss, epoch)
            writer.add_scalar('eval_acc', eval_acc, epoch)

            logger.info('eval_loss = {:,.4f}, eval_acc = {:,.4f}'.format(eval_loss, eval_acc))

            # save model
            if eval_acc > best_eval_acc:
                best_eval_acc = eval_acc
                early_stop_count = 0
                best_epoch = epoch

                if not args.do_mask: # save vanilla finetuned model for probe
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(args.output_dir, 'finetuned_{}.pth'.format(args.model))
                    torch.save(model_to_save.state_dict(), output_model_file)

            else:
                early_stop_count += 1
                logger.info('No improvement: early_stop_count = {}'.format(early_stop_count))
                if early_stop_count == args.patient:
                    logger.info('Early Stop!')
                    logger.info(
                        'best_epoch = {}, best_eval_acc = {:,.4f} \n'.format(
                            best_epoch, best_eval_acc))
                    break
        logger.info('best_epoch = {}, best_eval_acc = {:,.4f} \n'.format(best_epoch, best_eval_acc))
    writer.close()
 
def test_process(args, model, test_dataloader, device, grad_all_masks=None):
    model.eval()

    test_loss = 0.
    preds = None
    annos = None
    epoch_steps = len(test_dataloader)
    for step, datas in enumerate(test_dataloader):
        input_ids, attn_mask, labels, indexs = [data.to(device) for data in datas]

        extend_attn_mask = attn_mask.unsqueeze(1).unsqueeze(2).expand([-1, args.heads, args.max_len, -1])

        extend_attn_mask_list = [extend_attn_mask for i in range(args.layers)]
        if grad_all_masks is not None: # used for probe experiment
            for layer in args.mask_layer:
                layer_masks = torch.tensor(grad_all_masks[indexs.detach().cpu().numpy()]).to(device)
                extend_attn_mask_list[layer] = layer_masks

        extend_attn_mask_list = torch.stack(extend_attn_mask_list, dim=0)

        out = model(input_ids,
                    attention_mask=extend_attn_mask_list,
                    # token_type_ids=segment_ids,
                    labels=labels)
        loss = out[0]
        outputs = out[1]
        # attention = out[2]
        if preds is None:
            preds = outputs.detach().cpu().numpy()
            annos = labels.detach().cpu().numpy()
        else:
            preds = np.append(preds, outputs.detach().cpu().numpy(), axis=0)
            annos = np.append(annos, labels.detach().cpu().numpy(), axis=0)

        test_loss +=  loss.item()

    preds_id = np.argmax(preds, axis=1)

    test_acc = accuracy_score(preds_id, annos)

    return test_loss/epoch_steps, test_acc, preds_id, annos

def later_analysis(args, model, eval_dataloader, device):

    grad_all_masks, eval_loss_org, eval_acc_org, pred_org, _ = compute_grad_mask(args, eval_dataloader, model, device)

    if args.dropping_rate == 0.1:
        logger.info('eval_loss_orginal = {:,.4f}, eval_loss_orginal = {:,.4f}'.format(eval_loss_org, eval_acc_org))
    eval_loss, eval_acc, pred, _ = test_process(args, model, eval_dataloader, device, grad_all_masks)
    logger.info('layer = {}, rate = {}, eval_loss = {:,.4f}, eval_acc = {:,.4f}'.format(args.mask_layer[0], args.dropping_rate, eval_loss, eval_acc))

def compute_grad_mask(args, eval_dataloader, model, device):
    model.eval()
    grad_all_masks = []
    test_loss = 0.
    preds = None
    annos = None
    epoch_steps = len(eval_dataloader)
    for step, datas in enumerate(eval_dataloader):
        input_ids, attn_mask, labels, _ = [data.to(device) for data in datas]

        extend_attn_mask = attn_mask.unsqueeze(1).unsqueeze(2).expand([-1, args.heads, args.max_len, -1])
        extend_attn_mask_list = [extend_attn_mask for i in range(args.layers)]
        extend_attn_mask_list = torch.stack(extend_attn_mask_list, dim=0)

        out = model(input_ids,
                    attention_mask=extend_attn_mask_list,
                    # token_type_ids=segment_ids,
                    labels=labels)
        loss = out[0]
        outputs = out[1]
        attention = out[2]

        if preds is None:
            preds = outputs.detach().cpu().numpy()
            annos = labels.detach().cpu().numpy()
        else:
            preds = np.append(preds, outputs.detach().cpu().numpy(), axis=0)
            annos = np.append(annos, labels.detach().cpu().numpy(), axis=0)

        test_loss +=  loss.item()

        outputs_true = outputs.gather(1, labels.view(-1, 1)).sum() # logit output of gold label

        for layer in args.mask_layer:
            attention[layer].retain_grad()

            (grad,) = torch.autograd.grad(outputs_true.view(-1), attention[layer], retain_graph=True)

            if args.dropping_method == 'low':
                grad_masks = generate_mask_low(args, grad, extend_attn_mask, attn_mask, device)  # [B, H, L, L]
            elif args.dropping_method == 'high':
                grad_masks = generate_mask_high(args, grad, extend_attn_mask, attn_mask, device)  # [B, H, L, L]
            else:
                raise NotImplementedError

            grad_all_masks.append(grad_masks.detach().cpu().numpy()) # mask for all samples


    preds_id = np.argmax(preds, axis=1)

    test_acc = accuracy_score(preds_id, annos)

    return np.concatenate(grad_all_masks,axis=0), test_loss/epoch_steps, test_acc, preds_id, annos




