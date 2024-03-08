import math
import sys
import contextlib

import torch
import torch.distributed as dist

import util.misc as misc
import util.lr_sched as lr_sched

def train_one_epoch(model: torch.nn.Module,
                    data_loader, optimizer: torch.optim.Optimizer,
                    epoch: int, start_iter, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    for data_iter_step, data_img  in enumerate(
        metric_logger.log_every(data_loader, print_freq, header, start_iter), start=start_iter
    ):
        if len(data_img) == 4:
            examples, labels, image, modal = data_img
        elif len(data_img) == 3:
            examples, labels, modal = data_img
            image = None
        if data_iter_step % accum_iter == 0:
            # lr_sched.adjust_learning_rate(optimizer, data_iter_step, args)
            lr_sched.adjust_learning_rate_epoch(optimizer, data_iter_step / len(data_loader) + epoch, args)
        update_grad = (data_iter_step + 1) % accum_iter == 0

        autocast_ctx = {
            "bf16": torch.cuda.amp.autocast(dtype=torch.bfloat16),
            "fp16": torch.cuda.amp.autocast(dtype=torch.float16),
            "tf32": contextlib.nullcontext(),
        }[args.precision]
        backward_ctx = contextlib.nullcontext() if update_grad else model.no_sync()
        
        with autocast_ctx:
            i_loss = model(examples, labels, image, modal)
        i_loss_value = i_loss.item()
        if not math.isfinite(i_loss_value):
            print("[Rank {}] i_loss is {}, stopping training".format(dist.get_rank(), i_loss_value), force=True)
            # print(image_paths, force=True)
            sys.exit(1)
        loss_value = i_loss_value
        with backward_ctx:
            grad_norm = loss_scaler(
                i_loss / accum_iter, optimizer, model,
                parameters=model.parameters(),
                update_grad=update_grad,
                clip_grad=None if args.clip_grad <= 0 else args.clip_grad,
            )
        if update_grad:
            assert grad_norm is not None
            metric_logger.update(grad_norm=grad_norm)

        if update_grad:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(iloss=i_loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        # save checkpoint
        if data_iter_step % 1000 == 0 and data_iter_step != 0:
            misc.save_model(
                output_dir=args.output_dir,
                args=args, epoch=epoch, iteration=data_iter_step, model=model, optimizer=optimizer,
                loss_scaler=loss_scaler, dataset_state=None)

        if update_grad:
            loss_value_reduce = misc.all_reduce_mean(loss_value)
            i_loss_value_reduce = misc.all_reduce_mean(i_loss_value)
            if update_grad:
                grad_norm_reduce = misc.all_reduce_mean(grad_norm)

        if log_writer is not None and update_grad:
            log_writer.add_scalar('train_loss', loss_value_reduce, data_iter_step)
            log_writer.add_scalar('i_train_loss', i_loss_value_reduce, data_iter_step)
            if update_grad:
                log_writer.add_scalar('grad_norm', grad_norm_reduce, data_iter_step)
            log_writer.add_scalar('lr', lr, data_iter_step)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
