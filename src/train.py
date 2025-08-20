import glob
import os
import sys
import time

import attr
import numpy as np
import pytz
import torch
import yaml
import torch.nn as nn
from src.model import SiameseUNetSMPShared

torch.backends.cudnn.benchmark = True
import datetime
import logging

# sys.path.extend(["../", "../.."])


def train(hps, train_loader, val_loader):
    """
    Trains a single or multi class segmentation model given a hps and a train and val_loader.
    """
    logging.shutdown()
    early_patience = hps.patience * 2 + 2 if hps.patience > 1 else 4

    # Set Saving and Logging path
    curr_time = datetime.datetime.now(pytz.timezone("Europe/Amsterdam")).strftime("%Y-%m-%d_%H-%M-%S")
    trained_models_folder = "trained_models"
    log_path = f"{trained_models_folder}/{hps.name}/fold_{hps.fold_nb}/{curr_time}"

    weights_already_trained = glob.glob(f"{trained_models_folder}/{hps.name}/fold_{hps.fold_nb}/*/*pt")
    if len(weights_already_trained) > 0:
        raise ValueError(
            f"For the experiment {hps.name} there already exist weight paths ({weights_already_trained}). "
            f"Choose a different experiment name or delete the existing weight paths"
        )
    os.makedirs(log_path, exist_ok=True)

    # Initialize logging
    log = logging.getLogger()  # root logger - Good to get it only once.
    for hdlr in log.handlers[:]:  # remove the existing file handlers
        log.removeHandler(hdlr)
    formatter = logging.Formatter("%(asctime)s - %(message)s", "%d-%b-%y %H:%M:%S")
    filehandler = logging.FileHandler(os.path.join(log_path, "logs.log"), "w")
    filehandler.setFormatter(formatter)
    log.addHandler(filehandler)  # set the new handler
    streamhandler = logging.StreamHandler()
    streamhandler.setFormatter(formatter)
    log.addHandler(streamhandler)
    log.setLevel(logging.INFO)

    # Initialize model, loss, optimizer and lr schedule
    model = SiameseUNetSMPShared(
        in_channels=hps.input_channel,
        classes=hps.num_classes + 1,
        encoder_name=hps.backbone, #"timm_efficientnet_b1",  # underscore or hyphen accepted
        encoder_weights=hps.pretrained, #"noisy-student", #"imagenet",
        encoder_depth=5,
        decoder_channels=(256, 128, 64, 32, 16),
        time_fusion_mode="concat_diff",
        # time_fusion_mode="diff",
    )

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    logging.info(f"Training with {torch.cuda.device_count()} GPUS")

    loss_func = XEDiceLoss(alpha=hps.alpha, num_classes=hps.num_classes, ignore_index=255)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=hps.lr * torch.cuda.device_count(), weight_decay=hps.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=hps.patience,
        threshold=0.0015,
    )
    model.to(device="cuda")

    if hps.use_fp16:
        scaler = torch.amp.GradScaler()
        logging.info(f"Training is done with mixed precision")

    if len(hps.resume) > 0:
        checkpoint = torch.load(hps.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Resume checkpoint %s" % hps.resume)
        if "optim_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optim_state_dict"])

    save_dict = {"model_name": hps.name, "model_time": curr_time, "hps": attr.asdict(hps)}
    best_metric, best_metric_epoch = -1, 0
    early_stop_counter, best_val_early_stopping_metric, early_stop_flag = 0, 0, False
    start_time, yaml_saved, lrs = time.time(), False, []

    for curr_epoch_num in range(1, hps.n_epochs):
        if curr_epoch_num > 1 and lr < lrs[-1]:
            logging.info(f"LR was reduced to {optimizer.param_groups[0]['lr']:.4e}")
        lrs.append(optimizer.param_groups[0]["lr"])
        model.train()
        torch.set_grad_enabled(True)

        num_batches = hps.num_batches if hps.num_batches else len(train_loader)
        losses, batch_time, data_time = AverageMeter(), AverageMeter(), AverageMeter()
        gpu_da_time = AverageMeter()
        end = time.time()
        for iter_num, data in enumerate(train_loader):
            data_time.update(time.time() - end)
            x_data = data["image"].to(device="cuda", non_blocking=True)
            targets = data["mask"].long().to(device="cuda", non_blocking=True)

            start = time.time()
            if hps.gpu_da_params[0] != 0:
                x_data, targets = gpu_da(x_data, targets, hps.gpu_da_params)
            gpu_da_time.update(time.time() - start)

            optimizer.zero_grad()
            if hps.use_fp16:
                with torch.amp.autocast(device_type="cuda"):
                    preds = model(x_data[:, :hps.input_channel], x_data[:, hps.input_channel:])
                    loss = loss_func(preds, targets)

                # Scales loss. Calls backward() on scaled loss to create scaled gradients.
                scaler.scale(loss).backward()
                # Unscales the gradients, then optimizer.step() is called if gradients are not inf/nan,
                scaler.step(optimizer)
                # Updates the scale for next iteration.
                scaler.update()
            else:
                preds = model(x_data[:, :hps.input_channel], x_data[:, hps.input_channel:])
                loss = loss_func(preds, targets)
                loss.backward()
                optimizer.step()

            losses.update(loss.detach().item(), x_data.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if iter_num > 0 and iter_num % hps.print_freq == 0:
                logging.info(
                    f"Ep: [{curr_epoch_num}] B: [{iter_num:}/{num_batches}] TotalT: {(time.time() - start_time) / 60:.1f} min, "
                    f"BatchT: {batch_time.avg:.3f}s, DataT: {data_time.avg:.3f}s, GpuDaT: {gpu_da_time.avg:.3f}s, Loss: {losses.avg:.4f}"
                )
            if iter_num == num_batches:
                break

        logging.info(
            f"Ep: [{curr_epoch_num}] TotalT: {(time.time() - start_time) / 60:.1f} min, "
            f"BatchT: {batch_time.avg:.3f}s, DataT: {data_time.avg:.3f}s, GpuDaT: {gpu_da_time.avg:.3f}s, Loss: {losses.avg:.4f}"
        )
        lr = optimizer.param_groups[0]["lr"]

        ## VAL PHASE
        model.eval()
        torch.set_grad_enabled(False)

        batch_time, data_time, losses = AverageMeter(), AverageMeter(), AverageMeter()
        metrics = Metrics()

        end = time.time()
        for iter_num, data in enumerate(val_loader):
            data_time.update(time.time() - end)
            x_data = data["image"].to(device="cuda", non_blocking=True)
            targets = data["mask"].long().to(device="cuda", non_blocking=True)

            if hps.use_fp16:
                with torch.amp.autocast(device_type="cuda"):
                    preds = model(x_data[:, :hps.input_channel], x_data[:, hps.input_channel:])
                    loss = loss_func(preds, targets)
            else:
                preds = model(x_data[:, :hps.input_channel], x_data[:, hps.input_channel:])
                loss = loss_func(preds, targets)
            losses.update(loss.detach().item(), x_data.size(0))

            preds = (torch.softmax(preds, dim=1)[:, 1:] > 0.5) * 1

            # calculate metrics on all water grps
            metrics.update_metrics(preds, targets)

            batch_time.update(time.time() - end)
            end = time.time()

        # Calculating IoUs in all subgroups
        metrics.calc_ious()

        # Log results
        logging.info(
            f"Ep: [{curr_epoch_num}]  ValT: {(batch_time.avg * len(val_loader)) / 60:.2f} min, BatchT: {batch_time.avg:.3f}s, "
            f"DataT: {data_time.avg:.3f}s, Loss: {losses.avg:.4f}, IoU: {metrics.iou:.4f} (val)"
        )

        # Learning Rate Scheduler
        if curr_epoch_num > hps.epoch_start_scheduler:
            scheduler.step(metrics.early_stopping_metric)

        # Early Stopping and model saving
        if metrics.early_stopping_metric > best_metric:
            best_metric, best_metric_epoch = metrics.early_stopping_metric, curr_epoch_num
            save_dict.update(
                {
                    "Combination_metric": best_metric,
                    "epoch_num": curr_epoch_num,
                    "model_state_dict": model.state_dict(),
                }
            )

            old_model = glob.glob(f"{log_path}/best_metric*")  # delete old best model
            if len(old_model) > 0:
                os.remove(old_model[0])

            save_path = f"{log_path}/best_metric_{curr_epoch_num}_{best_metric:.4f}.pt"
            torch.save(save_dict, save_path)
            if not yaml_saved:  # save hyperparams once as .yaml
                with open("/".join(save_path.split("/")[:-1]) + "/config.yaml", "w") as file:
                    yaml.dump(attr.asdict(hps), file)
                yaml_saved = True

        if metrics.early_stopping_metric < best_val_early_stopping_metric + 0.0001:
            early_stop_counter += 1
            lr = optimizer.param_groups[0]["lr"]
            if early_stop_counter > early_patience and lr < 5e-5:  # only stop training when lr is low
                logging.info("Early Stopping")
                early_stop_flag = True
        else:
            best_val_early_stopping_metric, early_stop_counter = metrics.early_stopping_metric, 0

        if early_stop_flag:
            break
    torch.cuda.empty_cache()

    logging.info(f"Best validation combination metric of {best_metric:.5f} in epoch {best_metric_epoch}.")
    return best_metric, best_metric_epoch


def rotate(input, degrees=90):
    """(..., H, W) input expected"""
    if degrees == 90:
        return input.transpose(-2, -1).flip(-2)
    if degrees == 180:
        return input.flip(-2).flip(-1)
    if degrees == 270:
        return input.transpose(-2, -1).flip(-1)


def transpose(input):
    """(..., H, W) input expected"""
    return input.transpose(-2, -1)


def apply_rotate_transpose(input, rot90, rot180, rot270, transpose):
    transformed: torch.Tensor = input.clone()
    to_rot90 = rot90.to(input.device)
    transformed[to_rot90] = rotate(input[to_rot90], degrees=90)
    to_rot180 = rot180.to(input.device)
    transformed[to_rot180] = rotate(input[to_rot180], degrees=180)
    to_rot270 = rot270.to(input.device)
    transformed[to_rot270] = rotate(input[to_rot270], degrees=270)
    to_transpose = transpose.to(input.device)
    transformed[to_transpose] = transformed[to_transpose].transpose(-2, -1)
    return transformed


def gpu_da(x_data, y_data, gpu_da_params):
    with torch.no_grad():
        bs = y_data.size(0)

        no_dihedral_p = gpu_da_params[0]
        transpose, rot90, rot180, rot270 = get_transpose_rot_boolean_lists(bs, no_dihedral_p)

        transpose, rot90, rot180, rot270 = (
            torch.tensor(transpose),
            torch.tensor(rot90),
            torch.tensor(rot180),
            torch.tensor(rot270),
        )
        debug_show = False
        if debug_show:
            raise NotImplementedError()
        else:
            x_data = apply_rotate_transpose(x_data, rot90, rot180, rot270, transpose)
            y_data = apply_rotate_transpose(y_data, rot90, rot180, rot270, transpose)
            return x_data, y_data


def get_transpose_rot_boolean_lists(bs, no_dihedral_p):
    """
    In no_dihedral_p % do nothing, in (1-no_dihedral_p) % / 7 do one of the 7 possible transpose/rot combinations.
    """
    transpose, rot90, rot180, rot270 = [False] * bs, [False] * bs, [False] * bs, [False] * bs
    perc_for_each_combination = (1 - no_dihedral_p) / 7
    for k in range(bs):
        rand_float = np.random.random()
        if rand_float < perc_for_each_combination:
            rot90[k] = True
        elif rand_float < 2 * perc_for_each_combination:
            rot180[k] = True
        elif rand_float < 3 * perc_for_each_combination:
            rot270[k] = True
        elif rand_float < 4 * perc_for_each_combination:
            rot90[k] = True
            transpose[k] = True
        elif rand_float < 5 * perc_for_each_combination:
            rot180[k] = True
            transpose[k] = True
        elif rand_float < 6 * perc_for_each_combination:
            rot270[k] = True
            transpose[k] = True
        elif rand_float < 7 * perc_for_each_combination:
            transpose[k] = True
        else:
            pass
    # print(f"transpose: {sum(transpose)}/{bs}, 90degree rot: {sum(rot90)}/{bs}, 180degree rot: {sum(rot180)}/{bs}, 270degree rot: {sum(rot270)}/{bs}")
    return transpose, rot90, rot180, rot270


EPS = 1e-7


class XEDiceLoss(nn.Module):
    """
    Mixture of alpha * CrossEntropy and (1 - alpha) * DiceLoss.
    """

    def __init__(self, alpha=0.5, num_classes=1, debug=False, ignore_index=255):
        super().__init__()
        self.xe = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.alpha = alpha
        self.num_classes = num_classes
        self.debug = debug
        self.ignore_index = ignore_index

    def forward(self, preds, targets):
        xe_loss = self.xe(preds, targets) if self.alpha != 0 else 0
        dice_loss = 0
        no_ignore = targets.ne(self.ignore_index)
        targets = targets.masked_select(no_ignore)

        preds = torch.softmax(preds, dim=1)
        for j in range(self.num_classes):
            pred = preds[:, j + 1]
            pred = pred.masked_select(no_ignore)
            y_dat = (targets == j + 1).float() if pred.dtype == torch.float32 else (targets == j + 1).half()
            dice_loss += 1 - (2.0 * torch.sum(pred * y_dat)) / (torch.sum(pred + y_dat) + EPS)
            # if self.debug:
        #     print(f"Dice for class {j+1}: {1 - (2. * torch.sum(pred * y_dat)) / (torch.sum(pred + y_dat) + EPS):.3f}")
        dice_loss /= self.num_classes
        # print(f"XE: {xe_loss:.3f}, Dice: {dice_loss:.3f}")

        return self.alpha * xe_loss + (1 - self.alpha) * dice_loss

    def get_name(self):
        return "XEDiceLoss"


class Metrics:
    """
    Computes and stores segmentation related metrices for training
    """

    def __init__(self) -> None:
        self.tps, self.fps, self.fns, self.iou = 0, 0, 0, 0

    def update_metrics(self, preds, targets):
        tps, fps, fns = tp_fp_fn_with_ignore(preds, targets)
        self.tps += tps
        self.fps += fps
        self.fns += fns

    def calc_ious(self):
        """
        Calculates IoUs per class and biome, mean biome IoUs, penalty and final metric used for early stopping
        """
        self.iou = self.tps / (self.tps + self.fps + self.fns)
        self.early_stopping_metric = self.iou


def tp_fp_fn_with_ignore(preds, targets):
    """
    Calculates True Positives, False Positives and False Negatives ignoring pixels where the target is 255.

    Args:
        preds (float tensor): Prediction tensor
        targets (long tensor): Target tensor
        c_i (int, optional): Class value of target for the positive class. Defaults to 1.

    Returns:
        tps, fps, fns: True Positives, False Positives and False Negatives
    """
    preds = preds.flatten()
    targets = targets.flatten()

    # ignore missing label pixels
    no_ignore = targets.ne(255)
    preds = preds.masked_select(no_ignore)
    targets = targets.masked_select(no_ignore)

    # calculate TPs/FPs/FNs on all water
    tps = torch.sum(preds * (targets == 1))
    fps = torch.sum(preds) - tps
    fns = torch.sum(targets == 1) - tps

    return tps, fps, fns


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count