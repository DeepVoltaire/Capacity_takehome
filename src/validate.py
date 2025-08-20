import glob
import torch
import pandas as pd
from tqdm import tqdm
import os
import tifffile
import matplotlib.pyplot as plt
import numpy as np

from src.model import SiameseUNetSMPShared
from src.train import tp_fp_fn_with_ignore
from src.data import get_dataloaders
from src.visualize import visualize_s2_concat_bands_path, visualize_s1_path, visualize_s2_concat_bands_path_new


def predict_and_save(models, val_loader):
    """
    Predicts given a val_loader and a list of loaded models and hps
    """
    probs = []
    torch.set_grad_enabled(False)
    for iter_num, data in tqdm(enumerate(val_loader), total=len(val_loader)):
        x_data = data["image"].cuda(non_blocking=True)
        with torch.amp.autocast(device_type="cuda"):
            preds = torch.softmax(models[0](x_data), dim=1)[:, 1:]
            for model_nb in range(1, len(models)):
                preds += torch.softmax(models[model_nb](x_data), dim=1)[:, 1:]

        preds /= len(models)
        if preds.size(1) == 1:
            preds = preds[:, 0]
        probs.append(preds.cpu().numpy())
    # Concatenate predictions with potentially unequal batch size for the last batch
    probs_last = probs[-1]
    probs = np.array(probs[:-1])
    probs = probs.reshape((-1,))
    probs = np.concatenate((probs, probs_last), axis=0)
    return probs


def find_and_load_models(hps_list, fold_list=[0, 1, 2]):
    """Given a list of hps and folds, returns a list of build models with their best weights loaded."""
    models = []
    for fold_nb in fold_list:
        for hps in hps_list:
            model = SiameseUNetSMPShared(
                in_channels=hps.input_channel,
                classes=hps.num_classes + 1,
                encoder_name=hps.backbone,
                encoder_weights=hps.pretrained,
                encoder_depth=5,
                decoder_channels=(256, 128, 64, 32, 16),
                time_fusion_mode="concat_diff",
            )

            model = model.cuda()
            model.eval()

            model_weights_paths = glob.glob(f"trained_models/{hps.name}/fold_{fold_nb}/*/*pt")
            if len(model_weights_paths) > 1:
                raise ValueError(f"Multiple weight paths: {model_weights_paths}")
            cp = torch.load(model_weights_paths[0], weights_only=True)
            model.load_state_dict(cp["model_state_dict"])
            print(f"Loaded {model_weights_paths[0]} at epoch {cp['epoch_num']}")
            if torch.cuda.device_count() > 1:  # Multi GPU
                model = torch.nn.DataParallel(model)
            models.append(model)
    return models


def return_metrics_all_folds(
    hps_list,
    threshold=0.5,
    output_spatial_size=(512, 512),
    also_save_preds=False,
    fold_nbs=[0,1,2,3,4],
    preloaded_models=None,
):
    """
    Return tps, fps and fns per image per class given a threshold per class and a list of hps.
    """
    if also_save_preds:
        probs = []

    losses, tps, fps, fns = [], [], [], []

    for fold_nb in fold_nbs:
        # Initialize models, load in weights
        if preloaded_models:
            models = preloaded_models
        else:
            models = find_and_load_models(hps_list, fold_list=[fold_nb])

        # For the current val fold, get dataloader, predict and gather metrics
        hps = hps_list[0]
        hps.only_val = True
        hps.fold_nb = fold_nb
        hps.val_batch_size = 4
        _, val_loader = get_dataloaders(hps=hps)

        torch.set_grad_enabled(False)
        for iter_num, data in enumerate(val_loader):
            x_data = data["image"].cuda(non_blocking=True)
            targets = data["mask"].cuda(non_blocking=True)

            # predict
            with torch.amp.autocast(device_type="cuda"):
                preds = torch.softmax(models[0](x_data[:, :hps.input_channel], x_data[:, hps.input_channel:]), dim=1)[:, 1]
                for model_nb in range(1, len(models)):
                    preds += torch.softmax(models[model_nb](x_data[:, :hps.input_channel], x_data[:, hps.input_channel:]), dim=1)[:, 1]
            preds /= len(models)

            if also_save_preds:  # save preds as uint8
                probs.append(preds.cpu().numpy())

            # gather metrics
            preds = (preds > threshold) * 1

            for k in range(len(targets)):
                tp, fp, fn = tp_fp_fn_with_ignore(preds[k], targets[k])
                assert tp >= 0 and fp >= 0 and fn >= 0, f"Negative tp/fp/fn: tp: {tp}, fp: {fp}, fn: {fn}"
                tps.append(tp.item())
                fps.append(fp.item())
                fns.append(fn.item())

    if also_save_preds:
        probs = np.concatenate(probs, axis=0)
    return probs, losses, tps, fps, fns


def visualize_single_model(
    df,
    tps,
    fps,
    fns,
    preds_list,
    threshold,
    img_paths_before,
    img_paths_after,
    GT_paths,
    export_how_many=15,
    skip_how_many=0,
    alpha=0.5,
    selection_criteria="FP+FN",
    fontsize=20,
):
    """
    Visualizes the worst or random predictions on validation images given lists of tps, fps, fns
    """
    if selection_criteria == "FP+FN":
        idxs_to_use = np.argsort([fp + fn for fp, fn in zip(fps, fns)])[::-1]
    elif selection_criteria == "FN":
        idxs_to_use = np.argsort(fns)[::-1]
    elif selection_criteria == "FP":
        idxs_to_use = np.argsort(fps)[::-1]
    elif selection_criteria == "TP":
        idxs_to_use = np.argsort(tps)[::-1]
    elif selection_criteria == "FP+FN ratio":
        idxs_to_use = np.argsort([fn / (fp + 1e-6) for fp, fn in zip(fps, fns)])[::-1]
    elif selection_criteria == "(FP+FN)/TP ratio":
        idxs_to_use = np.argsort([(fp + fn) / (tp + 1e-6) for fp, fn, tp in zip(fps, fns, tps)])[::-1]
    else:
        idxs_to_use = list(range(len(fns)))
        np.random.shuffle(idxs_to_use)

    tp_sum, fp_sum, fn_sum = sum(tps) + 1, sum(fps) + 1, sum(fns) + 1
    count = 0
    for idx_, i in enumerate(idxs_to_use):
        count += 1
        if count <= skip_how_many:
            continue
        pred = np.array([preds[i] for preds in preds_list])
        pred = (pred.mean(axis=0) > threshold) * 1

        fold, clouds_nodata_best_before, clouds_nodata_best_after = df.loc[df["path_best_before"] == img_paths_before[i], [
            "fold", "clouds_nodata_best_before", "clouds_nodata_best_after"]].iloc[0]

        print(f"{img_paths_before[i]}, Fold {fold}")
        print(f"{img_paths_after[i]}")
        print(f"{GT_paths[i]}")
        print(f"FN/FP ratio      : {fns[i]/(fps[i]+1):.1f}")
        print(f"FP/TP ratio      : {fps[i]/(tps[i]+1):.1f}")
        print(f"FN/TP ratio      : {fns[i]/(tps[i]+1):.1f}")
        print(f"FP+FN / TP ratio : {(fps[i] + fns[i])/(tps[i]+1):.1f}")
        print("#" * 100)
        print(f"{i}")

        if "_sentinel2_" in img_paths_before[i]:
            img_before = visualize_s2_concat_bands_path_new(img_paths_before[i])
            img_after = visualize_s2_concat_bands_path_new(img_paths_after[i])
        else:
            img_before = visualize_s1_path(img_paths_before[i])
            img_after = visualize_s1_path(img_paths_after[i])

        f, ax = plt.subplots(1, 4, figsize=(35, 10))
        before_date = img_paths_before[i].replace("_uncompressed", "").split("/")[-1].split('.')[0].split('_')[-1]
        before_date_str = f"{before_date[:4]}-{before_date[4:6]}-{before_date[6:8]}"
        ax[0].set_title(f"Before: {before_date_str}", fontsize=fontsize)
        ax[0].imshow(img_before)

        ax[1].imshow(img_after)
        after_date = img_paths_after[i].replace("_uncompressed", "").split("/")[-1].split('.')[0].split('_')[-1]
        after_date_str = f"{after_date[:4]}-{after_date[4:6]}-{after_date[6:8]}"
        ax[1].set_title(f"After: {after_date_str}", fontsize=fontsize)

        ax[2].imshow(img_after)
        ax[2].imshow(np.ma.masked_where(pred == 0, pred), cmap="autumn", alpha=alpha)
        title_string = f"FPs: {fps[i]:.0f} = {100 * fps[i] / fp_sum:.1f}%, FNs: {fns[i]:.0f} = {100 * fns[i] / fn_sum:.1f}%, TPs: {tps[i]:.0f}"
        ax[2].set_title(f"(Pred) {title_string}", fontsize=fontsize)

        if GT_paths[i] == "empty":
            label = np.zeros((400, 400), dtype=np.uint8)
            deforest_date = ""
        else:
            label = tifffile.imread(GT_paths[i])
            deforest_date = GT_paths[i].split("/")[-1].split("_")[1]
            deforest_date = f"{deforest_date[:4]}-{deforest_date[4:6]}-{deforest_date[6:8]}"
        label = (label == 255).astype(np.uint8)
        ax[3].imshow(img_after)
        ax[3].imshow(np.ma.masked_where(label != 1, label), cmap="autumn", alpha=alpha)
        ax[3].set_title(f"Deforestation mask {deforest_date}", fontsize=fontsize)

        # before_date = img_paths_before[i].replace("_uncompressed", "").split("/")[-1].split('.')[0].split('_')[-1]
        # before_date_str = f"{before_date[:4]}-{before_date[4:6]}-{before_date[6:8]}"
        # ax[1,0].set_title(f"Before: {before_date_str}", fontsize=fontsize)
        # ax[1,0].imshow(img_before)
        # ax[1,1].imshow(img_after)
        # ax[1,1].set_title(f"After: {after_date_str}", fontsize=fontsize)

        # fps_ = ((pred == 1) & (label == 0)) * 1
        # fns_ = ((pred == 0) & (label == 1)) * 1
        # ax[1,2].imshow(np.ma.masked_where(fps_ != 1, fps_), cmap="autumn", alpha=1)
        # ax[1,2].imshow(np.ma.masked_where(fns_ != 1, fns_), cmap="gray", alpha=1)
        # ax[1,2].set_title(f"FPs = RED ({fps_.sum()}), FNs = BLACK ({fns_.sum()})", fontsize=fontsize)
        plt.tight_layout()
        plt.show()

        if count >= export_how_many + skip_how_many:
            break