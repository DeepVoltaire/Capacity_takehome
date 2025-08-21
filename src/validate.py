import glob
import torch
import pandas as pd
from tqdm import tqdm
import os
import tifffile
import matplotlib.pyplot as plt
import numpy as np

from src.model import build_model
from src.train import tp_fp_fn_with_ignore
from src.data import get_dataloaders
from src.visualize import visualize_s2_concat_bands_path, visualize_s1_path


def find_and_load_models(hps_list, fold_list=[0, 1, 2]):
    """Given a list of hps and folds, returns a list of build models with their best weights loaded."""
    models = []
    for fold_nb in fold_list:
        for hps in hps_list:
            model = build_model(hps)
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
                if hps.model == "siamese_unet":
                    preds = torch.softmax(models[0](x_data[:, :hps.input_channel], x_data[:, hps.input_channel:]), dim=1)[:, 1]
                else:
                    preds = torch.softmax(models[0](x_data), dim=1)[:, 1]
                for model_nb in range(1, len(models)):
                    if hps.model == "siamese_unet":
                        preds += torch.softmax(models[model_nb](x_data[:, :hps.input_channel], x_data[:, hps.input_channel:]), dim=1)[:, 1]
                    else:
                        preds += torch.softmax(models[model_nb](x_data), dim=1)[:, 1]
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
    preds,
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
        pred = (preds[i] > threshold) * 1

        fold, clouds_nodata_best_before, clouds_nodata_best_after = df.loc[df["path_best_before"] == img_paths_before[i], [
            "fold", "clouds_nodata_best_before", "clouds_nodata_best_after"]].iloc[0]

        print(f"{img_paths_before[i]}, Fold {int(fold)}")
        print(f"{img_paths_after[i]}")
        print(f"{GT_paths[i]}")
        print(f"FN/FP ratio      : {fns[i]/(fps[i]+1):.1f}")
        print(f"FP/TP ratio      : {fps[i]/(tps[i]+1):.1f}")
        print(f"FN/TP ratio      : {fns[i]/(tps[i]+1):.1f}")
        print(f"FP+FN / TP ratio : {(fps[i] + fns[i])/(tps[i]+1):.1f}")
        print("#" * 100)
        print(f"{i}")

        if "_sentinel2_" in img_paths_before[i]:
            img_before = visualize_s2_concat_bands_path(img_paths_before[i])
            img_after = visualize_s2_concat_bands_path(img_paths_after[i])
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
        plt.tight_layout()
        plt.show()

        if count >= export_how_many + skip_how_many:
            break


def visualize_S2_S1_model(
    df,
    tps,
    fps,
    fns,
    preds,
    threshold,
    img_paths_before_s1,
    img_paths_after_s1,
    img_paths_before_s2,
    img_paths_after_s2,
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
        pred = (preds[i] > threshold) * 1

        fold, clouds_nodata_best_before, clouds_nodata_best_after = df.loc[
            (df["s1_after_path"] == img_paths_after_s1[i]) & (df["s2_after_path"] == img_paths_after_s2[i]), [
            "fold", "clouds_nodata_best_before", "clouds_nodata_best_after"]].iloc[0]

        print(f"{img_paths_before_s1[i]}, Fold {int(fold)}")
        print(f"{img_paths_after_s1[i]}")
        print(f"{img_paths_before_s2[i]}")
        print(f"{img_paths_after_s2[i]}")
        print(f"{GT_paths[i]}")
        print(f"FN/FP ratio      : {fns[i]/(fps[i]+1):.1f}")
        print(f"FP/TP ratio      : {fps[i]/(tps[i]+1):.1f}")
        print(f"FN/TP ratio      : {fns[i]/(tps[i]+1):.1f}")
        print(f"FP+FN / TP ratio : {(fps[i] + fns[i])/(tps[i]+1):.1f}")
        print("#" * 100)
        print(f"{i}")

        after_sensor = "S2"
        if img_paths_before_s1[i] == "empty":
            img_before_s1 = np.zeros((400, 400))
        else:
            img_before_s1 = visualize_s1_path(img_paths_before_s1[i])
        if img_paths_after_s1[i] == "empty":
            img_after_s1 = np.zeros((400, 400))
        else:
            after_sensor = "S1"
            img_after_s1 = visualize_s1_path(img_paths_after_s1[i])

        if img_paths_before_s2[i] == "empty":
            img_before_s2 = np.zeros((400, 400))
        else:
            img_before_s2 = visualize_s2_concat_bands_path(img_paths_before_s2[i])
        if img_paths_after_s2[i] == "empty":
            img_after_s2 = np.zeros((400, 400))
        else:
            img_after_s2 = visualize_s2_concat_bands_path(img_paths_after_s2[i])

        before_date_s1 = img_paths_before_s1[i].replace("_uncompressed", "").split("/")[-1].split('.')[0].split('_')[-1]
        before_date_s1 = f"{before_date_s1[:4]}-{before_date_s1[4:6]}-{before_date_s1[6:8]}"
        before_date_s2 = img_paths_before_s2[i].replace("_uncompressed", "").split("/")[-1].split('.')[0].split('_')[-1]
        before_date_s2 = f"{before_date_s2[:4]}-{before_date_s2[4:6]}-{before_date_s2[6:8]}"
        after_date_s1 = img_paths_after_s1[i].replace("_uncompressed", "").split("/")[-1].split('.')[0].split('_')[-1]
        after_date_s1 = f"{after_date_s1[:4]}-{after_date_s1[4:6]}-{after_date_s1[6:8]}"
        after_date_s2 = img_paths_after_s2[i].replace("_uncompressed", "").split("/")[-1].split('.')[0].split('_')[-1]
        after_date_s2 = f"{after_date_s2[:4]}-{after_date_s2[4:6]}-{after_date_s2[6:8]}"

        f, ax = plt.subplots(1, 4, figsize=(35, 10))
        if after_sensor == "S1":
            ax[0].imshow(img_before_s1)
            ax[0].set_title(f"S1 Before: {before_date_s1}", fontsize=fontsize)
            ax[1].imshow(img_after_s1)
            ax[1].set_title(f"S1 After: {after_date_s1}", fontsize=fontsize)
            ax[2].imshow(img_after_s1)
            ax[3].imshow(img_after_s1)
        else:
            ax[0].imshow(img_before_s2)
            ax[0].set_title(f"S2 Before: {before_date_s2}", fontsize=fontsize)
            ax[1].imshow(img_after_s2)
            ax[1].set_title(f"S2 After: {after_date_s2}", fontsize=fontsize)
            ax[2].imshow(img_after_s2)
            ax[3].imshow(img_after_s2)
        
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
        # ax[3].imshow(img_after)
        ax[3].imshow(np.ma.masked_where(label != 1, label), cmap="autumn", alpha=alpha)
        ax[3].set_title(f"Deforestation mask {deforest_date}", fontsize=fontsize)
        plt.tight_layout()
        plt.show()

        # Show S1/S2 Before, S1/S2 After
        f, ax = plt.subplots(1, 4, figsize=(35, 10))
        ax[0].imshow(img_before_s1)
        ax[0].set_title(f"S1 Before: {before_date_s1}", fontsize=fontsize)

        ax[1].imshow(img_before_s2)
        ax[1].set_title(f"S2 Before: {before_date_s2}", fontsize=fontsize)

        ax[2].imshow(img_after_s1)
        ax[2].set_title(f"S1 After: {after_date_s1}", fontsize=fontsize)

        ax[3].imshow(img_after_s2)
        ax[3].set_title(f"S2 After: {after_date_s2}", fontsize=fontsize)
        plt.tight_layout()
        plt.show()

        if count >= export_how_many + skip_how_many:
            break