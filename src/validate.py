import glob
import torch
import pandas as pd

from tqdm import tqdm
import os
import tifffile
import matplotlib.pyplot as plt
import numpy as np

from src.model import SiameseUNetShared
from src.train import build_model, tp_fp_fn_with_ignore
from src.data import get_dataloaders
from src.visualize import visualize_s2_concat_bands_path


def predict_and_save(
    models,
    val_loader,
    on_gpu=True,
    output_spatial_size=(512, 512),
    save_path=None,
):
    """
    Predict and save as uint8 (0-255) to 'save_path' given a val_loader and a list of loaded models and hps
    """
    probs = []
    torch.set_grad_enabled(False)
    for iter_num, data in tqdm(enumerate(val_loader), total=len(val_loader)):
        x_data = data["image"]
        if on_gpu:
            x_data = x_data.cuda(non_blocking=True)

        with torch.cuda.amp.autocast():
            preds = torch.softmax(models[0](x_data), dim=1)[:, 1:]
            for model_nb in range(1, len(models)):
                preds += torch.softmax(models[model_nb](x_data), dim=1)[:, 1:]

        preds /= len(models)
        if preds.size(1) == 1:
            preds = preds[:, 0]

        if preds.size(-1) != output_spatial_size[-1]:
            raise ValueError(
                f"You want to save predictions of size ({preds.size(-2)}, {preds.size(-1)}) into size {output_spatial_size}. Specify the correct size."
            )
        probs.append(preds.cpu().numpy())  # * 255).astype(np.uint8))
    # Concatenate predictions with potentially unequal batch size for the last batch
    probs_last = probs[-1]
    probs = np.array(probs[:-1])
    probs = probs.reshape((-1,))
    probs = np.concatenate((probs, probs_last), axis=0)
    return probs


def find_and_load_models(hps_list, fold_list=[0, 1, 2], trained_models_dir=None):
    """Given a list of hps and folds, returns a list of build models with their best weights loaded."""
    if trained_models_dir is None:
        trained_models_dir = "trained_models"
    models = []
    for fold_nb in fold_list:
        for hps in hps_list:
            model = SiameseUNetShared(in_ch=hps.input_channel, base_ch=32, fusion_mode="concat_diff", out_ch=2)
            # model = build_model(hps)
            model = model.cuda()
            model.eval()

            model_weights_paths = glob.glob(f"{trained_models_dir}/{hps.name}/fold_{fold_nb}/*/*pt")
            if len(model_weights_paths) > 1:
                raise ValueError(f"Multiple weight paths: {model_weights_paths}")
            cp = torch.load(model_weights_paths[0])
            model.load_state_dict(cp["model_state_dict"])
            print(f"Loaded {model_weights_paths[0]} at epoch {cp['epoch_num']}")
            if torch.cuda.device_count() > 1:  # Multi GPU
                model = torch.nn.DataParallel(model)
            models.append(model)
    return models


def return_metrics_all_folds(
    hps_list,
    threshold=0.5,
    on_gpu=True,
    output_spatial_size=(512, 512),
    also_save_preds=False,
    save_path=None,
    fold_nbs=[0,1,2,3,4],
    preloaded_models=None,
):
    """
    Return tps, fps and fns per image per class given a threshold per class and a list of hps.
    """
    if also_save_preds:  # Initialize bcolz array for saving preds as uint8
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
            x_data = data["image"]
            targets = data["mask"]
            if on_gpu:
                x_data = x_data.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)

            # predict
            with torch.cuda.amp.autocast():
                preds = torch.softmax(models[0](x_data[:, :hps.input_channel], x_data[:, hps.input_channel:]), dim=1)[:, 1]
                for model_nb in range(1, len(models)):
                    preds += torch.softmax(models[model_nb](x_data[:, :hps.input_channel], x_data[:, hps.input_channel:]), dim=1)[:, 1]
            preds /= len(models)

            if also_save_preds:  # save preds as uint8
                if preds.size(-1) != output_spatial_size[-1]:
                    raise ValueError(
                        f"You want to save predictions of size ({preds.size(-2)}, {preds.size(-1)}) "
                        f"into size {output_spatial_size}. Specify the correct size."
                    )
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


def return_metrics(
    models,
    hps_list,
    val_loader,
    thresholds=[0.5],
    on_gpu=True,
    output_spatial_size=(512, 512),
    also_save_preds=False,
    save_path=None,
):
    """
    Return tps, fps and fns per image per class for each of the thresholds given a val_loader and a list of loaded models and hps.
    """
    hps = hps_list[0]
    num_classes = hps_list[0].num_classes
    if also_save_preds:
        if save_path is None:
            raise ValueError("Please provide a 'save_path' if you want to save predictions to disk")
        if num_classes == 1:
            save_shape = (0, output_spatial_size[0], output_spatial_size[1])
        else:
            save_shape = (0, num_classes, output_spatial_size[0], output_spatial_size[1])
        probs = bcolz.carray(np.zeros(save_shape, "uint8"), mode="w", rootdir=save_path)

    losses = []
    tps = {t: [] for t in thresholds}
    fps = {t: [] for t in thresholds}
    fns = {t: [] for t in thresholds}

    torch.set_grad_enabled(False)
    for iter_num, data in enumerate(val_loader):
        x_data = data["image"]
        targets = data["mask"]
        if on_gpu:
            x_data = x_data.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

        with torch.cuda.amp.autocast():
            preds = torch.softmax(models[0](x_data[:, :hps.input_channel], x_data[:, hps.input_channel:]), dim=1)[:, 1:]
            for model_nb in range(1, len(models)):
                preds += torch.softmax(models[model_nb](x_data[:, :hps.input_channel], x_data[:, hps.input_channel:]), dim=1)[:, 1:]
        preds /= len(models)

        for threshold in thresholds:
            pred = (preds[:, 0] > threshold) * 1
            for k in range(len(targets)):
                tp, fp, fn = tp_fp_fn_with_ignore(pred[k], targets[k])
                assert tp >= 0 and fp >= 0 and fn >= 0, f"Negative tp/fp/fn: tp: {tp}, fp: {fp}, fn: {fn}"
                tps[threshold].append(tp.item())
                fps[threshold].append(fp.item())
                fns[threshold].append(fn.item())

        if also_save_preds:
            if num_classes == 1:
                preds = preds[:, 0]
            if preds.size(-1) != output_spatial_size[0]:
                raise ValueError(
                    f"You want to save predictions of size ({preds.size(-2)}, {preds.size(-1)}) into size {output_spatial_size}. Specify the correct size."
                )
            probs.append((preds.cpu().numpy() * 255).astype(np.uint8))
            if iter_num % 10 == 9:
                probs.flush()
    if also_save_preds:
        probs.flush()
    if len(thresholds) == 1:
        tps, fps, fns = tps[thresholds[0]], fps[thresholds[0]], fns[thresholds[0]]
    return losses, tps, fps, fns


def cross_validation(
    hps_list,
    thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    debug=False,
    preloaded_models=None,
):
    """
    Finds the best thresholds per class by calculating IoUs given a list of thresholds and a list of hps over all
    validation folds. If the list of hps has length more than one, an ensemble of multiple models is cross validated.
    """
    df = pd.read_pickle(hps_list[0].df_path)
    # df = df.iloc[:500]
    df = df[df["fold"].notnull()]

    hps_names = [hps.name for hps in hps_list]
    # Get tps, fps, fns on all water and all water subgroups for all folds
    for fold_nb in range(int(df.loc[df["fold"] != "train", "fold"].max()) + 1):
        if preloaded_models:
            models = preloaded_models
        else:
            models = find_and_load_models(hps_list, fold_list=[fold_nb])

        hps = hps_list[0]
        hps.only_val = True
        hps.fold_nb = fold_nb
        _, val_loader = get_dataloaders(hps=hps)

        _, tps, fps, fns = return_metrics(models, hps_list, val_loader, thresholds=thresholds)
        for t in thresholds:
            df.loc[df["fold"] == fold_nb, f"tp@{t:.2f}"] = tps[t]
            df.loc[df["fold"] == fold_nb, f"fp@{t:.2f}"] = fps[t]
            df.loc[df["fold"] == fold_nb, f"fn@{t:.2f}"] = fns[t]

    # Calculate IoUs on all water and for all subgroups for all thresholds --> Find the best threshold
    final_metric, best_threshold, best_comb_score, best_stats = {}, {}, {}, {}
    miou, final_metric, best_threshold, best_comb_score = {}, -100, 0.5, -100
    best_stats = {}
    for t in thresholds:
        tp_str, fp_str = f"tp@{t:.2f}", f"fp@{t:.2f}"
        fn_str = f"fn@{t:.2f}"
        stats = df[[tp_str, fp_str, fn_str]].sum()
        stats["IoU"] = stats[tp_str] / (stats[tp_str] + stats[fp_str] + stats[fn_str])
        miou = stats["IoU"].mean()

        if debug:
            print("#" * 100)
            print(f"Performance @ t = {t:.2f} ({hps_names})")
            print("=" * 100)
            stats["recall"] = stats[tp_str] / (stats[tp_str] + stats[fn_str])
            stats["precision"] = stats[tp_str] / (stats[tp_str] + stats[fp_str])
            print(stats[["recall", "precision", "IoU"]])

        final_metric = miou
        if final_metric > best_comb_score:
            best_threshold, best_comb_score = t, final_metric
        else:
            if not debug:
                break

    # Given the best threshold, now we can print out the crossvalidation results for all water and flood water
    t = best_threshold
    tp_str, fp_str = f"tp@{t:.2f}", f"fp@{t:.2f}"
    fn_str = f"fn@{t:.2f}"
    stats = df[[tp_str, fp_str, fn_str]].sum()
    stats["IoU"] = stats[tp_str] / (stats[tp_str] + stats[fp_str] + stats[fn_str])
    best_stats = stats.copy()
    miou = stats["IoU"].mean()

    print("#" * 100)
    print(f"Performance @ t = {t:.2f} ({hps_names})")
    print("=" * 100)
    stats["recall"] = stats[tp_str] / (stats[tp_str] + stats[fn_str])
    stats["precision"] = stats[tp_str] / (stats[tp_str] + stats[fp_str])
    print(stats[["recall", "precision", "IoU"]])
    print("=" * 100)
    print(f"Best IoU of @{t:.2f}: {best_comb_score:.4f}")

    return df, best_stats, best_threshold


def calc_ious(group, calc_more_columns=False):
    """Given a dataframe group object, calculate and return IoUs and optionally Precision and Recall"""
    result = {}
    result["img_count"] = group.shape[0]
    cols1, cols2 = [], []
    tps = group["tps"].sum() + 1e-12
    fps = group["fps"].sum() + 1e-12
    fns = group["fns"].sum() + 1e-12
    precision = tps / (tps + fps)
    recall = tps / (tps + fns)
    if sum([tps, fps, fns]) < 500:
        # print(tps, fps, fns)
        result[f"IoU"] = np.nan
    else:
        result[f"IoU"] = tps / (tps + fps + fns)
    if calc_more_columns:
        result[f"Precision"] = precision
        result[f"Recall"] = recall
        result[f"TPS"] = tps / 1e6
        result[f"FPS"] = fps / 1e6
        result[f"FNS"] = fns / 1e6
    cols1.append(f"IoU")

    if not calc_more_columns:
        return pd.Series(result, index=["img_count"] + cols1)
    else:
        cols2.extend([f"Precision", f"Recall", "TPS", "FPS", "FNS"])
        return pd.Series(result, index=["img_count"] + cols1 + cols2)


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
        # np.random.seed(222911)
        np.random.shuffle(idxs_to_use)
    #     idxs_to_use = [
    #         781,3268,1471,1717,4632,3643,2098,4114,3017,2388,2292,1281,4932,452,3011,2330,3673,6639,4109,733,446,1924,
    # 4932,1933,6211,2000,2307,4113,3840,2082,1209,3542,2866,3713,4167,2769,4274,2298,2487,5392,1584,
    #     ]

    tp_sum, fp_sum, fn_sum = sum(tps) + 1, sum(fps) + 1, sum(fns) + 1
    looked_at, looked_at_idx = [], []
    count = 0
    # locations = sorted(df["location"].unique().tolist())
    location_count = 0
    for idx_, i in enumerate(idxs_to_use):
        # print(i)
        # if not "47.0625,15.4226" in img_paths[i]:
        #     continue
        # if fps[i] < 400:
        #     continue
        # if fps[i] >= 700:
        #     continue
        # if "_improved" in GT_paths[i]:
        #     continue


        # pred = np.array([(preds[i].astype(np.float32) / 255.0) for preds in preds_list])
        pred = np.array([preds[i] for preds in preds_list])
        # pred = (preds[i] > threshold).astype(np.uint8)

        # pred_std = np.std(pred, axis=0)
        # pred_std_02 = (pred_std > 0.15).astype(np.uint8)
        # pred_std_02_sum = pred_std_02.sum()

        pred = (pred.mean(axis=0) > threshold) * 1
        # pred_sum = pred.sum()
        # std_to_pred_ratio = pred_std_02_sum / (pred_sum+1)
        # if std_to_pred_ratio < .5:
        #     continue
        # if pred_std_02_sum + pred_sum <= 2000:
        #     continue
        # print(f"{pred_std_02_sum=}")
        # print(f"{pred_sum=}")
        # print(f"{std_to_pred_ratio=:.2f}")
        count += 1
        if count <= skip_how_many:
            continue

        fold, clouds_nodata_best_before, clouds_nodata_best_after = df.loc[df["path_best_before"] == img_paths_before[i], [
            "fold", "clouds_nodata_best_before", "clouds_nodata_best_after"]].iloc[0]

        looked_at.append(img_paths_before[i])
        looked_at_idx.append(i)
        print(f"{img_paths_before[i]}, Fold {fold}")
        print(f"{img_paths_after[i]}, Fold {fold}")
        print(f"{GT_paths[i]}")
        # print(f"FP/FN ratio      : {fps[i]/(fns[i]+1):.1f}")
        print(f"FN/FP ratio      : {fns[i]/(fps[i]+1):.1f}")
        print(f"FP/TP ratio      : {fps[i]/(tps[i]+1):.1f}")
        print(f"FN/TP ratio      : {fns[i]/(tps[i]+1):.1f}")
        print(f"FP+FN / TP ratio : {(fps[i] + fns[i])/(tps[i]+1):.1f}")
        # print(f"Max(FP,FN) / Min(FP,FN) ratio : {max_ratio:.1f}")
        print("#" * 100)
        print(f"{i}")  # - {location:20} - B2 > 7000: {high_b2/1000:.1f}k")
        f, ax = plt.subplots(2, 3, figsize=(35, 20))

        img_before = visualize_s2_concat_bands_path(img_paths_before[i])
        img_after = visualize_s2_concat_bands_path(img_paths_after[i])

        ax[0,0].imshow(img_after)
        after_date = img_paths_after[i].split("/")[-1].split('.')[0].split('_')[-1]
        after_date_str = f"{after_date[:4]}-{after_date[4:6]}-{after_date[6:8]}"
        ax[0,0].set_title(f"After: {after_date_str}", fontsize=fontsize)
        # pred = ((preds[i].astype(np.float32) / 255.0) > threshold) * 1
        ax[0,1].imshow(img_after)
        ax[0,1].imshow(np.ma.masked_where(pred == 0, pred), cmap="autumn", alpha=alpha)
        title_string = f"FPs: {fps[i]:.0f} = {100 * fps[i] / fp_sum:.1f}%, FNs: {fns[i]:.0f} = {100 * fns[i] / fn_sum:.1f}%, TPs: {tps[i]:.0f}"
        ax[0,1].set_title(f"(Pred) {title_string}", fontsize=fontsize)

        label = tifffile.imread(GT_paths[i])
        # nan_mask = label == 255
        label = (label == 255).astype(np.uint8)
        # label[nan_mask] = 0
        ax[0,2].imshow(img_after)
        ax[0,2].imshow(np.ma.masked_where(label != 1, label), cmap="autumn", alpha=alpha)
        ax[0,2].set_title(f"image with GT mask", fontsize=fontsize)

        before_date = img_paths_before[i].split("/")[-1].split('.')[0].split('_')[-1]
        before_date_str = f"{before_date[:4]}-{before_date[4:6]}-{before_date[6:8]}"
        ax[1,0].set_title(f"Before: {before_date_str}", fontsize=fontsize)
        ax[1,0].imshow(img_before)
        ax[1,1].imshow(img_after)
        ax[1,1].set_title(f"After: {after_date_str}", fontsize=fontsize)

        # ax[3].imshow(img_first_ax)
        # ax[3].imshow(
        #     scale_S2_img(arr_x[:, :, 0:1], min_values=np.array([100]), max_values=np.array([3500]))[:, :, 0]
        # )
        # ax[3].set_title(f"Blue", fontsize=15)
        # ax[3].imshow(np.ma.masked_where(pred_std_02 != 1, pred_std_02), cmap="autumn", alpha=alpha)
        # ax[3].set_title(f"Std > 0.2: {100 * pred_std_02_sum/512**2:.1f}%", fontsize=15)
        # plt.tight_layout()
        # plt.show()

        # f, ax = plt.subplots(1, 2, figsize=(35, 12))
        # ax[0].imshow(img_first_ax)
        # ax[0].set_title(f"Image", fontsize=fontsize)

        # ax[2].imshow(img_first_ax)
        # blue_clouds = ((arr_x[:, :, 0] > 2300) * 1).copy()
        # ax[2].imshow(np.ma.masked_where(pred_std_02 != 1, pred_std_02), cmap="autumn", alpha=alpha)
        # ax[2].set_title(f"B2 > 2300", fontsize=15)
        # plt.tight_layout()
        # plt.show()

        # f, ax = plt.subplots(1, 2, figsize=(35, 15))
        # ax[0].imshow(img_first_ax)
        fps_ = ((pred == 1) & (label == 0)) * 1
        fns_ = ((pred == 0) & (label == 1)) * 1
        # ax[1].imshow(np.ma.masked_where(fps_ != 1, fps_), cmap="autumn", alpha=1)
        # ax[1].imshow(np.ma.masked_where(fns_ != 1, fns_), cmap="gray", alpha=1)
        # ax[1].set_title(f"FPs = RED ({fps_.sum()}), FNs = BLACK ({fns_.sum()})", fontsize=fontsize)
        ax[1,2].imshow(np.ma.masked_where(fps_ != 1, fps_), cmap="autumn", alpha=1)
        ax[1,2].imshow(np.ma.masked_where(fns_ != 1, fns_), cmap="gray", alpha=1)
        ax[1,2].set_title(f"FPs = RED ({fps_.sum()}), FNs = BLACK ({fns_.sum()})", fontsize=fontsize)
        plt.tight_layout()
        plt.show()

        if count >= export_how_many + skip_how_many:
            break
    return looked_at, looked_at_idx


# fcc_to_bands = {
#     "SWIR": ["B12", "B8", "B4"],
#     "SWIRP": ["B8", "B11", "B4"],
#     "RGB": ["B4", "B3", "B2"],
#     "CIR": ["B8", "B4", "B3"],
# }
# fcc_to_band_indices = {
#     "SWIR": [12, 7, 3],
#     "SWIRP": [7, 11, 3],
#     "RGB": [3, 2, 1],
#     "CIR": [7, 3, 2],
# }
# min_max = {
#     "SWIR": {"min": [100, 100, 100], "max": [3500] * 3},
#     "SWIRP": {"min": [100, 100, 100], "max": [3500] * 3},
#     "RGB": {"min": [100, 100, 100], "max": [2500] * 3},
#     "CIR": {"min": [100, 100, 100], "max": [3500] * 3},
# }

# def scale_S2_img(matrix, min_values, max_values):
#     w, h, d = matrix.shape
#     if min_values is None:
#         min_values = np.array([100, 100, 100])
#         max_values = np.array([3500, 3500, 3500])

#     flat = np.reshape(matrix, (w * h, d)).astype(np.float64)
#     flat = (flat - min_values[None, :]) / (max_values[None, :] - min_values[None, :])
#     out = np.reshape(flat, (w, h, d))
#     return out.clip(0, 1)

# def visualize_s2_path(path_b2, profile = "SWIR", downsample = 0):
#     img = tifffile.imread([path_b2.replace("B2", band) for band in fcc_to_bands[profile]])
#     if img.shape[0] < 15:
#         img = np.transpose(img, (1, 2, 0))
#     if downsample > 0:
#         img = img[::downsample, ::downsample]
#     return scale_S2_img(img, np.array(min_max[profile]["min"]), np.array(min_max[profile]["max"]))

# def visualize_s2_concat_bands_path(concat_bands_path, profile = "SWIR", downsample = 0):
#     img = tifffile.imread(concat_bands_path)
#     if downsample > 0:
#         img = img[::downsample, ::downsample]
#     return scale_S2_img(img[:, :, fcc_to_band_indices[profile]], np.array(min_max[profile]["min"]), np.array(min_max[profile]["max"]))

def predict_and_save_multiple_separate_models(hps, models):
    """
    Predict and save multiple models to separate save_paths given a val_loader and a list of loaded models.
    Hyperparameters are assumed to be the same. Predictions are saved as uint8 (0-255).
    """
    probs = []
    for k in range(len(models)):
        probs.append([])

    torch.set_grad_enabled(False)
    hps.only_val = True
    _, val_loader = get_dataloaders(hps=hps)
    
    for iter_num, data in tqdm(enumerate(val_loader), total=len(val_loader)):
        # pass
        x_data = data["image"].cuda(non_blocking=True)
        for k in range(len(models)):
            # preds = torch.sigmoid(models[k](x_data)[:, 0])
            preds = torch.softmax(models[k](x_data[:, :hps.input_channel], x_data[:, hps.input_channel:]), dim=1)[:, 1:]
            # probs[k].append((preds.cpu().numpy() * 255).round(0).astype(np.uint8))
            probs[k].append(preds.cpu().numpy())
    for k in range(len(models)):
        probs[k] = np.concatenate(probs[k], axis=0)
    return probs


indices_to_pred = [
    213, 112, 
]

indices_to_nan = [
    102, 94,
]