import torch
import pandas as pd
import albumentations
import numpy as np
import tifffile
import cv2
import matplotlib.pyplot as plt

from src.visualize import visualize_s2_concat_bands_path, visualize_s1_path


class SingleSensorDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        img_paths_before,
        img_paths_after,
        mask_paths,
        transforms_only_img=None,
        transforms=None,
        val=False,
        test=False,
    ):
        """Dataset for training, validating and testing models.

        Args:
            img_paths (list of str): Paths to the input B02 path.
            mask_paths (list of str): Paths to the labels for the S2 images.
            transforms_only_img (albumentation.transforms, optional): Transforms to apply to the images only. Defaults to None.
            transforms (albumentation.transforms, optional): Transforms to apply to the images/masks. Defaults to None.
            val (bool, optional): If True, this dataset is used for validation.
                Defaults to False.
            test (bool, optional): If True, we don't provide the label, because we are testing. Defaults to False.
        """
        self.img_paths_before = img_paths_before
        self.img_paths_after = img_paths_after
        self.mask_paths = mask_paths
        self.transforms = transforms
        self.transforms_only_img = transforms_only_img
        self.val = val
        self.test = test

    def __getitem__(self, idx):
        sample = {}
        # # Load in image
        arr_x_before = tifffile.imread(self.img_paths_before[idx])
        if arr_x_before.shape[0] < 50:
            arr_x_before = arr_x_before.transpose((1, 2, 0))
        if arr_x_before.shape[0] != 400 or arr_x_before.shape[1] != 400:
            arr_x_before = cv2.resize(arr_x_before, (400, 400), interpolation=cv2.INTER_NEAREST)

        arr_x_after = tifffile.imread(self.img_paths_after[idx])
        if arr_x_after.shape[0] < 50:
            arr_x_after = arr_x_after.transpose((1, 2, 0))
        if arr_x_after.shape[0] != 400 or arr_x_after.shape[1] != 400:
            arr_x_after = cv2.resize(arr_x_after, (400, 400), interpolation=cv2.INTER_NEAREST)

        if "L2A" in self.img_paths_before[idx]:
            # Drop AOT, WVP bands from S2 L2A, 16d does not have them
            idxs_to_use = list(range(1, arr_x_before.shape[2] - 1))
            arr_x_before = arr_x_before[:, :, idxs_to_use]

        if "_sentinel2_" in self.img_paths_before[idx]:
            # Rescaling based on SCL multiplier
            scl_before = arr_x_before[:, :, -1]
            if scl_before.max() > 11:
                if scl_before.max() > 50:
                    factor = 6
                elif scl_before.max() > 45:
                    factor = 5
                elif scl_before.max() >= 40:
                    factor = 4.5
                else:
                    factor = 4
            else:
                factor = 1

            arr_x_before = (arr_x_before / factor).round(0).astype(np.float32)
            # low values, but all minimum above 1000 ==> Subtract by 1000
            if (np.nanmin(arr_x_before[:, :, [10, 7, 3]], axis=(0,1)) > 1000).sum() == 3:
                arr_x_before = arr_x_before - 1000
                arr_x_before[arr_x_before < 0] = 0

        if "L2A" in self.img_paths_after[idx]:
            # Drop AOT, WVP bands from S2 L2A, 16d does not have them
            idxs_to_use = list(range(1, arr_x_after.shape[2] - 1))
            arr_x_after = arr_x_after[:, :, idxs_to_use]

        if "_sentinel2_" in self.img_paths_after[idx]:
            # Rescaling based on SCL multiplier
            scl_after = arr_x_after[:, :, -1]
            if scl_after.max() > 11:
                if scl_after.max() > 50:
                    factor = 6
                elif scl_after.max() > 45:
                    factor = 5
                elif scl_after.max() >= 40:
                    factor = 4.5
                else:
                    factor = 4
            else:
                factor = 1

            arr_x_after = (arr_x_after / factor).round(0).astype(np.float32)
            # low values, but all minimum above 1000 ==> Subtract by 1000
            if (np.nanmin(arr_x_after[:, :, [10, 7, 3]], axis=(0,1)) > 1000).sum() == 3:
                arr_x_after = arr_x_after - 1000
                arr_x_after[arr_x_after < 0] = 0

        arr_x = np.concatenate([arr_x_before, arr_x_after], axis=-1)

        arr_x = np.nan_to_num(arr_x)
        sample["image"] = arr_x
        nan_mask_x = arr_x[:, :, 0] == 0

        # Load in and preprocess label mask
        if not self.test:
            if self.mask_paths[idx] == "empty":
                arr_y = np.zeros((400, 400), dtype=np.int32)
            else:
                arr_y = tifffile.imread(self.mask_paths[idx]).squeeze()
                arr_y = (arr_y == 255).astype(np.int32)
            arr_y[nan_mask_x] = 255  # set nan pixels to 255
            sample["mask"] = arr_y

        # Apply Data Augmentation
        if self.transforms:
            sample = self.transforms(image=sample["image"], mask=sample["mask"])
        if self.transforms_only_img:
            sample["image"] = self.transforms_only_img(image=sample["image"])["image"]
        if sample["image"].shape[-1] < 50:
            sample["image"] = sample["image"].transpose((2, 0, 1))

        if "_sentinel2_" in self.img_paths_before[idx]:
            sample["image"] = (sample["image"] / 2**15).astype(np.float32)
        elif "_sentinel1_" in self.img_paths_before[idx]:
            sample["image"] = (sample["image"] / 2**11).astype(np.float32) #2048
        if sample["image"].max() > 1: 
            print(f"Warning: Image max > 1 ({sample['image'].max():.3f}), Idx: {idx}\nBefore: {self.img_paths_before[idx]}\nAfter: {self.img_paths_after[idx]}")

        sample["img_path_before"] = self.img_paths_before[idx]
        sample["img_path_after"] = self.img_paths_after[idx]
        return sample

    def __len__(self):
        return len(self.img_paths_before)

    def visualize(self, how_many=1, show_specific_index=None):
        """Visualize a number of images from the dataset. The images are randomly selected unless show_specific_index is passed.

        Args:
            how_many (int, optional): number of images to visualize. Defaults to 1.
            show_specific_index (int, optional): If passed, only show the image corresponding to this index. Defaults to None.
        """
        for _ in range(how_many):
            rand_int = np.random.randint(len(self.img_paths_before))
            if show_specific_index is not None:
                rand_int = show_specific_index
            sample = self.__getitem__(rand_int)
            print(self.img_paths_before[rand_int], rand_int)
            print(self.img_paths_after[rand_int])
            print(self.mask_paths[rand_int])
            f, axarr = plt.subplots(1, 4, figsize=(30, 12))

            if "_sentinel2_" in self.img_paths_before[rand_int]:
                img_string = "S2"
                axarr[0].imshow(visualize_s2_concat_bands_path(self.img_paths_before[rand_int]))
                axarr[0].set_title(f"{img_string} Before Image")
                axarr[1].imshow(visualize_s2_concat_bands_path(self.img_paths_after[rand_int]))
                axarr[1].set_title(f"{img_string} After Image")
                img = sample["image"] * 2**15
                axarr[2].imshow(img[-7])
                axarr[2].set_title(
                    f"Mean: {img[-7].mean():.0f}, Min: {img[-7].min():.0f}, Max: {img[-7].max():.0f}", fontsize=15
                )
            else:
                img_string = "S1"
                axarr[0].imshow(visualize_s1_path(self.img_paths_before[rand_int]))
                axarr[0].set_title(f"{img_string} Before Image")
                axarr[1].imshow(visualize_s1_path(self.img_paths_after[rand_int]))
                axarr[1].set_title(f"{img_string} After Image")
                img = sample["image"] * 2**11
                axarr[2].imshow(img[1])
                axarr[2].set_title(
                    f"VH Mean: {img[1].mean():.0f}, Min: {img[1].min():.0f}, Max: {img[1].max():.0f}", fontsize=15
                )

            if "mask" in sample.keys():
                axarr[3].imshow(img[1])
                mask = sample["mask"]
                print(f"Mask unique values: {np.unique(mask)}")
                axarr[3].set_title(f"Mask==1 px: {(mask == 1).sum()}", fontsize=15)
                axarr[3].imshow(np.ma.masked_where(mask != 1, mask), cmap="autumn", alpha=0.5)
            plt.tight_layout()
            plt.show()


class S2S1Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        img_paths_before_s1,
        img_paths_before_s2,
        img_paths_after_s1,
        img_paths_after_s2,
        mask_paths,
        transforms_only_img=None,
        transforms=None,
        val=False,
        test=False,
    ):
        """Dataset for training, validating and testing models.

        Args:
            img_paths (list of str): Paths to the input B02 path.
            mask_paths (list of str): Paths to the labels for the S2 images.
            transforms_only_img (albumentation.transforms, optional): Transforms to apply to the images only. Defaults to None.
            transforms (albumentation.transforms, optional): Transforms to apply to the images/masks. Defaults to None.
            val (bool, optional): If True, this dataset is used for validation.
                Defaults to False.
            test (bool, optional): If True, we don't provide the label, because we are testing. Defaults to False.
        """
        self.img_paths_before_s1 = img_paths_before_s1
        self.img_paths_before_s2 = img_paths_before_s2
        self.img_paths_after_s1 = img_paths_after_s1
        self.img_paths_after_s2 = img_paths_after_s2
        self.mask_paths = mask_paths
        self.transforms = transforms
        self.transforms_only_img = transforms_only_img
        self.val = val
        self.test = test

    def __getitem__(self, idx):
        sample = {}
        
        # SENTINEL 1 BEFORE/AFTER
        if self.img_paths_before_s1[idx] == "empty":
            arr_x_before_s1 = np.ones((400, 400, 2), dtype=np.float32) * 2**11
        else:
            arr_x_before_s1 = tifffile.imread(self.img_paths_before_s1[idx])
            if arr_x_before_s1.shape[0] != 400 or arr_x_before_s1.shape[1] != 400:
                arr_x_before_s1 = cv2.resize(arr_x_before_s1, (400, 400), interpolation=cv2.INTER_NEAREST)

        if self.img_paths_after_s1[idx] == "empty":
            arr_x_after_s1 = np.ones((400, 400, 2), dtype=np.float32) * 2**11
        else:
            arr_x_after_s1 = tifffile.imread(self.img_paths_after_s1[idx])
            if arr_x_after_s1.shape[0] != 400 or arr_x_after_s1.shape[1] != 400:
                arr_x_after_s1 = cv2.resize(arr_x_after_s1, (400, 400), interpolation=cv2.INTER_NEAREST)
            nan_mask_x = arr_x_after_s1[:, :, 0] == 0

        arr_x_before_s1 = (arr_x_before_s1 / 2**11).astype(np.float32)
        arr_x_after_s1 = (arr_x_after_s1 / 2**11).astype(np.float32)

        # SENTINEL 2 BEFORE/AFTER
        if self.img_paths_before_s2[idx] == "empty":
            arr_x_before_s2 = np.ones((400, 400, 13), dtype=np.float32) * 2**15
        else:
            arr_x_before_s2 = tifffile.imread(self.img_paths_before_s2[idx])
            if arr_x_before_s2.shape[0] != 400 or arr_x_before_s2.shape[1] != 400:
                arr_x_before_s2 = cv2.resize(arr_x_before_s2, (400, 400), interpolation=cv2.INTER_NEAREST)
            if "L2A" in self.img_paths_before_s2[idx]:
                # Drop AOT, WVP bands from S2 L2A, 16d does not have them
                idxs_to_use = list(range(1, arr_x_before_s2.shape[2] - 1))
                arr_x_before_s2 = arr_x_before_s2[:, :, idxs_to_use]

            # Rescaling based on SCL multiplier
            scl_before = arr_x_before_s2[:, :, -1]
            if scl_before.max() > 11:
                if scl_before.max() > 50:
                    factor = 6
                elif scl_before.max() > 45:
                    factor = 5
                elif scl_before.max() >= 40:
                    factor = 4.5
                else:
                    factor = 4
            else:
                factor = 1

            arr_x_before_s2 = (arr_x_before_s2 / factor).round(0).astype(np.float32)
            # low values, but all minimum above 1000 ==> Subtract by 1000
            if (np.nanmin(arr_x_before_s2[:, :, [10, 7, 3]], axis=(0,1)) > 1000).sum() == 3:
                arr_x_before_s2 = arr_x_before_s2 - 1000
                arr_x_before_s2[arr_x_before_s2 < 0] = 0

        if self.img_paths_after_s2[idx] == "empty":
            arr_x_after_s2 = np.ones((400, 400, 13), dtype=np.float32) * 2**15
        else:
            arr_x_after_s2 = tifffile.imread(self.img_paths_after_s2[idx])
            if arr_x_after_s2.shape[0] != 400 or arr_x_after_s2.shape[1] != 400:
                arr_x_after_s2 = cv2.resize(arr_x_after_s2, (400, 400), interpolation=cv2.INTER_NEAREST)
            nan_mask_x = arr_x_after_s2[:, :, 0] == 0

            if "L2A" in self.img_paths_after_s2[idx]:
                # Drop AOT, WVP bands from S2 L2A, 16d does not have them
                idxs_to_use = list(range(1, arr_x_after_s2.shape[2] - 1))
                arr_x_after_s2 = arr_x_after_s2[:, :, idxs_to_use]

            # Rescaling based on SCL multiplier
            scl_before = arr_x_after_s2[:, :, -1]
            if scl_before.max() > 11:
                if scl_before.max() > 50:
                    factor = 6
                elif scl_before.max() > 45:
                    factor = 5
                elif scl_before.max() >= 40:
                    factor = 4.5
                else:
                    factor = 4
            else:
                factor = 1

            arr_x_after_s2 = (arr_x_after_s2 / factor).round(0).astype(np.float32)
            # low values, but all minimum above 1000 ==> Subtract by 1000
            if (np.nanmin(arr_x_after_s2[:, :, [10, 7, 3]], axis=(0,1)) > 1000).sum() == 3:
                arr_x_after_s2 = arr_x_after_s2 - 1000
                arr_x_after_s2[arr_x_after_s2 < 0] = 0
        arr_x_before_s2 = (arr_x_before_s2 / 2**15).astype(np.float32)
        arr_x_after_s2 = (arr_x_after_s2 / 2**15).astype(np.float32)

        # CONCAT ALL 4 IMAGES TOGETHER
        arr_x = np.concatenate([arr_x_before_s1, arr_x_before_s2, arr_x_after_s1, arr_x_after_s2], axis=-1)

        arr_x = np.nan_to_num(arr_x)
        sample["image"] = arr_x

        # Load in and preprocess label mask
        if not self.test:
            if self.mask_paths[idx] == "empty":
                arr_y = np.zeros((400, 400), dtype=np.int32)
            else:
                arr_y = tifffile.imread(self.mask_paths[idx]).squeeze()
                arr_y = (arr_y == 255).astype(np.int32)
            arr_y[nan_mask_x] = 255  # set nan pixels to 255
            sample["mask"] = arr_y

        # Apply Data Augmentation
        if self.transforms:
            sample = self.transforms(image=sample["image"], mask=sample["mask"])
        if self.transforms_only_img:
            sample["image"] = self.transforms_only_img(image=sample["image"])["image"]
        if sample["image"].shape[-1] < 50:
            sample["image"] = sample["image"].transpose((2, 0, 1))

        if sample["image"].max() > 1: 
            print(f"Warning: Image max > 1 ({sample['image'].max():.3f}), Idx: {idx}\n"
            f"Before S1: {self.img_paths_before_s1[idx]}\nAfter S1: {self.img_paths_after_s1[idx]}"
            f"Before S2: {self.img_paths_before_s2[idx]}\nAfter S2: {self.img_paths_after_s2[idx]}")

        sample["img_path_before_s1"] = self.img_paths_before_s1[idx]
        sample["img_path_before_s2"] = self.img_paths_before_s2[idx]
        sample["img_path_after_s1"] = self.img_paths_after_s1[idx]
        sample["img_path_after_s2"] = self.img_paths_after_s2[idx]
        return sample

    def __len__(self):
        return len(self.img_paths_before_s1)

    def visualize(self, how_many=1, show_specific_index=None):
        """Visualize a number of images from the dataset. The images are randomly selected unless show_specific_index is passed.

        Args:
            how_many (int, optional): number of images to visualize. Defaults to 1.
            show_specific_index (int, optional): If passed, only show the image corresponding to this index. Defaults to None.
        """
        for _ in range(how_many):
            rand_int = np.random.randint(len(self.img_paths_before_s1))
            if show_specific_index is not None:
                rand_int = show_specific_index
            sample = self.__getitem__(rand_int)
            print(self.img_paths_before_s1[rand_int], rand_int)
            print(self.img_paths_after_s1[rand_int])
            print(self.img_paths_before_s2[rand_int])
            print(self.img_paths_after_s2[rand_int])
            print(self.mask_paths[rand_int])
            f, axarr = plt.subplots(1, 5, figsize=(30, 12))

            img_string = "S2"
            if self.img_paths_before_s2[rand_int] == "empty":
                img = np.zeros((400, 400))
            else:
                img = visualize_s2_concat_bands_path(self.img_paths_before_s2[rand_int])
            axarr[0].imshow(img)
            axarr[0].set_title(f"{img_string} Before Image")
            if self.img_paths_after_s2[rand_int] == "empty":
                img = np.zeros((400, 400))
            else:
                img = visualize_s2_concat_bands_path(self.img_paths_after_s2[rand_int])
            axarr[1].imshow(img)
            axarr[1].set_title(f"{img_string} After Image")

            img_string = "S1"
            if self.img_paths_before_s1[rand_int] == "empty":
                img = np.zeros((400, 400))
            else:
                img = visualize_s1_path(self.img_paths_before_s1[rand_int])
            axarr[2].imshow(img)
            axarr[2].set_title(f"{img_string} Before Image")
            if self.img_paths_after_s1[rand_int] == "empty":
                img = np.zeros((400, 400))
            else:
                img = visualize_s1_path(self.img_paths_after_s1[rand_int])
            axarr[3].imshow(img)
            axarr[3].set_title(f"{img_string} After Image")
            if "mask" in sample.keys():
                mask = sample["mask"]
                print(f"Mask unique values: {np.unique(mask)}")
                axarr[4].set_title(f"Mask==1 px: {(mask == 1).sum()}", fontsize=15)
                axarr[4].imshow(np.ma.masked_where(mask != 1, mask), cmap="autumn", alpha=0.5)
            plt.tight_layout()
            plt.show()

def get_dataloaders(hps):
    """Builds datasets and dataloaders for training/validation"""
    df = pd.read_pickle(hps.df_path)
    num_workers = 4

    ## Setup the correct image and label paths
    mask_paths_val = df.loc[df["fold"] == hps.fold_nb, "label_path"].tolist()
    val, test = True, False

    if "path_best_before" in df.columns:
        img_paths_before_val = df.loc[df["fold"] == hps.fold_nb, "path_best_before"].tolist()
        img_paths_after_val = df.loc[df["fold"] == hps.fold_nb, "path_best_after"].tolist()
        val_dataset = SingleSensorDataset(
            img_paths_before_val,
            img_paths_after_val,
            mask_paths_val,
            val=val,
            test=test,
            transforms=albumentations.Compose([albumentations.CenterCrop(384, 384)]),
        )
    else:
        img_paths_before_s1_val = df.loc[df["fold"] == hps.fold_nb, "s1_before_path"].tolist()
        img_paths_before_s2_val = df.loc[df["fold"] == hps.fold_nb, "s2_before_path"].tolist()
        img_paths_after_s1_val = df.loc[df["fold"] == hps.fold_nb, "s1_after_path"].tolist()
        img_paths_after_s2_val = df.loc[df["fold"] == hps.fold_nb, "s2_after_path"].tolist()
        val_dataset = S2S1Dataset(
            img_paths_before_s1_val,
            img_paths_before_s2_val,
            img_paths_after_s1_val,
            img_paths_after_s2_val,
            mask_paths_val,
            val=val,
            test=test,
            transforms=albumentations.Compose([albumentations.CenterCrop(384, 384)]),
        )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=hps.val_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    if hps.only_val:
        return val_dataset, val_loader

    ## Building the training dataset and dataloader
    # Data Augmentations
    train_transforms = get_train_transforms(hps)
    train_img_transforms = get_train_img_transforms(hps)
    print(f"Train Data Augmentations: {train_transforms}")
    print(f"Train Image Data Augmentations: {train_img_transforms}")
    print("#" * 100)

    if "path_best_before" in df.columns:
        img_paths_before_train = df.loc[df["fold"] != hps.fold_nb, "path_best_before"].tolist()
        img_paths_after_train = df.loc[df["fold"] != hps.fold_nb, "path_best_after"].tolist()
        if "weight" in df.columns:
            weights_all = df.loc[df["fold"] != hps.fold_nb, "weight"].tolist()
        else:
            weights_all = [1] * len(img_paths_before_train)  # np.full(len(img_paths_before_train), 1)
    else:
        img_paths_before_s1_train = df.loc[df["fold"] != hps.fold_nb, "s1_before_path"].tolist()
        img_paths_before_s2_train = df.loc[df["fold"] != hps.fold_nb, "s2_before_path"].tolist()
        img_paths_after_s1_train = df.loc[df["fold"] != hps.fold_nb, "s1_after_path"].tolist()
        img_paths_after_s2_train = df.loc[df["fold"] != hps.fold_nb, "s2_after_path"].tolist()

    if "weight" in df.columns:
        weights_all = df.loc[df["fold"] != hps.fold_nb, "weight"].tolist()
    else:
        weights_all = [1] * len(df.loc[df["fold"] != hps.fold_nb])  # np.full(len(img_paths_before_train), 1)
    mask_paths_train = df.loc[df["fold"] != hps.fold_nb, "label_path"].tolist()

    print("#" * 100)
    print(f"Fold {hps.fold_nb} --> Train: {len(df[df['fold'] != hps.fold_nb])}, Val: {len(val_dataset)}")
    print("#" * 100)

    if "path_best_before" in df.columns:
        # repeat all paths if we need longer train epochs
        while hps.num_batches > int(len(img_paths_before_train) / hps.train_batch_size):
            print(
                f"We want to train {hps.num_batches} batches in each epoch, but our current dataset is only "
                f"{int(len(img_paths_before_train) / hps.train_batch_size)} batches long. Doubling dataset size"
            )
            img_paths_before_train = img_paths_before_train * 2
            img_paths_after_train = img_paths_after_train * 2
            mask_paths_train = mask_paths_train * 2
            weights_all = weights_all * 2

        train_dataset = SingleSensorDataset(
            img_paths_before_train,
            img_paths_after_train,
            mask_paths_train,
            transforms_only_img=albumentations.Compose(train_img_transforms),
            transforms=albumentations.Compose(train_transforms),
        )
    else:
        # repeat all paths if we need longer train epochs
        while hps.num_batches > int(len(img_paths_before_s1_train) / hps.train_batch_size):
            print(
                f"We want to train {hps.num_batches} batches in each epoch, but our current dataset is only "
                f"{int(len(img_paths_before_s1_train) / hps.train_batch_size)} batches long. Doubling dataset size"
            )
            img_paths_before_s1_train = img_paths_before_s1_train * 2
            img_paths_before_s2_train = img_paths_before_s2_train * 2
            img_paths_after_s1_train = img_paths_after_s1_train * 2
            img_paths_after_s2_train = img_paths_after_s2_train * 2
            mask_paths_train = mask_paths_train * 2
            weights_all = weights_all * 2

        train_dataset = S2S1Dataset(
            img_paths_before_s1_train,
            img_paths_before_s2_train,
            img_paths_after_s1_train,
            img_paths_after_s2_train,
            mask_paths_train,
            transforms_only_img=albumentations.Compose(train_img_transforms),
            transforms=albumentations.Compose(train_transforms),
        )

    # weights = [1] * len(weights_all) # for debugging
    print(f"Last 5 sample weights: {weights_all[-5:]}")
    weights_all = torch.Tensor(weights_all)
    weights_all = weights_all.double()
    msg = f"Training dataset and sampling weights have not the same length: {len(train_dataset)} and {len(weights_all)}"
    assert len(train_dataset) == len(weights_all), msg
    if sum(weights_all) == len(weights_all):
        sampler, train_shuffle = None, True
    else:
        sampler, train_shuffle = (
            torch.utils.data.sampler.WeightedRandomSampler(weights_all, len(weights_all)),
            False,
        )
        print("Using a Weighted Random Sampler with different weights for different batches of data")
        print("#" * 100)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=hps.train_batch_size,
        sampler=sampler,
        shuffle=train_shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
    )

    return train_dataset, train_loader, val_dataset, val_loader


def get_train_transforms(hps):
    """Returns train transforms using information from hps"""
    train_transforms = []
    if hps.random_sized_params[0]:
        train_transforms.append(
            albumentations.RandomSizedCrop(  # default params [0.875, 1.125]
                min_max_height=(
                    int(hps.train_crop_size * hps.random_sized_params[0]),
                    int(hps.train_crop_size * hps.random_sized_params[1]),
                ),
                height=hps.train_crop_size,
                width=hps.train_crop_size,
                interpolation=hps.randomsizecrop_interpolation,
            )
        )  # cv2.INTER_NEAREST faster than cv2.INTER_LINEAR faster than cv2.INTER_CUBIC
    else:
        train_transforms.append(albumentations.RandomCrop(hps.train_crop_size, hps.train_crop_size))
    return train_transforms


def get_train_img_transforms(hps):
    """Returns train transforms using information from hps"""
    train_img_transforms = []
    if hps.da_brightness_magnitude or hps.da_contrast_magnitude:
        train_img_transforms.append(
            albumentations.RandomBrightnessContrast(
                brightness_limit=hps.da_brightness_magnitude,
                contrast_limit=hps.da_contrast_magnitude,
                brightness_by_max=True,
                always_apply=False,
                p=1.0,
            )
        )
    return train_img_transforms