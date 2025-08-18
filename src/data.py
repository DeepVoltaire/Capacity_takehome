import torch
import pandas as pd
import albumentations
import numpy as np
import tifffile
import cv2
import matplotlib.pyplot as plt


class LoadTifDataset(torch.utils.data.Dataset):
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
        """Dataset for training, validating and testing S2 models.

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

        arr_x = np.concatenate([arr_x_before, arr_x_after], axis=-1)

        arr_x = np.nan_to_num(arr_x)
        sample["image"] = arr_x
        nan_mask_x = arr_x[:, :, 0] == 0

        # Load in and preprocess label mask
        if not self.test:
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

        sample["image"] = (sample["image"] / 2**15).astype(np.float32)

        sample["img_path_before"] = self.img_paths_before[idx]
        # assert sample["image"].shape[0] == 3, f"Image shape is {sample['image']}, expected 3 bands."
        # print(f"{self.mask_paths[idx]} - {sample['mask'].shape}, {self.img_paths[idx]} - {sample['image'].shape}")
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
            rand_int = np.random.randint(len(self.img_paths))
            if show_specific_index is not None:
                rand_int = show_specific_index
            sample = self.__getitem__(rand_int)
            print(self.img_paths[rand_int], rand_int)
            print(self.mask_paths[rand_int])
            f, axarr = plt.subplots(1, 3, figsize=(30, 12))

            img_string = "S2"
            arr_x = tifffile.imread([self.img_paths[rand_int].replace("B02.tif", f"B0{k}.tif") for k in [2, 3, 4]])
            if arr_x.shape[0] < 20:
                arr_x = arr_x.transpose((1, 2, 0))
            axarr[0].imshow(scale_S2_img(arr_x))

            img = sample["image"] * 2**16
            axarr[0].set_title(f"{img_string} Image")  # . Min: {img.min():.4f}, Max: {img.max():.4f}")
            # axarr[1].imshow(arr_x[:, :, 0])
            # axarr[1].set_title(f"Min: {arr_x[:, :, 0].min():.0f}, Max: {arr_x[:, :, 0].max():.0f}", fontsize=15)
            axarr[1].imshow(img[0])
            axarr[1].set_title(
                f"Mean: {img[0].mean():.0f}, Min: {img[0].min():.0f}, Max: {img[0].max():.0f}", fontsize=15
            )

            if "mask" in sample.keys():
                axarr[2].imshow(img[0])
                mask = sample["mask"]
                print(f"Mask unique values: {np.unique(mask)}")
                axarr[2].set_title(f"Mask==1 px: {(mask == 1).sum()}", fontsize=15)
                axarr[2].imshow(np.ma.masked_where(mask == 0, mask), cmap="spring", alpha=0.4)
            plt.tight_layout()
            plt.show()


def scale_S2_img(matrix, min_values=None, max_values=None):
    """Returns a scaled (H,W,D) image which is more easily visually inspectable. Image is linearly scaled between
    min and max_value of by channel"""
    w, h, d = matrix.shape
    if min_values is None:
        min_values = np.array([100, 100, 100])
        max_values = np.array([3500, 3500, 3500])

    matrix = np.reshape(matrix, [w * h, d]).astype(np.float64)
    matrix = (matrix - min_values[None, :]) / (max_values[None, :] - min_values[None, :])
    matrix = np.reshape(matrix, [w, h, d])

    matrix = matrix.clip(0, 1)
    return matrix


class TestDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        img_paths,
        transforms=None,
    ):
        """Dataset for predicting with S1/S2 models where each image loaded is automatically padded.

        Args:
            img_paths (list of list of str): Paths to the input S1 or S2 images.
            transforms (albumentation.transforms, optional): Transforms to apply to the images/masks. Defaults to None.
        """
        self.img_paths = img_paths
        self.transforms = transforms
        self.size_multiplier = 32

    def __getitem__(self, idx):
        # Load in image
        arr_x = np.nan_to_num(tifffile.imread(self.img_paths[idx]))

        if arr_x.shape[0] < 20:
            arr_x = arr_x.transpose((1, 2, 0))

        arr_x = (arr_x / 2 ** 15).astype(np.float32)
        # arr_x = (arr_x / 255).astype(np.float32)

        sample = {"image": arr_x}

        # Apply Data Augmentation
        if self.transforms:
            if sample["image"].shape[0] < 20:
                sample["image"] = sample["image"].transpose((1, 2, 0))
            sample = self.transforms(image=sample["image"], mask=sample["mask"])
        if sample["image"].shape[-1] < 20:
            sample["image"] = sample["image"].transpose((2, 0, 1))
        print(f"{self.mask_paths[idx]} - {sample['mask'].shape}, {self.img_paths[idx]} - {sample['image'].shape}")
        sample["path"] = self.img_paths[idx]
        return sample

    def __len__(self):
        return len(self.img_paths)


def get_dataloaders(hps):
    """Builds datasets and dataloaders for training/validation"""
    df = pd.read_pickle(hps.df_path)
    # df = df.iloc[:500]
    num_workers = 4
    ## Setup the correct image and label paths

    img_paths_before_val = df.loc[df["fold"] == hps.fold_nb, "path_best_before"].tolist()
    img_paths_after_val = df.loc[df["fold"] == hps.fold_nb, "path_best_after"].tolist()
    mask_paths_val = df.loc[df["fold"] == hps.fold_nb, "label_path"].tolist()
    val, test = True, False

    val_dataset = LoadTifDataset(
        img_paths_before_val,
        img_paths_after_val,
        mask_paths_val,
        val=val,
        test=test,
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
    img_paths_before_train = df.loc[df["fold"] != hps.fold_nb, "path_best_before"].tolist()
    img_paths_after_train = df.loc[df["fold"] != hps.fold_nb, "path_best_after"].tolist()
    mask_paths_train = df.loc[df["fold"] != hps.fold_nb, "label_path"].tolist()
    if "weight" in df.columns:
        weights_all = df.loc[df["fold"] != hps.fold_nb, "weight"].tolist()
    else:
        weights_all = [1] * len(img_paths_before_train)  # np.full(len(img_paths_before_train), 1)

    print("#" * 100)
    print(f"Fold {hps.fold_nb} --> Train: {len(img_paths_before_train)}, Val: {len(val_dataset)}")
    print("#" * 100)

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

    train_dataset = LoadTifDataset(
        img_paths_before_train,
        img_paths_after_train,
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
    if hps.da_p_cutout:
        train_transforms.append(
            albumentations.Cutout(
                num_holes=8,
                max_h_size=int(hps.train_crop_size * 0.1875),
                max_w_size=int(hps.train_crop_size * 0.1875),
                p=hps.da_p_cutout,
            )
        )
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