import tifffile
import rasterio
import numpy as np

def scale_percentile(
    matrix,
    min_percentile=2,
    max_percentile=98,
    min_min=100,
    max_min=1000,
    min_max=1000,
    max_max=3500,
    bounded=False,
    nodata=0,
):
    """Returns a scaled (H,W,D) image which is more easily visually inspectable. Image is linearly scaled between
    min and max_percentile of pixel values by channel"""
    transpose_back = False
    if matrix.shape[0] < 20:
        matrix = matrix.transpose((1, 2, 0))
        transpose_back = True

    if matrix.max() < 1:
        matrix = matrix * 2**16
    w, h, d = matrix.shape
    matrix = np.reshape(matrix, [w * h, d]).astype(np.float64)
    _matrix = matrix.copy()  # make a copy where we fill nodata with nan for percentile computation
    _matrix[_matrix == nodata] = np.nan
    mins = np.nanpercentile(_matrix, min_percentile, axis=0)
    maxs = np.nanpercentile(_matrix, max_percentile, axis=0)
    if bounded:
        mins = mins.clip(min_min, max_min)
        maxs = maxs.clip(min_max, max_max)
    maxs = maxs - mins
    matrix = (matrix - mins[None, :]) / maxs[None, :]
    matrix = np.reshape(matrix, [w, h, d])
    matrix = matrix.clip(0, 1)
    if transpose_back:
        matrix = matrix.transpose((2, 0, 1))
    return matrix


fcc_to_bands = {
    "SWIR": ["B12", "B8", "B4"],
    "SWIRP": ["B8", "B11", "B4"],
    "RGB": ["B4", "B3", "B2"],
    "CIR": ["B8", "B4", "B3"],
}
fcc_to_band_indices_16d = {
    "SWIR": [10, 7, 3],
}
fcc_to_band_indices_l2a = {
    "SWIR": [11, 8, 4],
}

min_max_l2a = {
    # "SWIR": {"min": [100, 100, 100], "max": [3500] * 3},
    "SWIR": {"min": [100, 100, 100], "max": [4500] * 3},
    "SWIRP": {"min": [100, 100, 100], "max": [3500] * 3},
    "RGB": {"min": [100, 100, 100], "max": [2500] * 3},
    "CIR": {"min": [100, 100, 100], "max": [3500] * 3},
}
min_max_16d = {
    "SWIR": {"min": [100, 100, 100], "max": [4500] * 3},
}

fcc_to_ls_band_indices = {
    "SWIR": [7, 3, 0], # swir22, nir08, red
}
min_max_ls = {
    "SWIR": {"min": [0, 0, 0], "max": [0.5] * 3},
}

def scale_img(matrix, min_values, max_values):
    w, h, d = matrix.shape
    flat = np.reshape(matrix, (w * h, d)).astype(np.float32)
    flat = (flat - min_values[None, :]) / (max_values[None, :] - min_values[None, :])
    out = np.reshape(flat, (w, h, d))
    return out.clip(0, 1)

# def visualize_s2_path(path_b2, profile = "SWIR", downsample = 0):
#     img = tifffile.imread([path_b2.replace("B2", band) for band in fcc_to_bands[profile]])
#     if img.shape[0] < 15:
#         img = np.transpose(img, (1, 2, 0))
#     if downsample > 0:
#         img = img[::downsample, ::downsample]
#     return scale_img(img, np.array(min_max[profile]["min"]), np.array(min_max[profile]["max"]))

def visualize_s2_concat_bands_path(concat_bands_path, profile = "SWIR"):
    # img = tifffile.imread(concat_bands_path)
    img = rasterio.open(concat_bands_path).read()
    if img.shape[0] < 25:
        img = np.transpose(img, (1, 2, 0))
    if "L2A" in concat_bands_path:
        fcc_to_band_indices = fcc_to_band_indices_l2a
        min_max = min_max_l2a
        img = img.astype(np.float32)
        img = img - 1000
        img[img < 0] = 0
    else:
        fcc_to_band_indices = fcc_to_band_indices_16d
        min_max = min_max_16d
        # import pdb; pdb.set_trace()
        _img = img[:, :, fcc_to_band_indices[profile]].copy()
        _img = _img.astype(np.float32)
        _img[_img == 0] = np.nan
        mins = np.nanpercentile(_img, 2, axis=(0,1))
        maxs = np.nanpercentile(_img, 98, axis=(0,1))
        print(f"16D: {mins=}, {maxs=}")
        return scale_img(img[:, :, fcc_to_band_indices[profile]], np.array(min_max[profile]["min"]), np.array(min_max[profile]["max"]))
        # return scale_img(img[:, :, fcc_to_band_indices[profile]], mins, maxs)
    return scale_img(img[:, :, fcc_to_band_indices[profile]], np.array(min_max[profile]["min"]), np.array(min_max[profile]["max"]))

def visualize_ls_concat_bands_path(concat_bands_path, profile = "SWIR", downsample = 0):
    # img = tifffile.imread(concat_bands_path).astype(np.float32)
    img = rasterio.open(concat_bands_path).read().astype(np.float32)
    if img.shape[0] < 25:
        img = np.transpose(img, (1, 2, 0))
    img = img * 0.0000275 - 0.2
    if downsample > 0:
        img = img[::downsample, ::downsample]
    return scale_img(img[:, :, fcc_to_ls_band_indices[profile]], np.array(min_max_ls[profile]["min"]), np.array(min_max_ls[profile]["max"]))

def visualize_cb_mux_path(concat_bands_path):
    # img = tifffile.imread(concat_bands_path)
    img = rasterio.open(concat_bands_path).read()
    if img.shape[0] < 25:
        img = np.transpose(img, (1, 2, 0))
    return scale_img(img[:, :, [3, 2, 1]], np.array([0, 0, 0]), np.array([100, 100, 100]))

def visualize_cb_wpm_path(concat_bands_path):
    # img = tifffile.imread(concat_bands_path)
    img = rasterio.open(concat_bands_path).read()
    if img.shape[0] < 25:
        img = np.transpose(img, (1, 2, 0))
    return scale_img(img, np.array([0, 0, 0]), np.array([50000, 50000, 50000]))

def visualize_s1_path(path):
    # VV and VH channel are in one path
    # s1_img = tifffile.imread(path).astype(np.float32)
    s1_img = rasterio.open(path).read().astype(np.float32)
    if s1_img.shape[0] < 25:
        s1_img = np.transpose(s1_img, (1, 2, 0))
    if s1_img.shape[-1] == 3:
        s1_img = s1_img[:, :, :2]

    img = np.zeros((s1_img.shape[0], s1_img.shape[1], 3), dtype=np.float32)
    img[:, :, :2] = s1_img
    img[:, :, 2] = s1_img[:, :, 0] / s1_img[:, :, 1]
    _img = img.copy()  # make a copy where we fill nodata with nan for percentile computation
    _img[_img == 0] = np.nan
    mins = np.nanpercentile(_img, 2, axis=(0,1))
    maxs = np.nanpercentile(_img, 98, axis=(0,1))
    # print(f"S1: {mins=}, {maxs=}")
    return scale_img(img, mins, maxs)
    # return scale_img(img, np.array([120, 60, 1.1]), np.array([400, 200, 3.5]))

