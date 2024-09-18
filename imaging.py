import numpy as np
import cv2
import random


def clahe(data, clip_limit=0.01, tile_grid_size=(8, 8)):
    """CLAHEを適用する

    Args:
        data (ndarray): HSデータ
        clip_limit (float, optional): clip limit. Defaults to 0.01.
        tile_grid_size (tuple, optional): tile grid size. Defaults to (8, 8).

    Returns:
        ndarray: CLAHE適用後のHSデータ
    """
    data = data.astype(np.uint16)
    if len(data.shape) > 2:
        for i in range(data.shape[-1]):
            data[..., i] = cv2.createCLAHE(
                clipLimit=clip_limit, tileGridSize=tile_grid_size
            ).apply(data[..., i])
    else:
        data = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size).apply(
            data
        )
    return data


def equalize_hist(data):
    """ヒストグラム平坦化を適用する

    Args:
        data (ndarray): HSデータ

    Returns:
        ndarray: ヒストグラム平坦化適用後のHSデータ
    """
    data = data / np.max(data) * 255
    data = data.astype(np.uint8)
    if len(data.shape) > 2:
        for i in range(data.shape[-1]):
            data[..., i] = cv2.equalizeHist(data[..., i])
    else:
        data = cv2.equalizeHist(data)
    return data


def gaussian_filter(data, sigma=1):
    """ガウシアンフィルタを適用する

    Args:
        data (ndarray): HSデータ
        sigma (float, optional): sigma. Defaults to 1.

    Returns:
        ndarray: ガウシアンフィルタ適用後のHSデータ
    """
    if len(data.shape) > 2:
        for i in range(data.shape[-1]):
            data[..., i] = cv2.GaussianBlur(data[..., i], (0, 0), sigma)
    else:
        data = cv2.GaussianBlur(data, (0, 0), sigma)
    return data


def median_filter(data, kernel_size=3):
    """メディアンフィルタを適用する

    Args:
        data (ndarray): HSデータ
        kernel_size (int, optional): カーネルサイズ. Defaults to 3.

    Returns:
        ndarray: メディアンフィルタ適用後のHSデータ
    """
    if len(data.shape) > 2:
        for i in range(data.shape[-1]):
            data[..., i] = cv2.medianBlur(data[..., i], kernel_size)
    else:
        data = cv2.medianBlur(data, kernel_size)
    return data


def split_into_patches(data, patch_size=(10, 10), patch_select=None):
    """HSデータをパッチに分割する

    Args:
        data (ndarray): HSデータ
        patch_size (tuple, optional): パッチサイズ. Defaults to (10, 10).
        patch_select (int, optional): 選択するパッチ数. Defaults to None. (Noneの場合は全てのパッチを選択)

    Returns:
        ndarray: パッチに分割されたHSデータ
    """
    h, w, _ = data.shape
    patches = []
    for i in range(0, h, patch_size[0]):
        for j in range(0, w, patch_size[1]):
            patch = data[i : i + patch_size[0], j : j + patch_size[1], :]
            if patch.shape[:2] != patch_size:
                continue
            patches.append(patch)
    if patch_select is not None:
        random.shuffle(patches)
        patches = patches[:patch_select]
    return np.array(patches).astype(data.dtype)


def zscore_normalization(data, axis=(0, 1), eps=1e-6):
    """Z-score normalizationを行う

    Args:
        data (ndarray): HSデータ

    Returns:
        ndarray: Z-score normalizationされたHSデータ
    """
    return (data - np.mean(data, axis=axis)) / (np.std(data, axis=axis) + eps)


def morphological_operations(data, kernel_size=3, operation="dilate"):
    """モルフォロジー演算を行う

    Args:
        data (ndarray): HSデータ
        kernel_size (int, optional): カーネルサイズ. Defaults to 3.
        operation (str, optional): モルフォロジー演算の種類.[dilate, erode] Defaults to "dilate".

    Returns:
        ndarray: モルフォロジー演算後のHSデータ
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    if operation == "dilate":
        return cv2.dilate(data, kernel, iterations=1)
    elif operation == "erode":
        return cv2.erode(data, kernel, iterations=1)
    else:
        raise ValueError(f"Invalid operation: {operation}")
