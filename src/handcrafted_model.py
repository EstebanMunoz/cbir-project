import cv2
import numpy as np
from scipy.signal import convolve
from sklearn.cluster import KMeans
from skimage.util.shape import view_as_blocks


def resize_img(img: np.ndarray, size: int) -> np.ndarray:
    new_shape = (size, size)
    resized = cv2.resize(img, new_shape, interpolation=cv2.INTER_LINEAR)
    return resized


def get_block_statistics(img_blocks: np.ndarray) -> tuple:
    dims = img_blocks.ndim
    idxs = tuple([idx for idx in range(dims//2, dims)])
    statistics = (
        img_blocks.min(axis=idxs),
        img_blocks.max(axis=idxs),
        img_blocks.mean(axis=idxs)
    )

    return statistics


def get_color_codebook(colors: np.ndarray, num_colors: int, seed: int = None) -> tuple:
    
    reshaped_colors = colors.reshape(-1, 3)/255

    kmeans = KMeans(n_clusters=num_colors, random_state=seed).fit(reshaped_colors)

    codebook = kmeans.cluster_centers_
    codebook = (255*codebook).astype(np.uint8)

    return codebook


def get_color_indexing(block_colors: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    h, w, _ = block_colors.shape
    color_indexing = np.zeros((h,w))

    for i in range(h):
        for j in range(w):
            color_indexing[i,j] = np.argmin(np.linalg.norm(block_colors[i,j] - codebook, axis=1))
    
    return color_indexing.astype(np.uint)


def get_ccf(img: np.ndarray, block_shape: tuple, codebook_size: int, resize: int = None, seed: int = 1) -> np.ndarray:
    if resize:
        img = resize_img(img, resize)
    img_blocks = view_as_blocks(img, block_shape)

    x_min, x_max, _ = get_block_statistics(img_blocks)

    min_codebook = get_color_codebook(x_min, codebook_size, seed)
    max_codebook = get_color_codebook(x_max, codebook_size, seed)

    min_color_indexing = get_color_indexing(x_min, min_codebook)
    max_color_indexing = get_color_indexing(x_max, max_codebook)

    ccf_matrix = np.zeros((codebook_size,codebook_size))

    h, w = min_color_indexing.shape
    for i in range(h):
        for j in range(w):
            min_index = min_color_indexing[i,j]
            max_index = max_color_indexing[i,j]
            ccf_matrix[min_index, max_index] += 1

    ccf_matrix /= h*w
    return ccf_matrix.sum(axis=1)


# Reshape image blocks to the original image.
def reorder_blocks(img_blocks: np.ndarray) -> np.ndarray:
    dims = img_blocks.ndim//2

    reshaped = img_blocks
    for dim in range(dims):
        blocks_list = np.split(reshaped, img_blocks.shape[dim], axis=dim)
        reshaped = np.concatenate(blocks_list, axis=dim+dims)
    
    return reshaped.squeeze()


def get_bitmap(img: np.ndarray, block_shape: tuple, kernel: np.ndarray) -> np.ndarray:
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_img_blocks = view_as_blocks(gray_img, block_shape)
    gray_x_min, gray_x_max, gray_x_mean = get_block_statistics(gray_img_blocks)
    
    estimation = np.zeros(gray_img_blocks.shape)
    m, n, _, _ = gray_img_blocks.shape
    for i in range(m):
        for j in range(n):
            estimation[i,j] = np.where(
                gray_img_blocks[i,j] >= gray_x_mean[i,j],
                gray_x_max[i,j],
                gray_x_min[i,j]
            )
    
    estimation = reorder_blocks(estimation)
    error = gray_img - estimation
    update = gray_img + convolve(error, kernel, mode='same')
    update_blocks = view_as_blocks(update, block_shape)

    bitmap = np.zeros(update_blocks.shape)
    m, n, _, _ = update_blocks.shape
    for i in range(m):
        for j in range(n):
            bitmap[i,j] = np.where(
                gray_img_blocks[i,j] >= gray_x_mean[i,j],
                1,
                0
            )

    return bitmap
    

def get_pattern_codebook(bitmap: np.ndarray, num_patterns: int, seed: int = None) -> np.ndarray:
    
    bitmap_shape = bitmap.shape
    num_pixels = bitmap_shape[-1]*bitmap_shape[-2]
    reshaped_bitmap = bitmap.reshape(-1, num_pixels)

    kmeans = KMeans(n_clusters=num_patterns, random_state=seed).fit(reshaped_bitmap)

    codebook = kmeans.cluster_centers_
    codebook = (codebook >= 0.5).astype(np.uint).reshape(-1, 8, 8)
    return codebook


def get_bpf(img: np.ndarray, block_shape: tuple, codebook_size: int, kernel: np.ndarray, resize: int = None, seed: int = 1):
    if resize:
        img = resize_img(img, resize)
    bitmap = get_bitmap(img, block_shape, kernel)
    pattern_codebook = get_pattern_codebook(bitmap, codebook_size, seed)

    m, n, _, _ = bitmap.shape
    min_distances = np.zeros(m*n)
    for i in range(m):
        for j in range(n):
            min_distances[n*i+j] = np.argmin(np.abs(pattern_codebook - bitmap[i,j]).sum(axis=(1,2)))

    return np.bincount(min_distances.astype(np.uint))/(m*n)


def extract_features(
    img: np.ndarray,
    color_block_shape: tuple,
    gray_block_shape: tuple,
    codebook_size: int,
    kernel: np.ndarray,
    resize: int,
    seed: int = 1):

    ccf = get_ccf(img, color_block_shape, codebook_size, resize, seed)
    bpf = get_bpf(img, gray_block_shape, codebook_size, kernel, resize, seed)

    return np.concatenate((ccf, bpf))
