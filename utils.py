import numpy as np
import einops
import os 
import PIL
from PIL import Image, ImageDraw 

#loading images
def load(dirname, start_slice, slices): 
    flow_data = []

    fname = (os.listdir(dirname))
    for i in range(start_slice, start_slice + slices):
        im = Image.open(os.path.join(dirname, fname[i]))
        imarray = np.array(im)
        flow_data.append(imarray)

    # convert to a 3D array and normalise so data is between 0 and 1 
    flow_data  = np.asarray(flow_data) 
    flow_data  = preprocess(flow_data)
    return(flow_data)


def preprocess(img: np.ndarray, normalize_axis=None) -> np.ndarray:
    """Converts image to 8-bit image. Optionally normalizes along `normalize_axis`."""

    assert isinstance(img, np.ndarray)
    assert img.dtype in (np.uint8, np.uint16, np.float32, np.float64), f"Unsupported dtype: {img.dtype}"
    assert normalize_axis is None or 0 <= normalize_axis % img.ndim < img.ndim

    if img.dtype in (np.float32, np.float64):
        img = (img - img.min()) / (img.max() - img.min()) * 255
        #mn = np.min(img, axis=normalize_axis, keepdims=True)
        #mx = np.max(img, axis=normalize_axis, keepdims=True)
        #np.subtract(img, mn, out=img)  # img = img - img.min()
        #np.divide(img, mx - mn, out=img)
        #np.multiply(img, 255, out=img)
        img = img.astype(np.uint8)
    
    if img.dtype == np.uint16:
        img = (img - img.min()) / (img.max() - img.min()) * 255
        #mn = np.min(img, axis=normalize_axis, keepdims=True)
        #mx = np.max(img, axis=normalize_axis, keepdims=True)
        #np.subtract(img, mn, out=img)  # img = img - img.min()
        #np.divide(img, mx - mn, out=img)
        #np.multiply(img, 255, out=img)
        img = img.astype(np.uint8)
    
    assert isinstance(img, np.ndarray)
    assert img.dtype == np.uint8
    
    return img


# function to mask with the dry scan 
def mask_with_dry(img, dry_scan):
    if img.max() > 1:
        img = (img - img.min()) / (img.max() - img.min()) * 255
        img = img.astype(np.uint8)

    # creating mask from segmented dry scan
    mask = (dry_scan == 0)

    assert img.shape == mask.shape
    assert mask.dtype == np.bool8
    # mask image 
    foreground = img.copy()
    foreground[mask] = 255 

    # create a composite image using the alpha layer
    masked_img = np.array(foreground, dtype=np.uint8)
    return masked_img 


def sanity_check(img: np.ndarray, mask: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    assert isinstance(mask, np.ndarray)
    assert mask.dtype == np.bool8
    assert isinstance(img, np.ndarray)
    assert img.dtype == np.uint8
    assert len(mask.shape) == 2
    assert len(img.shape) == 2  # grayscale

    background = einops.repeat(img, "h w -> h w c", c=3)  # grayscale to rgb

    foreground = background.copy()
    foreground[mask] = [255, 0, 0]

    foreground = foreground.astype(np.float16)
    background = background.astype(np.float16)

    composite = background * (1.0 - alpha) + foreground * alpha
    composite = np.array(composite, dtype=np.uint8)

    assert isinstance(composite, np.ndarray)
    assert composite.dtype == np.uint8
    assert len(composite.shape) == 3  # rgb

    return composite


def simple_thresholding(img: np.array, min_threshold: float, max_threshold: float) -> np.array:
    return ((img.max() - img.min()) * min_threshold + img.min() <= img) & (img <= (img.max() - img.min()) * max_threshold + img.min())
