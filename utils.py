import numpy as np
import einops


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
