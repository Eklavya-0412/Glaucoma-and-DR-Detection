"""Module K - edge detection utilities (Sobel, Prewitt, Canny, Laplacian).

All functions take a uint8 grayscale image and return a uint8 edge-strength
map (Canny returns a binary map). Parameters are returned alongside so that
pipelines can record them.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _norm(mag: np.ndarray) -> np.ndarray:
    m = mag.max()
    return np.zeros(mag.shape, np.uint8) if m <= 0 else (mag / m * 255).astype(np.uint8)


def sobel(gray: np.ndarray, ksize: int = 3) -> tuple[np.ndarray, dict[str, Any]]:
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    return _norm(np.hypot(gx, gy)), {"kernel_size": ksize, "output": "gradient magnitude sqrt(Gx^2+Gy^2)"}


def prewitt(gray: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], np.float64)
    gx = cv2.filter2D(gray.astype(np.float64), -1, kx)
    gy = cv2.filter2D(gray.astype(np.float64), -1, kx.T)
    return _norm(np.hypot(gx, gy)), {"kernel": "3x3 Prewitt", "output": "gradient magnitude"}


def canny(gray: np.ndarray, low: int | None = None, high: int | None = None,
          blur_sigma: float = 1.4) -> tuple[np.ndarray, dict[str, Any]]:
    g = cv2.GaussianBlur(gray, (0, 0), blur_sigma) if blur_sigma > 0 else gray
    rule = "user supplied"
    if low is None or high is None:
        # Automatic hysteresis thresholds from the gradient distribution:
        # high = 90th percentile of the (L1) Sobel magnitude, low = 0.4 * high.
        mag = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1))
        nz = mag[mag > 0]
        high = int(np.percentile(nz, 90)) if nz.size else 50
        high, low = max(high, 10), int(max(5, 0.4 * high))
        rule = "high = P90 of Sobel |G|, low = 0.4*high"
    return cv2.Canny(g, low, high), {"gaussian_sigma": blur_sigma, "low_threshold": low,
                                     "high_threshold": high, "threshold_rule": rule}


def laplacian(gray: np.ndarray, ksize: int = 3, blur_sigma: float = 1.0) -> tuple[np.ndarray, dict[str, Any]]:
    g = cv2.GaussianBlur(gray, (0, 0), blur_sigma) if blur_sigma > 0 else gray
    lap = cv2.Laplacian(g, cv2.CV_64F, ksize=ksize)
    return _norm(np.abs(lap)), {"kernel_size": ksize, "gaussian_sigma": blur_sigma, "output": "|Laplacian of Gaussian|"}


DETECTORS = {"Sobel": sobel, "Prewitt": prewitt, "Canny": canny, "Laplacian": laplacian}
