from typing import List, Tuple

import numpy as np


def connected_component(r: np.ndarray, c: np.ndarray) -> List[List[int]]:
    """Find connected components in the given row and column indices.

    Args:
    ----
        r (np.ndarray): Row indices.
        c (np.ndarray): Column indices.

    Yields:
    ------
        List[int]: Indices of connected components.

    """
    indices = [0]
    for i in range(1, r.size):
        if r[i] == r[indices[-1]] and c[i] == c[indices[-1]] + 1:
            indices.append(i)
        else:
            yield indices
            indices = [i]
    yield indices


def nms_horizontal(ratio: np.ndarray, threshold: float) -> np.ndarray:
    """Apply Non-Maximum Suppression (NMS) horizontally on the given ratio matrix.

    Args:
    ----
        ratio (np.ndarray): Input ratio matrix.
        threshold (float): Threshold for NMS.

    Returns:
    -------
        np.ndarray: Binary mask after applying NMS.

    """
    mask = np.zeros_like(ratio, dtype=bool)
    r, c = np.nonzero(ratio > threshold)
    if len(r) == 0:
        return mask
    for ids in connected_component(r, c):
        values = [ratio[r[i], c[i]] for i in ids]
        mi = np.argmax(values)
        mask[r[ids[mi]], c[ids[mi]]] = True
    return mask


def nms_vertical(ratio: np.ndarray, threshold: float) -> np.ndarray:
    """Apply Non-Maximum Suppression (NMS) vertically on the given ratio matrix.

    Args:
    ----
        ratio (np.ndarray): Input ratio matrix.
        threshold (float): Threshold for NMS.

    Returns:
    -------
        np.ndarray: Binary mask after applying NMS.

    """
    return np.transpose(nms_horizontal(np.transpose(ratio), threshold))


def fgbg_depth(
    d: np.ndarray, t: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find foreground-background relations between neighboring pixels.

    Args:
    ----
        d (np.ndarray): Depth matrix.
        t (float): Threshold for comparison.

    Returns:
    -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Four matrices indicating
        left, top, right, and bottom foreground-background relations.

    """
    right_is_big_enough = (d[..., :, 1:] / d[..., :, :-1]) > t
    left_is_big_enough = (d[..., :, :-1] / d[..., :, 1:]) > t
    bottom_is_big_enough = (d[..., 1:, :] / d[..., :-1, :]) > t
    top_is_big_enough = (d[..., :-1, :] / d[..., 1:, :]) > t
    return (
        left_is_big_enough,
        top_is_big_enough,
        right_is_big_enough,
        bottom_is_big_enough,
    )


def fgbg_depth_thinned(
    d: np.ndarray, t: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find foreground-background relations between neighboring pixels with Non-Maximum Suppression.

    Args:
    ----
        d (np.ndarray): Depth matrix.
        t (float): Threshold for NMS.

    Returns:
    -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Four matrices indicating
        left, top, right, and bottom foreground-background relations with NMS applied.

    """
    right_is_big_enough = nms_horizontal(d[..., :, 1:] / d[..., :, :-1], t)
    left_is_big_enough = nms_horizontal(d[..., :, :-1] / d[..., :, 1:], t)
    bottom_is_big_enough = nms_vertical(d[..., 1:, :] / d[..., :-1, :], t)
    top_is_big_enough = nms_vertical(d[..., :-1, :] / d[..., 1:, :], t)
    return (
        left_is_big_enough,
        top_is_big_enough,
        right_is_big_enough,
        bottom_is_big_enough,
    )


def fgbg_binary_mask(
    d: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find foreground-background relations between neighboring pixels in binary masks.

    Args:
    ----
        d (np.ndarray): Binary depth matrix.

    Returns:
    -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Four matrices indicating
        left, top, right, and bottom foreground-background relations in binary masks.

    """
    assert d.dtype == bool
    right_is_big_enough = d[..., :, 1:] & ~d[..., :, :-1]
    left_is_big_enough = d[..., :, :-1] & ~d[..., :, 1:]
    bottom_is_big_enough = d[..., 1:, :] & ~d[..., :-1, :]
    top_is_big_enough = d[..., :-1, :] & ~d[..., 1:, :]
    return (
        left_is_big_enough,
        top_is_big_enough,
        right_is_big_enough,
        bottom_is_big_enough,
    )


def edge_recall_matting(pr: np.ndarray, gt: np.ndarray, t: float) -> float:
    """Calculate edge recall for image matting.

    Args:
    ----
        pr (np.ndarray): Predicted depth matrix.
        gt (np.ndarray): Ground truth binary mask.
        t (float): Threshold for NMS.

    Returns:
    -------
        float: Edge recall value.

    """
    assert gt.dtype == bool
    ap, bp, cp, dp = fgbg_depth_thinned(pr, t)
    ag, bg, cg, dg = fgbg_binary_mask(gt)
    return 0.25 * (
        np.count_nonzero(ap & ag) / max(np.count_nonzero(ag), 1)
        + np.count_nonzero(bp & bg) / max(np.count_nonzero(bg), 1)
        + np.count_nonzero(cp & cg) / max(np.count_nonzero(cg), 1)
        + np.count_nonzero(dp & dg) / max(np.count_nonzero(dg), 1)
    )


def boundary_f1(
    pr: np.ndarray,
    gt: np.ndarray,
    t: float,
    return_p: bool = False,
    return_r: bool = False,
) -> float:
    """Calculate Boundary F1 score.

    Args:
    ----
        pr (np.ndarray): Predicted depth matrix.
        gt (np.ndarray): Ground truth depth matrix.
        t (float): Threshold for comparison.
        return_p (bool, optional): If True, return precision. Defaults to False.
        return_r (bool, optional): If True, return recall. Defaults to False.

    Returns:
    -------
        float: Boundary F1 score, or precision, or recall depending on the flags.

    """
    ap, bp, cp, dp = fgbg_depth(pr, t)
    ag, bg, cg, dg = fgbg_depth(gt, t)

    r = 0.25 * (
        np.count_nonzero(ap & ag) / max(np.count_nonzero(ag), 1)
        + np.count_nonzero(bp & bg) / max(np.count_nonzero(bg), 1)
        + np.count_nonzero(cp & cg) / max(np.count_nonzero(cg), 1)
        + np.count_nonzero(dp & dg) / max(np.count_nonzero(dg), 1)
    )
    p = 0.25 * (
        np.count_nonzero(ap & ag) / max(np.count_nonzero(ap), 1)
        + np.count_nonzero(bp & bg) / max(np.count_nonzero(bp), 1)
        + np.count_nonzero(cp & cg) / max(np.count_nonzero(cp), 1)
        + np.count_nonzero(dp & dg) / max(np.count_nonzero(dp), 1)
    )
    if r + p == 0:
        return 0.0
    if return_p:
        return p
    if return_r:
        return r
    return 2 * (r * p) / (r + p)


def get_thresholds_and_weights(
    t_min: float, t_max: float, N: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate thresholds and weights for the given range.

    Args:
    ----
        t_min (float): Minimum threshold.
        t_max (float): Maximum threshold.
        N (int): Number of thresholds.

    Returns:
    -------
        Tuple[np.ndarray, np.ndarray]: Array of thresholds and corresponding weights.

    """
    thresholds = np.linspace(t_min, t_max, N)
    weights = thresholds / thresholds.sum()
    return thresholds, weights


def invert_depth(depth: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Inverts a depth map with numerical stability.

    Args:
    ----
        depth (np.ndarray): Depth map to be inverted.
        eps (float): Minimum value to avoid division by zero (default is 1e-6).

    Returns:
    -------
    np.ndarray: Inverted depth map.

    """
    inverse_depth = 1.0 / depth.clip(min=eps)
    return inverse_depth


def SI_boundary_F1(
    predicted_depth: np.ndarray,
    target_depth: np.ndarray,
    t_min: float = 1.05,
    t_max: float = 1.25,
    N: int = 10,
) -> float:
    """Calculate Scale-Invariant Boundary F1 Score for depth-based ground-truth.

    Args:
    ----
        predicted_depth (np.ndarray): Predicted depth matrix.
        target_depth (np.ndarray): Ground truth depth matrix.
        t_min (float, optional): Minimum threshold. Defaults to 1.05.
        t_max (float, optional): Maximum threshold. Defaults to 1.25.
        N (int, optional): Number of thresholds. Defaults to 10.

    Returns:
    -------
        float: Scale-Invariant Boundary F1 Score.

    """
    assert predicted_depth.ndim == target_depth.ndim == 2
    thresholds, weights = get_thresholds_and_weights(t_min, t_max, N)
    f1_scores = np.array(
        [
            boundary_f1(invert_depth(predicted_depth), invert_depth(target_depth), t)
            for t in thresholds
        ]
    )
    return np.sum(f1_scores * weights)


def SI_boundary_Recall(
    predicted_depth: np.ndarray,
    target_mask: np.ndarray,
    t_min: float = 1.05,
    t_max: float = 1.25,
    N: int = 10,
    alpha_threshold: float = 0.1,
) -> float:
    """Calculate Scale-Invariant Boundary Recall Score for mask-based ground-truth.

    Args:
    ----
        predicted_depth (np.ndarray): Predicted depth matrix.
        target_mask (np.ndarray): Ground truth binary mask.
        t_min (float, optional): Minimum threshold. Defaults to 1.05.
        t_max (float, optional): Maximum threshold. Defaults to 1.25.
        N (int, optional): Number of thresholds. Defaults to 10.
        alpha_threshold (float, optional): Threshold for alpha masking. Defaults to 0.1.

    Returns:
    -------
        float: Scale-Invariant Boundary Recall Score.

    """
    assert predicted_depth.ndim == target_mask.ndim == 2
    thresholds, weights = get_thresholds_and_weights(t_min, t_max, N)
    thresholded_target = target_mask > alpha_threshold

    recall_scores = np.array(
        [
            edge_recall_matting(
                invert_depth(predicted_depth), thresholded_target, t=float(t)
            )
            for t in thresholds
        ]
    )
    weighted_recall = np.sum(recall_scores * weights)
    return weighted_recall
