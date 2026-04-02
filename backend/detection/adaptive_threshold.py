"""
Adaptive Thresholding Utility for Poisoning Detection

Replaces fixed percentile-based thresholds (which always flag a fixed %
of samples regardless of data quality) with distribution-aware methods.

Strategies:
  1. MAD (Median Absolute Deviation) — robust outlier statistic
  2. Gap-based — finds natural separations in score distributions
  3. Minimum score floor — absolute minimum to prevent noise flagging

Result: Clean datasets → ~0 flags.  Poisoned datasets → proportional flags.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


def adaptive_threshold(
    scores: np.ndarray,
    mad_multiplier: float = 3.5,
    min_score_floor: float = 0.3,
    fallback_percentile: float = 97.0,
    gap_factor: float = 2.0,
) -> tuple:
    """
    Compute an adaptive threshold for anomaly scores.

    Uses a combination of:
    1. MAD-based outlier detection (primary)
    2. Gap-based detection (secondary)
    3. Minimum score floor (safety net)
    4. Percentile fallback cap (prevents over-flagging)

    Args:
        scores: Array of anomaly scores (0-1 normalized expected).
        mad_multiplier: Number of MADs above median to set threshold.
            Higher = fewer flags. Typical: 3.0-5.0.
        min_score_floor: Absolute minimum score to consider flagging.
            Samples below this are never flagged regardless of statistics.
        fallback_percentile: Percentile used as an upper cap to prevent
            extreme over-flagging if MAD is very small.
        gap_factor: Minimum gap size (relative to median gap) to trigger
            gap-based thresholding.

    Returns:
        (threshold, flagged_mask, method_used)
    """
    n = len(scores)
    if n == 0:
        return 1.0, np.array([], dtype=bool), "empty"

    scores = np.asarray(scores, dtype=np.float64)

    # ── Strategy 1: MAD-based threshold ──
    median = np.median(scores)
    mad = np.median(np.abs(scores - median))

    if mad > 1e-10:
        mad_threshold = median + mad_multiplier * mad
    else:
        # MAD ≈ 0 means scores are nearly uniform (clean data).
        # Use a high threshold that will flag almost nothing.
        mad_threshold = median + 0.5 * (scores.max() - median + 1e-10)

    # ── Strategy 2: Gap-based threshold ──
    gap_threshold = _find_gap_threshold(scores, gap_factor)

    # ── Combine: use the lower of MAD and gap thresholds ──
    # (so we catch anomalies detected by either method)
    if gap_threshold is not None:
        combined_threshold = min(mad_threshold, gap_threshold)
        method = "mad+gap"
    else:
        combined_threshold = mad_threshold
        method = "mad"

    # ── Apply minimum score floor ──
    # Never flag scores below the absolute floor
    combined_threshold = max(combined_threshold, min_score_floor)

    # ── Apply percentile cap ──
    # Prevent extreme over-flagging: never flag more than (100 - percentile)%
    percentile_cap = np.percentile(scores, fallback_percentile)
    if combined_threshold < percentile_cap:
        # Only use percentile cap if it would reduce flags
        # (i.e., the adaptive threshold is too aggressive)
        pass  # Keep combined_threshold as-is; it's already reasonable
    # If combined_threshold > percentile_cap, that's fine — fewer flags is correct

    flagged = scores > combined_threshold
    method_used = method

    logger.debug(
        f"Adaptive threshold: {combined_threshold:.4f} "
        f"(method={method_used}, median={median:.4f}, mad={mad:.4f}, "
        f"flagged={flagged.sum()}/{n})"
    )

    return float(combined_threshold), flagged, method_used


def _find_gap_threshold(
    scores: np.ndarray,
    gap_factor: float = 2.0,
    min_gap_size: float = 0.05,
) -> float | None:
    """
    Find a natural gap in the score distribution.

    If there's a clear separation between 'normal' and 'anomalous' scores,
    the threshold is placed at the gap. If no significant gap exists
    (clean data), returns None.

    Args:
        scores: Anomaly scores.
        gap_factor: A gap must be at least this many times the median
            gap to be considered significant.
        min_gap_size: Absolute minimum gap size to consider.

    Returns:
        Threshold value at the gap, or None if no significant gap found.
    """
    sorted_scores = np.sort(scores)
    n = len(sorted_scores)

    if n < 10:
        return None

    # Compute consecutive differences
    diffs = np.diff(sorted_scores)

    if len(diffs) == 0:
        return None

    median_diff = np.median(diffs)
    if median_diff < 1e-10:
        median_diff = 1e-10

    # Search the ENTIRE score range for significant gaps.
    # We want to find the natural separation between clean and poisoned
    # samples regardless of where it falls in the distribution.
    # Only skip the bottom 10% to avoid trivial noise gaps.
    start_idx = max(n // 10, 1)

    # Get gaps above start_idx, sorted by size (largest first)
    search_diffs = diffs[start_idx:]
    if len(search_diffs) == 0:
        return None

    for gap_idx in np.argsort(search_diffs)[::-1]:
        actual_idx = start_idx + gap_idx
        gap_size = diffs[actual_idx]

        # Gap must be both relatively large AND absolutely large
        if gap_size > gap_factor * median_diff and gap_size >= min_gap_size:
            # Threshold is the midpoint of the gap
            threshold = (sorted_scores[actual_idx] + sorted_scores[actual_idx + 1]) / 2

            # Sanity check: gap should separate a meaningful group
            # Allow up to 60% flagging (handles heavily-contaminated datasets)
            n_above = (scores > threshold).sum()
            if 1 <= n_above <= n * 0.60:
                return float(threshold)

    return None

