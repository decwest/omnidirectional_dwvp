"""Exact unweighted ray/box geometry from dwpp@28bc0207 (MIT)."""
import numpy as np


def calc_ray_box_intersection_alpha_range(ray_direction: np.ndarray, box_min: np.ndarray, box_max: np.ndarray) -> tuple[bool, float, float]:
    alpha_low = 0.0
    alpha_high = float("inf")
    eps = 1e-12

    for d, lower, upper in zip(ray_direction, box_min, box_max):
        if abs(d) < eps:
            # この軸は alpha に依存しない。0 が範囲外なら交差不可。
            if lower <= 0.0 <= upper:
                continue
            return False, 0.0, 0.0

        a1 = lower / d
        a2 = upper / d
        axis_low = min(a1, a2)
        axis_high = max(a1, a2)

        alpha_low = max(alpha_low, axis_low)
        alpha_high = min(alpha_high, axis_high)

        if alpha_low > alpha_high:
            return False, 0.0, 0.0

    if alpha_high < 0.0:
        return False, 0.0, 0.0

    alpha_low = max(alpha_low, 0.0)
    if alpha_low <= alpha_high:
        return True, alpha_low, alpha_high
    return False, 0.0, 0.0


def calc_closest_alpha_to_box_from_ray(
    ray_direction: np.ndarray,
    box_min: np.ndarray,
    box_max: np.ndarray,
    prefer_large_alpha: bool
) -> float:
    eps = 1e-12
    tol = 1e-10
    best_alpha = 0.0
    best_dist = float("inf")

    def eval_alpha(alpha: float):
        nonlocal best_alpha, best_dist
        point = alpha * ray_direction
        nearest = np.clip(point, box_min, box_max)
        dist = float(np.sum((point - nearest) ** 2))
        if dist + tol < best_dist:
            best_dist = dist
            best_alpha = alpha
            return
        if abs(dist - best_dist) <= tol:
            if prefer_large_alpha and alpha > best_alpha:
                best_alpha = alpha
            if (not prefer_large_alpha) and alpha < best_alpha:
                best_alpha = alpha

    # 区分点（各軸の範囲境界を ray が横切る alpha）
    breakpoints = [0.0]
    for d, lower, upper in zip(ray_direction, box_min, box_max):
        if abs(d) < eps:
            continue
        breakpoints.append(lower / d)
        breakpoints.append(upper / d)

    breakpoints = sorted(set(float(b) for b in breakpoints if np.isfinite(b) and b >= 0.0))
    if len(breakpoints) == 0:
        return 0.0

    for alpha in breakpoints:
        eval_alpha(alpha)

    def calc_interval_quadratic_coeff(a: float, b: float | None) -> tuple[float, float]:
        if b is None:
            probe = a + 1.0
        else:
            probe = 0.5 * (a + b)

        qa = 0.0
        qb = 0.0
        for d, lower, upper in zip(ray_direction, box_min, box_max):
            p = probe * d
            if p < lower - eps:
                qa += d ** 2
                qb += -2.0 * lower * d
            elif p > upper + eps:
                qa += d ** 2
                qb += -2.0 * upper * d
            # inside の場合は寄与ゼロ
        return qa, qb

    for i in range(len(breakpoints) - 1):
        left = breakpoints[i]
        right = breakpoints[i + 1]
        qa, qb = calc_interval_quadratic_coeff(left, right)
        if qa <= eps:
            continue
        alpha_star = -qb / (2.0 * qa)
        if left <= alpha_star <= right:
            eval_alpha(alpha_star)

    # 最終区間 [last, +inf)
    last = breakpoints[-1]
    qa, qb = calc_interval_quadratic_coeff(last, None)
    if qa > eps:
        alpha_star = -qb / (2.0 * qa)
        if alpha_star >= last:
            eval_alpha(alpha_star)

    return best_alpha
