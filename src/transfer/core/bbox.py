from typing import List, Dict, Tuple, Optional
import numpy as np


def iou(a: List[float], b: List[float]) -> float:
	xA = max(a[0], b[0])
	yA = max(a[1], b[1])
	xB = min(a[2], b[2])
	yB = min(a[3], b[3])

	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	boxAArea = (a[2] - a[0] + 1) * (a[3] - a[1] + 1)
	boxBArea = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)

	return interArea / float(boxAArea + boxBArea - interArea)


def iou_with_nans(a: Optional[List[float]], b: Optional[List[float]]) -> float:
	if a is None and b is None:
		return np.nan
	if a is None or b is None:
		return 0.0
	return iou(a, b)


def iou_1d(lg, rg, lp, rp) -> float:
	left_union, right_union = min(lg, lp), max(rg, rp)
	left_inter, right_inter = max(lg, lp), min(rg, rp)
	union_w = (right_union - left_union)
	inter_w = (right_inter - left_inter)
	return inter_w / union_w


def iou_xy(a: List[float], b: List[float]) -> Tuple[float, float]:
	dx = iou_1d(a[0], a[2], b[0], b[2])
	dy = iou_1d(a[1], a[3], b[1], b[3])
	return dx, dy


def move_dist_xy(a: List[float], b: List[float]) -> Tuple[float, float]:
	dx = abs(a[0] - b[0]) + abs(a[2] + b[2])
	dy = abs(a[1] - b[1]) + abs(a[3] + b[3])
	return dx, dy


def fields_iou(fa: Dict, fb: Dict) -> float:
	return iou(fa["bbox"], fb["bbox"])
