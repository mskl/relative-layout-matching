from typing import List, Optional, Union
import tensorflow as tf
import numpy as np

from core.function import mean_embs, query_embedding_distance
from core.imgutils import best_component


def get_closest_field(fieldtype: str, target_fields: List[dict], pred_bbox: np.ndarray) -> Optional[dict]:
	"""Get the closest candidate field from the fields on the target document."""
	candidate_golds = [
		f for f in target_fields if f and f["fieldtype"] == fieldtype and f["fieldtype"] != "unk"
	]
	if not candidate_golds:
		return None
	candidate_distances = [
		np.linalg.norm(np.array(f["bbox_small"]) - pred_bbox) for f in candidate_golds
	]
	return candidate_golds[np.argmin(candidate_distances)]


def transfer(
	source_emb: Union[tf.Tensor, np.ndarray],
	target_emb: Union[tf.Tensor, np.ndarray],
	source_fields: List[dict],
	target_fields: List[dict],
	source_mask: Union[tf.Tensor, np.ndarray],
	source_docid: str,
	target_docid: str,
	threshold: float = 0.72,
	metric: str = "cosine",
) -> List[dict]:
	"""Use the query fields against the target embeddings to extract the bboxes."""
	source_query = mean_embs(
		tf.expand_dims(source_emb, axis=0),
		tf.expand_dims(source_mask, axis=0),
	)
	source_query = tf.math.l2_normalize(source_query, axis=-1)
	pred_distances = query_embedding_distance(source_query, target_emb, metric=metric)

	results = []
	for query_index, query_field in enumerate(source_fields):
		fieldtype = query_field["fieldtype"] if query_field else None
		if not query_field or fieldtype == "unk":
			continue

		# Distances between the query field and target embeddings
		pred_segmap = pred_distances[:, :, query_index]
		pred_bbox = best_component(pred_segmap, threshold=threshold)

		# Get a field of the same fieldtype from all golds
		closest_gold = get_closest_field(fieldtype, target_fields, pred_bbox)
		gold_bbox = closest_gold["bbox_small"] if closest_gold else None

		# Predict None if prediction seems too big
		mask_area = np.prod(source_mask.shape[:-1])
		pred_area = (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1])
		if pred_area > 0.35 * mask_area:
			pred_bbox = None

		results.append(
			{
				"source_docid": source_docid,
				"target_docid": target_docid,
				"fieldtype": query_field["fieldtype"],
				"query_bbox": query_field["bbox_small"],
				"gold_bbox": gold_bbox,
				"pred_bbox": pred_bbox,
			}
		)

	return results
