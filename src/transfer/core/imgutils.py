import io
from typing import Tuple, Union, Optional
import numpy as np
import skimage.measure
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import skimage.measure

from core.function import mean_embs, query_embedding_distance


def hash_to_rgb(param: str) -> Tuple[int, int, int]:
	h = hash(param)
	r = (h & 0xFF0000) >> 16
	g = (h & 0x00FF00) >> 8
	b = h & 0x0000FF
	return (r, g, b)


def plot_to_image(figure):
	"""
	Converts the matplotlib plot specified by 'figure' to a PNG image and
	returns it. The supplied figure is closed and inaccessible after this call.

	src: https://www.tensorflow.org/tensorboard/image_summaries
	"""
	# Save the plot to a PNG in memory.
	buf = io.BytesIO()
	plt.savefig(buf, format='png')
	# Closing the figure prevents it from being displayed directly inside the notebook.
	plt.close(figure)
	buf.seek(0)
	# Convert PNG buffer to TF image
	image = tf.image.decode_png(buf.getvalue(), channels=4)
	# Add the batch dimension
	image = tf.expand_dims(image, 0)
	return image


def largest_component(mask, threshold=0.8, return_bbox=True):
	"""Return bbox of the largest connected component. Bbox is in order of (l, t, r, b)."""
	img_bw = np.asarray(mask > threshold)
	labels = skimage.measure.label(img_bw, return_num=False)
	component = (labels == np.argmax(np.bincount(labels.flat, weights=img_bw.flat)))
	if return_bbox:
		x, y = np.where(component)
		return [x.min(), y.min(), x.max(), y.max()]
	return component


def best_component(
	mask: np.ndarray, threshold: float = 0.8, return_bbox: bool = True
) -> Union[list, np.ndarray]:
	"""Return the component seeded at the minimal value of mask."""
	mask = np.asarray(mask)
	best_seed = np.unravel_index(mask.argmax(), mask.shape)

	img_bw = np.asarray(mask > threshold)
	labels = skimage.measure.label(img_bw, return_num=False)
	best_label = labels[best_seed]

	component = np.asarray(labels == best_label)
	if return_bbox:
		x, y = np.where(component)
		return [x.min(), y.min(), x.max(), y.max()]
	return component


def show_query_image(
	model: tf.keras.models.Model,
	d1: "Document",
	d2: "Document",
	include_bos: bool = False,
	metric: str = "cosine",
	backbone: Optional[str] = None
):
	if include_bos:
		i1 = d1.pageimage_with_bos(downscale=2, backbone=backbone)
		i2 = d2.pageimage_with_bos(downscale=2, backbone=backbone)
	else:
		i1 = d1.processed_pageimage(downscale=2, backbone=backbone)
		i2 = d2.processed_pageimage(downscale=2, backbone=backbone)

	emb1 = model(i1[np.newaxis])
	emb2 = model(i2[np.newaxis])

	cls1, fields_source = d1.get_fieldmasks()
	cls2, fields_target = d2.get_fieldmasks()

	# Add the batch dimension
	cls1 = cls1[np.newaxis]
	cls2 = cls2[np.newaxis]

	memb1 = tf.math.l2_normalize(mean_embs(emb1, cls1), axis=1)

	sim = query_embedding_distance(memb1, emb2[0], metric=metric)
	fig, ax = plt.subplots(nrows=len(fields_source), ncols=3, figsize=(8, len(fields_source) * 2))
	for i, source_field in enumerate(fields_source):
		ax[i, 0].imshow(cls1[0, :, :, i])
		ax[i, 0].axis('off')
		ax[i, 0].set_title("query")

		ax[i, 1].imshow(sim[:, :, i])
		ax[i, 1].axis('off')
		ax[i, 1].set_title(source_field["fieldtype"] if source_field else "none")

		ax[i, 2].imshow(cls2[0, :, :, i])
		ax[i, 2].axis('off')
		ax[i, 2].set_title("target")
	plt.tight_layout()
	return fig


def show_pairwise_dists(model: tf.keras.Model, d1: "Document", d2: "Document", include_bos: bool = False):
	if include_bos:
		i1 = d1.pageimage_with_bos(downscale=2)
		i2 = d2.pageimage_with_bos(downscale=2)
	else:
		i1 = d1.processed_pageimage(downscale=2)
		i2 = d2.processed_pageimage(downscale=2)

	source_emb = model.backbone(i1[np.newaxis])
	target_emb = model.backbone(i2[np.newaxis])

	source_masks = d1.get_fieldmasks()
	target_masks = d2.get_fieldmasks()

	mean_emb1 = mean_embs(source_emb, source_masks[np.newaxis])
	mean_emb2 = mean_embs(target_emb, target_masks[np.newaxis])

	mean_emb1 = tf.math.l2_normalize(mean_emb1, axis=1)
	mean_emb2 = tf.math.l2_normalize(mean_emb2, axis=1)

	cos_sim = tf.matmul(mean_emb1, mean_emb2, transpose_a=True)
	euc_sim = tf.linalg.norm(mean_emb1 - tf.transpose(mean_emb2), axis=1)

	fig, ax = plt.subplots(ncols=2, figsize=(8, 4))
	sns.heatmap(cos_sim[0], ax=ax[0], vmin=0, vmax=1)
	ax[0].set_title("cosine sim")
	sns.heatmap(euc_sim, ax=ax[1], vmin=0, vmax=1)
	ax[1].set_title("l2 distance")
	plt.tight_layout()

	return fig
