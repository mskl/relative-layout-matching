import logging
import os
from typing import Tuple, Optional
import numpy as np
import pandas as pd
import tensorflow as tf
import tqdm as tqdm

from core.datagen import PairDataset
from core.function import mean_embs
from core.transfer import transfer
from core.bbox import iou_with_nans
from core.document import Document
from core.imgutils import plot_to_image, show_query_image
from core.utils import get_valid_filename


logger = logging.getLogger(__name__)

MODELS_PATH = "/models/"


def transfer_fields(
	pred: np.ndarray, valid_generator: PairDataset, threshold: float = 0.72, metric: str = "cosine"
) -> Tuple[pd.DataFrame, float]:
	"""Obtain micro and macro mean iou for given fields."""
	assert valid_generator.shuffle is False, "Trying to evaluate with shuffle=True"

	distances = []
	results = []
	for bi in tqdm.trange(len(valid_generator)):
		batch_index_start = bi * valid_generator.batch_size
		masks, fields, docid_pairs = valid_generator.get_batch(bi, only_labels=True)
		for pi, pair in enumerate(docid_pairs):
			pair_index_start = 2 * (batch_index_start + pi)
			source_emb = pred[pair_index_start]
			target_emb = pred[pair_index_start + 1]
			source_docid = pair[0]
			target_docid = pair[1]
			source_mask = masks[2 * pi]
			target_mask = masks[2 * pi + 1]

			# FIXME: Obtain all gold fields, not only the subsampled ones
			source_fields = fields[2 * pi]
			target_fields = fields[2 * pi + 1]

			# Calculate mean inter-cluster distance
			# FIXME: This could be a) batched and b) is duplicate
			source_query = mean_embs(
				tf.expand_dims(source_emb, axis=0),
				tf.expand_dims(source_mask, axis=0),
			)
			source_query = tf.math.l2_normalize(source_query, axis=-1)
			target_query = mean_embs(
				tf.expand_dims(target_emb, axis=0),
				tf.expand_dims(target_mask, axis=0),
			)
			target_query = tf.math.l2_normalize(target_query, axis=-1)
			sm = source_mask.sum(axis=(0, 1)) > 0
			tm = target_mask.sum(axis=(0, 1)) > 0
			mean_dist = -tf.keras.losses.cosine_similarity(
				target_query[0], source_query[0], axis=0
			).numpy()[sm * tm].mean()
			distances.append(mean_dist)

			res = transfer(
				source_emb,
				target_emb,
				source_fields,
				target_fields,
				source_mask,
				source_docid,
				target_docid,
				threshold=threshold,
				metric=metric,
			)
			results.extend(res)

	df = pd.DataFrame.from_dict(results)
	df["transfer_iou"] = df.apply(lambda x: iou_with_nans(x["gold_bbox"], x["pred_bbox"]), axis=1)
	df["copypaste_iou"] = df.apply(lambda x: iou_with_nans(x["gold_bbox"], x["query_bbox"]), axis=1)
	df["hit"] = (df.transfer_iou > 0.35) | (df.pred_bbox.isna() & df.gold_bbox.isna())
	return df, np.mean(distances)


class SparseCallback(tf.keras.callbacks.Callback):
	def __init__(self, eval_per_n_epochs: int = 1):
		"""Use a callable that has 1 parameter being the model."""
		super().__init__()
		self.eval_per_n_epochs = eval_per_n_epochs

	def on_epoch_end(self, epoch, logs=None):
		if epoch == 0 or (epoch + 1) % self.eval_per_n_epochs == 0:
			self._on_epoch_end(epoch, logs)

	def _on_epoch_end(self, epoch, logs=None):
		raise NotImplementedError


class SegmentationImageCallback(SparseCallback):
	def __init__(
		self,
		d1: Document,
		d2: Document,
		summary_writer,
		eval_per_n_epochs: int = 1,
		include_bos: bool = False,
		backbone: Optional[str] = None,
	):
		super().__init__(eval_per_n_epochs)
		self.d1 = d1
		self.d2 = d2
		self.summary_writer = summary_writer
		self.include_bos = include_bos
		self.backbone = backbone

	def _on_epoch_end(self, epoch, logs=None):
		logger.info(f"SegmentationImageCallback: on epoch end {epoch}")
		fig = show_query_image(
			self.model, self.d1, self.d2, include_bos=self.include_bos, backbone=self.backbone
		)
		cm_image = plot_to_image(fig)
		with self.summary_writer.as_default():
			tf.summary.image("Class Segmentations", cm_image, step=epoch)
		logger.info(f"SegmentationImageCallback: Written class segmentations")


class TransferCallback(SparseCallback):
	def __init__(
		self,
		generator: PairDataset,
		summary_writer,
		argstring: str,
		threshold: float = 0.5,
		eval_per_n_epochs: int = 1,
	):
		super().__init__(eval_per_n_epochs)
		self.generator = generator
		self.summary_writer = summary_writer
		self.argstring = argstring
		self.threshold = threshold

		# Path to a model that was previously saved by this generator
		self.previously_saved_model = None

		self.best_score = 0
		self.best_epoch = 0
		self.last_epoch = -1

	def save_model(self, epoch, score):
		"""Save the model and delete the previously saved model."""
		save_path = MODELS_PATH + get_valid_filename(
			f"{score:0.4f}_{epoch}_{self.argstring}.h5"
		)

		if not os.path.exists(MODELS_PATH):
			os.mkdir(MODELS_PATH)

		if self.previously_saved_model is not None:
			try:
				os.remove(self.previously_saved_model)
			except FileNotFoundError:
				print(f"Failed to remove saved model in file {self.previously_saved_model}!")

		self.model.save_weights(save_path)
		self.previously_saved_model = save_path

		print(f"[{epoch}] Saved model as {save_path}")

	@staticmethod
	def pred_transfer(model, generator, threshold) -> Tuple[pd.DataFrame, float]:
		pred = model.predict(generator, workers=3, use_multiprocessing=True, verbose=1, max_queue_size=2)
		pred = tf.math.l2_normalize(pred, axis=-1)
		return transfer_fields(pred, generator, threshold)

	@classmethod
	def evaluate(cls, model, generator, threshold) -> Tuple[float, float, float, float]:
		df, mean_dist = cls.pred_transfer(model, generator, threshold)
		macro_iou = df.groupby(by="fieldtype").mean().transfer_iou.mean()
		micro_iou = df.transfer_iou.mean()
		mean_hit = df.hit.mean()

		return micro_iou, macro_iou, mean_dist, mean_hit

	def _on_epoch_end(self, epoch, logs=None):
		logger.info(f"TransferCallback: On epoch end {epoch}")
		micro, macro, dist, mean_hit = self.evaluate(self.model, self.generator, self.threshold)
		with self.summary_writer.as_default():
			print(f"micro: {micro:.4f} macro: {macro:.4f} dist: {dist:.4f} hit: {mean_hit: .4f}")
			tf.summary.scalar('micro_iou', micro, step=epoch)
			tf.summary.scalar('macro_iou', macro, step=epoch)
			tf.summary.scalar('mean_dist', dist, step=epoch)
			tf.summary.scalar('mean_hit', mean_hit, step=epoch)

		if mean_hit > self.best_score:
			print(f"[{epoch}] Best mean hit score achieved {mean_hit:0.4f}.")
			self.best_score = mean_hit
			self.best_epoch = epoch
			self.save_model(epoch, mean_hit)
