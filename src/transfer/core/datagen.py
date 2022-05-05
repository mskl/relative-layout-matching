from tensorflow.keras.utils import Sequence
import numpy as np
from typing import List, Tuple, Optional
import random

from core.dataset import Dataset


class PairDataset(Sequence):
	def __init__(
		self,
		dataset: Dataset,
		clusters: List[List[str]],
		batch_size: int = 32,
		shuffle: bool = True,
		include_unk: bool = True,
		num_fields: int = 12,
		backbone: Optional[str] = None,
		include_bos: bool = False,
		n_repeats: int = 1,
	):
		self.dataset = dataset
		self.clusters = clusters
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.include_unk = include_unk
		self.num_fields = num_fields
		self.backbone = backbone
		self.include_bos = include_bos
		self.n_repeats = n_repeats

		# Allocate on the level of object, reallocation takes time
		self.color_channels = 3 if not self.include_bos else 51

		self.indexes = None
		self.on_epoch_end()

	def get_cluster_pair(self, cluster: set) -> Tuple[str, str]:
		"""Randomly select one pair from set."""
		if self.shuffle:
			a = random.sample(cluster, 1)[0]
			b = random.sample(set(cluster) - {a}, 1)[0]
			return a, b
		# WARNING: If shuffle is false, only first 2 documents are returned!
		sorted_cluster = sorted(cluster)
		return sorted_cluster[0], sorted_cluster[1]

	def on_epoch_end(self):
		"""Shuffle the classes on end of each epoch."""
		self.indexes = np.arange(len(self.clusters))
		if self.shuffle:
			np.random.shuffle(self.indexes)

	def batch_len(self) -> int:
		return int(np.floor(len(self.clusters) / self.batch_size))

	def __len__(self):
		"""Number of batches per epoch."""
		return self.batch_len() * self.n_repeats

	def get_batch_docids(self, index: int) -> List[Tuple[str, str]]:
		"""Return list of pairs of document id's."""
		indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
		base_clusters = [self.clusters[k] for k in indexes]
		return [self.get_cluster_pair(c) for c in base_clusters]

	def get_batch(self, index: int, include_labels: bool = False, only_labels: bool = False):
		"""Generate one batch of data. Return a Tuple."""
		docid_pairs = self.get_batch_docids(index % self.batch_len())

		f = [None] * (self.batch_size * 2)
		X = np.zeros((self.batch_size * 2, 624, 880, self.color_channels), dtype="float32")
		y = np.zeros((self.batch_size * 2, 78, 110, self.num_fields), dtype="int32")

		for i, (ida, idb) in enumerate(docid_pairs):
			docA1 = self.dataset[ida]
			docA2 = self.dataset[idb]
			if not only_labels:
				if self.include_bos:
					feats1 = docA1.pageimage_with_bos(downscale=2, backbone=self.backbone)
					feats2 = docA1.pageimage_with_bos(downscale=2, backbone=self.backbone)
				else:
					feats1 = docA1.processed_pageimage(downscale=2, backbone=self.backbone)
					feats2 = docA2.processed_pageimage(downscale=2, backbone=self.backbone)
				X[2 * i] = feats1[np.newaxis]
				X[2 * i + 1] = feats2[np.newaxis]
			y[2 * i], f[2 * i] = docA1.get_fieldmasks(num_fields=self.num_fields)
			y[2 * i + 1], f[2 * i + 1] = docA2.get_fieldmasks(num_fields=self.num_fields)

		if only_labels:
			# Used for evaluation when x is not needed
			return (y.astype("int32"), f, docid_pairs)

		if include_labels:
			# Used for debugging
			return (X.astype("float32"), y.astype("int32"), f)

		return (X.astype("float32"), y.astype("int32"))

	def __getitem__(self, index) -> Tuple[np.ndarray, np.ndarray]:
		return self.get_batch(index)
