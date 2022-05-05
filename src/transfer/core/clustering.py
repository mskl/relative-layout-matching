from typing import List, Tuple, Callable
import hashlib
import uuid

import numpy as np


class Cluster:
	def __init__(self, ids: List[str], embs: np.ndarray = None):
		self.ids = ids
		self.embs = embs

	def clusterid(self) -> str:
		all_ids = "".join(sorted(self.ids))
		m = hashlib.md5()
		m.update(all_ids.encode('utf-8'))
		return str(uuid.UUID(m.hexdigest()))

	def contains(self, iid: str) -> bool:
		return iid in self.ids

	def get_emb(self, iid: str) -> np.ndarray:
		assert self.contains(iid)
		return [e for (i, e) in zip(self.embs, self.ids) if i == iid][0]

	def sorted_neighbors(self, iid: str, distance: Callable) -> List[Tuple[int, str]]:
		assert self.contains(iid)
		if not distance:
			def distance(x, y):
				return np.linalg.norm(x - y)

		target_emb = self.get_emb(iid)
		distances = [distance(target_emb, y) for y in self.embs]
		return sorted(zip(distances, self.ids))[1:]


class Clustering:
	def __init__(self, clusters: List[Cluster]):
		self.clusters = clusters

		# Reverse map maps id to cluster index
		self.reverse_map = {}
		for cluster_index, cluster in enumerate(self.clusters):
			for cid in cluster.ids:
				self.reverse_map[cid] = cluster_index

	def neighbors(self, cid) -> List[str]:
		return self.clusters[self.reverse_map[cid]].ids
