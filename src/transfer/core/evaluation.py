from abc import ABC

import faiss
import itertools
from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

from core import SELECTED_FIELDS
from core.bbox import iou, iou_with_nans
from core.dataset import Dataset
from core.document import Document
from core.function import mean_embs
from core.imgutils import best_component
from core.models import backbone_factory
from core.models.keras import PairModel
from core.transfer import transfer, get_closest_field

DEFAULT_STEPS = (0, 1, 2, 3, 4, 8, 16)


class EvaluationBase:
	def __init__(self, clusters: List[List[str]]):
		self.clusters = clusters

		self.all_docids = list(itertools.chain(*clusters))
		self.n_docs = len(self.all_docids)
		self.dataset = Dataset(self.all_docids)

		self.transfer_methods = {
			"copypaste": self.evaluate_copypaste_at_n,
		}

		self.similarity_embeddings = None
		self.document_embeddings = None

	def get_k_closest(self, cluster: List[str], unknown_doc_index: int):
		raise NotImplementedError

	def get_document_embeddings(self) -> None:
		raise NotImplementedError

	def doc_pair_at_n(self, cluster: List[str], n: int = 0) -> Tuple[str, str]:
		"""Get the best match given the document. Only up to len(cluster)-1 makes sense."""
		s = min(n, len(cluster)-1)

		unknown_doc = cluster[s]
		unknown_doc_index = self.dataset.doc2index[unknown_doc]

		other_docs = [d for d in cluster if d != unknown_doc]
		other_docs_indexes = [self.dataset.doc2index[d] for d in other_docs]

		# Restricts self and all other docs from cluster except for first n
		# If n=0, self and all other documents from the same cluster are banned
		blacklisted_ids = np.array([unknown_doc_index, *other_docs_indexes[s:]])

		# Query the faiss index in order to find similar documents
		doc_indexes = self.get_k_closest(cluster, unknown_doc_index)
		best_filtered_result = np.argmin(np.isin(doc_indexes, blacklisted_ids))

		best_doc_index = doc_indexes[best_filtered_result]
		best_doc_docid = self.dataset.index2doc[best_doc_index]

		return unknown_doc, best_doc_docid

	def evaluate_copypaste_at_n(self, n: int = 0, multilabel: bool = False) -> List[Dict]:
		records = list()
		for cluster in tqdm(self.clusters):
			# Transfer onto doc unknown from source
			docid_target, docid_source = self.doc_pair_at_n(cluster, n)
			if not docid_target and not docid_source:
				continue

			doc_target = self.dataset[docid_target]
			doc_source = self.dataset[docid_source]

			source_masks, source_fields = doc_source.segnet_fieldmasks()
			target_masks, target_fields = doc_target.segnet_fieldmasks()

			for (source_field, target_field) in zip(source_fields, target_fields):
				if source_field is None or source_field["fieldtype"] == "unk":
					continue
				records.append({
					"source_docid": doc_source.docid,
					"target_docid": doc_target.docid,
					"fieldtype": source_field.get("fieldtype"),
					"query_bbox": source_field.get("bbox_small"),
					"gold_bbox": target_field.get("bbox_small") if target_field else None,
					"pred_bbox": source_field.get("bbox_small"),
					"eval_at": n
				})
		return records

	@staticmethod
	def closest_field(
		pred_fieldtype: str, pred_bbox: np.ndarray, source_fields: List[dict]
	) -> Optional[dict]:
		"""Get the closest candidate field from the fields on the target document."""
		candidate_golds = [f for f in source_fields if f and f["fieldtype"] == pred_fieldtype]
		if not candidate_golds:
			return None
		candidate_distances = [
			np.linalg.norm(np.array(f["bbox"]) - pred_bbox) for f in candidate_golds
		]
		return candidate_golds[np.argmin(candidate_distances)]

	def get_evaluation_df(
		self,
		transfer_type: str = "copypaste",
		measured_steps: Tuple[int, ...] = DEFAULT_STEPS,
		**kwargs,
	) -> pd.DataFrame:
		"""Get dataframe with evaluation results applied at different database depths."""
		records = []
		for n in measured_steps:
			records.extend(self.transfer_methods[transfer_type](n, **kwargs))

		df = pd.DataFrame.from_dict(records)
		df["iou"] = df.apply(lambda x: iou_with_nans(x.get("gold_bbox"), x.get("pred_bbox")), axis=1)
		df["hit"] = (df.iou > 0.35) | (df.pred_bbox.isna() & df.gold_bbox.isna())
		return df


class OracleCopyPasteEvaluation(EvaluationBase):
	def __init__(self, clusters: List[List[str]], embedding_kind: str = "dejavu"):
		super().__init__(clusters)

		self.embedding_kind = embedding_kind
		self.similarity_embeddings = np.array(
			[
				doc.embedding(embedding_kind)
				for doc in tqdm(self.dataset, desc="Loading similarity embeddings")
			]
		)

	def doc_pair_at_n(self, cluster: List[str], n: int = 0) -> Tuple[str, str]:
		"""Select the closest document from the same cluster."""
		s = min(n, len(cluster)-1)

		if n == 0:
			return None, None

		unknown_doc = cluster[s]
		unknown_doc_index = self.dataset.doc2index[unknown_doc]

		other_docs = [d for d in cluster if d != unknown_doc]
		other_docs_indexes = [self.dataset.doc2index[d] for d in other_docs]

		allowed_docs = other_docs[:s]
		allowed_docs_indexes = other_docs_indexes[:s]

		source_emb = self.similarity_embeddings[unknown_doc_index]
		cluster_embs = self.similarity_embeddings[allowed_docs_indexes]

		dists = np.linalg.norm(cluster_embs - source_emb, axis=1)
		best_doc_docid = allowed_docs[np.argmin(dists)]

		return unknown_doc, best_doc_docid


class PageCopypasteEvaluation(EvaluationBase):
	def __init__(self, clusters: List[List[str]], embedding_kind: str = "dejavu"):
		super().__init__(clusters)

		self.embedding_kind = embedding_kind
		self.similarity_embeddings = np.array(
			[
				doc.embedding(embedding_kind)
				for doc in tqdm(self.dataset, desc="Loading similarity embeddings")
			]
		)
		embedding_dim = self.similarity_embeddings.shape[-1]
		self.faiss_index = faiss.IndexFlatL2(embedding_dim)
		self.faiss_index.add(self.similarity_embeddings)

	def get_k_closest(self, cluster: List[str], unknown_doc_index: int):
		"""Get k closest documents based on the similarity with query document."""
		query_embedding = np.expand_dims(self.similarity_embeddings[unknown_doc_index], axis=0)
		distances, doc_indexes = self.faiss_index.search(query_embedding, k=len(cluster) + 1)
		return np.squeeze(doc_indexes)


class EmbeddingPageEvaluation(EvaluationBase, ABC):
	def __init__(
		self,
		clusters: List[List[str]],
		emb_w: int,
		emb_h: int,
		emb_dim: int,
		model_kind: str,
	):
		super().__init__(clusters)

		self.emb_w = emb_w
		self.emb_h = emb_h
		self.emb_dim = emb_dim
		self.model_kind = model_kind

		self.field_embeddings = np.zeros((self.n_docs, emb_dim, 12), dtype="float32")
		self.document_embeddings = np.zeros((self.n_docs, emb_w, emb_h, emb_dim), dtype="float32")
		self.pairwise_sim = np.zeros((self.n_docs, self.n_docs), dtype="float32")

		self.masks = np.zeros((self.n_docs, emb_w, emb_h, 12), dtype="float32")

		# NOTE: We don't really need this, but it helps with debugging
		self.doc_embedding_indexes = [None] * self.n_docs

		self.transfer_methods["prediction"] = self.evaluate_prediction_at_n

	def build(self):
		raise NotImplementedError

	def evaluate_prediction_at_n(self, n: int = 0, threshold: float = 0.94) -> List[Dict]:
		"""Return a list of objects with n objects in database."""
		records = list()
		for cluster in tqdm(self.clusters):
			docid_target, docid_source = self.doc_pair_at_n(cluster, n)

			doc_target = self.dataset[docid_target]
			doc_target_index = self.dataset.doc2index[docid_target]
			doc_source = self.dataset[docid_source]
			doc_source_index = self.dataset.doc2index[docid_source]

			source_emb = self.document_embeddings[doc_source_index]
			target_emb = self.document_embeddings[doc_target_index]

			if self.model_kind == "segnet":
				metric = "euclidean"
				source_masks, source_fields = doc_source.segnet_fieldmasks()
				target_masks, target_fields = doc_target.segnet_fieldmasks()
			elif self.model_kind == "model":
				metric = "cosine"
				source_masks, source_fields = doc_source.get_fieldmasks(num_fields=12)
				target_masks, target_fields = doc_target.get_fieldmasks(num_fields=12)
			else:
				raise ValueError(f"Unknown embedding model kind {self.model_kind}")

			pred_fields = transfer(
				source_emb=source_emb,
				target_emb=target_emb,
				source_fields=source_fields,
				target_fields=target_fields,
				source_mask=source_masks,
				source_docid=docid_source,
				target_docid=docid_target,
				threshold=threshold,
				metric=metric,
			)
			for field in pred_fields:
				field["eval_at"] = n
			records.extend(pred_fields)
		return records

	def get_k_closest(self, cluster: List[str], unknown_doc_index: int):
		partially_sorted = np.argpartition(self.pairwise_sim[unknown_doc_index], len(cluster))
		return partially_sorted[:len(cluster)]

	def compute_pairwise_sim(self, n_embs: int = 10, use_unk: bool = True):
		for i in tqdm(range(len(self.all_docids)), desc="Computing pairwise similarity"):
			for j in range(len(self.all_docids)):
				if i == j:
					self.pairwise_sim[i, j] = np.inf
					continue
				# Similarity of i -> j where i(query) -> j(document)
				q = self.field_embeddings[i].transpose().copy()
				m = self.masks[i].sum(axis=(0, 1)) > 0
				dist, _ = self.doc_embedding_indexes[j].search(q, n_embs)
				if use_unk is False:
					dist, m = dist[:-1], m[:-1]

				if self.model_kind == "model":
					self.pairwise_sim[i, j] = 1 - dist[m].mean()
				else:
					self.pairwise_sim[i, j] = dist[m].mean()


class SegNetPageEvaluation(EmbeddingPageEvaluation):
	def __init__(self, clusters: List[List[str]]):
		super().__init__(clusters, model_kind="segnet", emb_w=206, emb_h=292, emb_dim=96)

	def build(self) -> None:
		self.document_embeddings = [
			doc.embedding("segnet") for doc
			in tqdm(self.dataset, desc="Loading embeddings")
		]

		for i, doc in enumerate(tqdm(self.dataset, desc="Computing page features")):
			emb = self.document_embeddings[i]
			masks, fields = doc.segnet_fieldmasks()
			self.masks[i] = masks
			# NOTE: This could be computed in batches
			self.field_embeddings[i] = mean_embs(emb[np.newaxis], masks[np.newaxis])

		for i in tqdm(range(len(self.all_docids)), desc="Creating faiss indexes"):
			doc_emb = self.document_embeddings[i].reshape((-1, 96))
			faiss_index = faiss.IndexFlatL2(96)
			faiss_index.add(doc_emb)
			self.doc_embedding_indexes[i] = faiss_index

		self.compute_pairwise_sim()


class ModelPageEvaluation(EmbeddingPageEvaluation):
	def __init__(
		self,
		clusters: List[List[str]],
		backbone_name: str,
		model_path: str,
		emb_dim: int,
		emb_w: int = 78,
		emb_h: int = 110,
		use_bos: bool = False,
	):
		super().__init__(clusters, model_kind="model", emb_w=emb_w, emb_h=emb_h, emb_dim=emb_dim)

		self.use_bos = use_bos
		if backbone_name != "segnet":
			self.backbone_name = backbone_name
			self.model = self.load_model(backbone_name, model_path, emb_dim, use_bos)

	@staticmethod
	def load_model(backbone_name: str, model_path: str, emb_dim: int, use_bos: bool = False) -> Model:
		backbone = backbone_factory(backbone_name, emb_dim, use_bos=use_bos)

		pair_model = PairModel(backbone_model=backbone)
		channels = 51 if use_bos else 3
		pair_model.build(input_shape=(None, 624, 880, channels))
		pair_model.load_weights(model_path)

		return pair_model

	def _get_document_embeddings(self, doc: Document) -> Tuple[tf.Tensor, tf.Tensor]:
		if self.use_bos:
			image = doc.pageimage_with_bos(downscale=2, backbone=self.backbone_name)
		else:
			image = doc.processed_pageimage(downscale=2, backbone=self.backbone_name)
		masks, fields = doc.get_fieldmasks(num_fields=12)

		# Add a batch dim for the model
		image = image[np.newaxis]
		masks = masks[np.newaxis]

		# NOTE: This could be calculated much faster in batches
		embspace = self.model(image)
		return embspace, masks

	def build(self) -> None:
		for i, doc in enumerate(tqdm(self.dataset, desc="Computing page features")):
			embspace, masks = self._get_document_embeddings(doc)
			self.document_embeddings[i] = embspace[0]
			self.field_embeddings[i] = tf.math.l2_normalize(mean_embs(embspace, masks)[0], axis=0)
			self.masks[i] = masks[0]

		for i in tqdm(range(len(self.all_docids)), desc="Creating faiss indexes"):
			doc_emb = self.document_embeddings[i].reshape((-1, self.emb_dim))
			faiss_index = faiss.IndexFlatIP(self.emb_dim)
			faiss_index.add(doc_emb)
			self.doc_embedding_indexes[i] = faiss_index

		self.compute_pairwise_sim(n_embs=1)


class SingleFieldEvaluation(ModelPageEvaluation):
	def __init__(
		self,
		clusters: List[List[str]],
		backbone_name: str,
		model_path: str,
		emb_dim: int,
		emb_w: int = 78,
		emb_h: int = 110,
	):
		super().__init__(
			clusters, backbone_name=backbone_name, model_path=model_path,
			emb_dim=emb_dim, emb_w=emb_w, emb_h=emb_h
		)
		if backbone_name == "segnet":
			self.index_type = faiss.IndexFlatL2
		else:
			self.index_type = faiss.IndexFlatIP

	def build(self) -> None:
		for i, doc in enumerate(tqdm(self.dataset, desc="Computing page features")):
			embspace, masks = self._get_document_embeddings(doc)
			self.document_embeddings[i] = embspace[0]
			field_feats = mean_embs(embspace, masks)[0]
			self.field_embeddings[i] = tf.math.l2_normalize(field_feats, axis=0)
			self.masks[i] = masks[0]

	def evaluate_prediction_at_n(self, n: int = 0, threshold: float = 0.95, knn: int = 1) -> List[Dict]:
		records = list()
		for cluster in tqdm(self.clusters):
			segmaps = self.pred_segmentations(cluster, n, knn)

			m = min(n, len(cluster) - 1)

			docid_target = cluster[m]
			doc_target = self.dataset[docid_target]
			masks, fields = doc_target.get_fieldmasks()

			for i, fieldtype in enumerate(SELECTED_FIELDS):
				pred_segmap = segmaps[:, :, i]
				pred_bbox = best_component(pred_segmap, threshold=threshold)

				closest_gold = get_closest_field(fieldtype, fields, pred_bbox)
				gold_bbox = closest_gold["bbox_small"] if closest_gold else None

				# Predict None if prediction seems too big
				mask_area = np.prod(masks.shape[:-1])
				pred_area = (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1])
				if pred_area > 0.35 * mask_area:
					pred_bbox = None

				records.append(
					{
						"target_docid": doc_target.docid,
						"fieldtype": fieldtype,
						"gold_bbox": gold_bbox,
						"pred_bbox": pred_bbox,
						"eval_at": n
					}
				)
		return records

	def pred_segmentations(self, cluster: List[List[str]], n: int, knn: int) -> np.ndarray:
		m = min(n, len(cluster) - 1)
		docid_target = cluster[m]
		index_target = self.dataset.doc2index[docid_target]
		docids_banned = cluster[max(m - 1, 0):]
		indexs_banned = [self.dataset.doc2index[b] for b in docids_banned]

		sel = np.ones((len(self.dataset),), dtype=bool)
		sel[indexs_banned] = False
		selected_embs = self.field_embeddings[sel].transpose((0, 2, 1)).reshape((-1, self.emb_dim))

		doc_emb = self.document_embeddings[index_target]
		doc_emb_flat = doc_emb.reshape((-1, self.emb_dim))

		index = self.index_type(self.emb_dim)
		index.add(selected_embs)

		dist, embidxs = index.search(doc_emb_flat, k=knn)
		cls_num = embidxs % 12

		cat = to_categorical(cls_num, num_classes=12)
		res = cat.T * dist.T

		# Fix the shape when n is 0
		if len(cat.shape) == 2:
			cat = cat[:, np.newaxis, :]
			res = res[:, np.newaxis, :]

		cts = cat.sum(axis=1)
		cts = np.where(cts == 0, 1, cts)

		weighted = (res.sum(axis=1) / cts.T).T
		return weighted.reshape((*doc_emb.shape[:2], 12))


class SegnetSingleFieldEvaluation(SingleFieldEvaluation):
	def __init__(self, clusters: List[List[str]], emb_dim: int):
		super().__init__(clusters, backbone_name="segnet", model_path=None, emb_dim=emb_dim, emb_w=206, emb_h=292)

	def build(self) -> None:
		self.document_embeddings = [
			doc.embedding("segnet") for doc
			in tqdm(self.dataset, desc="Loading segnet embeddings")
		]

		for i, doc in enumerate(tqdm(self.dataset, desc="Computing page features")):
			emb = self.document_embeddings[i]
			masks, fields = doc.segnet_fieldmasks()
			self.masks[i] = masks
			# NOTE: This could be computed in batches
			self.field_embeddings[i] = mean_embs(emb[np.newaxis], masks[np.newaxis])
