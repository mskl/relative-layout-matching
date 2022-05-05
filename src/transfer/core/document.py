import json
import os.path
from scipy import sparse
import numpy as np
import tensorflow as tf
from PIL import ImageDraw, Image
from functools import lru_cache
from itertools import groupby
from typing import List, Dict, Set, Optional, Tuple

from core.imgutils import hash_to_rgb
from core.settings import embedding_path, annotation_path, pageimage_path, segnet_path, pic2bos_path
from core import SELECTED_FIELDS

BOS_SHAPE = (624, 880, 48)


class Document:
	def __init__(self, docid: str, dataset_path: str, metadata: Optional[dict] = None, preload_bos: bool = False):
		self.docid = docid
		self.dataset_path = dataset_path
		self.metadata = metadata or dict()

		doc_annotations_path = annotation_path(self.docid, self.dataset_path)
		with open(doc_annotations_path) as handle:
			self.raw = json.load(handle)

		self.sparse_bos = None
		if preload_bos:
			self.get_bos()

	@property
	def pageimage(self) -> Image:
		path = pageimage_path(self.docid, self.dataset_path)
		return Image.open(path)

	def draw_fields(self, fieldtypes: Optional[Set[str]] = None) -> Image:
		img = self.pageimage.copy()
		draw = ImageDraw.Draw(img)

		for field in self.fields:
			if fieldtypes and field["fieldtype"] not in fieldtypes:
				continue
			color = hash_to_rgb(field["fieldtype"])
			draw.rectangle(field["bbox"], fill=None, outline=color, width=2)

		return img

	def get_bos(
		self, render_width: int = 1248, render_height: int = 1760, downsample: int = 2
	) -> np.ndarray:
		# LRU Cache is known to have issues with multiprocessing
		if self.sparse_bos is not None:
			return self.sparse_bos.toarray().reshape(BOS_SHAPE)

		path = pic2bos_path(self.docid, self.dataset_path)
		if not os.path.exists(path):
			raise ValueError(f"Pic2Bos not available for document {self.docid}")

		path = pic2bos_path(self.docid, self.dataset_path)
		obj = np.load(path, mmap_mode="r")
		bos = obj["p2b"].astype("bool")

		# FIXME: This could be unified with pageimage resize
		bos = bos[:render_width, :render_height]
		x_pad = np.max(render_width - bos.shape[0], 0)
		y_pad = np.max(render_height - bos.shape[1], 0)

		bos = np.pad(bos, [(0, x_pad), (0, y_pad), (0, 0)], "constant")
		shape = (bos.shape[0] // downsample, bos.shape[1] // downsample, bos.shape[-1])
		target = (shape[0], bos.shape[0] // shape[0], shape[1], bos.shape[1] // shape[1], shape[2])

		# Use a max downsample
		res = bos.reshape(target).max(-2).max(1)
		sps = res.reshape((BOS_SHAPE[0] * BOS_SHAPE[1], BOS_SHAPE[2]))
		self.sparse_bos = sparse.coo_matrix(sps)
		return res

	@lru_cache(1)
	def _naive_emb(self, emb_size: int = 3999) -> np.ndarray:
		# Resize to roughly match desired emb size
		# s = np.array(self.pageimage.size)
		# Divide by 3 due to 3 channels
		# a = np.sqrt((emb_size / 3) / np.prod(s))
		# r = np.round(a * s).astype(int)
		# img = self.pageimage.resize(size=r)
		img = self.pageimage.resize(size=(62 // 2, 87 // 2))
		arr = np.asarray(img).flatten() / 255.0
		return arr.astype("float32")

	def embedding(self, kind: str = "dejavu", **kwargs) -> np.ndarray:
		if kind == "dejavu":
			path = embedding_path(self.docid, self.dataset_path)
			return np.fromfile(path, dtype="float32")
		elif kind == "naive":
			return self._naive_emb(**kwargs)
		elif kind == "segnet":
			path = segnet_path(self.docid, self.dataset_path)
			return np.load(path)
		else:
			raise KeyError(f"Unknown embedding kind '{kind}'!")

	def raw_to_fields(self, field_extractions=None) -> List[Dict]:
		return [
			{
				"docid": self.docid,
				"fieldtype": f["fieldtype"],
				"bbox": [int(v) for v in f["bbox"]],
				"parsed_val": f["parsed_val"],
				"text": f["text"]
			} for f in field_extractions
		]

	@property
	def fields(self) -> List[Dict]:
		fields = self.raw_to_fields(self.raw["field_extractions"])
		# Sort by top, left, bottom, right
		return sorted(fields, key=lambda x: tuple(np.array(x["bbox"]).take([1, 0, 2, 3])))

	def filtered_fields(self, fieldtypes: List[str] = SELECTED_FIELDS, multilabel: bool = False) -> List[Dict]:
		selected = [f for f in self.fields if f["fieldtype"] in fieldtypes]
		if multilabel:
			return selected
		fieldtypes_to_find = set(SELECTED_FIELDS)
		filtered_fields = []
		for field in selected:
			if field["fieldtype"] in fieldtypes_to_find:
				filtered_fields.append(field)
				fieldtypes_to_find -= {field["fieldtype"]}
		return filtered_fields

	@property
	def unique_fields(self) -> List[Dict]:
		sorted_extractions = sorted(self.raw["field_extractions"], key=lambda x: x["bbox"])
		return self.raw_to_fields(
			[list(g)[0] for k, g in groupby(sorted_extractions, key=lambda x: x["fieldtype"])]
		)

	@property
	def attributes(self) -> Dict[str, List[str]]:
		return {a["name"]: a["choices"] for a in self.raw["attributes"]}

	def get_field(self, fieldtype: str) -> dict:
		return [f for f in self.fields if f["fieldtype"] == fieldtype][0]

	def get_fieldmasks(
		self,
		render_width: int = 1248,
		render_height: int = 1760,
		downsample: int = 16,
		num_fields: int = 12,
	) -> Tuple[np.ndarray, List[Optional[dict]]]:
		"""Get class masks and return them as array or dictionary."""
		assert num_fields - 1 == len(SELECTED_FIELDS)
		all_fields = self.fields.copy()

		c2f = {f: None for f in SELECTED_FIELDS}
		for field in all_fields:
			fieldtype = field["fieldtype"]
			if fieldtype in SELECTED_FIELDS and not c2f.get(fieldtype):
				c2f[field["fieldtype"]] = field
		field_tuples = list(c2f.items())

		m_width, m_height = render_width//downsample, render_height//downsample
		fieldmasks = np.zeros((num_fields, m_width, m_height), dtype="int32")

		for i, (fieldtype, field) in enumerate(field_tuples):
			if field is None:
				continue
			bbox = field["bbox"]
			bbox_small = [
				bbox[0] // downsample,
				bbox[1] // downsample,
				bbox[2] // downsample,
				bbox[3] // downsample,
			]
			field["bbox_small"] = bbox_small
			fieldmasks[i][
				bbox_small[0]: bbox_small[2],
				bbox_small[1]: bbox_small[3]
			] = 1

		# Overwrite the last slot with unknown class
		field_tuples.append(("unk", {"fieldtype": "unk"}))
		fields = [v for k, v in field_tuples]

		fieldmasks[-1] = 1 - np.max(fieldmasks, axis=0)
		fieldmasks = np.einsum("cwh -> whc", fieldmasks)

		return fieldmasks, fields

	def pageimage_with_bos(self, backbone: Optional[str], downscale: int = 2) -> np.ndarray:
		img = self.processed_pageimage(downscale=downscale, backbone=backbone)
		bos = self.get_bos(downsample=downscale)
		return np.concatenate((img, bos), axis=2)

	@lru_cache(1)
	def processed_pageimage(
		self,
		downscale: int = 1,
		render_width: int = 1248,
		render_height: int = 1760,
		backbone: Optional[str] = None,
	) -> np.ndarray:
		"""Get pageimage preprocessed as numpy array padded to the required size."""
		img_array = np.asarray(self.pageimage)

		if backbone and "resnet" in backbone:
			img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
		elif backbone and "vgg" in backbone:
			img_array = tf.keras.applications.vgg16.preprocess_input(img_array)
		else:
			img_array = img_array / 255.0

		img_array = np.swapaxes(img_array, 0, 1)
		img_array = img_array[:render_width, :render_height]

		x_pad = np.max(render_width - img_array.shape[0], 0)
		y_pad = np.max(render_height - img_array.shape[1], 0)
		img_array = np.pad(img_array, [(0, x_pad), (0, y_pad), (0, 0)], "constant")

		shape = (img_array.shape[0] // downscale, img_array.shape[1] // downscale, 3)
		sh = shape[0], img_array.shape[0] // shape[0], shape[1], img_array.shape[1] // shape[1], 3
		img_array = img_array.reshape(sh).mean(3).mean(1)

		return img_array

	def segnet_pageimage(self) -> np.ndarray:
		return self.processed_pageimage(downscale=6, render_width=1236, render_height=1752)

	def segnet_fieldmasks(self) -> Tuple[np.ndarray, List[Optional[dict]]]:
		return self.get_fieldmasks(downsample=6, render_width=1236, render_height=1752)
