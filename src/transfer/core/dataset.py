from typing import Iterable, Optional, Union

from joblib import Parallel, delayed
from tqdm import tqdm
from core.document import Document
from core.settings import PROPRIETARY_DATASET_PATH


class Dataset:
	def __init__(
		self,
		docids: Iterable[str],
		dataset_path=PROPRIETARY_DATASET_PATH,
		preload_bos: bool = False,
		n_jobs: int = 1,
	):
		self.docids = list(docids)
		self.dataset_path = dataset_path
		self.docs = {}

		if preload_bos and n_jobs > 1:
			loaded_docs = Parallel(n_jobs=n_jobs)(
				delayed(Document)(docid, dataset_path, preload_bos=preload_bos)
				for docid in tqdm(self.docids, desc="Dataset")
			)
			self.docs = {docid: document for docid, document in zip(self.docids, loaded_docs)}
		else:
			for docid in tqdm(self.docids, desc="Dataset"):
				self.docs[docid] = Document(docid, dataset_path, preload_bos=preload_bos)

		self.doc2index = {d: i for (i, d) in enumerate(docids)}
		self.index2doc = {i: d for (i, d) in enumerate(docids)}

	def __getitem__(self, id_or_pos: Union[str, int]) -> Optional[Document]:
		if isinstance(id_or_pos, str):
			return self.docs.get(id_or_pos)
		elif isinstance(id_or_pos, int):
			return self[self.docids[id_or_pos]]

	def __iter__(self):
		for docid in self.docids:
			yield self.docs.get(docid)

	def __len__(self) -> int:
		return len(self.docids)
