import json
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from typing import Optional, Tuple, List

from tqdm import tqdm

from annotator.clusters import Clusters
from core.dataset import Dataset


class NNGenerator:
    def __init__(self, df: pd.DataFrame, already_known: Optional[set] = None):
        # Expected df.emb (emb), df.id (docid), df.candidates (list of docids)
        self.df = df

        X = np.array([*self.df.emb.values])
        self.neigh = NearestNeighbors().fit(X)

        already_known = already_known or set()

        # Prioritize showing documents that have known candidates
        self.candidate_queue = self.df[self.df.candidates.map(len) > 0].id.to_list()
        self.candidate_queue = list(set(self.candidate_queue) - already_known)

    @classmethod
    def from_csv(cls, dataset_name: str, already_known: Optional[set] = None) -> "NNGenerator":
        df = pd.read_csv(f"/data/{dataset_name}.csv")

        # Load serialized json strings into lists
        df.candidates = df.candidates.apply(json.loads)
        df.candidates = df.candidates.apply(set)

        dataset = Dataset(df.id.unique(), dataset_path=f"/data/{dataset_name}")
        id2emb = {
            d.docid: d.embedding("dejavu")
            for d in tqdm(dataset, desc="Loading embeddings")
        }
        df["emb"] = df.id.map(id2emb)
        return cls(df, already_known)

    def random(self, already_known: Optional[set] = None) -> str:
        if self.candidate_queue:
            print(f"Candidate queue contains {len(self.candidate_queue)}")
            if already_known:
                self.candidate_queue = list(set(self.candidate_queue) - already_known)
            return self.candidate_queue.pop()
        return self.df.sample(n=1).id.iloc[0]

    def get_emb_knn(self, docid: str, knn=10) -> Tuple[List[float], List[str]]:
        """Get k nearest neighbors of the docid."""
        row = self.df[self.df.id == docid].iloc[0]
        distances, indexes = self.neigh.kneighbors([row.emb], knn)
        neighbors = self.df.id.iloc[indexes[0]].values
        return distances, neighbors

    def knn(self, clusters: Clusters, docid: str) -> str:
        known_relationships = (
            clusters.known_true(docid) | clusters.known_false(docid) | clusters.known_reject(docid)
        )

        row = self.df[self.df.id == docid].iloc[0]
        if row.candidates:
            unexplored = row.candidates - known_relationships
            if unexplored:
                print(f"{len(unexplored)} unexplored candidates remain.")
                return unexplored.pop()

        distances, indexes = self.neigh.kneighbors([row.emb], 60)
        neighbors = self.df.id[indexes[0][1:]].values
        candidates = [_ for _ in neighbors if _ not in known_relationships]

        print(f"Found {len(candidates)} candidates.")
        return candidates[0] if len(candidates) > 0 else self.random()
