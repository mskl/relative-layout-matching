import os
import pandas as pd
from collections import defaultdict
from typing import Optional, List, Set


class Clusters:
    def __init__(self, records_path: str = None):
        self.same = defaultdict(set)
        self.interesting = defaultdict(set)
        self.different = defaultdict(set)
        self.reject = defaultdict(set)

        if records_path and not os.path.exists(records_path):
            print(f"Records file does not exists. Creating a new file at {records_path}")
            handle = open(records_path, "a")
            handle.close()

        if records_path:
            self.df = pd.read_csv(records_path, sep=";", names=["timestamp", "d1", "d2", "rel"])
            self.df.apply(lambda x: self.set_relationship(x.d1, x.d2, x.rel), axis=1)

    @staticmethod
    def double_add(cluster: dict, docid1: str, docid2: str) -> None:
        """Add a binary relationship between two documents into a clusterdict."""
        cluster[docid1] |= {docid2}
        cluster[docid2] |= {docid1}

    @staticmethod
    def double_rm(cluster: dict, docid1: str, docid2: str) -> None:
        """Remove binary relationship between two documents into a clusterdict."""
        cluster[docid1] -= {docid2}
        cluster[docid2] -= {docid1}

    def get_relationship(self, d1: str, d2: str) -> Optional[str]:
        """Get relationships between d1 and d2."""
        if d1 in self.same[d2]:
            return "same"
        if d1 in self.interesting[d2]:
            return "interesting"
        if d1 in self.different[d2]:
            return "different"
        if d1 in self.reject[d2]:
            return "reject"
        return None

    def set_relationship(self, d1: str, d2: str, rel: str) -> None:
        """Set a relationship between d1 and d2. Overrides existing rel."""
        if rel == "same":
            self.double_add(self.same, d1, d2)
            self.double_rm(self.different, d1, d2)
            self.double_rm(self.interesting, d1, d2)
            self.double_rm(self.reject, d1, d2)
        elif rel == "interesting":
            self.double_rm(self.same, d1, d2)
            self.double_rm(self.different, d1, d2)
            self.double_add(self.interesting, d1, d2)
            self.double_rm(self.reject, d1, d2)
        elif rel == "different":
            self.double_rm(self.same, d1, d2)
            self.double_add(self.different, d1, d2)
            self.double_rm(self.interesting, d1, d2)
            self.double_rm(self.reject, d1, d2)
        elif rel == "reject":
            self.double_rm(self.same, d1, d2)
            self.double_rm(self.different, d1, d2)
            self.double_rm(self.interesting, d1, d2)
            self.double_add(self.reject, d1, d2)
        else:
            raise ValueError("Unknown relationship", rel)

    def known_true(self, docid: str) -> Set[str]:
        """Known related documents to the docid."""
        known_true = self.same[docid] | self.interesting[docid] | {docid}
        known_true |= self.find_connected_component(
            sets=[self.same, self.interesting],
            unknown={docid}
        )
        return known_true

    def known_reject(self, docid: str) -> Set[str]:
        """Rejected documents related to the docid."""
        return self.reject[docid]

    def known_false(self, docid: str) -> Set[str]:
        """Known unrelated documents to the docid."""
        known_false = set()
        for different_docid in self.different[docid]:
            if different_docid in known_false:
                continue
            known_false |= self.find_connected_component(
                sets=[self.different],
                unknown={different_docid}
            )
        return known_false

    def n_clusters(self) -> int:
        """Number of distinct clusters discovered."""
        seen = self.same.keys() | self.interesting.keys()
        found_clusters = {frozenset(self.known_true(docid)) for docid in seen}
        nontrivial = [_ for _ in found_clusters if len(_) > 1]
        return len(nontrivial)

    @classmethod
    def find_connected_component(
        cls, sets: List[dict], known: Optional[set] = None, unknown: Optional[set] = None
    ) -> Set[str]:
        """Find a connected component in sets that were passed in."""
        if not unknown:
            return known

        known = known or set()
        unknown = unknown or set()
        unexplored = set()

        for i_set in sets:
            for r in unknown - known:
                unexplored |= i_set[r]

        return cls.find_connected_component(sets, known | unknown, unexplored)
