from typing import List
import json
import re


def get_valid_filename(s) -> str:
    s = str(s).strip().replace(' ', '_')
    return re.sub(r'(?u)[^-\w.]', '_', s)


def load_clusters(filepath="/data/clusters_v3_train.json") -> List[List[str]]:
    with open(filepath, "r") as fp:
        return json.load(fp)
