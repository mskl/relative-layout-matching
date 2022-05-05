PROPRIETARY_DATASET_PATH = "/data/combined"
ELECTIONS_DATASET_PATH = "/data/elections"


def pageimage_path(docid: str, dataset_path: str, page: int = 0) -> str:
    return f"{dataset_path}/{docid}_{page}_pageimage.png"


def embedding_path(docid: str, dataset_path: str, page: int = 0) -> str:
    return f"{dataset_path}/{docid}_{page}_embedding.txt"


def segnet_path(docid: str, dataset_path: str, page: int = 0) -> str:
    return f"{dataset_path}/{docid}_{page}_segnet_convy.npy"


def pic2bos_path(docid: str, dataset_path: str, page: int = 0) -> str:
    return f"{dataset_path}/{docid}_{page}_pic2bos.npz"


def annotation_path(docid: str, dataset_path: str) -> str:
    # NOTE: Annotations are combined for all pages.
    return f"{dataset_path}/{docid}_annotation.json"
