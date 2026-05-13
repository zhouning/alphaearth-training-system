from pathlib import Path
import pandas as pd

LABEL_COLUMNS = ["dataset_id", "patch_path", "label", "reviewer", "labeled_at"]


def load_patch_index(dataset_dir: Path) -> pd.DataFrame:
    """Load patch index from _index.parquet."""
    dataset_dir = Path(dataset_dir)
    index_path = dataset_dir / "_index.parquet"
    return pd.read_parquet(index_path)


def load_labels(dataset_dir: Path) -> pd.DataFrame:
    """Load labels from _labels.parquet. Return empty DataFrame with LABEL_COLUMNS if missing."""
    dataset_dir = Path(dataset_dir)
    labels_path = dataset_dir / "_labels.parquet"

    if not labels_path.exists():
        return pd.DataFrame(columns=LABEL_COLUMNS)

    return pd.read_parquet(labels_path)
