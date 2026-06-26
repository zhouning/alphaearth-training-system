import json
import sys
from pathlib import Path


repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "ae_backend"))


def test_summarize_cached_linhe_change_reads_heatmap(tmp_path: Path):
    from app.services.model_hub_change import summarize_cached_linhe_change

    change_dir = tmp_path / "linhe_change"
    pair_dir = change_dir / "2025Q1_vs_2025Q4"
    pair_dir.mkdir(parents=True)
    heatmap = change_dir / "change_heatmap_2025Q1_vs_2025Q4.geojson"
    heatmap.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [107.1, 40.8]},
                        "properties": {
                            "mean_pca_score": 0.42,
                            "mean_rgb_diff": 0.12,
                            "patch_a": "p_00001_00002.npz",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (pair_dir / "pair_visual_00001_00002.png").write_bytes(b"png")

    result = summarize_cached_linhe_change(
        options={"change_dir": str(change_dir), "top": 1},
    )

    assert result["result"]["task"] == "change_detection"
    assert result["result"]["summary"]["n_features"] == 1
    assert result["result"]["summary"]["top_mean_pca_score"] == 0.42
    artifact_kinds = {artifact["kind"] for artifact in result["artifacts"]}
    assert {"geojson", "png"}.issubset(artifact_kinds)
