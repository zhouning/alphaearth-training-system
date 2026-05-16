import json
from pathlib import Path

import pytest


def test_merge_writes_direction_field(tmp_path):
    from geoadapter.bench.run_loveda_crossdomain import merge_direction_results

    u2r_file = tmp_path / "u2r.json"
    r2u_file = tmp_path / "r2u.json"
    out_file = tmp_path / "loveda_lulc_seg.json"

    u2r_file.write_text(json.dumps([
        {"method": "linear_probe", "modality": "rgb_3band", "seed": 42,
         "trainable_params": 4614, "mIoU": 0.20},
        {"method": "houlsby", "modality": "rgb_3band", "seed": 42,
         "trainable_params": 1194246, "mIoU": 0.31},
    ]))
    r2u_file.write_text(json.dumps([
        {"method": "linear_probe", "modality": "rgb_3band", "seed": 42,
         "trainable_params": 4614, "mIoU": 0.18},
        {"method": "houlsby", "modality": "rgb_3band", "seed": 42,
         "trainable_params": 1194246, "mIoU": 0.28},
    ]))

    merge_direction_results(u2r_file, r2u_file, out_file)

    merged = json.loads(out_file.read_text())
    assert len(merged) == 4
    u2r_rows = [r for r in merged if r["direction"] == "U->R"]
    r2u_rows = [r for r in merged if r["direction"] == "R->U"]
    assert len(u2r_rows) == 2
    assert len(r2u_rows) == 2
    # Confirm U->R Houlsby was 0.31, not 0.28
    assert next(r["mIoU"] for r in u2r_rows if r["method"] == "houlsby") == pytest.approx(0.31)


def test_merge_handles_missing_input_file(tmp_path):
    from geoadapter.bench.run_loveda_crossdomain import merge_direction_results

    out_file = tmp_path / "loveda_lulc_seg.json"
    with pytest.raises(FileNotFoundError):
        merge_direction_results(tmp_path / "nope.json", tmp_path / "also-nope.json", out_file)
