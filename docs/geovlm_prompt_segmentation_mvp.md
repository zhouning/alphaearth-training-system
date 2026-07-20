# Paper12 GeoVLM Prompt Segmentation MVP

## Status

The implementation and offline tests are complete. Real model evidence is not.
The project must report `mvp_status: incomplete` until real LandCoverAI runs
produce all required prompt and baseline rows. Synthetic fixtures validate only
software plumbing and cannot be cited as model performance.

No GeoVLM result is currently incorporated into the Paper12 manuscript. This
work does not change the backend, frontend, ArcGIS packaging, or existing result
JSON files.

## Validated scope

- Dataset: LandCoverAI RGB imagery.
- Prompt language: English only.
- Supervised concepts: `building`, `road`, and `water`.
- Output: one binary probability map and thresholded mask for one prompt.
- Out of scope: woodland training, arbitrary open-vocabulary concepts,
  captioning, VQA, counting, referring expressions, Chinese prompts, and ArcGIS
  product parity.

The official source taxonomy is preserved:

| Class id | Source class | MVP role |
| ---: | --- | --- |
| 0 | background | negative pixels |
| 1 | building | target |
| 2 | woodland | out-of-scope diagnostic |
| 3 | water | target |
| 4 | road | target |

## Architecture

The prompt method is
`siglip_film_dense_similarity_houlsby`:

1. Normalize RGB values to `[0, 1]` and zero-pad three channels to the six
   Prithvi input channels.
2. Load Prithvi-100M and opt into checkpoint positional embeddings using the
   fixed temporal-mean and bilinear interpolation policy.
3. Freeze the Prithvi base and train Houlsby adapters with bottleneck dimension
   64.
4. Freeze `google/siglip-base-patch16-224` and train only the text projection.
5. Condition spatial visual tokens through FiLM and a dense cosine-similarity
   map, then decode one binary logit per pixel.

The comparison method,
`no_text_three_binary_heads_houlsby`, uses the same visual path and three binary
decoder channels. It is selected by target class for evaluation and does not
accept natural-language prompts.

The prompt model never maps prompt keywords to fixed class ids. Arbitrary
non-empty text is passed directly to the frozen text tower, while the validated
semantic claim remains limited to the three supervised concepts.

## Repository artifacts

- Config: `geoadapter/bench/configs/geovlm_prompt_segmentation.yaml`
- Prompts: `geoadapter/bench/configs/geovlm_prompts.yaml`
- Runner: `geoadapter/bench/run_geovlm_prompt_segmentation.py`
- Summary builder: `geoadapter/bench/geovlm_prompt_summary.py`
- Inference service: `geoadapter/inference/prompt_segmentation.py`
- CLI: `scripts/run_geovlm_prompt_segmentation.py`
- Colab: `colab/paper12_geovlm_prompt_segmentation_colab.ipynb`

LandCoverAI data, Prithvi weights, SigLIP caches, and trained checkpoints remain
outside Git.

## Local verification

Install core and GeoVLM dependencies:

```bash
pip install -e '.[geovlm]'
pip install torchgeo
```

Run the complete offline-focused set:

```bash
python -m pytest tests/test_prompt_segmentation_data.py tests/test_prithvi_position_embeddings.py tests/test_prompt_segmentation_model.py tests/test_prompt_segmentation_engine.py tests/test_geovlm_prompt_summary.py tests/test_geovlm_prompt_runner.py tests/test_geovlm_prompt_inference.py tests/test_paper12_colab_notebooks.py -v
```

These tests inject tiny local models and datasets. They do not download or
evaluate SigLIP or LandCoverAI.

## Real Colab execution

Open `colab/paper12_geovlm_prompt_segmentation_colab.ipynb` from the repository
master branch. The notebook:

1. records the Git commit and dependency versions;
2. stages and hashes `Prithvi_100M.pt`;
3. resolves and caches the exact SigLIP revision;
4. downloads LandCoverAI and verifies mask values are within `{0,1,2,3,4}`;
5. writes an absolute-path config with `allow_synthetic_fallback: false`;
6. runs the seed-42 smoke stage;
7. enables the full matrix only through `RUN_FULL_MATRIX = True`;
8. persists resumable checkpoints, previews, raw rows, and the summary to
   Google Drive.

Drive locations:

```text
/content/drive/MyDrive/paper12_results
/content/drive/MyDrive/paper12_checkpoints/geovlm_prompt_segmentation
/content/drive/MyDrive/paper12_previews/geovlm_prompt_segmentation
/content/drive/MyDrive/huggingface_cache/paper12_geovlm
```

## Runner commands

Seed-42 smoke stage:

```bash
python -m geoadapter.bench.run_geovlm_prompt_segmentation \
  --config geoadapter/bench/configs/geovlm_prompt_segmentation.yaml \
  --output paper12_results/geovlm_prompt_segmentation.json \
  --summary-output paper12_results/geovlm_prompt_segmentation_summary.json \
  --checkpoint-dir /path/to/checkpoints \
  --preview-dir /path/to/previews \
  --stage seed42
```

Full matrix after seed 42 passes:

```bash
python -m geoadapter.bench.run_geovlm_prompt_segmentation \
  --config geoadapter/bench/configs/geovlm_prompt_segmentation.yaml \
  --output paper12_results/geovlm_prompt_segmentation.json \
  --summary-output paper12_results/geovlm_prompt_segmentation_summary.json \
  --checkpoint-dir /path/to/checkpoints \
  --preview-dir /path/to/previews \
  --stage full
```

Rebuild the summary independently:

```bash
python -m geoadapter.bench.geovlm_prompt_summary \
  --input paper12_results/geovlm_prompt_segmentation.json \
  --output paper12_results/geovlm_prompt_segmentation_summary.json \
  --bootstrap-iterations 1000
```

## Acceptance contract

The raw matrix is complete only when both methods have rows for seeds 42, 123,
and 456 and each method/seed pair has building, road, and water rows. The prompt
method passes only when all of the following are true:

- mean foreground IoU is at least 0.40;
- every target class IoU is at least 0.25;
- mean held-out-prompt IoU retains at least 90% of seen-prompt IoU;
- correct-minus-wrong prompt IoU is at least 0.10;
- the paired-bootstrap 95% confidence interval has a lower bound above zero;
- mean absolute probability-map change is at least 0.05;
- every checkpoint reload reproduces stored evaluation logits;
- no row uses synthetic fallback or missing-weight substitution.

If the matrix is incomplete, the summary remains `incomplete`. If the matrix is
complete but any gate fails, it reports `failed` and lists the exact failed
gates. Only a complete matrix with every gate true may report `passed`.

## Offline inference

```bash
python scripts/run_geovlm_prompt_segmentation.py \
  --image sample.tif \
  --prompt "segment all water bodies" \
  --checkpoint /path/to/checkpoint.pt \
  --output-dir results/geovlm_prompt_inference \
  --threshold 0.5 \
  --local-files-only
```

For PNG/JPEG inputs the CLI writes a PNG mask, a float32 NumPy probability
array, a preview, and metadata JSON. For a GeoTIFF with CRS and affine transform
it writes a one-band uint8 GeoTIFF mask preserving the source georeferencing.
It never invents spatial metadata for non-georeferenced inputs.
