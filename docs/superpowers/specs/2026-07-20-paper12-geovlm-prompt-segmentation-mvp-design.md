# Paper12 GeoVLM Prompt Segmentation MVP Design

Date: 2026-07-20
Workspace: `D:\adk\AlphaEarth-System`
Branch: `master`

## Goal

Build and evaluate a real text-conditioned remote-sensing segmentation MVP for
Paper12. Given one LandCoverAI RGB image and an English prompt about buildings,
roads, or water, the model must produce a binary mask for the prompted concept.
The text embedding must participate in the model computation and change the
pixel logits; a keyword-to-existing-class lookup is not an acceptable
implementation.

The first delivery is an offline model evidence chain: reusable model code, a
training configuration, a generated Colab experiment, checkpoint contracts, an
offline inference CLI, metrics, and prediction previews. Backend and frontend
integration are explicitly deferred until the model passes the real-data gates.

## Current Context

Paper12 already provides a Prithvi-100M dense-prediction path, Houlsby PEFT,
RGB-to-six-channel bridging, and a `conv_lite` segmentation decoder. The latest
LandCoverAI decoder ablation reaches mean mIoU 0.7246 with
`houlsby_conv_lite_d128`, so the repository has a credible visual segmentation
base.

The current path is nevertheless fixed-class segmentation, not GeoVLM:

- `SegmentationHead` does not accept text;
- the project has no tokenizer or text encoder dependency;
- `PrithviBackbone` does not currently load `encoder.pos_embed` from the
  Prithvi checkpoint;
- existing LandCoverAI configs declare six output classes even though the
  official TorchGeo taxonomy and the recorded per-class arrays contain five:
  background, building, woodland, water, and road.

The local Prithvi checkpoint contains `encoder.pos_embed` with shape
`[1, 589, 768]`. This corresponds to one CLS token plus a 3 x 14 x 14 source
grid and can supply an explicit spatial prior for the new model without changing
historical Paper12 runs.

## Scope

### In Scope

- LandCoverAI RGB imagery and official five-class mask semantics.
- English prompts for `building`, `road`, and `water`.
- Frozen SigLIP text encoding using
  `google/siglip-base-patch16-224`.
- Frozen Prithvi base parameters with trainable Houlsby adapters.
- Checkpoint-derived, interpolated Prithvi positional embeddings enabled only
  for the new prompt model.
- FiLM text conditioning, dense image-text similarity, and a binary
  `conv_lite` decoder.
- Seen-prompt, held-out-prompt, and counterfactual wrong-prompt evaluation.
- A no-text three-channel binary segmentation baseline.
- One-seed smoke training followed by a three-seed result matrix.
- Offline checkpoint inference and reproducible Colab execution.

### Out of Scope

- Chinese or mixed-language prompts.
- Woodland as a trained target class.
- Open-vocabulary claims beyond the three supervised concepts.
- Captioning, visual question answering, counting, referring expressions, or
  autoregressive language generation.
- ArcGIS EMD, `.dlpk`, ArcPy, backend API, Model Hub, or frontend integration.
- Committing LandCoverAI data, SigLIP weights, Prithvi weights, model
  checkpoints, or Hugging Face caches to Git.
- Rewriting existing Paper12 result JSON files or retroactively changing the
  behavior of historical segmentation experiments.

## Selected Architecture

### Why This Approach

The selected design combines SigLIP, FiLM, and a dense similarity map. A pure
similarity decoder is smaller but is likely to underfit boundaries. A
multi-layer cross-attention transformer is more expressive but adds substantial
training risk for only three supervised concepts. FiLM plus similarity keeps
the text path explicit and testable while reusing the strongest existing
LandCoverAI decoder pattern.

### Vision Path

1. Load Prithvi-100M with the existing checkpoint remapping.
2. Convert LandCoverAI RGB to the six-channel Prithvi input using the existing
   deterministic zero-pad adapter. Do not add a trainable GeoAdapter in this
   MVP; the existing LandCoverAI baseline uses zero padding, and adding a second
   trainable modality component would confound the text-conditioning result.
3. Insert Houlsby bottleneck adapters with dimension 64 into the transformer
   blocks. Keep all original Prithvi parameters frozen.
4. Return the spatial patch tokens and grid dimensions.

The existing `PrithviBackbone` default remains position-free so prior results
are reproducible. The new prompt model opts into checkpoint positional
embeddings through an explicit mode. The loader must:

- separate the CLS vector from the 588 patch vectors;
- validate that the patch count is exactly `3 * 14 * 14` for the staged
  Prithvi-100M checkpoint;
- reshape to `[1, 3, 14, 14, 768]`;
- average the three temporal positions because the MVP consumes one RGB image,
  yielding a single 14 x 14 spatial grid;
- bilinearly interpolate that grid to the runtime patch grid;
- add the interpolated positions to patch tokens and the stored CLS position
  to the CLS token.

Unexpected positional shapes must raise a compatibility error rather than be
silently ignored.

### Text Path

Use `google/siglip-base-patch16-224` through Hugging Face `transformers`.
Tokenization and the SigLIP text tower run with frozen weights. The pooled text
embedding is projected to a 256-dimensional conditioning space and L2
normalized. The projection remains trainable.

The text encoder is a replaceable module behind a narrow interface that accepts
`list[str]` and returns `[batch, text_dim]`. Unit and integration tests inject a
small deterministic fake encoder, so local tests never download weights or
require network access.

### Conditioning and Decoder

For visual tokens `V` and normalized text embedding `T`:

1. Project `V` to a 256-dimensional normalized similarity space.
2. Compute a learned-temperature cosine similarity between every patch token
   and `T`, producing one scalar map per image.
3. Project the original visual tokens to 128 decoder channels.
4. Map `T` through a small MLP to 128 scale values and 128 bias values.
5. Apply FiLM as `V_film = V_decoder * (1 + gamma) + beta`.
6. Concatenate the FiLM feature map and the one-channel similarity map.
7. Decode with a compact 3 x 3 convolutional path and bilinearly upsample to
   the input resolution, producing one binary logit per pixel.

Trainable parameters are limited to Houlsby adapters, the visual and text
projections, the FiLM MLP, the similarity temperature, and the decoder. The
base Prithvi and SigLIP parameters remain frozen and must be reported separately
from trainable parameters.

## Data and Prompt Contract

### Taxonomy

Use the official TorchGeo LandCoverAI indices exactly:

| Index | Class | MVP role |
|---:|---|---|
| 0 | background | negative pixels |
| 1 | building | target |
| 2 | woodland | out-of-scope diagnostic |
| 3 | water | target |
| 4 | road | target |

All new configs use `num_classes: 5` for source-mask validation. Existing
historical configs and result files are not rewritten as part of this work.

For each training sample, a prompt class is sampled uniformly from building,
water, and road. The source multiclass mask is converted to
`target = (mask == class_index)`, producing a binary target without tripling the
dataset on disk or in the dataset index.

The prompt sampler operates after batching so it can cap empty-target examples
at 25% when at least one supported target is present in the image. If an image
contains none of the three supported targets, it remains a valid empty example.
Class sampling statistics and empty-target share are written into every result
file.

### Prompt Sets

Prompt definitions live in a versioned YAML file. Training and held-out strings
are disjoint and are validated as such.

| Class | Training prompts | Held-out prompts |
|---|---|---|
| building | `segment all buildings`; `find the buildings`; `map building footprints`; `show built structures` | `extract every building visible in this aerial image`; `identify roofed structures` |
| road | `segment all roads`; `find the roads`; `map road surfaces`; `show the road network` | `extract the visible transportation routes`; `identify paved routes` |
| water | `segment all water bodies`; `find the water`; `map surface water`; `show lakes and rivers` | `extract visible aquatic areas`; `identify open water` |

Training randomly selects only from the training column. Held-out prompts are
used only during validation and final evaluation. Woodland prompts are not
trained against false empty masks; they are retained only as an explicitly
out-of-scope diagnostic and cannot contribute to the MVP pass decision.

The model and CLI accept arbitrary non-empty English text and do not parse
keywords. Output metadata states that validated semantic scope is limited to
the three target concepts.

## Training

### Objective

Use a binary segmentation objective:

`loss = weighted_bce_with_logits + dice_loss`

Dataset-level positive weights are computed independently for building, water,
and road on the training split, clipped to `[1, 20]`, and selected per example
according to its prompt class. Dice loss is computed per example with a small
epsilon and averaged across the batch. Initial coefficients are 1.0 for both
terms.

The allowed tuning sequence, if the real-data gates fail, is deliberately
narrow:

1. adjust BCE/Dice coefficients;
2. adjust the empty-target cap;
3. adjust the conditioning dimension or decoder hidden dimension.

Changing datasets, adding a keyword lookup, or reporting fixed-class output as
prompt-conditioned output is not an allowed recovery path.

### Experiment Stages

Stage 1 is a single-seed run with seed 42. It verifies that loss decreases,
each target class produces nonconstant masks, prompt changes alter logits, and
the pipeline saves and reloads a usable checkpoint.

Stage 2 runs seeds 42, 123, and 456 for both:

- `siglip_film_dense_similarity_houlsby`, the selected prompt model;
- `no_text_three_binary_heads_houlsby`, a baseline with the same visual path
  and a three-channel decoder but no text input.

The baseline channel is selected by target class only for evaluation. It
measures how much is gained or lost by language conditioning without pretending
to accept natural language.

## Evaluation

### Metrics

Report foreground IoU and Dice for every target class under:

- seen training prompts;
- held-out prompts;
- correct prompts;
- both wrong target prompts on the same images.

Also report foreground pixel share, empty-mask rate, mean absolute probability
change between prompt pairs, trainable parameter count, inference latency, and
per-seed values. Aggregate means and standard deviations across three seeds.

Counterfactual prompt sensitivity is evaluated on samples containing at least
one supported target. For each image/class pair, evaluate the correct class
prompt and both wrong class prompts against the correct class mask. The prompt
effect passes only when:

- the mean correct-minus-wrong foreground-IoU delta is at least 0.10; and
- a paired bootstrap with 1,000 resamples gives a 95% confidence interval whose
  lower bound is greater than zero; and
- the mean absolute probability-map change across class prompts is at least
  0.05.

These checks prevent a model that ignores text and emits the same mask for every
prompt from passing on class-averaged IoU alone.

### MVP Acceptance Gates

The three-seed prompt model passes only if all conditions hold:

- mean foreground IoU over building, road, and water is at least 0.40;
- each class has foreground IoU of at least 0.25;
- mean held-out-prompt IoU is at least 90% of mean seen-prompt IoU;
- all counterfactual prompt-sensitivity conditions above pass;
- checkpoints reload and reproduce stored evaluation logits within numerical
  tolerance;
- no run uses synthetic fallback or missing-weight substitution.

If a gate fails, outputs must state `mvp_status: failed` with the failed checks.
The manuscript and project status must not claim that the GeoVLM MVP is
complete.

## Artifacts and Reproducibility

### Repository Artifacts

Commit only lightweight, reviewable artifacts:

- prompt-conditioned model modules;
- prompt dataset/sampler and metrics;
- experiment configuration and prompt YAML;
- Colab notebook generator and generated notebook;
- offline inference CLI;
- unit and integration tests;
- result-summary builder and schema documentation.

Add an optional dependency group for the real text path with a pinned
`transformers` major version plus the tokenizer/runtime packages required by
SigLIP. Core package imports must continue to work without that optional group.

### External Artifacts

LandCoverAI, the Prithvi checkpoint, SigLIP weights/cache, and trained model
checkpoints remain outside Git. The Colab notebook stages them on local SSD or
Google Drive and records exact model identifiers, dependency versions, and file
hashes in its run manifest.

The standard result locations are:

- `paper12_results/geovlm_prompt_segmentation.json` for raw per-seed metrics;
- `paper12_results/geovlm_prompt_segmentation_summary.json` for gates and
  aggregates;
- a Drive checkpoint directory for model states;
- a Drive preview directory for a bounded set of image, target, probability,
  and predicted-mask panels.

Real results are not mirrored into the current submission package until all
gates pass and the author explicitly decides to incorporate this new evidence
into Paper12.

### Checkpoint Contract

Each checkpoint contains:

- trainable prompt-model state;
- Houlsby adapter state;
- optimizer, scheduler, epoch, and seed for resumable training;
- Prithvi checkpoint hash and positional-embedding reduction policy;
- SigLIP model identifier and revision;
- prompt-config hash and official class mapping;
- conditioning/decoder dimensions and threshold;
- training dependency versions.

Frozen Prithvi and SigLIP weights are referenced by identifier and hash rather
than duplicated inside each checkpoint.

## Offline Inference CLI

Provide a CLI that accepts:

- one RGB `.tif`, `.tiff`, `.png`, `.jpg`, or `.jpeg` image;
- one non-empty English prompt;
- one trained MVP checkpoint;
- an output directory and optional probability threshold.

It writes a binary mask, floating-point probability array, preview image, and
metadata JSON. Metadata includes the prompt, validated semantic scope, model
and source-weight identifiers, input dimensions, threshold, foreground pixel
share, and output paths. GeoTIFF inputs preserve their CRS and affine transform
in the mask output. Non-georeferenced inputs produce PNG/NumPy outputs without
inventing spatial metadata.

The CLI does not map prompt text to a class id and does not silently fall back
to the no-text baseline.

## Failure Handling

Real training and inference fail early with actionable errors when any of the
following is missing or incompatible:

- LandCoverAI files or official five-class mask values;
- Prithvi checkpoint or expected checkpoint tensors;
- SigLIP weights/tokenizer when offline mode is requested;
- optional text dependencies;
- checkpoint hashes, dimensions, or prompt schema;
- finite image values or supported RGB channel count.

Synthetic data is allowed only inside tests with explicit fake components. It
is forbidden in the real experiment configuration and result generator.

## Testing

### Offline Unit Tests

Unit tests must cover:

- the official five-class mapping and three binary-mask conversions;
- balanced class selection and the 25% empty-target cap;
- disjoint training and held-out prompt sets;
- positional checkpoint parsing, CLS separation, temporal averaging, and 2D
  interpolation for square and rectangular runtime grids;
- explicit failure for incompatible positional token counts;
- frozen SigLIP parameters and text embedding shapes using a fake encoder;
- FiLM, dense similarity, and decoder shapes;
- changed text embeddings changing logits for identical visual tokens;
- gradients reaching Houlsby, projections, FiLM, temperature, and decoder but
  not frozen Prithvi/SigLIP parameters;
- BCE plus Dice behavior for positive and empty masks;
- checkpoint save/resume compatibility;
- result gate computation and failed-gate reasons;
- offline CLI output contracts with a tiny checkpoint and fake text encoder.

### Offline Integration Test

A small deterministic integration test uses a tiny visual backbone and fake
text encoder to train on synthetic geometric masks for several steps. It must
show decreasing loss, prompt-dependent logits, successful checkpoint reload,
and identical post-reload inference. This test validates plumbing only and is
never reported as model evidence.

### Real Verification

The generated Colab notebook must:

1. install pinned dependencies;
2. stage and hash Prithvi and SigLIP assets;
3. download/verify LandCoverAI;
4. run focused data and forward smoke checks;
5. run the seed-42 checkpointed experiment;
6. continue to the three-seed prompt/baseline matrix only after Stage 1 checks;
7. generate raw metrics, summary gates, manifests, and previews;
8. copy resumable checkpoints and outputs to Drive.

Because this work produces a PyTorch model rather than an ADK agent, ADK agent
evaluation is not applicable. Verification consists of pytest, real LandCoverAI
metrics, counterfactual prompt tests, checkpoint reproduction, and inference
artifacts.

## Delivery Boundary

Passing this design supports the bounded statement that Paper12 infrastructure
implements a LandCoverAI-trained English prompt segmentation MVP for buildings,
roads, and water. It does not support a claim of a complete ArcGIS GeoVLM,
general open-vocabulary segmentation, captioning, VQA, or cross-sensor
deployment.

Product integration is a separate phase that begins only after real three-seed
results pass all acceptance gates.
