# Cover Letter Draft

Dear Editor,

We are pleased to submit our manuscript, "Architecture-Aware Parameter-Efficient Adaptation of Prithvi-100M for Heterogeneous Remote Sensing Inputs: Benchmark, Production Validation, and Cross-Domain Stress Tests," for consideration as a research article in ISPRS Journal of Photogrammetry and Remote Sensing.

The manuscript addresses a practical but unresolved problem in remote sensing AI: whether parameter-efficient fine-tuning methods that work well in mainstream NLP and computer vision transfer reliably to geospatial foundation models under heterogeneous sensor inputs and domain shift. We present a systematic benchmark of PEFT methods on Prithvi-100M, covering EuroSAT, BigEarthNet-S2, LandCover.ai semantic segmentation, an operational Linhe County land-use/land-cover workflow, and LoveDA urban/rural cross-domain transfer.

The paper is well aligned with ISPRS JPRS because it combines remote sensing methodology, photogrammetry and computer vision practice, reproducible benchmarking, and operational land-cover validation. Its main contributions are:

1. A systematic evaluation of PEFT strategies for adapting Prithvi-100M to heterogeneous geospatial inputs.
2. A diagnostic separation between standard LoRA insertion failure under fused-QKV attention and the remaining post-fix low-rank ceiling.
3. Evidence that Houlsby adapters are more reliable than LoRA, BitFit, linear probing, and input-stage adaptation across classification, multi-label classification, and segmentation.
4. Production-style validation on a Linhe County LULC workflow with geographic scene-level splitting, Esri-derived supervisory labels, and a synthetic weak-label control that is not an independent manual validation set.
5. A LoveDA cross-domain replication supporting an adapter-capacity hypothesis in the tested Prithvi-100M setting.

All code, configuration files, logs, and reproducibility artifacts are prepared for release with the submission package. The manuscript has not been published, accepted, or submitted elsewhere.

Thank you for considering our manuscript.

Sincerely,

Zhouning
