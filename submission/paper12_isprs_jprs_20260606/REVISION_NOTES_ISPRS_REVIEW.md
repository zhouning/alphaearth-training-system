# Revision Notes After ISPRS JPRS-Style Review

## One-Sentence Revised Argument

In Prithvi-100M adaptation under heterogeneous remote sensing inputs, the manuscript shows that PEFT behavior depends on backbone architecture, input modality, and deployment domain, supported by public benchmarks, Linhe production-style validation, and LoveDA cross-domain stress tests, with conclusions bounded to the tested backbone until second-backbone and parameter-matched experiments are added.

## Main Text Changes

- Reframed the title around architecture-aware Prithvi-100M adaptation.
- Rewrote the abstract to avoid universal GeoFM claims.
- Rewrote the introduction contribution list to separate supported findings from hypotheses.
- Added explicit definitions of standard LoRA and split-QKV LoRA.
- Added the six-channel Prithvi input-bridge boundary.
- Reframed Linhe as production-style validation with Esri-derived supervisory labels.
- Clarified that Linhe split-QKV LoRA weakness is not merely the original fused-QKV insertion bug.
- Reframed the LoveDA capacity threshold as a deployment hypothesis.
- Updated discussion and conclusion limitations.

## Remaining High-Risk Reviewer Questions

- Does the PEFT ranking hold on any backbone other than Prithvi-100M?
- Is Houlsby's advantage architectural or mainly due to a larger parameter budget?
- How reliable are the Esri-derived Linhe labels for 2025 imagery?
- Are LoveDA cross-domain scores too close to background baselines to support a capacity-threshold claim?
- Does the six-channel bridge discard important information from `s2_full`?

