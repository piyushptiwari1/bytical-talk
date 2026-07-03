"""Architecture upgrades (opt-in, require retraining). See PLAN_C.

Planned modules:
  * film.py  — FiLM multi-scale audio injection into decoder levels (C1)
  * attn.py  — audio cross-attention / AdaIN fusion at the bottleneck (C2)

These break checkpoint compatibility (add parameters) but keep the inference
contract, so a model retrained with them still loads and runs unchanged.
"""
