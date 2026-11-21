# Implementation notes

The source code implements major ideas described in the accompanying RD-Fuse article and passes structural software tests. It was produced from the document description; it is **not a byte-for-byte copy of the author's experiment repository**. In particular:

- The optional `timm` PVTv2-B0 stage adapter has not been validated against the paper's original training environment.
- The detection head, positive-sample assignment, evaluation metric and attention layout are compact engineering implementations; original experiment-specific details were not supplied with the PDFs.
- Utility targets follow loss differences for forced global paths and single-window C0 interventions. Regret samples multi-window configurations and evaluates their joint losses.
- The route-profile JSON under `configs/demo_profile.json` contains only illustrative costs for testing. Use actual TensorRT device traces when studying published latency claims.
- The synthetic demo has no relation to the paper's two datasets. For real experiments use the paired dataset manifest contract in `DATA_PROTOCOL.md`.

The paper's original PDF is unchanged. Code-based results should be documented with their own training data, checkpoint, environment, profiler and evaluation protocol.
