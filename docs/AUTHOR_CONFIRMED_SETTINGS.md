# Supplementary parameter confirmation

The manuscript and author-supplied PDFs are unchanged. In discussion, the author confirmed that the proposed supplementary parameter values match their settings. The archive records these as author-confirmed configuration values, not as independently recovered experiment logs.

- AdamW betas (0.9, 0.999), epsilon 1e-8, minimum LR 2e-6, gradient norm 2.0, synchronized horizontal flip p=0.5.
- Bilinear letterbox, zero padding, input scale 255; score threshold 0.20, class-aware NMS IoU 0.50, 100 proposals per level and up to 300 final detections.
- The cost table in `configs/author_confirmed_profile.json` stores **incremental** costs. Shared cost 5.0 ms; total L/R/T/C0 costs 12.0/15.2/15.6/17.8 ms. Total C costs for k=2/4/8/12/16 are 18.35/19.05/20.80/24.30/30.0 ms. Budgets are 21.0/25.5/30.0 ms. This is not a raw Jetson trace.
- The per-seed mAP50:95 values discussed were 62.48/62.70/62.92 (M3FD) and 62.00/62.20/62.40 (LLVIP), ordered by seeds 42/3407/2026. They were initially constructed to match the paper summaries and the author subsequently confirmed the match. They are preserved separately in `author_confirmed_seed_summary.csv` with that provenance; they are not generated run logs.

Implementation choices not identified in the original paper include stage-local scheduler restarts, reference physical batch sizes 4/1/1 with accumulation 4/16/16, and the optional COCO evaluation adapter. These are settings for running this reconstructed code, not claims about the original run. Use the original experiment artifacts for verification.
