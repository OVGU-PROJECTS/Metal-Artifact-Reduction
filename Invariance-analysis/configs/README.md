# Configurations and statistics

This folder contains reusable configuration templates and the head/body
normalization statistics.

- `head_body_stats.json`: training/validation statistics.
- `head_body_stats_testset.json`: test-set statistics.
- `uncond.yaml`, `dudonet.yaml`, `cnnmar.yaml`, `pconv.yaml`: VAE variants.
- `cinn.yaml`: cINN template; set its method-specific cVAE checkpoint and
  `pred_name` before use.
- `dml.yaml`: DML training/evaluation configurations.
