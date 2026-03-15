# MVP2 Sprint 3 Step 1 Report

Date: 2026-03-12  
Scope: Config plumbing for Dual-Encoder parameters

## Done

1. Updated model factory kwargs handling:
   - [factory.py](C:\Users\korsr\PycharmProjects\bpm_prediction\src\domain\models\factory.py)
   - `create_model(...)` now filters kwargs by constructor signature.
   - This allows passing Sprint-3 params from CLI without breaking baseline models.

2. Wired new model config keys from YAML into model creation:
   - [cli.py](C:\Users\korsr\PycharmProjects\bpm_prediction\src\cli.py)
   - Added pass-through:
     - `struct_encoder_type`
     - `struct_hidden_dim`
     - `cross_attention_heads`

3. Extended EOPKG model config defaults:
   - [gatv2_eopkg.yaml](C:\Users\korsr\PycharmProjects\bpm_prediction\configs\model\gatv2_eopkg.yaml)
   - Added:
     - `struct_encoder_type: "GATv2Conv"`
     - `struct_hidden_dim: 64`
     - `cross_attention_heads: 4`

## Notes

1. Backward compatibility is preserved:
   - Baseline model constructors ignore new params via filtered kwargs.
2. No model architecture changes yet in this step.

