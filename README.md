# MAIRA-2 text-dominance experiments

Reproducible phrase-grounding experiments for testing whether MAIRA-2 produces
anatomically plausible boxes for unsupported or spatially corrupted chest-X-ray
claims. The code is designed to consume the corruption-pair CSVs produced by the
existing MedGrounder experiments.

MAIRA-2 is a research model and must not be used for clinical decision-making.

## Server setup

The tested server configuration is an RTX 3090 (24 GB), PyTorch 2.5.1 with CUDA
12.1, Python 3.10, and Transformers 4.51.3. Activate the existing environment:

```bash
conda activate /mnt/HDD4/qili0786/conda-envs/maira2
cd /mnt/HDD4/qili0786/maira2-text-dominance
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/mnt/HDD4/qili0786/huggingface_cache
python -m pip install -r requirements.txt
```

Accept the gated-model terms at <https://huggingface.co/microsoft/maira-2> and
authenticate using `hf auth login`. Never commit a Hugging Face token.

## 1. Smoke test

```bash
python scripts/smoke_test.py
```

This runs the official positive phrase plus a laterality corruption and a finding
reported as absent in the official example. Generated boxes are not probabilities
of claim truth.

## 2. Prepare a small transferable subset (local computer)

The following command selects complete pairs, copies only their images, and
rewrites image paths to be portable:

```powershell
python scripts/prepare_subset.py `
  --input-csv D:\github_projects\GMPG\outputs\claim_corruption_evaluation\constructed_claim_pairs.csv `
  --input-csv D:\github_projects\GMPG\outputs\finding_replacement_smoke\constructed_claim_pairs.csv `
  --output-dir D:\maira2-transfer\pilot `
  --max-pairs-per-type 5
```

Transfer the resulting `pilot` directory to the server, for example under
`/mnt/HDD4/qili0786/maira2-data/pilot`. Do not put PadChest images in GitHub.

## 3. Run deterministic batch inference

```bash
python scripts/run_corruption_test.py \
  --input-csv /mnt/HDD4/qili0786/maira2-data/pilot/corruption_pairs.csv \
  --output-csv /mnt/HDD4/qili0786/maira2-results/pilot/results.csv \
  --resume
```

Useful options:

- `--limit 4`: validate the pipeline before a full pilot.
- `--image-root PATH`: remap legacy image paths by filename.
- `--claim-column tested_claim`: select the phrase column.
- `--resume`: retain completed rows after interruption.

The runner uses greedy decoding (`do_sample=False`, `num_beams=1`) and saves
progress atomically every ten rows. It records raw model text, parser status,
boxes relative to the model crop, boxes corrected to the original image shape,
and patient-side/vertical labels. MAIRA-2 provides no calibrated region
confidence, so this project does not invent or compare a fake confidence score.

## 4. Evaluate and visualize

```bash
python scripts/evaluate_results.py \
  --input-csv /mnt/HDD4/qili0786/maira2-results/pilot/results.csv \
  --output-dir /mnt/HDD4/qili0786/maira2-results/pilot/statistics

python scripts/visualize_results.py \
  --input-csv /mnt/HDD4/qili0786/maira2-results/pilot/results.csv \
  --output-dir /mnt/HDD4/qili0786/maira2-results/pilot/visualizations \
  --image-root /mnt/HDD4/qili0786/maira2-data/pilot \
  --only-corrupted
```

Primary outcomes are valid-output rate, incorrect-claim box rate, rejection/no-box
rate, text-following rate for spatial corruptions, and paired image-sensitivity
changes. Absolute MedGrounder query scores must not be compared with MAIRA-2
generation outputs.

## Recommended experiment order

1. Four official-example phrases using `smoke_test.py`.
2. Twenty complete corruption pairs.
3. Inspect raw outputs and visualizations before scaling.
4. Run the fixed shared PadChest-GR subset.
5. Add mismatched-image and horizontal-flip controls to distinguish text following
   from genuine image conditioning.
