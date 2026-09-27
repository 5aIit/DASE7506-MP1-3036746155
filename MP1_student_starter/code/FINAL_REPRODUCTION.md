# Final frozen MP1 predictor: reproduction

Final complete-test CPU FP32 **BPB = 1.4583128252943809**. Validation BPB = **1.4435484111417993**. The model is frozen; no further score search is planned.

The checkpoint is `experiments/final-hybrid-v3/checkpoint.pt` in the workspace/release layout, SHA256:

```text
5b99639aa30264a10ec0d9a7608317fec5e926b521ce205b64c2b9bfde1183e1
```

The original course files remain under `MP1_student_starter/`. Run commands below from `MP1_student_starter/code/`, with Python 3.12. The provided `data/` must contain all three original WikiText-2 text files, tokenizer.json and manifest.json. This submission includes them unchanged. Extract the separate checkpoint bundle into the repository root; it adds `experiments/final-hybrid-v3/checkpoint.pt`.

## Environment and direct evaluation (no retraining)

Install PyTorch 2.7.1 for the selected device, NumPy 2.5.3 and tokenizers 0.21.4, as described in README.md. For CPU-only reproduction:

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy==2.5.3 tokenizers==0.21.4
python -m unittest discover -s tests -v
python verify_hybrid.py --checkpoint ../../experiments/final-hybrid-v3/checkpoint.pt
python evaluate.py --checkpoint ../../experiments/final-hybrid-v3/checkpoint.pt --device cpu --precision fp32 --threads 4 --split test
```

If the checkpoint is stored elsewhere, change only the checkpoint path. The saved checkpoint includes both neural members and the raw training-count table. No `.bin`, `.npz`, pretrained model, external text or network request is needed for inference. Do not use validation-analysis probability files as model assets. The supplied evaluator requires CPU FP32 and independent 256-target windows; do not change it.

Local environment: Windows; Python 3.12; PyTorch 2.7.1+cu126; NumPy 2.5.3; tokenizers 0.21.4; RTX 3050 Ti Laptop GPU for training, CPU with four PyTorch threads for final scoring. Cross-platform retraining need not produce an identical checkpoint hash. Direct evaluation should reproduce BPB within small FP32 numerical differences.

## Exact training recipe for the selected members

These commands document reconstruction; they were not launched again after freezing. Use fresh output directories. They use CUDA FP32; changing `--device cpu` is possible but substantially slower. Training saves the checkpoint with the lowest FP32 validation BPB.

```bash
python train.py --implementation student --config configs/scaled_256x6_dropout02.json --run-dir runs/wide-p02 --device cuda --precision fp32 --threads 4 --seed 17 --steps 16000 --batch-size 32 --learning-rate 0.0015 --weight-decay 0.1 --eval-every 1000

python train.py --implementation student --config configs/scaled_256x6_dropout02.json --init-checkpoint runs/wide-p02/checkpoint.pt --run-dir runs/wide-continued --device cuda --precision fp32 --threads 4 --seed 23 --steps 4000 --batch-size 32 --learning-rate 0.0002 --weight-decay 0.1 --eval-every 1000

python train.py --implementation student --config configs/scaled_160x4_dropout01.json --run-dir runs/medium --device cuda --precision fp32 --threads 4 --seed 23 --steps 16000 --batch-size 32 --learning-rate 0.002 --weight-decay 0.1 --eval-every 2000

python build_ngram_asset.py --output runs/train_counts.bin
```

The three selected runs use 131,072,000 + 32,768,000 + 131,072,000 = **294,912,000 processed training targets**. Training-only count construction reads 3,613,343 tokens separately. The count file is 30,522,560 raw bytes and should have SHA256 `1c318f7fb8669d9bacba4a73be691bb9deab1fad5735c7a90ee2a55e475a5d09`.

## Rebuild the frozen probability model

The following command evaluates **one fixed configuration**, not a new hyperparameter search. It regenerates a path/hash-consistent selection record for the newly trained members. The small-model temperature is **1.1**, matching the actual frozen checkpoint. An exploratory grid also found 1.05 nearly tied, but it is not the final setting.

```bash
python select_hybrid.py --large runs/wide-continued/checkpoint.pt --small runs/medium/checkpoint.pt --ngram-asset runs/train_counts.bin --output runs/fixed_selection.json --device cuda --batch-size 2 --large-temperature 1.1 --small-temperature 1.1 --small-weight 0.25 --ngram-method continuation --ngram-discount 0.5 --ngram-weight 0.125

python build_hybrid.py --large runs/wide-continued/checkpoint.pt --small runs/medium/checkpoint.pt --selection runs/fixed_selection.json --ngram-asset runs/train_counts.bin --output runs/rebuilt-hybrid/checkpoint.pt

python verify_hybrid.py --checkpoint runs/rebuilt-hybrid/checkpoint.pt
python evaluate.py --checkpoint runs/rebuilt-hybrid/checkpoint.pt --device cpu --precision fp32 --threads 4 --split validation
```

Temperature-scaled neural distributions use weights 0.75/0.25 and training log-frequency bias 0.05. The window-local suffix cache uses the fixed coefficients 0.05/0.4/0.2 in `local_cache.py`. The training 4-gram uses continuation-count discount 0.5 and probability weight 0.125. All tables derive only from the supplied training counts. The raw-count alpha field is retained for compatibility but unused by continuation smoothing.

`build_hybrid.py` requires matching member and count-asset paths and hashes. The original pre-test selection is preserved as `final-hybrid-v3/selection_original.json`; `rebuild_selection.json` adds the explicit asset path/hash needed by the current builder. Different metadata/serialization may change checkpoint bytes even when the predictor is equivalent. The released checkpoint hash above is the identity for exact reproduction.

## Final resources and verification

| Requirement | Actual final measurement | Result |
|---|---:|---|
| CPU test scoring <= 5 times original baseline | 51.423691 / 10.812416 = 4.755985 times | Pass |
| Peak evaluation RAM <= 4 GiB | 2.470680 GiB, conservative maximum of validation/test process-tree bounds | Pass |
| Uncompressed inference assets <= 64 MiB | 58,484,077 bytes = 55.774762 MiB | Pass |
| Full-test protocol | 428,405 targets; 1,292,013 raw UTF-8 bytes; CPU FP32 | Pass |
| Causality, normalization, independent examples/windows, gradients | Actual checkpoint contract checks plus seven provided tests | Pass |

The timing measurements are sequential, on the same computer with training and other Python probes stopped. They use the unchanged scorer's timed region for both baseline and final model. Windows RAM measurement includes the runner, Python launcher and evaluator descendants from imports/loading through result saving; the sum of per-process high-water marks is conservative. GPU memory is not substituted for RAM.

The asset total includes the checkpoint, required inference Python files, fixed tokenizer and data manifest. The standalone count file is not shipped a second time for inference. Supplied benchmark input text and runtime libraries are separate from learned inference assets. `FREEZE.json` records every dependency hash, checked before and after final scoring.

From the repository root, with the documented Python environment activated, the Windows-only measurement script can reproduce the measurements. Its package copy accepts any Python environment with the specified versions; it does not require the original author's directory layout. Use separate output directories to preserve the historical evidence:

```powershell
python .\run_baseline.py --checkpoint .\experiments\final-hybrid-v3\checkpoint.pt --output-dir .\verification_outputs\final
# Validation-only checks during development, with a separate output directory:
python .\run_baseline.py --checkpoint .\experiments\final-hybrid-v3\checkpoint.pt --split validation --output-dir .\verification_outputs\validation-repeat
```

Evidence: final `validation_cpu_fp32.json`, `test_cpu_fp32.json`, `resource_measurements.json`, `contract_checks.json`, `FREEZE.json`, `terminal_status.json`, and `verification_terminal.log`. The same-period baseline evidence is in `experiments/final-compliance-baseline/`. Model details, matched-target comparisons, ablation, failed methods, search-cost uncertainty and AI attribution are in REPORT.md / REPORT.pdf.

The saved technical checks passed locally. The report bundled here is the final eight-page English report. Historical evidence JSON and FREEZE.json retain the values and paths recorded during the original measurements; FREEZE.json was written before the final test. COMPLIANCE.json updates only the report metadata for this submission, not the measured model results.
