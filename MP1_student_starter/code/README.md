# MP1 code — installation and usage

Read [the project guide](../GUIDE.md) for the assignment, assessment, deadlines and peer review. This README contains the running instructions and technical rules. The experiment report and final reproduction record describe the student modifications.

All commands below run from **code/**. Data and the tokenizer are included. No API key, pretrained weights or additional dataset download is needed; after installing dependencies, training and evaluation work offline.

## 1. Install

Use **Python 3.12**. From the extracted package directory:

```bash
cd MP1_student_starter/code
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` instead.

Install PyTorch for **one** device:

```bash
# Linux/Windows CPU: recommended; no GPU needed
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
```

For an NVIDIA GPU with a compatible driver, use this command **instead**:

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
```

For macOS, install `torch==2.7.1` from the default PyPI index and run on CPU. After installing PyTorch, install the remaining dependencies and check the model:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The supplied Linux CPU commands use Python 3.12 and PyTorch 2.7.1+cpu. The local experiments use Windows, Python 3.12 and PyTorch 2.7.1+cu126; ranked scoring still uses CPU FP32 with four threads. macOS timings have not been measured.

## 2. Train and evaluate

**Quick installation check** — 10 training steps, then full-test evaluation:

```bash
python train.py --implementation model --steps 10 --run-dir runs/smoke
python evaluate.py --checkpoint runs/smoke/checkpoint.pt --split test
```

This checks that the pipeline works; its score is **not** the full baseline. Each training run needs a new output directory.

**Full baseline** — 1,200 training updates, then evaluation:

```bash
python train.py --implementation model --device cpu --threads 4 --seed 17 --run-dir runs/baseline
python evaluate.py --checkpoint runs/baseline/checkpoint.pt --device cpu --precision fp32 --split test
```

The baseline has four GPT blocks, width 128, four attention heads and **1,088,256 parameters**, and achieves approximately **2.10 test BPB**. On the reference four-thread Xeon Platinum 8457C, measured training took about **311 seconds** and scoring **5.92 seconds**, excluding installation and loading. These are reference measurements, not laptop guarantees or a fixed time allowance.

**Your model** — edit `student.py` and supporting files, then:

```bash
python train.py --implementation student --seed 17 --eval-every 300 --run-dir runs/my-model
python evaluate.py --checkpoint runs/my-model/checkpoint.pt --split validation
# Freeze the final method before testing:
python evaluate.py --checkpoint runs/my-model/checkpoint.pt --split test
```

Training writes `checkpoint.pt` and `metrics.json`. Evaluation writes `test_cpu_fp32.json` (or the corresponding device/split name) and per-window losses. Submit the **bpb** value from the complete-test JSON, not token perplexity or validation BPB. Default evaluation is FP32. Add `--device cuda` for GPU runs; training can use BF16, but ranked evaluation must use FP32 and remain reproducible on CPU. The supplied CUDA runner caps PyTorch allocation at 20 GB; driver overhead is additional.

## 3. Files and model interface

| Files | Use |
|---|---|
| `model.py`, `configs/baseline.json` | Runnable baseline; preserve for comparisons. |
| `student.py`, `train.py` | Your model factory and training recipe; add supporting code as needed. |
| `common.py`, `evaluate.py` | Fixed data checks, windows and scorer; keep unchanged. |
| `data/` | Supplied splits, tokenizer and dataset hashes; keep unchanged. |
| `tests/test_contract.py` | Checks your model's causality, normalization, independence and gradients. |
| `PACKAGE_MANIFEST.json` | Release hashes; paths are relative to the package root containing code/ and GUIDE.md. |

- `build_model(config)` returns a PyTorch model with `context=256`.
- The supplied trainer calls `forward(ids)` for unnormalized logits; the scorer calls `predict_log_probs(ids)` for finite, normalized natural-log probabilities. Both outputs have shape `[batch, time, 2048]`.
- A prediction at position t may use only the observed prefix through t. Reset temporary state between independent windows, examples and scoring passes. Compact training-derived assets may be reused across windows; evaluation-prefix state may not.
- Checkpoints record the implementation module and configuration. Include that module and every required asset so the evaluator can reconstruct the submitted predictor. No optimizer state is required for direct evaluation.
- Training length, architecture, optimizer, regularization, self-trained weight averaging and ensembles may change within the guide's constraints. Log all seeds, processed training targets, checkpoint ancestry and search costs; reusing a checkpoint does not erase its training cost. No particular seed or score improvement is mandated.

## 4. Benchmark and resource measurements

**Fixed score.** Protocol `7506-mp1-wt2-v2`: WikiText-2 raw text, train-fitted BPE-2048, independent windows of 256 targets, including the final short window. Every target except the first token of each split is scored once. Input windows share a boundary token but carry no state. BPB is summed negative log-base-2 next-token probability divided by the split's entire raw UTF-8 byte length, including the first token's bytes.

| Split | Scored targets | UTF-8 bytes |
|---|---:|---:|
| Validation | 376,599 | 1,148,007 |
| Test | 428,405 | 1,292,013 |

Use validation for all development and checkpoint/mixture selection. Weights, statistics and retrieval entries must derive only from training text. The public test text enables reproduction; it must not be used to tune the method. Once frozen, the same predictor may be evaluated repeatedly for timing or reproduction. Token perplexity is not directly comparable with published word-level perplexity.

Measure all three limits for the same frozen predictor:

- **CPU time ≤5× baseline:**
- **Peak RAM ≤4 GiB:**
- **Inference assets ≤64 MiB uncompressed:** 

## 5. Prepare your submission and reproduce a peer

The [guide](../GUIDE.md) specifies the deadline and website workflow. Include the following in your immutable code repository:

- **Report, at most 10 pages including figures, tables and references** 
- **Reproduction instructions**

Your final website submission must link to this code and the matching complete checkpoint bundle. The website generates the Issue JSON automatically. Keep all inference assets downloadable for verification.

To check a peer, obtain their exact code version and checkpoint, follow their installation instructions, and run their frozen model with the supplied evaluator:

```bash
python evaluate.py --checkpoint /path/to/peer-checkpoint.pt --device cpu --precision fp32 --split test --output peer-test.json
```

Compare reproduced BPB with the reported score. Submit **Peer Review Report** with the reproduced score; optionally include the command, environment, difference and evidence/log link.  The instructor adjudicates discrepancies. Confirmed discrepancies during the seven-day review earn bonus credit under the announced marking policy.

## 6. Data attribution

WikiText-2 was introduced by Stephen Merity, Caiming Xiong, James Bradbury and Richard Socher in [Pointer Sentinel Mixture Models](https://arxiv.org/abs/1609.07843). The text is by Wikipedia contributors. The [upstream dataset](https://huggingface.co/datasets/Salesforce/wikitext) identifies [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) and the [GNU Free Documentation License](https://www.gnu.org/licenses/fdl-1.3.html); retain these notices when redistributing the data.

The supplied `wikitext-2-raw-v1` splits preserve revision `b08601e04326c79dfdd32d625aee71d232d685c3`. Rows are joined with newlines and encoded as UTF-8; the tokenizer is fitted only to training text. Dataset hashes are in `data/manifest.json`. These dataset notices do not assign a new license to the surrounding classroom code.

## Final frozen candidate and reproduction

The final predictor combines two self-trained causal GPTs, a per-window copy distribution and a compact training-only continuation-count 4-gram model. Full-test CPU FP32 BPB is **1.458312825**; validation BPB is **1.443548411**. The final test result was measured after freezing and did not trigger further tuning.

All local technical requirements passed: test scoring **51.424 s / 10.812 s baseline = 4.756x** (limit 5x), conservative process-tree peak RAM **2.471 GiB** (limit 4 GiB), complete uncompressed inference assets **55.775 MiB** (limit 64 MiB). The actual-checkpoint causal contract and seven provided tests passed. Original data, tokenizer, baseline, common.py and evaluate.py hashes are unchanged.

Checkpoint: `experiments/final-hybrid-v3/checkpoint.pt`; SHA256 `5b99639aa30264a10ec0d9a7608317fec5e926b521ce205b64c2b9bfde1183e1`. It includes the neural weights and training counts, so inference needs no external count sidecar. Use the full course data directory for the unchanged scorer.

See [FINAL_REPRODUCTION.md](FINAL_REPRODUCTION.md) for exact installation, direct evaluation and fixed-setting training/build commands. [REPORT.md](REPORT.md) and [REPORT.pdf](REPORT.pdf) document matched-target comparisons, ablation, rejected methods, resources and training/search cost. The previous 1.539524 test ensemble is historical and superseded.

**AI assistance disclosure:** OpenAI Codex assisted with assignment interpretation, model and training implementation, validation experiments, debugging, CPU optimization and documentation. The submitter should review and understand the implementation. The original course code is retained and continuation-count smoothing is attributed to Kneser and Ney (1995) in continuation_ngrams.py.

The current submission report is the eight-page REPORT.pdf. Extract the separate checkpoint bundle at the repository root before direct evaluation. See the root README.md for the package layout and quick-start commands.
