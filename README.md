# DASE7506 MP1 - Meng Weitao (3036746155)

Full-test CPU FP32 BPB: **1.4583128252943809**. The fixed model combines two causal GPTs, window-local copying, and training-only continuation-count 4-grams.

## Contents

- [Report (8 pages)](MP1_student_starter/code/REPORT.pdf) and [editable report](MP1_student_starter/code/REPORT.md).
- [Exact installation, training, assembly, and evaluation instructions](MP1_student_starter/code/FINAL_REPRODUCTION.md).
- [Code README, data attribution, and AI assistance disclosure](MP1_student_starter/code/README.md).
- `MP1_student_starter/code/`: runnable models, fixed evaluator, training scripts, configurations, correctness tests, and unchanged course data.
- `experiments/`: compact recorded metrics and evidence supporting the report. It does not contain historical model weights or cached evaluation answers.
- `environment/checks/smoke-cuda/metrics.json`: one small historical training-cost record, not a Python environment.
- `run_baseline.py`: Windows process-tree RAM measurement wrapper. This packaged copy uses the active Python environment instead of requiring an author-specific virtual-environment path.
- [Submission verification](SUBMISSION_CHECKS.json): isolated extraction, seven tests, actual-checkpoint contracts, full-test BPB reproduction, and same-machine CPU timing.
- `SOURCE_MANIFEST.json`: SHA256 hashes of the source package, excluding this manifest itself.

## Evaluate without retraining

Download the matching checkpoint bundle from this version's release. Extract it into the repository root, retaining the archive's paths. The resulting file must be `experiments/final-hybrid-v3/checkpoint.pt` next to the existing evidence files.

Create and activate a Python 3.12 environment. In PowerShell, use `.venv/Scripts/Activate.ps1`; on Linux/macOS use `source .venv/bin/activate`.

```bash
python -m venv .venv
# Activate the environment, then:
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy==2.5.3 tokenizers==0.21.4
cd MP1_student_starter/code
python -m unittest discover -s tests -v
python evaluate.py --checkpoint ../../experiments/final-hybrid-v3/checkpoint.pt --device cpu --precision fp32 --threads 4 --split test --output ../../verification_outputs/test_cpu_fp32.json
```

For macOS, install `torch==2.7.1` from the default PyPI index instead. No data download, network access, external count file, or retraining is required for inference after dependencies are installed.

Checkpoint SHA256:
`5b99639aa30264a10ec0d9a7608317fec5e926b521ce205b64c2b9bfde1183e1`

## Resource evidence and package notes

Recorded same-machine CPU scoring ratio: 51.423691 / 10.812416 = 4.755985 (limit 5). Conservative peak process-tree RAM bound: 2.470680 GiB (limit 4 GiB). Uncompressed inference assets: 58,484,077 bytes, or 55.774762 MiB (limit 64 MiB). ZIP sizes are transfer sizes, not the inference-asset metric.

Historical records keep their original machine paths and hashes. Some paths identify development-only candidates that are not distributed; they are provenance, not inference dependencies. The 31 retained training metrics support the search-cost ledger. FREEZE.json describes the state before the final test; test_cpu_fp32.json and COMPLIANCE.json record the completed evaluation. COMPLIANCE.json's report metadata has been updated for the current report.

The checkpoint and all frozen inference modules are unchanged. Only submission documents, package manifests, and the portable environment check in the measurement wrapper have been updated.

## AI assistance and reused work

OpenAI Codex assisted with assignment interpretation, model and training implementation, validation experiments, debugging. The base code and fixed benchmark come from the course starter. Data notices are retained in the code README; continuation-count smoothing is attributed in continuation_ngrams.py.
