# SCA-Exchange: anonymous review artifact

Code and recorded experimental evidence for **Acquire or Generate? Cost-Aware Profiling Decisions for Side-Channel Evaluation**.

This repository supports the submitted paper's joint acquisition/construction cost model, 190 method-budget configurations on AES-PTv2 D1-D4 and AES_GPU, and its recovery, fidelity, and attack-block analyses. The supplied rank arrays reproduce Table II and Figures 1-3 offline. Training code is also provided for the 18 evaluated method variants.

Paper SHA-256: `c78f8659a9d5ccb7b8d3c36e6df8c5fba6987aeedd28793a045a84431f58a860`.

## Quick start: reproduce the paper without training

Use **Python 3.10 or newer** (3.12 recommended). From the repository root:

```bash
python -m pip install -r requirements.txt
python -B reproduce.py
```

This reads the included archives, verifies their hashes, recomputes recovery and cost decisions, and writes the table, figures, and verification record to `results/`. It needs no GPU, dataset download, checkpoint download, account, or network connection after dependencies are installed. A validated local run took about 12 seconds; runtime depends on the CPU and numerical libraries. Allow approximately 1 GB RAM.

```bash
python -B reproduce.py --no-plots
python -B reproduce.py --output runs/reanalysis
python -B scripts/verify.py
```

The third command verifies the default `results/` directory. Use `--output runs/reanalysis` to verify an alternate output. Plotting uses Times New Roman when installed and DejaVu Serif otherwise; the font fallback does not change numerical results.

## Paper-to-code and evidence map

| Paper location | Implementation | Inputs and reproduced output |
| --- | --- | --- |
| Section III-A/B, Eqs. (1)-(5): cost, joint Q/A choice, break-even prices | [`analysis/reanalyze.py`](analysis/reanalyze.py): `cost`, `winner`, `joint_envelope`, `pareto` | Recorded construction times plus complete rank curves; `results/extended-analysis.json`: `scenarios`, `equal_price_envelopes`, `attack_1ms_envelopes`, `bootstrap_crossings` |
| Section III-C, Eq. (6): assessment fidelity | [`analysis/revision_analysis.py`](analysis/revision_analysis.py): `fidelity` construction | `results/extended-analysis.json`: `fidelity`; Qmax=20,000 and kappa=2 |
| Section IV-A: splits, 32 fitting-only HW-SNR POIs, normalization, key hypotheses | [`scripts/prepare.py`](scripts/prepare.py), [`src/base.py`](src/base.py), [`src/aes.py`](src/aes.py) | [`data/frozen-splits-and-preprocessing.npz`](data/frozen-splits-and-preprocessing.npz), [`data/gpu-split.npz`](data/gpu-split.npz); separately obtained public traces |
| Section IV-B, Eq. (7): HW Gaussian regression and covariance selection | [`src/gaussian.py`](src/gaussian.py), [`src/hw_selection.py`](src/hw_selection.py) | Fixed ridge/shrinkage or five fitting-only folds; Table II rows `HW Gaussian` |
| Section IV-B, Eq. (8): HW-RL, Bit-Gauss, Bitwise-RL | [`src/likelihood.py`](src/likelihood.py) | Exact full residual bank; fitted choices and construction/scoring times in the residual result archives |
| Section IV-B, Eq. (9): RShare | [`src/residual.py`](src/residual.py), [`src/residual_numpy.py`](src/residual_numpy.py), [`scripts/train.py`](scripts/train.py) | Centered whole-vector residuals, uniformly sampled donors, and balanced target identities; Table II `RShare` |
| Section IV-B: classifiers, FM4/DDPM16, and augmentation controls | [`src/classifier.py`](src/classifier.py), [`src/generators.py`](src/generators.py), [`src/generator_training.py`](src/generator_training.py), [`src/smote.py`](src/smote.py), [`src/cnn.py`](src/cnn.py), [`scripts/train.py`](scripts/train.py) | Fixed 2,000-update MLP; separate 6,000/12,000-update generator endpoints; 20,000-row pools |
| Section IV-C: SR, GE, first crossing and uncertainty | [`src/ranking.py`](src/ranking.py), [`src/reference_ranks.py`](src/reference_ranks.py), [`analysis/reanalyze.py`](analysis/reanalyze.py), [`analysis/revision_analysis.py`](analysis/revision_analysis.py) | 20 orders per subset, ten subsets per configuration, 2,000 paired-subset bootstrap draws |
| Table II: attack traces to 95% observed SR | [`analysis/plots.py`](analysis/plots.py): `table` | [`results/tables/joint-attack-requirements.tex`](results/tables/joint-attack-requirements.tex); all 190 configurations are also in [`results/configurations.csv`](results/configurations.csv) |
| Figure 1: D1/GPU recovery curves | [`analysis/plots.py`](analysis/plots.py): `recovery` | [`results/figures/joint-recovery.pdf`](results/figures/joint-recovery.pdf) |
| Figure 2: cost curves and price-plane decisions | [`analysis/plots.py`](analysis/plots.py): `frontier` | [`results/figures/joint-frontier.pdf`](results/figures/joint-frontier.pdf); bootstrap and second-block decisions in `extended-analysis.json` |
| Figure 3: acquisition/construction decomposition | [`analysis/plots.py`](analysis/plots.py): `decomposition` | Stage times in [`data/frozen-records.json`](data/frozen-records.json); [`results/figures/joint-cost-decomposition.pdf`](results/figures/joint-cost-decomposition.pdf) |
| Section V-B: ChipWhisperer price calibration | [`data/capture-rates.json`](data/capture-rates.json) | Five sustained workflow rates; median 10.87 traces/s, or 91.9963 ms/trace |
| Section V-D: caps, sustained attainment, strict SR, and second block | [`analysis/revision_analysis.py`](analysis/revision_analysis.py) | `cap_sensitivity`, `second_block`, `A95_sustained`, `Astrict`, and subset-specific crossing distributions in `extended-analysis.json` |
| Section VI: collected-device and unsuccessful-condition scope controls | [`data/scope-controls.csv`](data/scope-controls.csv) | Recorded Q=2,000 subset endpoints for nine collected-device controls, derived noise, AES_HD and ASCAD; excluded from the five-configuration cost frontier |
| Table I and related work | Literature comparison in the submitted paper | These are reported settings/times from cited publications, not new benchmark results or reproductions of those systems. |

## What is included

```text
analysis/       Recovery aggregation, cost envelopes, bootstrap, table/figure rendering
src/            Method, preprocessing, and rank-scoring numerical kernels
scripts/        Public-data preparation, fitting/evaluation, and evidence verification
tests/          Numerical consistency checks
data/           Frozen rank archives, construction records, splits, and calibration
results/        Current paper outputs and verification records
reproduce.py    One-command offline rebuild
SHA256SUMS      Integrity manifest for the distributed files
```

| Rank archive | Evaluations | Groups | Role |
| --- | ---: | ---: | --- |
| `data/rank-curves.zip` | 1,490 | 149 | Original classifiers, augmentation methods, and HW Gaussian baselines |
| `data/residual-results.zip` | 380 | 38 | HW-RL and Bit-Gauss |
| `data/bitwise-rl-results.zip` | 30 | 3 | GPU Bitwise-RL |
| `data/second-block-results.zip` | 1,640 | 164 | D1-D4 replay on a disjoint attack block, with fitted models reused |

The first three archives contain **190 primary groups and 1,900 subset evaluations**. Including second-block replays gives **3,540 rank arrays**. Each array has shape `(20, 10000)` and stores twice the zero-based midrank as `int16`; divide by two for GE, and test equality to zero for successful recovery. NPY payload bytes and their recorded hashes are preserved. Location strings in JSON records are anonymized; `external-record/...` values identify historical files by opaque references and are not required local paths.

`results/extended-analysis.json` is the current combined analysis. `results/analysis.json` is an intermediate analysis of the original 149 groups and does not include the added likelihood baselines. The scripts regenerate both in this order.

## Expected checks

`scripts/verify.py` checks every rank-array hash, all 80 cells of Table II, finite-grid cost decisions, and the following paper results:

| Configuration | Equal 1 ms prices: winner / total seconds | Equal 91.9963 ms prices: winner / total seconds |
| --- | --- | --- |
| D1 | HW-RL, Q=5,000 / 4.31 | RShare, Q=5,000 / 325.23 |
| D2 | HW-RL, Q=2,000 / 0.73 | RShare, Q=2,000 / 33.40 |
| D3 | HW-RL, Q=2,000 / 1.07 | RShare, Q=2,000 / 46.20 |
| D4 | HW-RL, Q=2,000 / 1.25 | RShare, Q=2,000 / 48.66 |
| GPU | Bit-Gauss, Q=5,000 / 3.50 | Bitwise-RL, Q=5,000 / 308.52 |

At a target price of 1 ms, the primary-block neural crossings are **17.00 ms for D1** and **15.25 ms for GPU**. GPU Bitwise-RL at Q=5,000 reaches 95% observed SR at **297** traces. On the second D1 attack block, DDPM16 at Q=2,000 does not reach 95% SR by 10,000 traces, while HW-RL at Q=5,000 reaches it at **798**.

## Refit methods from public traces

Install the training dependencies in a separate environment. Use the PyTorch build appropriate for the available CUDA installation:

```bash
python -m pip install -r requirements-training.txt
python -B -m unittest discover -s tests -v
```

The recorded likelihood environment was Python 3.12.3, NumPy 2.2.6, PyTorch 2.9.0+cu128, and a shared NVIDIA A100-SXM4-80GB with two CPU threads. The exact recorded metadata is in [`data/recorded-environment.json`](data/recorded-environment.json). CPU execution supports inspection and checks; full-bank KDE, especially Bitwise-RL, is substantially faster on a GPU.

Obtain datasets from their original providers:

- **AES-PTv2 D1-D4:** [AESPTv2](https://github.com/urioja/AESPTv2). Use each unprotected STM32F4 HDF5 file, with groups `D*/Unprotected/Profiling` and `D*/Unprotected/Attack`, each containing `Traces`, `Labels`, and `MetaData`. The targeted label is `Sbox(plaintext[0] XOR key[0])`.
- **AES_GPU:** [AES_GPU](https://github.com/fr4nky4ng/AES_GPU), `Traces/AES_GPU.mat`. The recorded file contains 39,511 traces of 350 samples, with `trace`, `cipher`, and `key` arrays. SHA-256: `723fa4b2bc0b65631560b4f121293141e4643bdf47f5d7e3fb4361ef334cfa77`. The label is `InvSbox(cipher XOR last_round_key)`, using key byte 19.

For example, place the GPU file in `local-data/AES_GPU.mat`, then run:

```bash
python -B scripts/prepare.py --dataset AES_GPU --data local-data/AES_GPU.mat --budget 5000 --seed 13 --device cuda:0 --with-attack --output runs/gpu-q5000-s13
python -B scripts/train.py fit --input runs/gpu-q5000-s13 --method Bitwise-RL --device cuda:0 --output runs/gpu-q5000-s13-bitwise-rl
python -B scripts/train.py evaluate --input runs/gpu-q5000-s13 --device cuda:0 --output runs/gpu-q5000-s13-bitwise-rl
```

An AES-PTv2 example is:

```bash
python -B scripts/prepare.py --dataset D1 --data local-data/AES_PTv2_D1.h5 --budget 5000 --seed 13 --device cuda:0 --with-attack --output runs/d1-q5000-s13
python -B scripts/train.py fit --input runs/d1-q5000-s13 --method RShare-T --device cuda:0 --output runs/d1-q5000-s13-rshare
python -B scripts/train.py evaluate --input runs/d1-q5000-s13 --device cuda:0 --output runs/d1-q5000-s13-rshare
python -B scripts/train.py evaluate --input runs/d1-q5000-s13 --device cuda:0 --block second --output runs/d1-q5000-s13-rshare
```

Preparation verifies fitting-only POIs/normalization against the frozen parameters and preserves the exact recorded row indices. Attack inputs are optional and are never opened by the `fit` command. Evaluation verifies the saved model and fitting-input hashes. For a comparison campaign, complete all planned fits before scoring their attack blocks. Existing run directories are preserved; use a new output directory for another run.

Supported method names are:

| Paper name | `--method` |
| --- | --- |
| HW Gaussian; fitting-only CV variant | `HW-template-fixed`, `HW-template-cv` |
| HW-RL; Bit-Gauss; Bitwise-RL | `HW-RL`, `Bit-Gauss`, `Bitwise-RL` |
| Real MLP; RShare; Gaussian augmentation | `Real`, `RShare-T`, `Gauss` |
| Balanced replay; jitter; SMOTE; HW mixup | `Bal`, `Jitter`, `SMOTE`, `Mixup-HW` |
| CNN real; CNN RShare | `Zaid-Real`, `Zaid-RShare` |
| Flow matching, four sampling steps | `FM4-6000`, `FM4-12000` |
| Diffusion, 16 deterministic sampling steps | `DDPM16-6000`, `DDPM16-12000` |

Only recorded method/dataset/budget combinations are admitted by the fitting CLI; see `results/configurations.csv`. In particular, GPU FM4/DDPM16 at Q=5,000 were not fitted. This package does not fill missing cells with inferred results. The neural implementations are the paper's controlled configurations; they are not claimed reproductions of every system discussed in related work.

## Protocol and interpretation

- Profiling budgets include 20% validation. D1-D4 use Q in `{2000,5000,10000,20000}`; GPU uses the available Q in `{2000,5000,20000}`. Seeds are `13,37,73,101,137,211,251,307,401,503`. Every fifth selected position is held out. The generator, likelihood model, POIs, and normalization use only the other 80%.
- Each MLP pool contains 20,000 rows, beginning with all distinct fitting rows. The MLP is 32-64-64-9 with GELU, 2,000 AdamW updates, batch 256, learning rate 0.001, and weight decay 0.0001. Likelihood scores subtract the smoothed training-pool HW log prior.
- FM4/DDPM16 use width-128, two-layer SiLU generators, batch 128, learning rate 0.001 and weight decay 0.0001. Training endpoints are separate candidates. The adapter reloads each fixed endpoint before sampling; validation does not choose the reported endpoint.
- HW-RL and Bitwise-RL select bandwidth from `{.05,.1,.2,.4,.8,1.6,3.2}` and shrinkage from `{.01,.1,.5,.9,1}`, with ridge 0.01. Bit-Gauss additionally selects ridge from `{.001,.01,.1}`. Five fitting-only folds refit the mean, covariance, and residual bank. Scoring uses the exact full residual bank.
- SR averages 20 orders within each subset and then ten subsets. GE is zero-based midrank with tie tolerance `1e-9`. A95 is the first integer crossing of mean SR >= 0.95. Unattained crossings remain `null`; `n/a` means unfitted. The 5k/8k caps, sustained attainment, and strict SR=0.996875 use the same recorded curves.
- Cost is `(Q-2000)*tau_p + A*tau_t + C` for the reported B=1, zero-setup frontiers. Times/prices in JSON are **seconds**. The B=16 sensitivity multiplies homogeneous construction cost only; profiling and attack observations remain shared. It is not a full-key recovery measurement.
- Historical construction/stage times are retained. The portable training adapter reports fresh local timing, which is not a substitute for the recorded warmed A100 measurements. A different processor, library version, RNG backend, or timing boundary can change retraining time and neural results. Offline reanalysis uses the original measurements.

## Coverage and availability

The distributed artifact was checked by replaying all 3,540 rank arrays and all Table II cells, rebuilding Figures 1-3, running six numerical tests, and completing CPU fits of HW Gaussian, HW Gaussian CV, Real MLP, and RShare on a real GPU-dataset subset. A complete public `prepare -> fit -> evaluate` run for GPU HW Gaussian, Q=2,000, seed 13 reproduced its original `(20,10000)` rank array exactly. Full neural campaigns were not rerun during packaging. See [`results/verification.json`](results/verification.json) and [`results/training-verification.json`](results/training-verification.json).

Raw waveform datasets and trained checkpoints are not distributed here. They are unnecessary for the offline table/figure rebuild; obtaining the public datasets is necessary for refitting. The small split/preprocessing archives contain indices and transform parameters, not waveforms. Section VI's collected-device controls are supplied as recorded endpoints; their raw acquisitions are not included. The five capture rates are supplied calibration measurements; per-repeat timing logs and the exact capture-loop boundary are unavailable.

The experiments cover one byte and fixed keys. GPU has no unused second attack block. Bootstrap intervals are conditional on the recorded fitting subsets and attack block. They do not measure new-key or new-block uncertainty. The package contains no full-key or simulation-to-silicon experiment and assigns no measured per-trace simulation price.

## Attribution and anonymous packaging

The numerical kernels are extracted from the recorded experiment sources. Their source hashes and the paper identity are recorded in [`data/source-provenance.json`](data/source-provenance.json). The public entry points resolve paths inside this repository or from command-line arguments. Personal paths, host identifiers, author/contact fields, repository history, remote launchers, and unrelated research drafts are excluded.

External data and methods retain their attribution: Rioja et al., *Towards Human Dependency Elimination: AI Approach to SCA Robustness Assessment*, TIFS 2022 (AES-PTv2); the AES_GPU dataset repository; Zaid et al., *Methodology for Efficient CNN Architectures in Profiling Attacks*, TCHES 2020 (CNN control); Chawla et al., *SMOTE: Synthetic Minority Over-sampling Technique*, JAIR 2002; Lipman et al., *Flow Matching for Generative Modeling*, ICLR 2023; and Song et al., *Denoising Diffusion Implicit Models*, ICLR 2021. Full bibliographic context is in the submitted paper. Dataset licenses and access terms are those of the original providers. No public open-source license is asserted by this review package.
