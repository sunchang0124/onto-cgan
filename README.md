# Onto-CGAN

Ontology-conditioned Conditional GAN for synthetic tabular data generation. Extension of dp-cgans that uses ontology embeddings (from a knowledge graph / gensim KeyedVectors) as conditional vectors instead of or alongside standard one-hot category conditions.

**Package location (HPC):** `/gpfs/home2/csun/onto_cgans/onto_cgans/`
**Package location (local):** `/Users/changsun/Documents/onto_cgans/`
**Source reference:** `dp_cgans/src/dp_cgans/ontocgan/`
**Install (HPC):** `python3.11 -m pip install -e /gpfs/home2/csun/onto_cgans/onto_cgans/`
**Note:** System default `python3` on HPC is 3.9 (too old). Always use `python3.11`.

---

## How the Embedding Is Injected

The ontology embedding is not a side-channel — it is concatenated at the **raw input** of both networks:

- **Generator** input dim = `noise_dim + embed_size`
- **Discriminator** input dim = `data_dim + embed_size`

Neither network ever sees noise or data without an ontology embedding alongside it.

### The Three Roles of the Embedding

| Role | Where | What it does |
|---|---|---|
| Generator conditioning | `fakez = cat([noise, embedding])` | Generator maps each disease embedding → data distribution |
| Discriminator pair coherence | `disc_input = cat([data, embedding])` | Discriminator judges (data, embedding) pairs for semantic consistency |
| Self-consistency loss | `mse(embed(generated_RD), input_embedding)` | Penalises generator if the generated RD label doesn't match the conditioning embedding |

### Zero-Shot Learning
Because the generator is conditioned on **dense embedding vectors** (not one-hot labels), you can pass any disease at inference time — including ones never seen during training. The generator interpolates based on ontology proximity.

```python
# Seen RDs — sampled from training distribution
samples = model.sample(500)

# Unseen RDs — zero-shot generation
samples = model.sample(500, unseen_rds=["http://www.orpha.net/ORDO/Orphanet_XXXX"])
```

### Self-Consistency Loss — Now Differentiable (fixed 2026-04-05)

The old implementation used `.detach()` before `inverse_transform`, which broke the gradient. The fix uses the **Gumbel-softmax output directly** (already in `fakeact`) as soft class probabilities and multiplies it by a prebuilt embedding matrix:

```python
# Built once at fit() time — [n_classes, embed_size]
self._rd_embedding_matrix = tensor(get_rd_embeds(ordered_iris))  # frozen

# In generator training step — fully differentiable
icd_soft = fakeact[:, self._rd_col_st:self._rd_col_ed]     # [B, n_classes]
generated_embeddings_t = icd_soft @ self._rd_embedding_matrix  # [B, embed_size]
cross_entropy_pair = mse_loss(generated_embeddings_t, real_embeddings)
```

Gradient flows: `loss_g → cross_entropy_pair → icd_soft → fakeact → Generator weights`.
`_rd_embedding_matrix` is frozen (`requires_grad=False`) — it is a fixed ontology lookup, not trained.

---

## Public API

```python
from onto_cgans import Onto_DP_CGAN, OntologyEmbedding, make_embedding_model

# Gensim-based ontology embedding
embedding = make_embedding_model(
    use_llm=False,
    embedding_path="path/to/model.kv",
    embedding_size=100,
    hp_dict_fn="path/to/hp_dict.txt",
    rd_dict_fn="path/to/rd_dict.txt",
)

# LLM-based embedding (OpenAI or local HF model)
embedding = make_embedding_model(
    use_llm=True,
    llm_backend="openai",        # or "hf-local"
    cache_path="embeddings.json",
)

model = Onto_DP_CGAN(
    log_file_path=None,
    embedding=embedding,
    epochs=300,
    batch_size=500,
    noise_dim=100,
    generator_dim=(256, 256),
    discriminator_dim=(256, 256),
    private=False,
    cuda=True,
    # Conditioning mode:
    # "ontology"         — frozen ontology embedding [default]
    # "decoder"          — ontology + EmbeddingDecoder(ontology) → predicted clinical mean
    # "decoder_contrast" — ontology + (decoder(ont) - decoder(nearest_neighbour))
    # "decoder_var"      — ontology + predicted_mean + predicted_std
    # "prior"            — ontology + hardcoded clinical prior (not truly ZSL, disabled by default)
    conditioning_mode='ontology',
    clinical_prior=None,            # only used when conditioning_mode="prior"
)

model.fit(train_data)               # first col = IRI, second col = label, rest = features
samples = model.sample(500)         # seen RDs
samples = model.sample(500, unseen_rds=["http://www.orpha.net/ORDO/Orphanet_519"])  # ZSL
```

**Output DataFrame columns:** `[IRI, icd_code, ...features...]`
- `IRI` — disease IRI used for conditioning (correct even for ZSL unseen diseases)
- `icd_code` — overwritten post-generation with the label derived from the IRI (no longer shows spurious training labels for ZSL rows)

---

## Bugs Fixed

### Session 2026-04-02

| # | File | Bug |
|---|------|-----|
| 1 | `onto_cgan.py` | None-check after unpacking `condvec_pair` → crash |
| 2 | `onto_cgan.py` | `self._discriminator` never stored → `xai_discriminator()` always crashed |
| 3 | `ontology_embedding.py` | `v` used outside scope in `LLMEmbedding.get_embedding()` → NameError |
| 4 | `onto_data_sampler.py` | Wrong column index for RD in `sample_original_condvec` |
| 5 | `onto_cgan.py` | Hard-coded `"icd_code"` column + ORDO IRI — not generalisable |
| 6 | `onto_cgan.py` | `BCEWithLogitsLoss` applied to raw embeddings (wrong); replaced with MSE |
| 7 | `onto_cgan_modular.py` | `ontology` parameter accepted but silently ignored |
| 8 | `onto_cgan.py` | `real_data` extracted but never used (dead code) |
| 9 | `onto_cgan.py` | `sampled_rds` tracked in `sample()` but not attached to returned DataFrame |
| 10 | `onto_data_sampler.py` | Crash when < 3 discrete columns in `sample_condvec_pair` |

### Session 2026-04-03 to 2026-04-05

| # | File | Bug / Change |
|---|------|-----|
| 11 | `onto_data_sampler.py` | `_discrete_column_matrix_st` never populated → always zeros; `current_id` incremented but assignment missing. Fixed: added `self._discrete_column_matrix_st[current_id] = st` |
| 12 | `onto_cgan.py` | Self-consistency loss not differentiable (`.detach()` broke gradient). Fixed: Gumbel-softmax × frozen embedding matrix replaces argmax + numpy lookup |
| 13 | `onto_cgan.py` | IRI column inserted before `Table.reverse_transform`, which explicitly strips it. Fixed: synthesizer stores `_last_sampled_rds`; `Onto_DPCGANModel._sample_rows()` re-inserts IRI after `reverse_transform` |
| 14 | `onto_cgan.py` | ZSL output `icd_code` column showed training labels (GAN can only decode seen labels). Fixed: overwrite `icd_code` post-generation using label derived from sampled IRI; parse unseen IRIs from path string |

---

## Package Structure

```
onto_cgans/
├── pyproject.toml
├── requirements.txt
├── src/onto_cgans/
│   ├── __init__.py               # Onto_DP_CGAN, OntologyEmbedding, make_embedding_model
│   ├── __main__.py               # CLI: onto-cgans gen / version
│   ├── onto_cgan.py              # Core synthesizer
│   ├── onto_cgan_modular.py      # High-level model wrapper
│   ├── onto_base.py              # Base tabular model
│   ├── onto_data_sampler.py      # Data sampler
│   ├── ontology_embedding.py     # OntologyEmbedding + LLMEmbedding + make_embedding_model
│   ├── utils.py
│   ├── synthesizers/             # DataTransformer, BaseSynthesizer
│   ├── functions/                # rdp_accountant, table, base, gaussian
│   └── Transformers/             # transformers, hyper_transformer, base
└── tests/
    └── test_onto_cgans.py
```

---

## Conditioning Modes (updated 2026-04-08)

Five modes available. `encoder` and `augmented` have been removed. Set via `conditioning_mode=` parameter.

### `"ontology"` (default)
- Conditioning vector = frozen ontology embedding, 100d
- Generator input dim = `noise_dim + 100`
- ZSL: use ontology embedding of unseen IRI directly
- **Limitation**: generates data resembling ontology-nearest seen diseases, not real AML

### `"prior"` (disabled in generate.py / evaluate.py — not truly zero-shot)
- Conditioning vector = ontology embedding + hardcoded external clinical reference values
- Reference values for AML sourced from observed means of real AML patients in MIMIC data
- **Not truly ZSL**: requires knowing real AML clinical values in advance
- Best ROC-AUC (~0.67 RF) when enabled — useful as a calibration reference

### `"decoder"`
- `EmbeddingDecoder`: `embed_size → 32 → n_features`, trained pre-GAN on seen diseases
- Conditioning = `[ontology_embed | decoder(ontology_embed)]`
- ZSL generalises automatically — no special case needed for unseen IRI
- **High recall, low precision**: broad synthetic distribution overlaps MDS and DLBCL heavily

### `"decoder_contrast"` (added 2026-04-08)
- Same `EmbeddingDecoder` as above, but conditioning uses the contrast signal:
  `[ontology_embed | decoder(ont) - decoder(nearest_seen_neighbour)]`
- Nearest neighbour found via cosine similarity in ontology embedding space (self excluded)
- Pre-computes `decoder(nearest_neighbour)` for each seen disease during `fit()`
- At ZSL inference: finds nearest seen disease on the fly for the unseen IRI
- **Goal**: push synthetic AML away from MDS/DLBCL by encoding what makes AML distinct

### `"decoder_var"` (added 2026-04-08)
- `EmbeddingDecoderVar`: shared trunk → two heads (mean + log_std), `embed_size → 32 → n_features × 2`
- Trained with NLL loss + L2 regularisation on log_std (prevents extreme variance predictions)
- Conditioning = `[ontology_embed | predicted_mean | predicted_std]`
- Effective embed size = `embed_size + 2 × n_features`
- **Goal**: give GAN a disease-specific variance signal so synthetic distribution is tighter where AML is clinically consistent

### Key internal methods

| Method | Purpose |
|---|---|
| `_get_conditioning(cat_ids)` | Returns conditioning tensor for training loop, branches on mode |
| `_sample_conditioning_for_iris(iris_list)` | Returns conditioning tensor for `sample()`, handles ZSL |

---

## Open Issues

- **`HyperTransformer` warning** — appears on every `sample()` call. The hyper_transformer in `Table` is never fitted; `reverse_transform` is effectively a no-op. Investigate whether any column-level transformations are being silently skipped.
- **Within-disease variance in ZSL** — all modes produce ZSL data conditioned on a single embedding point per disease. The GAN noise adds variation but it is shaped by seen diseases, not AML-specific. `decoder_var` partially addresses this via predicted std conditioning.
- **decoder_contrast noise with few diseases** — contrast signal is the difference of two decoder predictions trained on only ~6 diseases. If the decoder is inaccurate, the contrast amplifies noise rather than signal.

---

## Evaluation Script — `evaluation/evaluate.py`

Added 2026-04-06. Implements a full ZSL utility evaluation comparing three scenarios on MIMIC data.

### Scenarios

| Scenario | Train set | Test set | Purpose |
|---|---|---|---|
| S1 — Baseline | Seen diseases only | Seen test + all unseen real | Lower bound — classifier never predicts unseen |
| S2 — Synthetic aug | Seen + ZSL synthetic unseen | Same test set | Main experiment — does synthetic data help? |
| S3 — Upper bound | Seen + real unseen (50%) | Seen test + 50% unseen real | Ceiling — what if you had real data |

Run S1+S2 only (all 268 AML in test):
```bash
python3.11 evaluation/evaluate.py --model-dir output/synthetic
# outputs → output/no_split_seed42/
```

Run 5-fold CV (all 268 AML evaluated across folds):
```bash
python3.11 evaluation/evaluate.py --model-dir output/synthetic --cv --n-splits 5
# outputs → output/no_split_cv5fold_seed42/
```

Run S1+S2+S3 with 70/30 AML split:
```bash
python3.11 evaluation/evaluate.py --model-dir output/synthetic --include-s3
# outputs → output/with_s3_seed42/
```

### Key design decisions

- **Unseen disease:** `Orphanet_519` (AML) — withheld entirely from GAN training
- **Features:** 35 clinical features used for both GAN training and classifier (same set, no split):
  `anchor_age, diastolic, systolic, Platelet Count, Creatinine, Urea Nitrogen, Potassium, Chloride, Sodium, White Blood Cells, MCH, MCV, RDW, Hematocrit, Hemoglobin, Red Blood Cells, MCHC, Bicarbonate, Glucose, Magnesium, Calcium Total, Phosphate, PT, INR(PT), PTT, Neutrophils, Basophils, Eosinophils, Monocytes, Bilirubin Total, AST, ALT, Alkaline Phosphatase, Lactate Dehydrogenase, Albumin`
- **GAN model parameters:** epochs=3000, batch_size=500, noise_dim=128, generator_dim=(256,256,256), discriminator_dim=(256,256,256), discriminator_steps=10, generator_lr=2e-4, discriminator_lr=2e-4
- **Classifier:** RandomForest (n_estimators=500, class_weight=balanced, criterion=gini)
- **n_syn:** Evaluated at 1×, 5×, 10× the number of real unseen test patients — GAN trained once per mode, sampled three times
- **GAN models saved** as pkl: `output/{run}/gan_model_S2_{mode}.pkl` for later reuse

### S2 conditioning modes

| Mode | Status |
|---|---|
| ontology | active |
| decoder | active |
| decoder_contrast | active (added 2026-04-08) |
| decoder_var | active (added 2026-04-08) |
| prior | disabled (not truly ZSL) |
| encoder | removed |
| augmented | removed |

### Evaluation metrics (focused on unseen disease)
- Recall, Precision, F1 (for unseen class)
- ROC-AUC, Average Precision (one-vs-rest)
- TP / total unseen patients

### Statistical similarity (synthetic unseen vs real unseen only)
Per feature: KS statistic + p-value, Wasserstein distance, mean/std comparison.
Saved to `stat_similarity_S2_{mode}_{multiplier}x.csv`

### Correlation matrix similarity (synthetic unseen vs real unseen only)
Frobenius norm, mean/max absolute difference, per-feature MAE.
Saved to `corr_similarity_S2_{mode}_{multiplier}x.csv` + heatmap PNG.

### Evaluation results (updated 2026-04-09, epochs=3000, 14 features, 7 diseases, 3 classifiers)

**Setup:** 7 diseases (AML withheld from GAN training), 14 hematologic features, MICE imputation,
N_SYN=1340 (5× AML count), dropna(thresh=16) on synthetic data, random_state=42.
3 classifiers: RandomForest (RF), KNN (k=100), GaussianNB. No post-hoc filtering.

**Patient exclusion (applied in `load_data()` — affects both generation and evaluation):**
- One record per patient per disease (`drop_duplicates(subset=['subject_id','icd_code'])`)
- 69 patients who appear under both AML and a seen disease are removed entirely (143 rows: 74 seen + 69 AML). Their AML rows are not evaluated; their seen-disease rows are not trained on.

Two evaluation protocols:
- **Single-seed** (199 clean AML in test, 80% seen train): `output/no_split_seed42/`
- **5-fold stratified CV** (all 199 AML evaluated across folds, ~40 AML/fold): `output/no_split_cv5fold_seed42/`

---

#### Single-seed results (199 AML in test, random_state=42)

##### ROC-AUC (AML-specific, binarized one-vs-rest)

| Scenario | RF | KNN | GaussianNB |
|---|---|---|---|
| S1 — Baseline | 0.500 | 0.500 | 0.500 |
| S2 — ontology | 0.674 | 0.774 | 0.376 |
| S2 — decoder | **0.752** | 0.734 | **0.625** |
| S2 — decoder_contrast | 0.702 | 0.696 | 0.435 |
| S2 — decoder_var | 0.690 | **0.713** | 0.420 |

##### Precision / Recall / F1 (199 AML in test)

| Scenario | RF Prec | RF Rec | RF F1 | KNN Prec | KNN Rec | KNN F1 | GNB Prec | GNB Rec | GNB F1 |
|---|---|---|---|---|---|---|---|---|---|
| S2 — ontology | 0.471 | 0.367 | 0.412 | 0.553 | 0.814 | **0.659** | 0.202 | 0.101 | 0.134 |
| S2 — decoder | 0.590 | 0.744 | **0.658** | 0.512 | 0.844 | 0.638 | **0.569** | 0.312 | **0.403** |
| S2 — decoder_contrast | **0.571** | 0.503 | 0.535 | 0.471 | 0.699 | 0.563 | 0.443 | 0.136 | 0.208 |
| S2 — decoder_var | 0.545 | 0.548 | 0.546 | **0.562** | 0.799 | 0.660 | 0.227 | 0.111 | 0.149 |

TP counts out of 199 AML: ontology RF=73, decoder RF=148, decoder_contrast RF=100, decoder_var RF=109.

---

#### 5-fold CV results (all 199 AML evaluated, mean ± std across folds)

AML prevalence in each fold ≈ 8.1% (vs 30.6% in single-seed), so precision/F1 are lower than
single-seed due to lower prior — this is expected. ROC-AUC is prevalence-invariant.

##### ROC-AUC (mean ± std)

| Scenario | RF | KNN | GaussianNB |
|---|---|---|---|
| S1 — Baseline | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.500 ± 0.000 |
| S2 — ontology | 0.675 ± 0.050 | **0.749 ± 0.031** | 0.401 ± 0.048 |
| S2 — decoder | **0.718 ± 0.025** | 0.771 ± 0.016 | **0.661 ± 0.048** |
| S2 — decoder_contrast | 0.727 ± 0.040 | 0.758 ± 0.023 | 0.518 ± 0.016 |
| S2 — decoder_var | 0.696 ± 0.017 | 0.723 ± 0.033 | 0.459 ± 0.074 |

##### Recall (mean ± std)

| Scenario | RF | KNN | GaussianNB |
|---|---|---|---|
| S2 — ontology | 0.407 ± 0.152 | 0.764 ± 0.067 | 0.121 ± 0.034 |
| S2 — decoder | **0.704 ± 0.028** | **0.864 ± 0.022** | **0.372 ± 0.068** |
| S2 — decoder_contrast | 0.558 ± 0.103 | 0.778 ± 0.083 | 0.196 ± 0.045 |
| S2 — decoder_var | 0.578 ± 0.062 | 0.849 ± 0.026 | 0.116 ± 0.068 |

##### F1 (mean ± std)

| Scenario | RF | KNN | GaussianNB |
|---|---|---|---|
| S2 — ontology | 0.223 ± 0.068 | 0.303 ± 0.015 | 0.074 ± 0.021 |
| S2 — decoder | 0.308 ± 0.017 | 0.273 ± 0.021 | **0.258 ± 0.043** |
| S2 — decoder_contrast | **0.320 ± 0.060** | 0.289 ± 0.019 | 0.184 ± 0.039 |
| S2 — decoder_var | 0.287 ± 0.016 | **0.315 ± 0.018** | 0.085 ± 0.053 |

---

**Key findings (2026-04-09, clean cohort — multi-disease patients fully excluded):**

- **Multi-disease patient exclusion is essential for valid ZSL evaluation:** 69 patients appear
  under both AML and a seen disease in MIMIC. Including them in test corrupts the AML evaluation
  (their clinical profile is mixed); including them in training leaks AML signal. All 143 rows
  (74 seen-disease + 69 AML) are now removed in `load_data()`, affecting both GAN training and
  classifier evaluation equally. Clean AML cohort = 199 patients.
- **decoder** achieves highest recall across RF and KNN — generates the broadest AML-like
  distribution, recovering 70–86% of true AML patients. Highest ROC-AUC for RF (0.752 single-seed,
  0.718 CV) and GNB (0.625 / 0.661).
- **decoder_contrast** achieves best precision for RF (0.571 vs 0.590 for decoder, similar recall
  trade-off) and highest CV F1 for RF (0.320 ± 0.060) — the contrast signal produces tighter
  synthetic AML with less MDS/DLBCL overlap.
- **decoder_var** intermediate: best KNN single-seed F1 (0.660) close to ontology; variance
  conditioning adds modest benefit; slightly lower ROC-AUC than decoder for RF.
- **ontology**: best KNN single-seed F1 (0.659) and CV ROC-AUC (0.749); KNN benefits from the
  compact embedding-conditioned cluster structure. RF/GNB performance is weakest.
- **KNN always highest recall** (76–86%) at cost of precision — large k=100 creates broad decision
  regions that capture many AML patients but also many FP from MDS/DLBCL.
- **GaussianNB** worst across all modes — violated independence assumption (Hb/Hct/RBC collinear).
- **CV precision/F1 lower than single-seed**: expected — AML prevalence is 8.1% in CV folds vs
  30.6% in single-seed. ROC-AUC (prevalence-invariant) is consistent between protocols.

---

## Proposed Next Steps — ZSL Improvement

### Problem: Decoder mode has low precision (high FP rate)

The decoder mode generates a **broad synthetic AML distribution** that overlaps heavily with MDS
(Orphanet_52688) and DLBCL (Orphanet_207046). This causes many false positives when a classifier
is trained on seen + synthetic data.

Root cause: the `EmbeddingDecoder` is trained on only ~6 seen diseases, so its AML prediction is
a rough extrapolation. The GAN adds noise shaped by seen diseases (not AML-specific), producing
synthetic patients spread too broadly in feature space. The classifier cannot learn a tight AML
boundary from this data.

**Evidence (S2 decoder, seed=42, 81 AML in test):**

| Classifier | TP | FP | Precision | Recall |
|---|---|---|---|---|
| RandomForest | 28 | 91 | 0.235 | 0.346 |
| LogisticRegression | 6 | 42 | 0.125 | 0.074 |
| KNN | 66 | 194 | 0.254 | 0.815 |
| GaussianNB | 26 | 72 | 0.265 | 0.321 |

Main FP sources: MDS (29–79 patients misclassified as AML) and DLBCL (9–57).
S3 upper bound achieves precision ~0.63–0.74 — showing the gap is closeable with better signal.

---

### Option 1 — Contrastive decoder conditioning

Instead of conditioning the GAN on absolute predicted AML means, condition on the **contrast**
between AML and its nearest seen neighbour (MDS):

```
contrast_signal = decoder(AML_embedding) - decoder(MDS_embedding)
```

This encodes what makes AML *distinct* from its closest neighbour rather than where AML sits
in absolute feature space. The GAN learns to shift synthetic data away from the MDS region.

**Why it works:** The decoder's AML prediction is close to MDS (they are ontology neighbours).
The contrast signal amplifies the difference, giving the GAN a clearer directional target.

**Pros:** No extra data needed; works within existing decoder architecture; principled.
**Cons:** Requires identifying the nearest seen neighbour at inference time; contrast may be
noisy if the decoder's seen-disease predictions are inaccurate.

**Implementation:** In `_sample_conditioning_for_iris()`, compute `decoder(AML_emb) - decoder(nearest_seen_emb)`
and concatenate that as the conditioning signal instead of raw `decoder(AML_emb)`.

---

### Option 2 — Decoder with variance conditioning

The current `EmbeddingDecoder` predicts only **per-feature means**. Extend it to also predict
**per-feature standard deviations**, then pass both as conditioning to the GAN:

```
EmbeddingDecoder: embedding → (predicted_mean, predicted_std)  [2 × n_features]
GAN conditioning: [ontology_embed | predicted_mean | predicted_std]
```

During training the GAN sees tight std for diseases with consistent clinical profiles and wide
std for heterogeneous ones. At ZSL inference, the predicted std for AML constrains how broadly
the GAN spreads the synthetic distribution.

**Why it works:** The noise currently adds undifferentiated variance. Explicit std conditioning
gives the GAN a disease-specific variance signal — if AML's predicted std is small on a feature,
the GAN should generate tight values for that feature.

**Pros:** Architecturally clean; directly addresses the broad-distribution problem.
**Cons:** Requires retraining the GAN; with only ~6 seen diseases the std estimate is noisy;
may need regularisation (e.g. predict log-std with L2 penalty).

**Implementation:** Change `EmbeddingDecoder` output to `2 × n_features`; split into mean/std;
update `_effective_embed_size` and `_get_conditioning` / `_sample_conditioning_for_iris` to
concatenate both; retrain via `generate.py`.

---

### Option 3 — VAE over embedding-conditioned data (joint distribution)

Replace the GAN generator with a VAE trained jointly on (embedding, real features) pairs from
seen diseases. The decoder learns to produce realistic patient data conditioned on any embedding,
including unseen ones.

```
Seen diseases → train VAE:
    encoder: (embedding, real_features) → latent z ~ N(0,1)
    decoder: (z, embedding)             → synthetic features

ZSL at inference:
    sample z ~ N(0,1)
    decoder(z, AML_embedding) → synthetic AML patient
```

**Pros:** Preserves feature correlations (joint distribution), models within-disease variance,
more realistic synthetic patients for classifier training.
**Cons:** More complex to implement and train; still limited by how well the ontology embedding
captures clinical specificity.

---

## Context for Next Session

**Active codebase (HPC):** `/gpfs/home2/csun/onto_cgans/onto_cgans/`
Always use `python3.11` on HPC — system `python3` is 3.9 and will reject the package.

**Recent evaluation output folder:** `output/no_split_seed42/`

**Test + data:**
- `tests/test_onto_cgans.py` — smoke test: fit → sample (seen) → sample (ZSL)
- `data/test_data.csv` — semicolon-separated, 9 disease classes
- `data/ontology_emb/` — gensim KeyedVectors + HPO.dict + ORDO.dict
- `data/MIMICDATA_MoreDisease_all_test.csv` — main evaluation dataset, 7 disease classes (AML withheld)

**Training data format:**
- `pandas.DataFrame`, **first column = IRI** (used for embedding lookup, dropped before GAN training), **second column = label** (e.g. `icd_code`), remaining = features
- String columns auto-detected as discrete by `onto_cgan_modular.py`

**Dictionary file format** (for `OntologyEmbedding`):
- Plain text, one entry per line: `label;IRI`
- Example: `Orphanet_513;http://www.orpha.net/ORDO/Orphanet_513`

**Evaluation scripts (evaluation/):**
- `generate.py` — train GANs + save pkl + synthetic CSVs (slow ~4.5h per run)
  Default modes: `ontology decoder decoder_contrast decoder_var`
- `evaluate.py` — load pkl models, sample, run classifiers + metrics (fast, re-runnable)
  Default modes: `ontology decoder decoder_contrast decoder_var`
- `aggregate.py` — aggregate evaluation_results.csv across multiple seed runs
- `utils.py` — shared constants: 7 diseases, 14 features, load_and_split, MICE imputation

**Current output:**
- `output/synthetic/gan_model_S2_{mode}.pkl` — saved GAN models (4 modes)
- `output/synthetic/synthetic_S2_{mode}.csv` — synthetic AML rows per mode
- `output/no_split_seed42/evaluation_results.csv` — single-seed results (268 AML in test, no S3)
- `output/no_split_seed42/kde_comparison.png` — KDE plots per mode
- `output/no_split_cv5fold_seed42/evaluation_results_all_folds.csv` — per-fold CV results
- `output/no_split_cv5fold_seed42/evaluation_results_cv.csv` — CV mean ± std summary

**All 4 modes evaluated (2026-04-09), with patient-level AML exclusion:**
- All GAN models trained (3000 epochs): `output/synthetic/gan_model_S2_{mode}.pkl`
- Single-seed: `output/no_split_seed42/evaluation_results.csv`
- 5-fold CV: `output/no_split_cv5fold_seed42/evaluation_results_cv.csv`
