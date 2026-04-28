# Onto-CGAN

**Ontology-conditioned Conditional GAN for synthetic tabular data generation.**

Onto-CGAN extends [DP-CGANS](https://github.com/sunchang0124/dp_cgans) by replacing standard one-hot category conditions with dense ontology embedding vectors. This enables **zero-shot learning (ZSL)**: the model can generate realistic synthetic patients for diseases it has never seen during training, by leveraging semantic relationships encoded in biomedical ontologies (ORDO, HPO).

---

## How It Works

The ontology embedding is concatenated at the **raw input** of both the generator and discriminator — neither network sees noise or data without an embedding alongside it.

| Component | Input | Role |
|---|---|---|
| Generator | `[noise \| embedding]` | Maps each disease embedding to a data distribution |
| Discriminator | `[data \| embedding]` | Judges `(data, embedding)` pairs for semantic consistency |
| Self-consistency loss | `mse(soft_embed(generated_label), input_embedding)` | Penalises the generator if the generated label doesn't match the conditioning embedding |

Because the generator is conditioned on **dense embedding vectors** rather than one-hot labels, you can pass any disease IRI at inference time — including diseases never seen during training — and the generator interpolates based on ontology proximity.

---

## Installation

```bash
# Basic install
pip install -e /path/to/onto_cgans/

# With LLM / HuggingFace embedding support
pip install -e /path/to/onto_cgans/[llm]
```

> **HPC note:** Use `python3.11` explicitly — the system default `python3` may be too old.

---

## Quick Start

```python
from onto_cgans import Onto_DP_CGAN, make_embedding_model

# 1. Create an embedding backend (choose one — see Embedding Backends below)
embedding = make_embedding_model(
    owl_path="embeddings/hpObo_hoom_ordo.owl",
    model_name="FremyCompany/BioLORD-2023",
    cache_path="embeddings/cache_biolord.json",
)

# 2. Instantiate the model
model = Onto_DP_CGAN(
    log_file_path=None,
    embedding=embedding,
    epochs=300,
    batch_size=500,
    noise_dim=128,
    generator_dim=(256, 256, 256),
    discriminator_dim=(256, 256, 256),
    discriminator_steps=10,
    cuda=True,
    conditioning_mode='decoder',   # see Conditioning Modes below
)

# 3. Fit
# train_data: DataFrame where col[0]=IRI, col[1]=label, remaining cols=features
model.fit(train_data)

# 4. Sample seen diseases
synthetic = model.sample(500)

# 5. Zero-shot: generate data for an unseen disease
synthetic_zsl = model.sample(500, unseen_rds=["http://www.orpha.net/ORDO/Orphanet_519"])
```

**Output DataFrame columns:** `[IRI, label, ...features...]`
- `IRI` — disease IRI used for conditioning (correct even for ZSL unseen diseases)
- `label` — disease label derived from the IRI

---

## Training Data Format

`pandas.DataFrame` with this column order:

| Position | Content | Example |
|---|---|---|
| col[0] | Disease IRI | `http://www.orpha.net/ORDO/Orphanet_519` |
| col[1] | Disease label | `icd_code` or `label` |
| col[2+] | Clinical features | age, lab values, … |

String columns are automatically detected as discrete (categorical); numeric columns are treated as continuous.

---

## Conditioning Modes

Set via `conditioning_mode=` in the constructor.

### `"ontology"` (default)
Uses the frozen ontology embedding directly as the conditioning vector.

- Simple and fast — no pre-training step required
- ZSL uses the unseen disease's ontology embedding directly
- Works best when ontology proximity correlates with clinical similarity

### `"decoder"`
Trains a small `EmbeddingDecoder` network (`embed_size → 128 → 64 → n_features`) on seen diseases before GAN training. Conditioning = `[ontology_embed | decoder(ontology_embed)]`.

- Decoder predicts the clinical feature means for each disease from its embedding
- ZSL generalises automatically — no special case needed for unseen IRIs
- Achieves higher recall than `ontology` in most evaluations


---

## Embeddings

All embeddings share the same interface: `get_embedding(iri: str) → np.ndarray`.
Pass any backend as `embedding=` to `Onto_DP_CGAN`.

### OntologyEmbedding

Used OWL2VEC* (which uses a pre-trained gensim KeyedVectors model trained with graph walks over ORDO + HPO). You can find the trained embeddings here (https://doi.org/10.6084/m9.figshare.27959826)

```python
embedding = make_embedding_model(
    embedding_path="embeddings/ontology.embeddings",
    embedding_size=100,
    hp_dict_fn="embeddings/HPO.dict",
    rd_dict_fn="embeddings/ORDO.dict",
)
```

### LLM Embeddings

Parses a combined OWL file (ORDO + HPO) and encodes disease labels and definitions using a sentence-transformers model. Results are cached to a JSON file.

```python
from onto_cgans import make_embedding_model

# Hierarchy-aware (HiT — trained on SnomedCT)
embedding = make_embedding_model(
    owl_path="embeddings/hpObo_hoom_ordo.owl",
    model_name="Hierarchy-Transformers/HiT-MiniLM-L12-SnomedCT",
    cache_path="embeddings/cache_hit.json",
)

# Biomedical entity linking (SapBERT)
embedding = make_embedding_model(
    owl_path="embeddings/hpObo_hoom_ordo.owl",
    model_name="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
    cache_path="embeddings/cache_sapbert.json",
)

# Rare-disease concepts (BioLORD-2023)
embedding = make_embedding_model(
    owl_path="embeddings/hpObo_hoom_ordo.owl",
    model_name="FremyCompany/BioLORD-2023",
    include_definition=True,
    cache_path="embeddings/cache_biolord.json",
)

# General biomedical text (PubMedBERT)
embedding = make_embedding_model(
    owl_path="embeddings/hpObo_hoom_ordo.owl",
    model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    cache_path="embeddings/cache_pubmedbert.json",
)
```

Dictionary file format — one entry per line: `label;IRI`


## Evaluation

The `evaluation/` directory contains scripts for a full ZSL utility evaluation.

### Training models

```bash
# Interactive — prompts for dataset path, unseen IRI, epochs, etc.
python3.11 evaluation/generate.py

# Any cgan-format CSV (IRI | label | features)
python3.11 evaluation/generate.py \
    --data-path /path/to/cgan_data.csv \
    --unseen-iri 519                      # plain ID, Orphanet_519, or full IRI

# Named experiment (pre-registered in evaluation/utils.py CONFIGS)
python3.11 evaluation/generate.py --disease aml --epochs 3000

# Subset of conditioning modes
python3.11 evaluation/generate.py --data-path data.csv --unseen-iri 519 \
    --modes ontology decoder

# Quick smoke test (epochs=10, batch_size=50)
python3.11 evaluation/generate.py --data-path data.csv --unseen-iri 519 --quick
```

`generate.py` only trains and saves GAN models as `.pkl` files. Sampling happens in `evaluate.py`, so you can re-sample with different seeds or multipliers without re-training.

### Running evaluation

```bash
# Single-seed evaluation (S1 + S2)
python3.11 evaluation/evaluate.py --model-dir output/aml/synthetic

# 5-fold cross-validation
python3.11 evaluation/evaluate.py --model-dir output/aml/synthetic --cv --n-splits 5

# Include S3 upper bound
python3.11 evaluation/evaluate.py --model-dir output/aml/synthetic --include-s3

# Named disease
python3.11 evaluation/evaluate.py --disease psc --model-dir output/psc/synthetic
```

### Evaluation scenarios

| Scenario | Train set | Test set | Purpose |
|---|---|---|---|
| S1 — Baseline | Seen diseases only | All test patients | Lower bound — classifier never predicts unseen disease |
| S2 — Synthetic augmentation | Seen + ZSL synthetic | All test patients | Main experiment |
| S3 — Upper bound | Seen + real unseen (50%) | Seen + 50% unseen | Ceiling with real data |

### Metrics

- Recall, Precision, F1 (unseen disease class)
- ROC-AUC, Average Precision (one-vs-rest)
- Per-feature KS statistic, Wasserstein distance, mean/std (statistical similarity)
- Frobenius norm, per-feature MAE of correlation matrix (correlation similarity)

### Output layout

```
output/{disease}/
├── synthetic/
│   ├── gan_model_S2_{mode}.pkl     # trained GAN model
│   └── synthetic_S2_{mode}.csv    # synthetic samples
├── no_split_seed42/
│   └── evaluation_results.csv
└── no_split_cv5fold_seed42/
    └── evaluation_results_cv.csv
```

### Adding a new disease

Add one entry to `CONFIGS` in `evaluation/utils.py`:

```python
CONFIGS["my_disease"] = ExperimentConfig(
    name="my_disease",
    data_path=Path("/path/to/cgan_my_disease.csv"),
    unseen_iri="http://www.orpha.net/ORDO/Orphanet_XXXX",
    label_col="label",
    feat_cols=None,   # None = auto-detect all non-meta columns
)
```

Then run `generate.py --disease my_disease` and `evaluate.py --disease my_disease`.

---

## Package Structure

```
onto_cgans/
├── pyproject.toml
├── embeddings/                       # ontology files — not tracked in git, populate locally
│   ├── HPO.dict
│   ├── ORDO.dict
│   ├── hpObo_hoom_ordo.owl
│   ├── cache_*.json                  # embedding caches (generated on first run)
│   └── ontology.embeddings (+ .npy) # gensim model (legacy backend)
├── src/onto_cgans/
│   ├── __init__.py                   # public API: Onto_DP_CGAN, make_embedding_model, ...
│   ├── __main__.py                   # CLI: onto-cgans gen / version
│   ├── onto_cgan.py                  # core synthesizer + EmbeddingDecoder
│   ├── onto_cgan_modular.py          # high-level model wrapper (Onto_DP_CGAN)
│   ├── onto_base.py                  # base tabular model
│   ├── onto_data_sampler.py          # data sampler
│   ├── ontology_embedding.py         # OntologyEmbedding, OWLEmbedding, LLMEmbedding
│   ├── synthesizers/                 # DataTransformer, BaseSynthesizer
│   ├── functions/                    # rdp_accountant, gaussian mechanism
│   └── Transformers/                 # HyperTransformer, column transformers
└── evaluation/
    ├── generate.py                   # train GANs + save .pkl
    ├── evaluate.py                   # sample + classifiers + metrics
    └── utils.py                      # ExperimentConfig, CONFIGS, shared loaders
```

---

## License

MIT License. See `LICENSE.txt`.
