"""Smoke test for onto_cgans: fit → sample → ZSL sample."""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from onto_cgans import Onto_DP_CGAN, OntologyEmbedding

OUTPUT_DIR = Path(__file__).parent.parent / "output"

DATA_DIR  = Path(__file__).parent.parent / "data"
EMB_DIR   = DATA_DIR / "ontology_emb"

ICD_TO_IRI = {
    2050.0: "http://www.orpha.net/ORDO/Orphanet_519",
    2040.0: "http://www.orpha.net/ORDO/Orphanet_513",
    2840.0: "http://www.orpha.net/ORDO/Orphanet_68383",
    2842.0: "http://www.orpha.net/ORDO/Orphanet_314399",
    2041.0: "http://www.orpha.net/ORDO/Orphanet_67038",
    2387.0: "http://www.orpha.net/ORDO/Orphanet_52688",
    2007.0: "http://www.orpha.net/ORDO/Orphanet_544",
    2005.0: "http://www.orpha.net/ORDO/Orphanet_46135",
    2028.0: "http://www.orpha.net/ORDO/Orphanet_207046",
}
ICD_TO_LABEL = {
    2050.0: "ORDO.Orphanet_519",
    2040.0: "ORDO.Orphanet_513",
    2840.0: "ORDO.Orphanet_68383",
    2842.0: "ORDO.Orphanet_314399",
    2041.0: "ORDO.Orphanet_67038",
    2387.0: "ORDO.Orphanet_52688",
    2007.0: "ORDO.Orphanet_544",
    2005.0: "ORDO.Orphanet_46135",
    2028.0: "ORDO.Orphanet_207046",
}
UNSEEN_RD = "http://www.orpha.net/ORDO/Orphanet_519"


def load_data():
    df = pd.read_csv(DATA_DIR / "test_data.csv", sep=';', index_col="Unnamed: 0")
    df['icd_code'] = df['icd_code'].apply(lambda x: np.floor(x / 10) if x > 9999 else np.floor(x))
    df = df.drop_duplicates().drop("subject_id", axis=1)
    df.replace({'gender': 0}, 'Male', inplace=True)
    df.replace({'gender': 1}, 'Female', inplace=True)
    df = df[df["icd_code"].isin(ICD_TO_IRI.keys())]
    df = df[['icd_code', 'gender', 'anchor_age', 'Weight', 'BMI',
             'diastolic', 'systolic', 'Platelet Count', 'Hematocrit']]
    df.insert(0, 'IRI', df['icd_code'].map(ICD_TO_IRI))
    df.replace({'icd_code': ICD_TO_LABEL}, inplace=True)
    train = df[df['icd_code'] != "ORDO.Orphanet_519"].copy()
    return train.head(200)   # cap at 200 rows for speed


def test_fit_sample_zsl():
    embedding = OntologyEmbedding(
        embedding_path=str(EMB_DIR / "ontology.embeddings"),
        embedding_size=100,
        hp_dict_fn=str(EMB_DIR / "HPO.dict"),
        rd_dict_fn=str(EMB_DIR / "ORDO.dict"),
    )

    train = load_data()
    print(f"\nTraining rows: {len(train)}, diseases: {train['icd_code'].unique()}")

    model = Onto_DP_CGAN(
        log_file_path=None,
        embedding=embedding,
        epochs=3,
        batch_size=50,
        noise_dim=32,
        generator_dim=(32, 32),
        discriminator_dim=(32, 32),
        generator_lr=2e-4,
        discriminator_lr=2e-4,
        private=False,
        cuda=False,
        verbose=True,
    )

    model.fit(train)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Seen RDs
    seen = model.sample(10)
    assert isinstance(seen, pd.DataFrame) and len(seen) == 10
    seen_path = OUTPUT_DIR / f"{ts}_seen_samples.csv"
    seen.to_csv(seen_path, index=False)
    print(f"Seen sample OK — shape {seen.shape} → {seen_path}")

    # ZSL — AML never seen during training
    zsl = model.sample(10, unseen_rds=[UNSEEN_RD])
    assert isinstance(zsl, pd.DataFrame) and len(zsl) == 10
    zsl_path = OUTPUT_DIR / f"{ts}_zsl_samples.csv"
    zsl.to_csv(zsl_path, index=False)
    print(f"ZSL sample OK  — shape {zsl.shape} → {zsl_path}")


if __name__ == "__main__":
    test_fit_sample_zsl()
    print("All checks passed.")
