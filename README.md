# Onto-CGAN (Ontology-enhanced Conditional Generative Adversarial Network)


**Abstract**: This repository presents a novel generative framework (Onto-CGAN) that combines knowledge drawn from disease ontologies, with conditional GAN models to generate unseen data for diseases not present in the training data. Our pre-print publication: [Generating Unseen Diseases Patient Data Using Ontology-enhanced Generative Adversarial Networks](https://www.researchsquare.com/article/rs-5043150/v1).

**Author**: Chang Sun, Institute of Data Science, Maastricht University

**Start date**: March-2024

**Status**: Continue developing [This repo keeps updating]


## 📥️ Installation 
To be updated (with a simple and runnable example)

You will need Python >=3.8+ and <=3.11

### 🐍 Use with python 

This library can also be used directly in python scripts

If your input is tabular data (e.g., csv):

 ```python
import os
import numpy as np
import pandas as pd
from datetime import datetime
from onto_cgan import Onto_CGAN, OntologyEmbedding

tabular_data=pd.read_csv("../YOUR_DATA_FILE.csv")

### Load your ontology embeddings ###
onto_embedding = OntologyEmbedding(embedding_path='ontology/ontology.embeddings',
                                   embedding_size=100,
                                   hp_dict_fn='ontology/HPO.dict',
                                   rd_dict_fn='ontology/ORDO.dict')

epochs = 2000
model = Onto_CGAN(
    embedding=onto_embedding,
    log_file_path=generated_files_path,
    epochs=epochs,
    batch_size=700,
    log_frequency=True,
    verbose=True,
    noise_dim=100,
    generator_dim=(256, 256, 256),
    discriminator_dim=(256, 256, 256),
    generator_lr=2e-5,
    discriminator_lr=2e-5,
    discriminator_steps=10,
    private=False,
    wandb=True
)

print(f'Start model training, for {epochs} epochs')
model.fit(real_data_rest)


now = datetime.now()
current_time = now.strftime("%Y_%m_%d_%H_%M_%S")
print('Training finished, saving the model')
# Saving the model for future sampling
model.save(f'{generated_files_path}/{current_time}_{epochs}_epochs_onto_cgan_unseen.pkl')

# Give the label of the unseen disease 
picked_unseen_rds=["http://www.orpha.net/ORDO/Orphanet_519",]
# ZSL sampling (unseen embeddings)
if len(picked_unseen_rds) > 0:
    fn = os.path.join(generated_files_path, f'{current_time}_{epochs}_epochs_unseen_sample_{nb_rows}_rows_519.csv')
    print(f'Sampling {nb_rows} unseen rows')
    sampled_unseen = model.sample(nb_rows, unseen_rds=picked_unseen_rds).to_csv(fn)
 ```

