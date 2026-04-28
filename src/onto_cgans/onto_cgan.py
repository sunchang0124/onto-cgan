# pos_weight in BCEWithLogitsLoss model #
# sigma = 5 #
# delta = 2e-6 #

import warnings
import os

import joblib
import numpy as np
import pandas as pd
import math
import random
import torch
from packaging import version
from torch import optim
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, ReLU, Sequential, functional, BCEWithLogitsLoss
from tqdm import tqdm

from onto_cgans.onto_data_sampler import Onto_DataSampler
from onto_cgans.synthesizers.data_transformer import DataTransformer
from onto_cgans.synthesizers.base import BaseSynthesizer


######## ADDED ########

from onto_cgans.functions.rdp_accountant import compute_rdp, get_privacy_spent

######## ADDED - wandb ########
from types import SimpleNamespace
from pathlib import Path

class Discriminator(Module):

    def __init__(self, input_dim, discriminator_dim, pac=10):
        super(Discriminator, self).__init__()
        dim = input_dim * pac
        self.pac = pac
        self.pacdim = dim
        seq = []
        for item in list(discriminator_dim):
            seq += [Linear(dim, item), LeakyReLU(0.2), Dropout(0.5)]
            dim = item

        seq += [Linear(dim, 1)]
        self.seq = Sequential(*seq)

    def calc_gradient_penalty(self, real_data, fake_data, device='cpu', pac=10, lambda_=10):
        alpha = torch.rand(real_data.size(0) // pac, 1, 1, device=device)
        alpha = alpha.repeat(1, pac, real_data.size(1))
        alpha = alpha.view(-1, real_data.size(1))

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        disc_interpolates = self(interpolates)

        gradients = torch.autograd.grad(
            outputs=disc_interpolates, inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=device),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        gradient_penalty = ((
            gradients.view(-1, pac * real_data.size(1)).norm(2, dim=1) - 1
        ) ** 2).mean() * lambda_

        return gradient_penalty

    def forward(self, input):
        assert input.size()[0] % self.pac == 0
        return self.seq(input.view(-1, self.pacdim))


class Residual(Module):

    def __init__(self, i, o):
        super(Residual, self).__init__()
        self.fc = Linear(i, o)
        self.bn = BatchNorm1d(o)
        self.relu = ReLU()

    def forward(self, input):
        out = self.fc(input)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input], dim=1)


class Generator(Module):

    def __init__(self, input_dim, generator_dim, data_dim):
        super(Generator, self).__init__()
        dim = input_dim
        seq = []
        for item in list(generator_dim):
            seq += [Residual(dim, item)]
            dim += item
        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)

    def forward(self, input):
        data = self.seq(input)
        return data


class EmbeddingDecoder(Module):
    """Maps ontology embedding → clinical feature means.

    Used in conditioning_mode='decoder' (Option C). Trained on seen diseases
    using (ontology_embedding, clinical_mean) pairs, then applied at ZSL time
    to predict the clinical profile of unseen diseases from their embedding.

    With few seen diseases, a small linear architecture with strong L2
    regularisation is used to avoid overfitting.
    """
    def __init__(self, embed_size, n_features, hidden_dim=128):
        super().__init__()
        self.net = Sequential(
            Linear(embed_size, hidden_dim),
            LeakyReLU(0.2),
            Linear(hidden_dim, hidden_dim // 2),
            LeakyReLU(0.2),
            Linear(hidden_dim // 2, n_features),
        )

    def forward(self, x):
        return self.net(x)


class Onto_DPCGANSynthesizer(BaseSynthesizer):
    """Conditional Table GAN Synthesizer.

    This is the core class of the CTGAN project, where the different components
    are orchestrated together.
    For more details about the process, please check the [Modeling Tabular data using
    Conditional GAN](https://arxiv.org/abs/1907.00503) paper.
    Args:
        log_file_path (str):
           Path to log the losses if verbose is True
        embedding (OntologyEmbedding):
            OntologyEmbedding instance to retrieve the ontology embeddings.
        noise_dim (int):
            Size of the random sample passed to the Generator. Defaults to 128.
        generator_dim (tuple or list of ints):
            Size of the output samples for each one of the Residuals. A Residual Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        discriminator_dim (tuple or list of ints):
            Size of the output samples for each one of the Discriminator Layers. A Linear Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        generator_lr (float):
            Learning rate for the generator. Defaults to 2e-4.
        generator_decay (float):
            Generator weight decay for the Adam Optimizer. Defaults to 1e-6.
        discriminator_lr (float):
            Learning rate for the discriminator. Defaults to 2e-4.
        discriminator_decay (float):
            Discriminator weight decay for the Adam Optimizer. Defaults to 1e-6.
        batch_size (int):
            Number of data samples to process in each step.
        discriminator_steps (int):
            Number of discriminator updates to do for each generator update.
            From the WGAN paper: https://arxiv.org/abs/1701.07875. WGAN paper
            default is 5. Default used is 1 to match original CTGAN implementation.
        log_frequency (boolean):
            Whether to use log frequency of categorical levels in conditional
            sampling. Defaults to ``True``.
        verbose (boolean):
            Whether to have print statements for progress results. Defaults to ``False``.
        epochs (int):
            Number of training epochs. Defaults to 300.
        pac (int):
            Number of samples to group together when applying the discriminator.
            Defaults to 10.
        cuda (bool):
            Whether to attempt to use cuda for GPU computation.
            If this is False or CUDA is not available, CPU will be used.
            Defaults to ``True``.
        private (bool): 
            Whether to use differential privacy
        wandb_config (dict):
            whether to use weights and bias tool to monitor the training
        conditional_columns (float):
            a matrix of embeddings
    """

    def __init__(self, log_file_path, embedding=None, noise_dim=100,
                 generator_dim=(256, 256), discriminator_dim=(256, 256),
                 generator_lr=2e-4, generator_decay=1e-6, discriminator_lr=2e-4,
                 discriminator_decay=1e-6, batch_size=500, discriminator_steps=1,
                 log_frequency=True, verbose=False, epochs=300, pac=10, cuda=True, private=False,
                 wandb=False, conditional_columns=None,
                 conditioning_mode='ontology',
                 clinical_prior=None,
                 saved_transformer=None):

        assert batch_size % 2 == 0

        self._embedding = embedding
        self._noise_dim = noise_dim

        self._log_file_path = log_file_path

        self._generator_dim = generator_dim
        self._discriminator_dim = discriminator_dim

        self._generator_lr = generator_lr
        self._generator_decay = generator_decay
        self._discriminator_lr = discriminator_lr
        self._discriminator_decay = discriminator_decay

        self._batch_size = batch_size
        self._discriminator_steps = discriminator_steps
        self._log_frequency = log_frequency
        self._verbose = verbose
        self._epochs = epochs
        self.pac = pac

        print(f'Verbose: {self._verbose}')

        self.private = private
        self.conditional_columns = conditional_columns
        self.wandb = wandb
        self._conditioning_mode = conditioning_mode
        self.saved_transformer = saved_transformer if saved_transformer is not None \
            else os.path.join(os.getcwd(), 'fitted_transformer.pkl')

        if not cuda or not torch.cuda.is_available():
            device = 'cpu'
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = 'cuda'

        if self._verbose:
            print(f'Using {device}')

        self._device = torch.device(device)

        # 👇 Add this block
        if self._device.type == "cuda":
            torch.cuda.init()                 # initialize CUDA runtime
            _ = torch.cuda.current_device()   # touch device 0
            torch.cuda.synchronize()          # ensure context is live
            print("CUDA context created on", torch.cuda.get_device_name(0))

        self._transformer = None
        self._data_sampler = None
        self._generator = None
        self._discriminator = None
    
        self.loss_values = pd.DataFrame(columns=['Epoch', 'Generator Loss', 'Distriminator Loss'])


    @staticmethod
    def _gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
        """Deals with the instability of the gumbel_softmax for older versions of torch.

        For more details about the issue:
        https://drive.google.com/file/d/1AA5wPfZ1kquaRtVruCd6BiYZGcDeNxyP/view?usp=sharing
        Args:
            logits:
                […, num_features] unnormalized log probabilities
            tau:
                non-negative scalar temperature
            hard:
                if True, the returned samples will be discretized as one-hot vectors,
                but will be differentiated as if it is the soft sample in autograd
            dim (int):
                a dimension along which softmax will be computed. Default: -1.
        Returns:
            Sampled tensor of same shape as logits from the Gumbel-Softmax distribution.
        """
        if version.parse(torch.__version__) < version.parse("1.2.0"):
            for i in range(10):
                transformed = functional.gumbel_softmax(logits, tau=tau, hard=hard,
                                                        eps=eps, dim=dim)
                if not torch.isnan(transformed).any():
                    return transformed
            raise ValueError("gumbel_softmax returning NaN.")

        return functional.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=dim)

    def _apply_activate(self, data):
        """Apply proper activation function to the output of the generator."""
        data_t = []
        st = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if span_info.activation_fn == 'tanh':
                    ed = st + span_info.dim
                    data_t.append(torch.tanh(data[:, st:ed]))
                    st = ed
                elif span_info.activation_fn == 'softmax':
                    ed = st + span_info.dim
                    transformed = self._gumbel_softmax(data[:, st:ed], tau=0.2)
                    data_t.append(transformed)
                    st = ed
                else:
                    raise ValueError(f'Unexpected activation function {span_info.activation_fn}.')

        return torch.cat(data_t, dim=1)

    def _get_conditioning(self, cat_ids):
        """Return conditioning tensor based on self._conditioning_mode.

        Args:
            cat_ids (np.ndarray): condvec batch [B, n_categories].

        Returns:
            torch.Tensor [B, effective_embed_size]
        """
        batch = len(cat_ids)
        ont_np = self._data_sampler.get_embeds_from_cat_ids(cat_ids, batch)
        ont_t = torch.from_numpy(ont_np).to(self._device, non_blocking=True)

        if self._conditioning_mode == "ontology":
            return ont_t

        elif self._conditioning_mode == "decoder":
            dec_out = self._embedding_decoder(ont_t)
            return torch.cat([ont_t, dec_out], dim=1)

        elif self._conditioning_mode == "decoder_contrast":
            irls = self._data_sampler.get_rds(cat_ids, batch)
            nn_dec_np = np.stack(
                [self._disease_nn_dec_out[iri] for iri in irls]).astype('float32')
            nn_dec_t = torch.from_numpy(nn_dec_np).to(self._device, non_blocking=True)
            dec_out = self._embedding_decoder(ont_t)
            return torch.cat([ont_t, dec_out - nn_dec_t], dim=1)

        else:
            raise ValueError(f"Unknown conditioning_mode: {self._conditioning_mode!r}")

    def _cond_loss_pair(self, data, c_pair, m_pair):

        loss=[]

        data  = torch.from_numpy(data).to(self._device)

        for each_dim in range(0, len(data[0])):
  
            criterion = BCEWithLogitsLoss(reduction='none')
            each_loss = criterion(data[:,each_dim],c_pair[:, each_dim])
            loss.append(each_loss)

        loss = torch.stack(loss, dim=1)

        return loss.sum() / len(loss[0]) / len(loss)  #(loss.sum() / len(loss[0]) / len(loss)) * 100
        
    def _validate_discrete_columns(self, train_data, discrete_columns):
        """Check whether ``discrete_columns`` exists in ``train_data``.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        if isinstance(train_data, pd.DataFrame):
            invalid_columns = set(discrete_columns) - set(train_data.columns)
        elif isinstance(train_data, np.ndarray):
            invalid_columns = []
            for column in discrete_columns:
                if column < 0 or column >= train_data.shape[1]:
                    invalid_columns.append(column)
        else:
            raise TypeError('``train_data`` should be either pd.DataFrame or np.array.')

        if invalid_columns:
            raise ValueError('Invalid columns found: {}'.format(invalid_columns))


    ############ Tensorflow Privacy Measurement ##############

    def fit(self, train_data, discrete_columns=tuple(), epochs=None):
        """Fit the CTGAN Synthesizer models to the training data.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
            epochs (int):
                Number of epochs to train the model for.
        """

        # if self.conditional_columns != None:
        #     if set(self.conditional_columns) <= set(discrete_columns):
        #         discrete_columns = self.conditional_columns
        #     else:
        #         raise NotImplementedError("Conditional columns are not in the valid columns.",discrete_columns)
        if self.wandb == True:
            config = SimpleNamespace(
                epochs=epochs, # number of training epochs
                batch_size=self._batch_size, # the size of each batch
                log_frequency=self._log_frequency,
                verbose=self._verbose,
                generator_dim=self._generator_dim,
                discriminator_dim=self._discriminator_dim,
                generator_lr=self._generator_lr,
                discriminator_lr=self._discriminator_lr,
                discriminator_steps=self._discriminator_steps, 
                private=self.private
                
            )


        self._rd_column_name = train_data.columns[0]   # IRI column (first, will be dropped)
        self._rd_label_col_name = train_data.columns[1] # label column (icd_code, stays in data)

        self._validate_discrete_columns(train_data, discrete_columns)

        if epochs is None:
            epochs = self._epochs
        else:
            warnings.warn(
                ('`epochs` argument in `fit` method has been deprecated and will be removed '
                 'in a future version. Please pass `epochs` to the constructor instead'),
                DeprecationWarning
            )

        # ── Build label→IRI mapping before IRI column is dropped ────────────────
        self._label_to_iri = dict(zip(train_data.iloc[:, 1], train_data.iloc[:, 0]))

        # Getting the list of unique RDs, sorted by order of appearance
        rds = train_data.iloc[:, 0].values
        _, idx = np.unique(rds, return_index=True)
        rds = rds[np.sort(idx)]

        iri_col_name = train_data.columns[0]

        if self.saved_transformer and os.path.exists(self.saved_transformer):
            print(f"Found existing fitted transformer - {self.saved_transformer}")
            print("Loading fitted transformer...")
            cache = joblib.load(self.saved_transformer)
            iri_transformer     = cache['iri_transformer']
            self._transformer   = cache['transformer']
            train_data_full     = cache['train_data_full']
            train_data_orig     = cache['train_data_orig']
            rds                 = cache['rds']
        else:
            # ── Fit IRI transformer separately (sampler only) ─────────────────
            iri_df = train_data[[iri_col_name]].copy()
            iri_transformer = DataTransformer()
            iri_transformer.fit(iri_df, [iri_col_name])
            train_iri = iri_transformer.transform(iri_df)

            # ── Fit main transformer on non-IRI data ──────────────────────────
            train_data = train_data.drop(columns=iri_col_name, axis=1)
            train_data_orig = train_data.copy()

            self._transformer = DataTransformer()
            self._transformer.fit(train_data, discrete_columns)
            train_data = self._transformer.transform(train_data)

            train_data_full = np.hstack([train_iri, train_data])

            cache = {
                'iri_transformer': iri_transformer,
                'transformer':     self._transformer,
                'train_data_full': train_data_full,
                'train_data_orig': train_data_orig,
                'rds':             rds,
            }
            joblib.dump(cache, self.saved_transformer)
            print(f"Saving fitted transformer to {self.saved_transformer}...")

        data_dim = self._transformer.output_dimensions

        # ── Combine output_info for sampler ──────────────────────────────────
        full_output_info_list = iri_transformer.output_info_list + self._transformer.output_info_list

        self._data_sampler = Onto_DataSampler(
            train_data_full,
            rds,
            full_output_info_list,
            self._log_frequency,
            self._embedding)

        # ── Build embedding matrix + column slice (differentiable self-consistency) ──
        st = 0
        self._rd_col_st = 0
        self._rd_col_ed = 0
        for col_info in self._transformer._column_transform_info_list:
            ed = st + col_info.output_dimensions
            if col_info.column_name == self._rd_label_col_name:
                self._rd_col_st = st
                self._rd_col_ed = ed
                ordered_labels = col_info.transform.dummies
                ordered_iris = [self._label_to_iri[lbl] for lbl in ordered_labels]
                emb_np = self._data_sampler.get_rd_embeds(ordered_iris)
                self._rd_embedding_matrix = torch.tensor(
                    emb_np, dtype=torch.float32, device=self._device
                )
                break
            st = ed

        # ── Mode-specific setup ──────────────────────────────────────────────────
        self._effective_embed_size = self._embedding.embed_size  # default

        if self._conditioning_mode in ("decoder", "decoder_contrast"):
            # Compute per-disease clinical means (shared setup for all decoder variants)
            cont_cols = [c for c in train_data_orig.columns if c not in discrete_columns]
            self._cont_cols = cont_cols
            iri_col = train_data_orig[self._rd_label_col_name].map(self._label_to_iri)
            means = {}
            for iri in ordered_iris:
                mask = iri_col == iri
                subset = train_data_orig.loc[mask, cont_cols]
                means[iri] = subset.mean().values.astype('float32')
            all_means = np.stack(list(means.values()))
            self._clinical_mean_center = all_means.mean(axis=0)
            self._clinical_mean_scale  = all_means.std(axis=0) + 1e-8
            self._disease_clinical_means = {
                iri: (v - self._clinical_mean_center) / self._clinical_mean_scale
                for iri, v in means.items()
            }
            # Build training data: (embedding, normalised_clinical_mean) per seen disease
            seen_embeds = np.stack(
                [self._data_sampler.get_rd_embeds([iri])[0] for iri in ordered_iris])
            seen_means  = np.stack(
                [self._disease_clinical_means[iri] for iri in ordered_iris])
            X_dec = torch.tensor(seen_embeds, dtype=torch.float32, device=self._device)
            Y_dec = torch.tensor(seen_means,  dtype=torch.float32, device=self._device)

            # Train EmbeddingDecoder: embedding → clinical mean
            self._embedding_decoder = EmbeddingDecoder(
                self._embedding.embed_size, len(cont_cols)
            ).to(self._device)
            opt_dec = optim.Adam(
                self._embedding_decoder.parameters(), lr=1e-3, weight_decay=1e-2)
            for _ in range(2000):
                opt_dec.zero_grad()
                pred = self._embedding_decoder(X_dec)
                loss_dec = torch.nn.functional.mse_loss(pred, Y_dec)
                loss_dec.backward()
                opt_dec.step()
            print(f"  EmbeddingDecoder trained — final MSE: {loss_dec.item():.4f}")
            self._effective_embed_size = self._embedding.embed_size + len(cont_cols)

            if self._conditioning_mode == "decoder_contrast":
                self._seen_embeds_np = seen_embeds
                self._seen_iris_list = list(ordered_iris)
                with torch.no_grad():
                    all_dec_out = self._embedding_decoder(X_dec).cpu().numpy()
                self._seen_dec_out_np = all_dec_out
                seen_norms = seen_embeds / (
                    np.linalg.norm(seen_embeds, axis=1, keepdims=True) + 1e-12)
                self._disease_nn_dec_out = {}
                for i, iri in enumerate(ordered_iris):
                    sims = seen_norms @ seen_norms[i]
                    sims[i] = -1
                    nn_idx = int(np.argmax(sims))
                    self._disease_nn_dec_out[iri] = all_dec_out[nn_idx]
                print(f"  decoder_contrast: precomputed nearest-neighbour decoder outputs.")

        # ── Networks ─────────────────────────────────────────────────────────────
        self._generator = Generator(
            self._noise_dim + self._effective_embed_size,
            self._generator_dim,
            data_dim
        ).to(self._device)

        self._discriminator = Discriminator(
            data_dim + self._effective_embed_size,
            self._discriminator_dim,
            pac=self.pac
        ).to(self._device)
        discriminator = self._discriminator

        optimizerG = optim.Adam(
            self._generator.parameters(), lr=self._generator_lr, betas=(0.5, 0.9),
            weight_decay=self._generator_decay
        )

        optimizerD = optim.Adam(
            discriminator.parameters(), lr=self._discriminator_lr,
            betas=(0.5, 0.9), weight_decay=self._discriminator_decay
        )

        mean = torch.zeros(self._batch_size, self._noise_dim, device=self._device)
        std = mean + 1

        self.loss_values = pd.DataFrame(columns=['Epoch', 'Generator Loss', 'Distriminator Loss'])
        
        epoch_iterator = tqdm(range(epochs), disable=(not self._verbose))
        if self._verbose:
            description = 'Gen. ({gen:.2f}) | Discrim. ({dis:.2f})'
            epoch_iterator.set_description(description.format(gen=0, dis=0))
            
        steps_per_epoch = max(len(train_data) // self._batch_size, 1)
        ######## ADDED ########
        
        for i in epoch_iterator:
            for step in range(steps_per_epoch):

                for n in range(self._discriminator_steps):
                    fakez = torch.normal(mean=mean, std=std)

                    condvec_pair = self._data_sampler.sample_condvec_pair(self._batch_size)

                    if condvec_pair is None:
                        c_pair_1, m_pair_1, col_pair_1, opt_pair_1 = None, None, None, None
                        real = self._data_sampler.sample_data_pair(self._batch_size, col_pair_1, opt_pair_1)
                        real = torch.from_numpy(real.astype('float32')).to(self._device, non_blocking=True)
                        fake = self._generator(fakez)
                        fakeact = self._apply_activate(fake)
                        real_cat = real
                        fake_cat = fakeact
                    else:
                        c_pair_1, m_pair_1, col_pair_1, opt_pair_1 = condvec_pair
                        perm = np.arange(self._batch_size)
                        np.random.shuffle(perm)
                        c_pair_2 = c_pair_1[perm]

                        real_embeddings = self._get_conditioning(c_pair_1)

                        fakez = torch.cat([fakez, real_embeddings], dim=1)
                        fake = self._generator(fakez)
                        fakeact = self._apply_activate(fake)

                        real = self._data_sampler.sample_data_pair(
                            self._batch_size, col_pair_1[perm], opt_pair_1[perm])
                        real = torch.from_numpy(real.astype('float32')).to(self._device, non_blocking=True)

                        real_embeddings_2 = self._get_conditioning(c_pair_2)

                        fake_cat = torch.cat([fakeact, real_embeddings], dim=1)
                        real_cat = torch.cat([real, real_embeddings_2], dim=1)
    

                    y_fake = discriminator(fake_cat)
                    y_real = discriminator(real_cat)

                    loss_d = -(torch.mean(y_real) - torch.mean(y_fake))


                    #### DP ####
                    if self.private:
                        sigma = 5
                        weight_clip = 0.01 

                        if sigma is not None:
                            for parameter in discriminator.parameters():
                                parameter.register_hook(
                                    lambda grad: grad + (1 / self._batch_size) * sigma
                                    * torch.randn(parameter.shape, device=self._device)
                                )
                    #### DP ####

                    pen = discriminator.calc_gradient_penalty(
                        real_cat, fake_cat, self._device, self.pac)

                    optimizerD.zero_grad()
                    pen.backward(retain_graph=True)  # https://machinelearningmastery.com/how-to-implement-wasserstein-loss-for-generative-adversarial-networks/ 
                    loss_d.backward()
                    optimizerD.step()

                    if self.private:
                        #### DP ####
                        # Weight clipping for privacy guarantee
                        for param in discriminator.parameters():
                            param.data.clamp_(-weight_clip, weight_clip)
                        #### DP ####

                fakez = torch.normal(mean=mean, std=std)
                condvec_pair = self._data_sampler.sample_condvec_pair(self._batch_size)

                if condvec_pair is None:
                    c_pair_1, m_pair_1, col_pair_1, opt_pair_1 = None, None, None, None
                    fake = self._generator(fakez)
                    fakeact = self._apply_activate(fake)
                    y_fake = discriminator(fakeact)
                    loss_g = -torch.mean(y_fake)
                else:
                    c_pair_1, m_pair_1, col_pair_1, opt_pair_1 = condvec_pair

                    real_embeddings = self._get_conditioning(c_pair_1)

                    c_pair_1 = torch.from_numpy(c_pair_1).to(self._device, non_blocking=True)
                    m_pair_1 = torch.from_numpy(m_pair_1).to(self._device, non_blocking=True)
                    fakez = torch.cat([fakez, real_embeddings], dim=1)

                    fake = self._generator(fakez)
                    fakeact = self._apply_activate(fake)

                    y_fake = discriminator(torch.cat([fakeact, real_embeddings], dim=1))

                    ##### ONTO-CGAN differentiable self-consistency loss #####
                    icd_soft = fakeact[:, self._rd_col_st:self._rd_col_ed]   # [B, n_classes]
                    generated_embeddings_t = icd_soft @ self._rd_embedding_matrix  # [B, embed_size]
                    # Compare against ontology embedding regardless of mode
                    ont_np = self._data_sampler.get_embeds_from_cat_ids(
                        c_pair_1.cpu().numpy(), self._batch_size)
                    ont_t = torch.from_numpy(ont_np).to(self._device, non_blocking=True)
                    cross_entropy_pair = torch.nn.functional.mse_loss(
                        generated_embeddings_t, ont_t)
                    #######################################################

                    loss_g = -torch.mean(y_fake) + cross_entropy_pair

                optimizerG.zero_grad(set_to_none=False)
                loss_g.backward()
                optimizerG.step()


            generator_loss = loss_g.detach().cpu()
            discriminator_loss = loss_d.detach().cpu()

    
            epoch_loss_df = pd.DataFrame({
                'Epoch': [i],
                'Generator Loss': [generator_loss],
                'Discriminator Loss': [discriminator_loss]
            })
            if not self.loss_values.empty:
                self.loss_values = pd.concat(
                    [self.loss_values, epoch_loss_df]
                ).reset_index(drop=True)
            else:
                self.loss_values = epoch_loss_df


                
            if self._verbose:
                epoch_iterator.set_description(
                    description.format(gen=generator_loss, dis=discriminator_loss)
                )


            if self.private:
                orders = [1 + x / 10. for x in range(1, 100)]
                sampling_probability = self._batch_size/len(train_data)
                delta = 2e-6
                rdp = compute_rdp(q=sampling_probability,
                                  noise_multiplier=sigma,
                                  steps=i * steps_per_epoch,
                                  orders=orders)
                epsilon, _, opt_order = get_privacy_spent(orders, rdp, target_delta=delta) # target_delta=1e-5

                # log_file.write(f'Differential privacy with eps = {epsilon:.3g} and delta = {delta}.\n')
                # log_file.write(f'The optimal RDP order is {opt_order}.\n')

                # if opt_order == max(orders) or opt_order == min(orders):
                #     log_file.write('The privacy estimate is likely to be improved by expanding the set of orders.\n')
            else:
                epsilon = np.nan

                    

                    

    def _sample_conditioning_for_iris(self, iris):
        """Return conditioning tensor [len(iris), effective_embed_size] for sample().

        - ontology:         frozen ontology embedding
        - decoder:          ontology + decoder(ontology) → predicted clinical mean
        - decoder_contrast: ontology + (decoder(ont) - decoder(nearest_seen))
        """
        ont_np = self._data_sampler.get_rd_embeds(iris)  # [B, embed_size]
        ont_t = torch.from_numpy(ont_np).to(self._device, non_blocking=True)

        if self._conditioning_mode == "ontology":
            return ont_t

        elif self._conditioning_mode == "decoder":
            with torch.no_grad():
                dec_out = self._embedding_decoder(ont_t)
            return torch.cat([ont_t, dec_out], dim=1)

        elif self._conditioning_mode == "decoder_contrast":
            with torch.no_grad():
                dec_out = self._embedding_decoder(ont_t)
            seen_norms = self._seen_embeds_np / (
                np.linalg.norm(self._seen_embeds_np, axis=1, keepdims=True) + 1e-12)
            nn_dec_rows = []
            for i, iri in enumerate(iris):
                if iri in self._disease_nn_dec_out:
                    nn_dec_rows.append(self._disease_nn_dec_out[iri])
                else:
                    q = ont_np[i] / (np.linalg.norm(ont_np[i]) + 1e-12)
                    sims = seen_norms @ q
                    nn_idx = int(np.argmax(sims))
                    nn_dec_rows.append(self._seen_dec_out_np[nn_idx])
            nn_dec_t = torch.tensor(
                np.stack(nn_dec_rows), dtype=torch.float32, device=self._device)
            return torch.cat([ont_t, dec_out - nn_dec_t], dim=1)

        else:
            raise ValueError(f"Unknown conditioning_mode: {self._conditioning_mode!r}")

    def sample(self, n, unseen_rds=[], sort=True):
        """Sample data similar to the training data.

        Choosing a condition_column and condition_value will increase the probability of the
        discrete condition_value happening in the condition_column.
        Args:
            n (int):
                Number of rows to sample.
            unseen_rds (list-like):
                List-like object containing names of unseen RDs to sample.
                Does not sample from seen_rds if there is at least one item in it.
            sort (bool):
                Whether to sort the resulting dataframe on RDs (alphabetically). Defaults to True.
        Returns:
            (Pandas.DataFrame):
                The sampled data.
        """

        steps = n // self._batch_size + 1
        data = []
        sampled_rds = []

        # Using unseen RDs for sampling if a list was passed as argument
        if len(unseen_rds) > 0:
            # DUplicate the rds to match batch_size
            unseen_rds = [rd for rd in unseen_rds for repetitions in range(math.ceil(self._batch_size/len(unseen_rds)))]
            unseen_rds = unseen_rds[:self._batch_size]

        for i in range(steps):
            mean = torch.zeros(self._batch_size, self._noise_dim, device=self._device)

            std = mean + 1
            fakez = torch.normal(mean=mean, std=std)

            if len(unseen_rds) > 0:
                random.shuffle(unseen_rds)
                real_embeddings = self._sample_conditioning_for_iris(unseen_rds)
                fakez = torch.cat([fakez, real_embeddings], dim=1)
                sampled_rds += unseen_rds
            else:
                condvec = self._data_sampler.sample_original_condvec(self._batch_size)

                if condvec is None:
                    pass
                else:
                    c1, m1 = condvec
                    rds = self._data_sampler.get_rds(cat_ids=c1, batch_size=self._batch_size)
                    real_embeddings = self._sample_conditioning_for_iris(rds)
                    fakez = torch.cat([fakez, real_embeddings], dim=1)
                    sampled_rds += rds

            fake = self._generator(fakez)
            fakeact = self._apply_activate(fake)

            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]
        sampled_rds = sampled_rds[:n]

        sampled_data = self._transformer.inverse_transform(data)

        # Overwrite the label column with the correct label for each sampled IRI.
        # The GAN's discrete column can only produce training labels, so for ZSL rows
        # (and for correctness in general) we derive the label from the sampled IRI.
        iri_to_label = {v: k for k, v in self._label_to_iri.items()}

        def _iri_to_label(iri):
            if iri in iri_to_label:
                return iri_to_label[iri]
            # Unseen IRI: derive label from the IRI path
            # e.g. http://www.orpha.net/ORDO/Orphanet_519 → ORDO.Orphanet_519
            token = iri.rstrip('/').split('/')[-1]
            if 'Orphanet_' in token:
                return 'ORDO.' + token
            return token

        sampled_data[self._rd_label_col_name] = [_iri_to_label(iri) for iri in sampled_rds]

        # Store sampled IRIs for re-attachment by Onto_DPCGANModel._sample_rows
        # (IRI is stripped by Table.reverse_transform, so we re-insert it after that step)
        self._last_sampled_rds = sampled_rds

        return sampled_data

    def set_device(self, device):
        self._device = torch.device(device) if not isinstance(device, torch.device) else device
        for attr in ('_generator', '_discriminator', '_embedding_decoder'):
            m = getattr(self, attr, None)
            if m is not None:
                m.to(self._device)
        if getattr(self, '_rd_embedding_matrix', None) is not None:
            self._rd_embedding_matrix = self._rd_embedding_matrix.to(self._device)

    def xai_discriminator(self, data_samples):

        # for exlain AI (SHAP) the single row from the pd.DataFrame needs to be transformed. 
        data_samples = pd.DataFrame(data_samples).T

        condvec_pair = self._data_sampler.sample_condvec_pair(len(data_samples))
        c_pair_1, m_pair_1, col_pair_1, opt_pair_1 = condvec_pair

        if condvec_pair is None:
            c_pair_1, m_pair_1, col_pair_1, opt_pair_1 = None, None, None, None
            real = self._data_sampler.sample_data_pair(len(data_samples), col_pair_1, opt_pair_1)
        else:
            c_pair_1, m_pair_1, col_pair_1, opt_pair_1 = condvec_pair
            c_pair_1 = torch.from_numpy(c_pair_1).to(self._device)
            m_pair_1 = torch.from_numpy(m_pair_1).to(self._device)

            perm = np.arange(len(data_samples))
            np.random.shuffle(perm)

            real = self._data_sampler.sample_data_pair(len(data_samples), col_pair_1[perm], opt_pair_1[perm])
            c_pair_2 = c_pair_1[perm]

        real = torch.from_numpy(real.astype('float32')).to(self._device)

        if col_pair_1 is not None:
            real_cat = torch.cat([real, c_pair_2], dim=1)
        else:
            real_cat = real

        ### Wassertein distance?? (a data point from real training data's wassertain distance means what?)
        discriminator_predict_score = self._discriminator(real_cat)

        return discriminator_predict_score