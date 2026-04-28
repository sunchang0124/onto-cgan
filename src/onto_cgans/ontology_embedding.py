
from gensim.models import KeyedVectors
from typing import Optional, Dict
import numpy as np
import os, json, re

# ---------------------------
# Existing OntologyEmbedding
# ---------------------------
class OntologyEmbedding:
    """Ontology embedding class using Gensim KeyedVectors (with JSON cache)."""
    def __init__(self, embedding_path, embedding_size, hp_dict_fn, rd_dict_fn,
                 cache_path: Optional[str] = None):
        self._embed_model = KeyedVectors.load(embedding_path)
        self.embed_size = getattr(self._embed_model, "vector_size", embedding_size)
        self.cache = _LocalEmbeddingCache(cache_path) if cache_path else None  # NEW

        self._iri_dict = {}
        with open(hp_dict_fn) as f:
            for line in f:
                (entity, iri) = line.strip().split(';')
                self._iri_dict[entity] = iri

        with open(rd_dict_fn) as f:
            for line in f:
                (entity, iri) = line.strip().split(';')
                self._iri_dict[entity] = iri

    def get_embedding(self, disease):
        key = disease #" ".join(str(entity).split())                      # NEW (stable key)
        # print(self._embed_model.wv["http://www.orpha.net/ORDO/Orphanet_513"])
        # 1) cache hit
        if self.cache:
            cached = self.cache.get(key)
            if cached is not None:
                return cached

        # 2) fetch from KeyedVectors
        try:
            vec = self._embed_model.wv[key]
            
        except KeyError:
            vec = np.zeros(self.embed_size, dtype=np.float32)

        # 3) persist
        if self.cache:
            self.cache.put(key, vec)
            self.cache.save()

        return vec

# class OntologyEmbedding:
#     """Ontology embedding class using Gensim KeyedVectors."""
#     def __init__(self, embedding_path, embedding_size, hp_dict_fn, rd_dict_fn):
#         self._embed_model = KeyedVectors.load(embedding_path)
#         self.embed_size = getattr(self._embed_model, "vector_size", embedding_size)
#         self._iri_dict = {}

#         with open(hp_dict_fn) as f:
#             for line in f:
#                 (entity, iri) = line.strip().split(';')
#                 self._iri_dict[entity] = iri

#         with open(rd_dict_fn) as f:
#             for line in f:
#                 (entity, iri) = line.strip().split(';')
#                 self._iri_dict[entity] = iri

#     def get_embedding(self, entity):
#         # If you later want to try IRI fallbacks, add them here.
#         try:
#             return self._embed_model.wv[entity]
#         except KeyError:
#             return np.zeros(self.embed_size, dtype=np.float32)


# ---------------------------
# Local JSON cache helper
# ---------------------------
class _LocalEmbeddingCache:
    def __init__(self, path: str ):
        self.path = path
        self._m = {}
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self._m = {k: np.array(v, dtype=np.float32) for k, v in raw.items()}

    def get(self, k):
        return self._m.get(k) if self._m is not None else None

    def put(self, k, v: np.ndarray):
        if self._m is not None:
            self._m[k] = v.astype(np.float32)

    def save(self):
        if self.path and self._m is not None:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump({k: v.tolist() for k, v in self._m.items()}, f)


# ---------------------------
# LLMEmbedding (OpenAI or SentenceTransformers)
# ---------------------------
class LLMEmbedding:
    """
    LLM-based embedding class with optional local persistence.
    - backend="openai": uses OpenAI embeddings (API)
    - backend="phi" or "sentence-transformers": local model (no API)
    """
    def __init__(
        self,
        backend: str = "openai",
        cache_path: str = "llm_embeddings.json",
        normalize: bool = False,
        **backend_kwargs
    ):
        self.backend = backend
        self.normalize = normalize
        self.cache = _LocalEmbeddingCache(cache_path)

        if backend == "openai":
            # backend_kwargs: model="text-embedding-3-small" | "text-embedding-3-large", etc.
            from openai import OpenAI
            self._client = OpenAI()
            self._model = backend_kwargs.get("model", "text-embedding-3-small")
            self.embed_size = 3072 if "large" in self._model else 1536
            self._encode_fn = self._encode_openai

        # elif backend in ("sentence-transformers", "st"):
        #     # backend_kwargs: model_name="sentence-transformers/all-MiniLM-L6-v2", ...
        #     from sentence_transformers import SentenceTransformer
        #     model_name = backend_kwargs.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
        #     self._st_model = SentenceTransformer(model_name)
        #     self.embed_size = int(self._st_model.get_sentence_embedding_dimension())
        #     self._encode_fn = self._encode_st

        elif backend in ("hf-local", "huggingface-local"):
            # backend_kwargs:
            #   model_name_or_path (str)  -> HF repo id or local dir (required)
            #   device (str)              -> "cuda" | "cpu" (default: auto)
            #   dtype (str)               -> "float16" | "bfloat16" | "float32" (default: float32)
            #   pooling (str)             -> "mean" | "cls" (default: "mean")
            #   max_length (int)          -> tokenizer max length (default: 512)
            #   instruction (str)         -> optional prefix (e.g., "passage: " for E5 models)
            from transformers import AutoModel, AutoTokenizer
            import torch

            model_name = backend_kwargs.get("model_name_or_path")
            if not model_name:
                raise ValueError("hf-local backend requires backend_kwargs['model_name_or_path'].")

            device = backend_kwargs.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")

            dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
            dtype = dtype_map.get(str(backend_kwargs.get("dtype", "float32")).lower(), torch.float32)

            self._hf_pooling = str(backend_kwargs.get("pooling", "mean")).lower()
            self._hf_max_len = int(backend_kwargs.get("max_length", 512))
            self._hf_instruction = backend_kwargs.get("instruction", "")

            self._hf_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            self._hf_model = AutoModel.from_pretrained(model_name, torch_dtype=dtype).to(device).eval()
            self._hf_device = device

            hidden_size = getattr(self._hf_model.config, "hidden_size", None)
            if hidden_size is None:
                with torch.inference_mode():
                    toks = self._hf_tokenizer("probe", return_tensors="pt", truncation=True, max_length=16)
                    toks = {k: v.to(self._hf_device) for k, v in toks.items()}
                    out = self._hf_model(**toks)
                    hidden_size = out.last_hidden_state.shape[-1]
            self.embed_size = int(hidden_size)

            self._encode_fn = self._encode_hf_local


        else:
            raise ValueError("backend must be 'openai' or 'other huggingface models'")

    def _encode_hf_local(self, text: str) -> np.ndarray:
        import torch, numpy as np
        prompt = f"{self._hf_instruction}{text}" if self._hf_instruction else text
        toks = self._hf_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self._hf_max_len)
        toks = {k: v.to(self._hf_device) for k, v in toks.items()}
        with torch.inference_mode():
            out = self._hf_model(**toks)
            hidden = out.last_hidden_state  # [1, T, H]
            if self._hf_pooling == "cls":
                vec = hidden[:, 0, :]
            else:
                attn = toks.get("attention_mask", torch.ones(hidden.shape[:2], device=hidden.device)).unsqueeze(-1)
                vec = (hidden * attn).sum(dim=1) / attn.sum(dim=1).clamp(min=1)
        return vec[0].detach().cpu().to(torch.float32).numpy()

    
    
    def _normalize(self, v: np.ndarray) -> np.ndarray:
        if not self.normalize:
            return v
        n = np.linalg.norm(v)
        return v / (n + 1e-12)

    def _encode_openai(self, text: str) -> np.ndarray:
        text = " ".join(str(text).split())
        resp = self._client.embeddings.create(model=self._model, input=text)
        return np.array(resp.data[0].embedding, dtype=np.float32)

    def _encode_st(self, text: str) -> np.ndarray:
        return self._st_model.encode([str(text)], convert_to_numpy=True)[0].astype(np.float32)

    def get_embedding(self, entity) -> np.ndarray:
        if isinstance(entity, str):
            key = " ".join(str(entity).split())

        elif isinstance(entity, tuple) and len(entity) == 2:
            # (key, value) from a dict iteration
            k, v = entity
            # you decide which one to embed — here I’ll embed the value
            key = " ".join(str(v).split())

        else:
            raise ValueError("Entity must be a str or a (key,value) tuple.")

        # 1) cache
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        # 2) compute
        try:
            vec = self._encode_fn(key)
        except Exception:
            # Fail-safe: consistent shape zeros
            return np.zeros(self.embed_size, dtype=np.float32)

        vec = self._normalize(vec)

        # 3) persist
        self.cache.put(key, vec)
        self.cache.save()
        return vec


# ---------------------------
# OWLEmbedding
# ---------------------------

# Pattern for ontology codes that are not human-readable names, e.g. "ORPHA:519", "HP:0001234"
_CODE_PATTERN = re.compile(r'^[A-Z]+:\d+$')


def _best_label(labels: list) -> Optional[str]:
    """Return the first label that is not an ontology code (e.g. ORPHA:54057)."""
    clean = [l for l in labels if not _CODE_PATTERN.match(str(l).strip())]
    if clean:
        return str(clean[0]).strip()
    if labels:
        return str(labels[0]).strip()
    return None


class OWLEmbedding:
    """Embed ontology classes from an OWL file using a sentence-transformers model.

    Parses the OWL file once with owlready2, builds an IRI → text mapping
    (label + optional definition + optional synonyms), then encodes each IRI
    on demand using the specified sentence-transformers model.

    This single class covers:
      - Hierarchy-aware embeddings:  model_name="Hierarchy-Transformers/HiT-MiniLM-L12-ORDO"
      - Biomedical entity linking:   model_name="cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
      - Rare-disease concepts:       model_name="FremyCompany/BioLORD-2023"
      - General biomedical text:     model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
      - General text baseline:       model_name="sentence-transformers/all-MiniLM-L6-v2"

    Parameters
    ----------
    owl_path : str
        Path to the combined OWL file (e.g. hpObo_hoom_ordo.owl).
    model_name : str
        HuggingFace model name or local path for sentence-transformers.
    include_definition : bool
        Append the class definition (IAO:0000115) to the label text. Default True.
    include_synonyms : bool
        Append alternative_term / synonym values. Default False.
    fallback_labels : dict
        {IRI: label} for any IRI missing from the OWL file. Useful for classes
        that were added after the OWL snapshot.
    cache_path : str
        Path to a JSON file for caching computed embeddings across runs.
    normalize : bool
        L2-normalise embeddings. Default False.
    namespaces : list of str
        Only index classes whose IRI contains one of these strings.
        Default ["Orphanet", "HP_"] — excludes HOOM mapping nodes.
    """

    def __init__(
        self,
        owl_path: str,
        model_name: str,
        include_definition: bool = True,
        include_synonyms: bool = False,
        fallback_labels: Optional[Dict[str, str]] = None,
        cache_path: Optional[str] = None,
        normalize: bool = False,
        namespaces: Optional[list] = None,
    ):
        self._normalize = normalize
        self._cache = _LocalEmbeddingCache(cache_path) if cache_path else None
        self._fallback = fallback_labels or {}

        # ── Load OWL and build IRI → text map ─────────────────────────────────
        _ns = namespaces if namespaces is not None else ["Orphanet", "HP_"]
        self._iri_to_text = self._build_text_map(
            owl_path, _ns, include_definition, include_synonyms)
        print(f"  OWLEmbedding: indexed {len(self._iri_to_text)} classes from {owl_path}")

        # ── Load sentence-transformers model ───────────────────────────────────
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)
        self.embed_size = self._model.get_sentence_embedding_dimension()
        print(f"  OWLEmbedding: model={model_name!r}  embed_size={self.embed_size}")

    @staticmethod
    def _build_text_map(
        owl_path: str,
        namespaces: list,
        include_definition: bool,
        include_synonyms: bool,
    ) -> Dict[str, str]:
        """Parse the OWL file and return {IRI: text_for_encoding}."""
        import owlready2
        onto = owlready2.get_ontology(owl_path).load()

        iri_to_text = {}
        for cls in onto.classes():
            iri = cls.iri
            if not any(ns in iri for ns in namespaces):
                continue

            # ── Label ──────────────────────────────────────────────────────────
            label = _best_label(list(cls.label))
            if not label:
                # fall back to comment if no label
                comments = list(cls.comment)
                label = str(comments[0]).strip() if comments else None
            if not label:
                continue  # skip classes with no usable name

            parts = [label]

            # ── Synonyms ───────────────────────────────────────────────────────
            if include_synonyms:
                syns = []
                for attr in ("alternative_term", "hasExactSynonym",
                             "hasBroadSynonym", "hasNarrowSynonym"):
                    syns += [str(s) for s in getattr(cls, attr, [])]
                if syns:
                    parts.append("Also known as: " + ", ".join(syns[:3]) + ".")

            # ── Definition ─────────────────────────────────────────────────────
            if include_definition:
                defn = getattr(cls, "definition", None)
                if defn:
                    defn_str = str(list(defn)[0]).strip() if hasattr(defn, "__iter__") else str(defn).strip()
                    if defn_str:
                        parts.append(defn_str)

            iri_to_text[iri] = " ".join(parts)

        return iri_to_text

    def _encode(self, text: str) -> np.ndarray:
        vec = self._model.encode(text, convert_to_numpy=True,
                                 show_progress_bar=False).astype(np.float32)
        if self._normalize:
            vec = vec / (np.linalg.norm(vec) + 1e-12)
        return vec

    def get_embedding(self, iri: str) -> np.ndarray:
        key = iri.strip()

        # 1) cache hit
        if self._cache:
            cached = self._cache.get(key)
            if cached is not None:
                return cached

        # 2) build text
        text = self._iri_to_text.get(key)
        if text is None:
            # fallback label provided by caller
            text = self._fallback.get(key)
        if text is None:
            # last resort: use the local name from the IRI
            text = key.rstrip("/").split("/")[-1].replace("_", " ")

        # 3) encode
        vec = self._encode(text)

        # 4) persist
        if self._cache:
            self._cache.put(key, vec)
            self._cache.save()

        return vec


# ---------------------------
# Simple factory helper
# ---------------------------
def make_embedding_model(
    use_llm: bool = False,
    *,
    # OntologyEmbedding (gensim, legacy) args
    embedding_path: Optional[str] = None,
    embedding_size: Optional[int] = None,
    hp_dict_fn: Optional[str] = None,
    rd_dict_fn: Optional[str] = None,
    # OWLEmbedding args  ← new
    owl_path: Optional[str] = None,
    model_name: Optional[str] = None,
    include_definition: bool = True,
    include_synonyms: bool = False,
    fallback_labels: Optional[Dict[str, str]] = None,
    namespaces: Optional[list] = None,
    # Shared args
    cache_path: Optional[str] = None,
    normalize: bool = False,
    # LLMEmbedding (OpenAI / hf-local) args — kept for backward compat
    llm_backend: str = "openai",
    **llm_kwargs
):
    """Return an embedding model with a unified get_embedding(iri) → np.ndarray interface.

    Three backends
    --------------
    1. OWLEmbedding (recommended — OWL file + sentence-transformers):
       Provide owl_path + model_name.

       Examples::

           # Hierarchy-aware (HiT, trained on ORDO)
           make_embedding_model(
               owl_path="data/ontology_emb/hpObo_hoom_ordo.owl",
               model_name="Hierarchy-Transformers/HiT-MiniLM-L12-ORDO",
               cache_path="cache_hit.json",
           )

           # Biomedical entity linking (SapBERT)
           make_embedding_model(
               owl_path="data/ontology_emb/hpObo_hoom_ordo.owl",
               model_name="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
               cache_path="cache_sapbert.json",
           )

           # Rare-disease concepts (BioLORD-2023)
           make_embedding_model(
               owl_path="data/ontology_emb/hpObo_hoom_ordo.owl",
               model_name="FremyCompany/BioLORD-2023",
               include_definition=True,
               cache_path="cache_biolord.json",
           )

           # General biomedical text (PubMedBERT)
           make_embedding_model(
               owl_path="data/ontology_emb/hpObo_hoom_ordo.owl",
               model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
               cache_path="cache_pubmedbert.json",
           )

    2. OntologyEmbedding (legacy gensim KeyedVectors):
       Provide embedding_path + embedding_size + hp_dict_fn + rd_dict_fn.

    3. LLMEmbedding (OpenAI API or raw HuggingFace):
       Set use_llm=True + llm_backend.
    """
    if owl_path and model_name:
        return OWLEmbedding(
            owl_path=owl_path,
            model_name=model_name,
            include_definition=include_definition,
            include_synonyms=include_synonyms,
            fallback_labels=fallback_labels,
            cache_path=cache_path,
            normalize=normalize,
            namespaces=namespaces,
        )
    elif use_llm:
        return LLMEmbedding(
            backend=llm_backend,
            cache_path=cache_path,
            normalize=normalize,
            **llm_kwargs
        )
    else:
        if not all([embedding_path, embedding_size, hp_dict_fn, rd_dict_fn]):
            raise ValueError(
                "Provide either (owl_path + model_name) for OWLEmbedding, "
                "or (embedding_path + embedding_size + hp_dict_fn + rd_dict_fn) "
                "for the legacy gensim OntologyEmbedding."
            )
        return OntologyEmbedding(
            embedding_path, embedding_size, hp_dict_fn, rd_dict_fn,
            cache_path=cache_path
        )




# from gensim.models import KeyedVectors
# import numpy as np


# class OntologyEmbedding():
#     """Ontology embedding class.

#     This class is specifically designed for embeddings with the RDs + HPs datasets.

#     Args:
#     embedding_path (str):
#         Path to the embeddings file, to be loaded with KeyedVectors (.embedding file).
#     embedding_size (int):
#         Size of each embedding vector.
#     hp_dict_fn (str):
#         Path to the HP label -> HP URI dictionary file.
#     rd_dict_fn (str):
#         Path to the RD label -> RD URI dictionary file.
#     """

#     def __init__(self, embedding_path, embedding_size, hp_dict_fn, rd_dict_fn):
#         self._embed_model = KeyedVectors.load(embedding_path)
#         self.embed_size = embedding_size
#         self._iri_dict = {}

#         with open(hp_dict_fn) as f:
#             for line in f:
#                 (entity, iri) = line.strip().split(';')
#                 self._iri_dict[entity] = iri

#         with open(rd_dict_fn) as f:
#             for line in f:
#                 (entity, iri) = line.strip().split(';')
#                 self._iri_dict[entity] = iri



#     def get_embedding(self, entity):
#         """Get embedding of entity.

#         Args:
#             entity (str):
#                 Label or URI of the class to get the embedding of.
#         Returns:
#             (numpy.ndarray): Embedding of the entity, array full of zeros if embedding not found.
#         """
#         # iri = self.get_iri(entity)
#         # if iri is not None:
#         #     return self._embed_model.wv[iri]

#         # print("http://www.orpha.net/ORDO/Orphanet_"+str(int(entity)))

#         # try: 
#         return self._embed_model.wv[entity]#["http://www.orpha.net/ORDO/Orphanet_"+str(int(entity))]
#         # except:
#         #     print("The disease iri %s is not found in the embeddings." %entity)
#         #     return np.zeros(self.embed_size)
