import json
import os
import sys

import torch

# Load the key from gpt_key.json (untracked) rather than hardcoding it here,
# so the secret never lands in a committed file. Checked in the repo root
# first, then its parent.
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPENAI_API_KEY = ''
for _candidate in (_repo_root, os.path.dirname(_repo_root)):
    _key_path = os.path.join(_candidate, 'gpt_key.json')
    if os.path.isfile(_key_path):
        with open(_key_path) as _f:
            OPENAI_API_KEY = json.load(_f)['gpt_key']
        break
GPT_MODEL = "gpt-5.4" #4.1
GEMINI_API_KEY = '' #YOUR GEMINI API KEY
GEMINI_MODEL = "gemini-pro-latest" #gemini-flash-latest", "2.5-flash"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Paths -----------------------------------------------------------------
# No absolute paths are baked in. Everything below is anchored to this file's
# location, so the checkout runs from any working directory and on any
# machine. Each entry honours a MOLLINGO_* environment variable override.
#
#   export MOLLINGO_DATA_DIR=/path/to/large/datasets   # default: ./data
#   export MOLLINGO_VOCAB_PATH=/path/to/vocab20.json   # default: ./toolset/vocab20.json
#   export MOLLINGO_MGLTOOLS_PATH=/path/to/conda/env   # default: the active env
#   export MOLLINGO_VINA_EXE=vina                      # name or full path

REPO_ROOT = _repo_root

# Root for the large datasets that are too big to ship in the repo
# (BPE training corpus, block search space, scaffold pairs).
DATA_DIR = os.environ.get("MOLLINGO_DATA_DIR", os.path.join(REPO_ROOT, "data"))

# Pretrained ADMET oracle checkpoints (DILI.pt, hERG.pt, ... and their
# *_norm_stats.pt). These used to be a bare relative "model_weights", so
# running from any directory other than the repo root made every ADMET
# prediction fail -- silently, because ADMET_Profile swallows the exception
# and only returns an empty dict.
MODEL_WEIGHTS_DIR = os.environ.get(
    "MOLLINGO_MODEL_WEIGHTS", os.path.join(REPO_ROOT, "model_weights")
)

# Block vocabulary for the BFE tokenizer, loaded at import time by
# toolset/tokenizer.py. Ships with the repo, so this resolves with no setup.
#
# Two builds are vendored, both with a prebuilt frequency cutoff of 20:
#   vocab20_937k.json   937,073 blocks, 256.1M occurrences  <- in use
#   vocab20.json      1,234,634 blocks, 522.7M occurrences  (larger corpus)
# They agree once normalized for corpus size, but the 937k build at
# MIN_BLOCK_FREQ=2000 is what reproduces the reference 5-block imatinib
# decomposition on the tokenizer's primary path. Reaching the same
# decomposition with the 1.23M build needs MIN_BLOCK_FREQ inside 3728-4145,
# a window fitted to one molecule, so the 937k build is preferred.
VOCAB_PATH = os.environ.get(
    "MOLLINGO_VOCAB_PATH", os.path.join(REPO_ROOT, "toolset", "vocab20_937k.json")
)

# Minimum corpus frequency for a block to stay in the vocabulary. The file is
# prebuilt with a cutoff of 20; this filters it further at load time.
# 2000 keeps 10,394 of the 937,073 blocks (1.11%) and is what yields the
# reference 5-block imatinib decomposition. Raising it does NOT give finer
# splits -- once nothing is admissible the best-effort tier in tokenizer.py
# returns the coarsest candidate instead. Block count is controlled by
# get_blocks(..., num_tokens=N), which floors it at N.
MIN_BLOCK_FREQ = int(os.environ.get("MOLLINGO_MIN_BLOCK_FREQ", 2000))

# Large datasets, used by the offline preprocessing entry points.
CORPUS_PATH = os.path.join(DATA_DIR, "drug_like.txt")            # BPE training corpus
CORPUS_ALL_PATH = os.path.join(DATA_DIR, "corpus_all.csv")       # drug-likeness filtering
VOCAB_SEARCH_SPACE_PATH = os.path.join(DATA_DIR, "vocab_search_space.pkl")
SCAFFOLD_MOLS_PATH = os.path.join(DATA_DIR, "mols_w_different_scaffolds.json")
GENERATOR_DATA_DIR = os.path.join(DATA_DIR, "generator_data")

# Conda env (or MGLTools install) providing prepare_receptor4.py /
# prepare_ligand4.py via bin/pythonsh. Defaults to the interpreter's own
# environment rather than a hardcoded path.
MGLTOOLS_PATH = os.environ.get("MOLLINGO_MGLTOOLS_PATH", sys.prefix)

# AutoDock Vina executable: a bare name is resolved on PATH.
VINA_EXE = os.environ.get("MOLLINGO_VINA_EXE", "vina")

# --- DrugCLIP (§2.3.1 de novo route) ---------------------------------------
# DrugCLIP is a third party model (ref [18]); it is not vendored here. Its
# checkpoint is 1.2 GB and its pre-encoded molecule library 1.4 GB, so the
# install stays where it is and MolLingo calls into it. It also needs its own
# environment: `unicore` lives in the `unimol` env, not in the `drugclip` one.
#
#   export MOLLINGO_DRUGCLIP_DIR=/path/to/DrugCLIP
#   export MOLLINGO_DRUGCLIP_PYTHON=/path/to/env/bin/python
DRUGCLIP_DIR = os.environ.get(
    "MOLLINGO_DRUGCLIP_DIR",
    os.path.join(os.path.dirname(os.path.dirname(REPO_ROOT)), "DrugCLIP"),
)
DRUGCLIP_PYTHON = os.environ.get(
    "MOLLINGO_DRUGCLIP_PYTHON",
    os.path.join(os.path.dirname(os.path.dirname(REPO_ROOT)),
                 "anaconda3", "envs", "unimol", "bin", "python"),
)