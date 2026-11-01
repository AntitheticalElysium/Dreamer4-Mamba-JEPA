"""Read-only source/checkpoint access, separate outputs for observational diagnosis."""
import importlib
import json
import hashlib
from pathlib import Path
import numpy as np
import torch
from saccade import data as D,model as M,train as TR
from saccade.eval import s1,gates as G

previous=importlib.import_module('artifacts.experiments.20261028_saccade_p1_dissection.common')
OLD=previous.OUT
OUT=Path(__file__).resolve().parent
ROOT,RUN,CKPT,READER=previous.ROOT,previous.RUN,previous.CKPT,previous.READER
HUD,HUD_CELLS=previous.HUD,previous.HUD_CELLS
labels,hud_values,episode_rows,crop_cells=previous.labels,previous.hud_values,previous.episode_rows,previous.crop_cells
load_model,model_hash,fingerprints=previous.load_model,previous.model_hash,previous.fingerprints


def save(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n')


def load_cache(split):return torch.load(OLD/f'real_{split}.pt',weights_only=False,mmap=True)


def load_crops(split):return torch.load(OLD/f'crops_{split}.pt',weights_only=False,mmap=True)
