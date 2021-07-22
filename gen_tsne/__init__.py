import logging

from gen_tsne.grid import build as build_grid
from gen_tsne.metric import calc_jaccard_index, calc_rmse

logging.basicConfig(level=logging.INFO)
