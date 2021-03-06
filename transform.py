import argparse
import logging
import pickle

import numpy as np
from PIL import Image
from umap.parametric_umap import load_ParametricUMAP

from gen_tsne.grid import load_features

logger = logging.getLogger(__name__)


def transform(paths, model_file, resize):
    logger.info("loading model")
    if model_file.endswith("umap"):
        model = load_ParametricUMAP(model_file)
    else:
        with open(model_file, "rb") as f:
            model = pickle.load(f)
    points = {}
    for f in paths:
        if f.endswith(".npz"):
            features = np.expand_dims(load_features(f), axis=0)
        else:
            image = Image.open(f)
            if resize:
                image = image.resize(resize)
            features = np.expand_dims(np.array(image) / 255, axis=0).reshape((1, -1))
        points[f] = model.transform(features)[0].tolist()
    logger.info("points: %s", points)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Use a fitted model to transform a image into coordinates.')
    parser.add_argument('-p', '--paths', action='append', help='Paths to images', required=True)
    parser.add_argument('-m', '--model-file', help='Path to the model', default="dim_reduction.pkl")
    parser.add_argument('-s', '--size', nargs='+', type=int, help='Resize image', default=None)
    args = parser.parse_args()
    transform(args.paths, args.model_file, args.size)
