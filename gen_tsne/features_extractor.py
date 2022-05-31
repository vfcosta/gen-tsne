import argparse
import logging
import os
from glob import glob

import numpy as np
from piq.feature_extractors import InceptionV3
from torchvision.io import read_image
from tqdm import tqdm

logger = logging.getLogger(__name__)

inception_model = None


def build_inception_model():
    """Build the inception model without normalization (convert the range from [0, 1] to [-1, 1]"""
    return InceptionV3(normalize_input=False)


def extract(image):
    global inception_model
    if inception_model is None:
        inception_model = build_inception_model()
    inception_model = inception_model.cuda()
    inception_model.eval()
    return inception_model(image)[0].detach().cpu().numpy().squeeze()


def extract_all(paths):
    logger.info("extracting features from paths %s", paths)
    for path in paths:
        images = glob(path)
        for p in tqdm(images):
            image = (read_image(p)/127.5) - 1
            features = extract(image.cuda().unsqueeze(0))
            np.savez_compressed(os.path.splitext(p)[0], features)
            logger.debug("activations shape: %s", features.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate features from image using InceptionV3.')
    parser.add_argument('-p', '--paths', action='append', help='Paths to images from generative models', required=True)
    args = parser.parse_args()
    extract_all(args.paths)
