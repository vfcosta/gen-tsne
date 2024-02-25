import argparse
import logging
import os
from glob import glob

import clip
import numpy as np
import torch
import torchvision
from piq.feature_extractors import InceptionV3
from torchvision.io import read_image
from tqdm import tqdm

logger = logging.getLogger(__name__)

inception_model = None
clip_model = None
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_normalize = torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                  (0.26862954, 0.26130258, 0.27577711))


def build_inception_model():
    """Build the inception model without normalization (convert the range from [0, 1] to [-1, 1]"""
    return InceptionV3(normalize_input=False)


def extract_inception(image):
    global inception_model
    if inception_model is None:
        inception_model = build_inception_model()
    inception_model = inception_model.cuda()
    inception_model.eval()
    return inception_model((image/127.5) - 1)[0].detach().cpu().numpy().squeeze()


def build_clip_model():
    perceptor, preprocess = clip.load('ViT-B/32', device)
    return perceptor


def extract_clip(image):
    global clip_model
    if clip_model is None:
        clip_model = build_clip_model()
    image = torch.nn.functional.interpolate(image, (224, 224), mode='nearest') / 255
    features = clip_model.encode_image(clip_normalize(image))
    return features.detach().cpu().numpy().squeeze()


def extract_all(paths, model):
    logger.info("extracting features from paths %s", paths)
    for path in paths:
        images = glob(path)
        for p in tqdm(images):
            image = read_image(p).cuda().unsqueeze(0)
            features = extract_clip(image) if model == "clip" else extract_inception(image)
            np.savez_compressed(os.path.splitext(p)[0], features)
            logger.debug("activations shape: %s", features.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate features from image using InceptionV3.')
    parser.add_argument('-p', '--paths', action='append', help='Paths to images from generative models', required=True)
    parser.add_argument('-m', '--model', help='Model for extracting features from images (clip|inception)',
                        default="inception")
    args = parser.parse_args()
    logger.info(args)
    extract_all(args.paths, args.model)
