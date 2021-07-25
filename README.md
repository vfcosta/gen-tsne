# Gen TSNE: Visualization and Metric for generative models

[![CI](https://github.com/vfcosta/gen-tsne/actions/workflows/ci.yml/badge.svg)](https://github.com/vfcosta/gen-tsne/actions/workflows/ci.yml)

## Instructions

First, put images from the dataset and from generative models into different folders.

Then, start the process with the following command:
```
python main.py -b <DATASET IMAGES> -p <IMAGES FROM MODEL 1> -p <IMAGES FROM MODEL 2>
```

Execute `python main.py --help` to see more options.

If you want to use features instead of image pixels in the grid calculation (`-f` argument), the directory should follow the same structure used in [test/assets/dataset](/test/assets/dataset) and [test/assets/model_a](/test/assets/model_a), i.e. store `.npz` (or `.npy`) files with the same name as each image that you want to evaluate.
