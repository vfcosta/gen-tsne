import argparse
from gen_tsne.gen_tsne import calculate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Apply Gen t-SNE metric.')
    parser.add_argument('-b', '--baseline', help='Path to images from the dataset (baseline)', required=False)
    parser.add_argument('-p', '--paths', action='append', help='Paths to images from generative models', required=True)
    parser.add_argument('-o', '--output', help='Output dir', default="./output")
    parser.add_argument('-r', "--rows", type=int, help='rows', default=60)
    parser.add_argument('-c', "--cols", type=int, help='cols', default=60)
    parser.add_argument('-k', "--perplexity", type=int, help='perplexity', default=30)
    parser.add_argument('-n', "--iter", type=int, help='iterations', default=1000)
    parser.add_argument('-f', "--use-features", default=False, action='store_true',
                        help='Use features to build the grid (<IMAGE_NAME>.npy or <IMAGE_NAME>.npz)')
    parser.add_argument('-s', '--size', nargs='+', type=int, help='Resize image', default=None)
    parser.add_argument('-d', '--dim-reduction', default="tsne",
                        help='Algorithm for dimensionality reduction (tsne|umap|parametric_umap)')
    parser.add_argument('-m', '--model-file', help='Trained model file')
    parser.add_argument('-j', "--jitter-win", type=int, help='jitter to reposition overlapping images', default=0)
    parser.add_argument('-t', "--calc-rmse", default=False, action='store_true', help='Calculate rmse')
    parser.add_argument('-e', "--executions", default=1, type=int, help='Number of runs for calculating the metric')
    args = parser.parse_args()
    calculate(([args.baseline] if args.baseline else []) + args.paths, args.output, frow=args.rows, fcol=args.cols,
              perplexity=args.perplexity, n_iter=args.iter, use_features=args.use_features, resize=args.size,
              dim_reduction=args.dim_reduction, model_file=args.model_file, jitter_win=args.jitter_win,
              enable_rmse=args.calc_rmse, executions=args.executions)
