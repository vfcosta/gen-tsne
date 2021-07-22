import argparse
from gen_tsne.gen_tsne import calculate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Apply Gen t-SNE metric.')
    parser.add_argument('-b', '--baseline', help='Path to images from the dataset (baseline)', required=True)
    parser.add_argument('-p', '--paths', action='append', help='Paths to images from generative models', required=True)
    parser.add_argument('-o', '--output', help='Output dir', default="./output")
    parser.add_argument('-r', "--rows", type=int, help='rows', default=60)
    parser.add_argument('-c', "--cols", type=int, help='cols', default=60)
    parser.add_argument('-k', "--perplexity", type=int, help='perplexity', default=30)
    parser.add_argument('-n', "--iter", type=int, help='iterations', default=1000)
    args = parser.parse_args()
    calculate([args.baseline] + args.paths, args.output, frow=args.rows, fcol=args.cols, perplexity=args.perplexity,
              n_iter=args.iter)
