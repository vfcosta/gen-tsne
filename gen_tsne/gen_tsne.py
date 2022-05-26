import logging
import os

from gen_tsne import build_grid, calc_jaccard_index, calc_rmse

logger = logging.getLogger(__name__)


def calculate(paths, output_dir, enable_rmse=True, pca_components=None, frow=60, fcol=60, perplexity=30,
              n_iter=1000, save_data=True, use_features=False, images_pattern=("*.png",), resize=None):
    df, image_shape = build_grid(
        paths, pca_components=pca_components, frow=frow, fcol=fcol, perplexity=perplexity, n_iter=n_iter,
        save_data=save_data, output_dir=output_dir, use_features=use_features, images_pattern=images_pattern,
        resize=resize)
    distance_threshold, stats_df = calc_jaccard_index(df)
    logger.info("stats %s", stats_df)
    stats_df.to_csv(os.path.join(output_dir, "stats.csv"), index=False)
    if enable_rmse:
        df_distances = calc_rmse(df, image_shape)
        df_distances.to_csv(os.path.join(output_dir, "distances.csv"), index=False)
    return stats_df
