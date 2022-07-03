import logging
import os

import pandas as pd

from gen_tsne import build_grid, calc_jaccard_index, calc_rmse

logger = logging.getLogger(__name__)


def calculate(paths, output_dir, enable_rmse=True, pca_components=None, frow=60, fcol=60, perplexity=30,
              n_iter=1000, save_data=True, use_features=False, resize=None, dim_reduction="tsne", model_file=None,
              jitter_win=0, save_grid_images=True, executions=1):
    logger.info("calculating for %d executions", executions)
    stats_df = pd.DataFrame()
    for i in range(executions):
        last_execution = i == executions - 1
        df, image_shape = build_grid(
            paths, pca_components=pca_components, frow=frow, fcol=fcol, perplexity=perplexity, n_iter=n_iter,
            save_data=last_execution and save_data, output_dir=output_dir, use_features=use_features, resize=resize,
            dim_reduction=dim_reduction, model_file=model_file, jitter_win=jitter_win,
            save_grid_images=last_execution and save_grid_images)
        distance_threshold, exec_stats_df = calc_jaccard_index(df)
        exec_stats_df["execution"] = i
        stats_df = pd.concat((stats_df, exec_stats_df), ignore_index=True, sort=False)
    logger.info("stats %s", stats_df)
    stats_df.to_csv(os.path.join(output_dir, "stats.csv"), index=False)
    if enable_rmse:
        df_distances = calc_rmse(df, image_shape)
        df_distances.to_csv(os.path.join(output_dir, "distances.csv"), index=False)
    return stats_df
