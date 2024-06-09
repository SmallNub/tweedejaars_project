from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm
import pandas as pd

from ..config import INTERIM_DATA_DIR, RAW_DATA_DIR, MAIN_DATA_FILE_NAME
from .fileloader import load_df, save_df
from .features import add_ids, add_realtime_target

app = typer.Typer()


def fix_target(df: pd.DataFrame):
    """Fix the target to be more correct and consistent."""
    def set_first_two_false(group):
        group.iloc[:] = group.any()
        group.iloc[:2] = False
        return group

    temp_df = pd.concat([df['datetime'], df['time_since_last_two_sided']], axis=1)
    temp_df.columns = ['datetime', 'time_since_last_two_sided']
    temp_df = add_ids(temp_df)
    temp_df = add_realtime_target(temp_df)

    temp_df['target_two_sided_ptu_realtime'] = temp_df.groupby('ptu_id')['target_two_sided_ptu_realtime'].transform(set_first_two_false)
    df['target_two_sided_ptu'] = temp_df['target_two_sided_ptu_realtime']
    return df


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / MAIN_DATA_FILE_NAME,
    output_path: Path = INTERIM_DATA_DIR / MAIN_DATA_FILE_NAME,
    # ----------------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Processing dataset...")
    df = load_df(input_path)

    for i in tqdm(range(10), total=10):
        if i == 0:
            df = fix_target(df)
            logger.info("Fixed target_two_sided_ptu.")

    df = save_df(df, output_path)
    logger.success("Processing dataset complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
