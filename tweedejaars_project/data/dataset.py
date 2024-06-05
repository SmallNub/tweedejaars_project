from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

from ..config import INTERIM_DATA_DIR, RAW_DATA_DIR, MAIN_DATA_FILE_NAME
from .fileloader import load_df, save_df

app = typer.Typer()


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
        if i == 5:
            logger.info("Something happened for iteration 5.")

    df = save_df(df, output_path)
    logger.success("Processing dataset complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
