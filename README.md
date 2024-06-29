# Tweedejaars Project

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A machine learning approach to predict price swings in the electricity grid of the Netherlands.

This project was made by: \
Nordin el Assassi \
Sakr Ismail \
Simen Veenman \
Steven Dong \
Tycho van Willigen


## Prerequisites

- <a target="_blank" href="https://docs.anaconda.com/miniconda/">Conda</a>
- This repository.
- Data provided by Eneco.


## Instructions

1. Place the raw data into the folder `data/raw`.
2. Open the terminal in the same folder as this file.
3. Run the make command `make create_environment` (might raise an error).
   This will create a new conda environment.
   - NOTE: this might take a very long time due to the amount of Python packages needed to be installed. \
     If this process is interrupted, run the make command `make requirements`.
   - If it raised an error, run `pip install python-dotenv` after step 4.
4. Run the conda command `conda activate tweedejaars_project`.
5. Run the pip command `pip install -e .`. This installs the package in an editable state.
6. Run the make command `make data`. This will preprocess the data and generate new features.
7. Navigate to the desired file using the Project Organization below.


## Project Organization

```
├── Makefile           <- Makefile with convenience commands like `make data`.
│
├── README.md          <- The top-level README for developers using this project.
│
├── data
│   │
│   ├── interim        <- Intermediate data that has been transformed.
│   │
│   ├── processed      <- The final, canonical data sets for modeling.
│   │
│   └── raw            <- The original, immutable data dump.
│
├── models             <- Trained and serialized models.
│
├── notebooks          <- Jupyter notebooks.
│   │
│   ├── legacy                          <- Folder containing Old notebooks.
│   │                                      Some may no longer run without changing the code manually.
│   │
│   ├── autoencoder.ipynb               <- Model for reconstructing data and detecting anomalies.
│   │
│   ├── autoregressive_rnn.ipynb        <- Autoregressive RNN model for using the entire history.
│   │
│   ├── custom_loss.ipynb               <- Simple FNN using a custom loss function.
│   │
│   ├── eneco_deliverable.ipynb         <- The final deliverable in the requested format by Eneco.
│   │                                      This file is made by combining all the code in the source code
│   │                                      and `main_model.ipynb` into a singular massive file.
│   │                                      This is not recommended, use `main_model.ipynb` instead.
│   │
│   ├── eneco_model.ipynb               <- A recreation of the model used by Eneco.
│   │
│   ├── main_model.ipynb                <- The best model, HistogramGradientBoostingRegressor.
│   │
│   ├── markovian_rnn.ipynb             <- Markovian RNN model for using a limited history.
│   │
│   └── price_prediction.ipynb          <- Model for predicting `settlement_price_realized`.
│
├── pyproject.toml     <- Project configuration file with package metadata for tweedejaars_project
│                         and configuration for tools like black.
│
├── environment.yml    <- The requirements file for reproducing the analysis environment, e.g.
│                         to recreate the conda environment `conda env create -f environment.yml`.
│
├── setup.cfg          <- Configuration file for flake8.
│
├── make_dataset.py    <- Simple wrapper script to process the data and generate features.
│
└── tweedejaars_project                 <- Source code for use in this project.
    │
    ├── __init__.py    <- Makes tweedejaars_project a Python module.
    │
    ├── config.py      <- Configuration of the Python module.
    │
    ├── data           <- Scripts to manage data.
    │   │
    │   ├── dataset.py                  <- Cleans and processes the data.
    │   │
    │   ├── features.py                 <- Creates new features.
    │   │
    │   ├── fileloader.py               <- Scripts for loading and saving.
    │   │
    │   └── split.py                    <- Splits the data into training, validation and testing.
    │
    ├── evaluation     <- Scripts to evaluate models and features.
    │   │
    │   ├── adjustment.py               <- Script for adjusting predictions.
    │   │
    │   ├── evaluate_model.py           <- High performance code for testing models and features.
    │   │
    │   └── metrics.py                  <- Simple metrics and custom metrics.
    │
    ├── utility        <- Utility scripts.
    │   │
    │   └── misc.py                     <- Script containing miscellaneous functions.
    │
    └── visualization  <- Scripts to visualize the data and metrics.
        │
        ├── analyse.py                  <- Script containing simple plotting functions.
        │
        ├── plot_metrics.py             <- Plot the results of the metrics.
        │
        └── visualize.py                <- Utility visualization functions.

```

--------

