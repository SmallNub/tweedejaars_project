# Tweedejaars Project

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Predicting price swings in electricity

## Project Organization

```
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
│
├── README.md          <- The top-level README for developers using this project.
│
├── data
│   ├── interim        <- Intermediate data that has been transformed.
│   │
│   ├── processed      <- The final, canonical data sets for modeling.
│   │
│   └── raw            <- The original, immutable data dump.
│
├── models             <- Trained and serialized models.
│
├── notebooks          <- Jupyter notebooks.
│
├── pyproject.toml     <- Project configuration file with package metadata for tweedejaars_project
│                         and configuration for tools like black
│
├── environment.yml    <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `conda env create -f environment.yml`
│
├── setup.cfg          <- Configuration file for flake8
│
├── make_dataset.py    <- Simple wrapper script to process the data and generate features.
│
└── tweedejaars_project                <- Source code for use in this project.
    │
    ├── __init__.py    <- Makes tweedejaars_project a Python module
    │
    ├── config.py      <- Configuration of the Python module
    │
    ├── data           <- Scripts to manage data
    │   │
    │   ├── dataset.py                 <- Cleans and processes the data
    │   │
    │   ├── features.py                <- Creates new features
    │   │
    │   ├── fileloader.py              <- Scripts for loading and saving
    │   │
    │   └── split.py                   <- Splits the data into training, validation and testing
    │
    ├── evaluation     <- Scripts to evaluate models and features
    │   │
    │   ├── adjustment.py              <- Script for adjusting predictions
    │   │
    │   ├── evaluate_model.py          <- High performance code for testing models and features
    │   │
    │   └── metrics.py                 <- Simple metrics and custom metrics
    │
    ├── utility        <- Utility scripts
    │   │
    │   └── misc.py                    <- Script containing miscellaneous functions
    │
    └── visualization  <- Scripts to visualize the data and metrics
        │
        ├── analyse.py                 <- Script containing simple plotting functions
        │
        ├── plot_metrics.py            <- Plot the results of the metrics
        │
        └── visualize.py               <- Utility visualization functions

```

--------

