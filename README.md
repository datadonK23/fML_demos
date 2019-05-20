[![Deon badge](https://img.shields.io/badge/ethics%20checklist-deon-brightgreen.svg?style=popout-square)](http://deon.drivendata.org/)

# fML_demos

Functional ML demos implemented with [fklearn](https://github.com/nubank/fklearn).

Demos:

* `nlp_classif_complaints.py` - Classification of customer complaints. Model performance indicators compared to baseline model (the one from [example in the fklearn docs](https://fklearn.readthedocs.io/en/latest/examples/nlp_classification.html)).
* tbd


### Prerequisites

Clone the Repo change to the root directory of the project.

```
git clone https://github.com/datadonK23/fML_demos
cd fML_demos/
```

Create the Python environment and activate it. 

```
conda env create -f environment.yaml
conda activate fML
```


### Installing

Download necessary datasets for the demos by running the `make_dataset.py`script.

```
python make_dataset.py
```

Afterwards you can run each demo by executing its script. E.g.:

```
python nlp_classif_complaints.py
```


Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project
    ├── ETHICS.md          <- Ethics checklist of project, generated with Deon
    │
    ├── data		       <- Datasets 
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── environment.yaml   <- Conda environment file to reproduce Python environment
    │
    ├── make_dataset.py             <- Downloads necessary datasets for demos
    ├── nlp_classif_complaints.py  	<- Binary classification of customer complaints demo
    └── tbd                <- tbd
    

## License

This project is licensed under an Apache License, v2.0 - see the [LICENSE.md](LICENSE.md) file for details.