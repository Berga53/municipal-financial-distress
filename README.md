# Data-driven solutions for Bank of Italy

Research code for predicting Italian municipality fiscal distress (financial
"dissesto") from balance-sheet, revenue, and indicator time series, using
classical ML models, a CNN, and a graph neural network over a
distance-based municipality graph.

## Repository layout

- `src/bankit/`: the importable Python package with the shared data,
  model, and plotting code.
  - `data.py`: panel construction (`create_data`, `create_data_map`) and
    the municipality registry builder (`create_final`). `create_data` (used
    by `classification.ipynb`) and `create_data_map` (used by `map.ipynb`)
    build similar-looking panels from different inputs and are kept as
    separate functions rather than merged.
  - `graph.py`: haversine distances, distance-matrix filtering
    (`filter_distances`, `inverse_distance`, `top_distances`), and
    `prepare_graph`, which builds the municipality adjacency graph used by
    the GNN.
  - `models.py`: the CNN (`cnn_train`) and the shared scoring/evaluation
    helpers (`evaluate_model` and friends) used by every model family.
  - `gnn.py`: the `GNN` module (CNN encoder + two `GCNConv` layers) and its
    training/evaluation loop (`gnn_train_evaluate`).
  - `training.py`: `run_training`, which fits and evaluates the CNN, GNN,
    and the classical sklearn models (Logistic Regression, Decision Tree,
    Random Forest, XGBoost) for one dataset/target/lag combination.
  - `utils.py`: `print_one_line`, a single-line progress printer.
  - `viz.py`: `summarize_metric` and `plot_confidence_ellipse`, used by
    `results.ipynb`.
- Notebooks (repository root):
  - `data.ipynb`: builds the raw per-year CSVs (`Spese`, `Entrate`,
    `Anticipazioni`, `Indicatori`), the municipality registry, population,
    and zone files under `data/`. Run this first.
  - `classification.ipynb`: the main experiment notebook. Defines the
    "Periodo 1" (2009-2016) and "Periodo 2" (2016-2024) experiment grids,
    trains all model families via `run_training`, and pickles results to
    `results/`. It also contains the LIME/MDI interpretability sections and
    the distance-graph exploration cells.
  - `map.ipynb`: builds the choropleth map of predicted distress
    probability from a fitted CNN and the `shapes/` shapefile.
  - `results.ipynb`: loads the pickled results, builds LaTeX summary
    tables, and produces the scatter/confidence-ellipse comparison plots.
- `data/`: raw and processed input data (balance sheet items, indicators,
  population, municipality registry).
- `shapes/`: the ISTAT municipality shapefile used by `map.ipynb`.
- `gi_comuni.csv`: municipality lat/lon lookup used to build the distance
  graph in `graph.prepare_graph`.
- `figures/`, `results/`: notebook outputs (plots and pickled
  results/tables). `results/` and `rss/` are gitignored; `figures/` is
  tracked since it holds the currently-published set of plots.
- `rss/`: a full parallel copy of `figures/` and `results/` produced by the
  RSS-region-excluded experiment (see below).

## The RSS toggle

`classification.ipynb` defines an alternative experiment where municipalities
in the special-statute regions (`RSS`, e.g. Valle d'Aosta, Trentino-Alto
Adige, Friuli Venezia Giulia, Sardegna) are excluded before training. The
`RSS` list and the filtered `comuni_rss`/`comuni_ind_rss` frames are defined
early in the notebook; re-running the cell `comuni = comuni_rss` (right
before the `# EXP` section) switches every downstream experiment cell to
that filtered dataset, producing the results under `rss/`. This is a
deliberate alternative run, not dead code — leave both the definition and
the toggle cell in place.

## Setup

```bash
make setup
```

This creates `.venv`, installs `requirements.txt`, and adds Jupyter.
Alternatively:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt jupyter
```

The IP-based GNN training uses PyTorch Geometric; on some platforms its
optional `torch-scatter`/`torch-sparse` extensions need to be installed
separately for your PyTorch/CUDA build (see the PyG install docs).

## Running the notebooks

Run everything from the repository root, in this order:

```bash
make notebook
```

1. `data.ipynb` to (re)build the CSVs under `data/`.
2. `classification.ipynb` to train all models and pickle results/figures.
3. `map.ipynb` to render the choropleth map (needs a saved CNN pickle).
4. `results.ipynb` to summarize and plot the pickled results.

Each notebook adds `src` to `sys.path` and imports the shared code from
`bankit`, e.g.:

```python
import sys
sys.path.insert(0, 'src')
from bankit import create_data, run_training
```
