# Evolutionary Prisoner’s Dilemma on Social Networks

## Project Overview
This repository contains a simulation pipeline for running evolutionary-game dynamics on large networks. It supports:

- Two datasets:
  - **Facebook** (undirected)
  - **Epinions** (directed)
- Multiple update rules:
  - Imitate-best-neighbor
  - Fermi update
  - Trust-aware update
  - All-neighbors trust-aware update
- Two game types (for Epinions):
  - Standard evolutionary game
  - Trust-based evolutionary game
- Initialization via a coin-flip distribution of cooperators vs. defectors

At each iteration, the code:
1. Plays one round of the chosen game on the graph.
2. Updates every node’s strategy according to the selected update rule.
3. Records the fraction of cooperators.
4. Saves intermediate GEXF snapshots (every _save_interval_ iterations) for visualization in Gephi.
5. Exports a final PNG plot showing “Fraction of Cooperators vs. Iteration”.

**Project Paper can be found [here](https://github.com/alexishyu/ECE-227-Final-Project/blob/main/ECE227FinalPaper.pdf)**

## Prerequisites
- **Python 3.9+**
- Required Python packages (install via `pip` or `conda`):
  - networkx
  - numpy
  - matplotlib

You should also have the raw edge lists under `data/`:
- `data/facebook_combined.txt`
- `data/epinion/soc-sign-epinions.txt`

Included in the repo, but also available on SNAP website.

## Installation
1. **Clone this repository**:
    ```bash
    git clone https://github.com/alexishyu/ECE-227-Final-Project.git
    ECE-227-Final-Project
    ```

2. **Create (and activate) a new Conda environment (Optional)**:
    ```bash
    conda create -n my_env python=3.10
    conda activate my_env
    ```
3. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

## File Structure

```
├── data/
│   ├── facebook_combined.txt
│   └── epinion/
│       └── soc-sign-epinions.txt
│
├── output/
│   └── (auto-generated directories and files containing GEXF snapshots + PNG plots)
│
├── src/
│   ├── game/
│   │   └── game_play.py
│   └── strategies/
│       ├── initial_state.py
│       └── update_rule.py
|
├── utils/
│   └── load_dataset.py
│
└── simulations.py    ← Main entry point
```

## Usage

1. **Ensure data files exist** under `data/`:

   * `data/facebook_combined.txt`
   * `data/epinion/soc-sign-epinions.txt`

2. **Run the script**:

   ```bash
   python simulations.py
   ```

   * This will loop over each combination of:

     * Dataset (`facebook` / `epinion`)
     * Game type (for `facebook`, only `evolutionary_game_round`; for `epinion`, both `evolutionary_game_round` and `game_round_trust`)
     * Update rule (depending on dataset)
     * Initial‐coin-flip probability (0.25, 0.50, 0.75)

3. **Inspect the output** in `output/`:

   ```
   output/
   ├── facebook/
   │   └── evolutionary_game_round/
   │       ├── imitate_best_neighbor/
   │       │   ├── p_0.25/
   │       │   │   ├── facebook_iter_0.gexf
   │       │   │   ├── facebook_iter_10.gexf
   │       │   │   ├── facebook_iter_20.gexf
   │       │   │   └── cooperator_fractions.png
   │       │   └── p_0.50/ ...
   │       └── fermi_update/ ...
   └── epinion/
       ├── evolutionary_game_round/
       │   └── imitate_best_neighbor/ ...
       ├── game_round_trust/
       │   └── trust_aware_update/ ...
       └── … (other update rules)
   ```

   * **`.gexf` files**: Snapshots of the network state (strategies stored as a node attribute). These can be opened in Gephi for visualization.
   * **`cooperator_fractions.png`**: A line plot showing the fraction of cooperators over all iterations.

## Configurable Parameters

All of these are at the top of `simulations.py`, so you can tweak them before running.

1. **`data`** (list of dataset names)

   ```python
   data = ['facebook', 'epinion']
   ```

   * To run on only one dataset, e.g., `['facebook']`.

2. **`num_iterations`** (integer)

   ```python
   num_iterations = 20
   ```

   * Total number of update iterations per scenario.

3. **`save_interval`** (integer)

   ```python
   save_interval = 10
   ```

   * Save a GEXF snapshot every `save_interval` iterations (plus at iteration 0 and the final iteration).

4. **`prob`** (list of floats)

   ```python
   prob = [0.25, 0.5, 0.75]
   ```

   * Initial probability (`p`) of assigning “cooperator” versus “defector” during coin-flip initialization. You can add or remove values here (e.g., `[0.1, 0.5, 0.9]`).

5. **`game_types`** (list of functions)

   ```python
   # For Facebook:
   game_types = [evolutionary_game_round]
   # For Epinions:
   game_types = [evolutionary_game_round, game_round_trust]
   ```

   * You can include/exclude `game_round_trust` (only relevant for the directed Epinions graph).

6. **`update_rules`** (list of functions)

   ```python
   # For Facebook:
    update_rules = [
        imitate_best_neighbor, 
        fermi_update
    ]
   # For Epinions:
    update_rules = [
        imitate_best_neighbor,
        fermi_update,
        trust_aware_update,
        all_neighbors_trust_aware_update,
    ]
   ```

   * To test only one rule, reduce the list accordingly. Each function implements a distinct strategy-update mechanism.

7. **`seed`**

   * A fixed integer (42 + iteration index) ensures reproducibility. Change or remove to introduce non-deterministic behavior.



## Notes

* Make sure that all imports (e.g., `utils.load_dataset`, `src.strategies.*`, `src.game.*`) resolve correctly. If you move files around, update the `PYTHONPATH` or adjust relative imports.
* The current implementation uses hardcoded seeds (`seed=42 + iteration`) to ensure reproducibility. Remove or randomize if you want noise between runs.
