import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import os

# 1) Point to your raw log file here:
file_path = Path('output/output.txt')
if not file_path.exists():
    raise FileNotFoundError(
        "The file 'output.txt' was not found. "
        "Please place your log file in the working directory."
    )

# 2) Read the entire log into a string
content = file_path.read_text()

# 3) Parse function: finds each “Running … p=…” header, then its iteration lines
def parse_results(content: str):
    data = []
    # Header lines look like: 
    #   Running facebook with evolutionary_game_round, imitate_best_neighbor, initial coinflip p=0.25
    header_pattern = re.compile(
        r"Running ([^ ]+) with ([^,]+), ([^,]+), initial ([^ ]+) p=(0\.\d+)"
    )
    # Iteration lines look like:
    #   Iteration 0 Strategy Distribution: 25.67% cooperators, 74.33% defectors
    iteration_pattern = re.compile(
        r"Iteration (\d+) Strategy Distribution: ([\d\.]+)% cooperators, ([\d\.]+)% defectors"
    )
    
    current = {}
    for line in content.splitlines():
        header_match = header_pattern.match(line)
        if header_match:
            dataset, game, update_rule, init, p = header_match.groups()
            current = {
                "dataset": dataset,
                "game": game,
                "update_rule": update_rule,
                "p": float(p),
                "iterations": {}
            }
            data.append(current)
        else:
            iter_match = iteration_pattern.match(line)
            if iter_match and current:
                iteration, coop_pct, defe_pct = iter_match.groups()
                current["iterations"][int(iteration)] = {
                    "cooperators": float(coop_pct),
                    "defectors": float(defe_pct)
                }
    return data

parsed_data = parse_results(content)

# 4) Build a single DataFrame from the parsed data (for plotting)
records = []
for entry in parsed_data:
    for iteration, values in entry["iterations"].items():
        records.append({
            "dataset": entry["dataset"],
            "game": entry["game"],
            "update_rule": entry["update_rule"],
            "p": entry["p"],
            "iteration": iteration,
            "cooperators": values["cooperators"],
            "defectors": values["defectors"]
        })

df = pd.DataFrame(records)

# 5) Create a 'plots/' directory (if not already present)
plots_dir = Path('Project Template for ECE227/figures/plots/')
os.makedirs(plots_dir, exist_ok=True)

# 6) For each unique (dataset, game, update_rule), make one plot
combinations = df[['dataset', 'game', 'update_rule']].drop_duplicates()

for _, combo in combinations.iterrows():
    ds, game, update_rule = combo
    subset = df[
        (df['dataset'] == ds) & 
        (df['game'] == game) & 
        (df['update_rule'] == update_rule)
    ]
    p_values = sorted(subset['p'].unique())
    
    plt.figure()
    for p in p_values:
        data_p = subset[subset['p'] == p].sort_values('iteration')
        plt.plot(
            data_p['iteration'],
            data_p['cooperators'],
            marker='o',
            label=f'p={p}'
        )
    if game == 'evolutionary_game_round':
        plt.title(f'{ds} | Standard Game | {update_rule}')
    else:
        plt.title(f'{ds} | Trust Aware Game | {update_rule}')
    plt.xlabel('Iteration')
    plt.ylabel('Cooperators (%)')
    plt.legend()
    plt.grid(True)
    
    # Save the plot as e.g. "plots/facebook_evolutionary_game_round_imitate_best_neighbor.png"
    filename = plots_dir / f"{ds}_{game}_{update_rule}.png".replace(" ", "_")
    plt.savefig(filename)
    plt.close()

# 7) Extract the final (iteration 20) split and compute convergence for each entry, then write to CSV
final_records = []
for entry in parsed_data:
    iterations = entry["iterations"]
    # Determine the maximum iteration number recorded
    max_iter = max(iterations.keys())
    final_values = iterations[max_iter]
    
    # Determine convergence: first iteration at which 'cooperators' percentage stops changing
    sorted_iters = sorted(iterations.keys())
    convergence_iter = max_iter
    previous_coop = iterations[sorted_iters[0]]["cooperators"]

    # tolerance of 0.05%
    tol = 0.05

    for idx in sorted_iters[1:]:
        current_coop = iterations[idx]["cooperators"]
        if abs(current_coop - previous_coop) < tol:
            convergence_iter = idx
            break
        previous_coop = current_coop
    
    final_records.append({
        "dataset": entry["dataset"],
        "game": entry["game"],
        "update_rule": entry["update_rule"],
        "p": entry["p"],
        "final_cooperators_pct": final_values["cooperators"],
        "final_defectors_pct": final_values["defectors"],
        "convergence_iteration": convergence_iter
    })

final_df = pd.DataFrame(final_records)

# Save to "final_splits.csv"
final_csv_path = 'output/final_splits.csv'
final_df.to_csv(final_csv_path, index=False)

# 8) Display the final splits (including convergence iteration)
pd.set_option('display.max_rows', None)
print(final_df)
