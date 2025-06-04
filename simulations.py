from typing import Callable, Dict, List, Tuple
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from utils.load_dataset import load_epinions
from src.strategies.initial_state import coin_flip_initializer, assign_strategies
from src.strategies.update_rule import imitate_best_neighbor, trust_aware_update, fermi_update, all_neighbors_trust_aware_update
from src.game.game_play import evolutionary_game_round, game_round_trust

import os



if __name__ == "__main__":
    data = ['facebook', 'epinion']

    num_iterations = 20
    cooperator_fractions = []
    save_interval = 10  # Save every 5 iterations
    prob = [0.25, 0.5, 0.75]


    for dataset in data:
        if dataset == 'facebook':
            G = nx.read_edgelist('data/facebook_combined.txt', nodetype=int)
            game_types= [evolutionary_game_round]  # Only use evolutionary game for Facebook
            update_rules = [imitate_best_neighbor, fermi_update]  # Use only these two update rules for Facebook
        elif dataset == 'epinion':
            G = load_epinions('data/epinion/soc-sign-epinions.txt', directed=True)
            game_types = [evolutionary_game_round, game_round_trust]  # Use both games for Epinions
            update_rules = [imitate_best_neighbor, fermi_update, trust_aware_update, all_neighbors_trust_aware_update]  # Use all three update rules for Epinions


        for game_type in game_types:
            for update_rule in update_rules:
                for p in prob:
                    print("-" * 100)
                    print(f"Running {dataset} with {game_type.__name__}, {update_rule.__name__}, initial coinflip p={p}")
                    cooperator_fractions = []  # Reset for each game type and update rule

                    # Initialize the graph with strategies
                    graph = G.copy()  # Use a copy of the graph for each game type and update rule

                    # Initialize strategies once at the start
                    assign_strategies(graph, coin_flip_initializer, p=p, seed=42)
                    # Record initial fraction of cooperators
                    strategies = nx.get_node_attributes(graph, "strategy")
                    strategy_counts = Counter(strategies.values())
                    total_count = len(strategies)
                    cooperator_fractions.append(strategy_counts[1] / total_count)
                    
                    output_path = f"output/{dataset}/{game_type.__name__}/{update_rule.__name__}/p_{p:.2f}"
                    os.makedirs(output_path, exist_ok=True)

                    print(f"Iteration 0 Strategy Distribution: {strategy_counts[1]/total_count:.2%} cooperators, {strategy_counts[0]/total_count:.2%} defectors, Total: {total_count}")
                    nx.write_gexf(graph, f"{output_path}/{data}_iter_0.gexf")


                    for i in range(num_iterations):
                        # Play one round and update strategies
                        if game_type == evolutionary_game_round:
                            evolutionary_game_round(graph, update_rule, seed=42+i)
                        elif game_type == game_round_trust:
                            game_round_trust(graph, update_rule, flip_prob=0.7, seed=42+i)

                        # Record fraction of cooperators
                        strategies = nx.get_node_attributes(graph, "strategy")
                        strategy_counts = Counter(strategies.values())
                        cooperator_fractions.append(strategy_counts[1] / total_count)
                        print(f"Iteration {i+1} Strategy Distribution: {strategy_counts[1]/total_count:.2%} cooperators, {strategy_counts[0]/total_count:.2%} defectors")

                        # Save the graph for Gephi
                        if (i+1) % save_interval == 0 or i == num_iterations - 1:
                            nx.write_gexf(graph, f"{output_path}/{data}_iter_{i+1}.gexf")

                    # Plot the results
                    plt.plot(range(num_iterations+1), cooperator_fractions)
                    plt.xlabel("Iteration")
                    plt.ylabel("Fraction of Cooperators")
                    plt.title("Evolution of Cooperation")
                    # plt.show()
                    # Save the plot
                    plt.savefig(f"{output_path}/cooperator_fractions.png")
                    plt.close()

