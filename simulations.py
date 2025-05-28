from typing import Callable, Dict, List, Tuple
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from utils.load_dataset import load_epinions
from src.strategies.initial_state import coin_flip_initializer, assign_strategies
from src.strategies.update_rule import imitate_best_neighbor
from src.game.game_play import evolutionary_game_round



if __name__ == "__main__":
    data = 'facebook'  # Change to 'epinion' for the Epinions dataset
    
    if data == 'facebook':
        G = nx.read_edgelist('data/facebook_combined.txt', nodetype=int)
    elif data == 'epinion':
        G = load_epinions('data/epinion/soc-sign-epinions.txt', directed=False)

    num_iterations = 10
    cooperator_fractions = []

    # Initialize strategies once at the start
    assign_strategies(G, coin_flip_initializer, seed=42)

    # Record initial fraction of cooperators
    strategies = nx.get_node_attributes(G, "strategy")
    strategy_counts = Counter(strategies.values())
    total_count = len(strategies)
    cooperator_fractions.append(strategy_counts[1] / total_count)

    print(f"Iteration 0 Strategy Distribution: {strategy_counts[1]/total_count:.2%} cooperators, {strategy_counts[0]/total_count:.2%} defectors, Total: {total_count}")
    nx.write_gexf(G, f"output/{data}/{data}_iter_0.gexf")

    for i in range(num_iterations):
        # Play one round and update strategies
        evolutionary_game_round(G, lambda G, rng: nx.get_node_attributes(G, "strategy"), imitate_best_neighbor, seed=42+i)

        # Record fraction of cooperators
        strategies = nx.get_node_attributes(G, "strategy")
        strategy_counts = Counter(strategies.values())
        cooperator_fractions.append(strategy_counts[1] / total_count)
        print(f"Iteration {i+1} Strategy Distribution: {strategy_counts[1]/total_count:.2%} cooperators, {strategy_counts[0]/total_count:.2%} defectors")

        # Save the graph for Gephi
        nx.write_gexf(G, f"output/{data}/{data}_iter_{i+1}.gexf")

    # Plot the results
    plt.plot(range(num_iterations+1), cooperator_fractions)
    plt.xlabel("Iteration")
    plt.ylabel("Fraction of Cooperators")
    plt.title("Evolution of Cooperation")
    plt.show()