'''
Module for flexible strategy-update mechanisms in an evolutionary-game simulation on networks.
Provides an abstraction for applying different update rules with the following strategies:
    - Imitate-best-neighbor
'''
from typing import Callable, Dict
import networkx as nx
import numpy as np

# Type aliases for clarity
Strategy = int  # 1 = Cooperate, 0 = Defect
PayoffMap = Dict[int, float]  # node ID -> accumulated payoff
UpdateRule = Callable[[nx.Graph, PayoffMap, np.random.Generator], Dict[int, Strategy]] # Function type for update rules

def update_strategies(
    G: nx.Graph,
    payoffs: PayoffMap,
    update_rule: UpdateRule,
    seed: int = None
) -> None:
    """
    Update each node's 'strategy' attribute in-place using the given update rule.

    Args:
        G (nx.Graph) : Graph with node attribute 'strategy' and precomputed payoffs.
        payoffs (Dict[int, float]) : Mapping from node ID to its accumulated payoff.
        update_rule (UpdateRule) : Function implementing the strategy-update logic.
        seed (int) : Seed for random decisions within the update rule.

    Raises:
        ValueError: If the update rule fails to return a new strategy for every node.
    """
    rng = np.random.default_rng(seed)
    new_strategy_map = update_rule(G, payoffs, rng)

    # Ensure the rule covers all nodes
    missing = set(G.nodes()) - set(new_strategy_map.keys())
    if missing:
        raise ValueError(f"Update rule did not provide strategies for nodes: {missing}")

    # Apply the updated strategies
    nx.set_node_attributes(G, new_strategy_map, "strategy")

def imitate_best_neighbor(
    G: nx.Graph,
    payoffs: PayoffMap,
    rng: np.random.Generator
) -> Dict[int, Strategy]:
    """
    Imitate-best-neighbor rule: each agent adopts the strategy of the neighbor (or itself)
    with the highest payoff. In case of ties, randomly select among the top performers.

    Args:
        G (nx.Graph) : The graph with strategy and payoff information.
        payoffs (Dict[int, float]) : Mapping of node ID to total payoff.
        rng (np.random.Generator) : Random generator for tie-breaking.

    Returns:
        Dict[int, Strategy] : Mapping from node ID to new strategy.
    """
    new_strat = {}
    for u in G.nodes():
        best = u
        best_pay = payoffs[u]
        for v in G.neighbors(u):
            p = payoffs[v]
            if p > best_pay:
                best = v
                best_pay = p
            elif p == best_pay and rng.random() < 0.5:
                best = v
        new_strat[u] = G.nodes[best]["strategy"]
    return new_strat

# Example usage
# G = nx.read_edgelist('facebook_combined.txt', nodetype=int)
# payoffs = {node: np.random.random() for node in G.nodes()}  # Dummy payoffs. Should be received from game logic.
# update_strategies(G, payoffs, imitate_best_neighbor, seed=42)
# print("Updated strategies:", nx.get_node_attributes(G, "strategy"))
