'''
Module for flexible assignment of initial strategies to agents in a network.
This abstraction allows plugging in different initialization schemes (e.g., random coin-flip,
feature-based) without modifying the core assignment routine.
'''
from typing import Callable, Dict
import networkx as nx
import numpy as np

# Type aliases for clarity
Strategy = int  # 1 = Cooperate, 0 = Defect
Initializer = Callable[[nx.Graph, np.random.Generator, float], Dict[int, Strategy]]


def assign_strategies(
    G: nx.Graph,
    initializer: Initializer,
    p: float = 0.5,  # Probability of assigning strategy=1 (Cooperate)
    seed: int = None
) -> None:
    """
    Assign a 'strategy' attribute to each node in the graph using a custom initializer.

    Args: 
        G (nx.Graph) : The networkx graph whose nodes will receive strategy assignments.
        initializer (Initializer) : A function taking (G, rng) and returning a dict mapping node IDs to strategies.
        seed (int) : Seed for the random number generator for perfect reproducibility (default: None).

    Raises:    
        ValueError: If the initializer does not assign a strategy for every node.
    """
    rng = np.random.default_rng(seed)
    strategy_map = initializer(G, rng, p)

    # Ensure complete coverage
    missing = set(G.nodes()) - set(strategy_map.keys())
    if missing:
        raise ValueError(f"Initializer did not assign strategies for nodes: {missing}")

    # Attach the computed strategies
    nx.set_node_attributes(G, strategy_map, "strategy")


def coin_flip_initializer(
    G: nx.Graph,
    rng: np.random.Generator,
    p: float = 0.5
) -> Dict[int, Strategy]:
    """
    Initialize strategies via independent Bernoulli trials (coin-flip).
    
    Args:
        G (nx.Graph) : The graph whose nodes will be assigned strategies.
        rng (np.random.Generator) : Numpy random generator for reproducibility.
        p (float) : Probability of assigning strategy=1 (Cooperate) to each node (default: 0.5).

    Returns:
        Dict[int, Strategy] : Mapping from node ID to assigned strategy (0 - defect or 1 - cooperate).    
    
    """
    nodes = list(G.nodes())
    samples = rng.random(len(nodes))

    # Assign strategy based on the coin flip
    return {node: int(samples[i] < p) for i, node in enumerate(nodes)}

# Example usage
# G = nx.read_edgelist('facebook_combined.txt', nodetype=int)
# assign_strategies(G, coin_flip_initializer, seed=42)

