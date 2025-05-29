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

def trust_aware_update(
    G: nx.Graph,               # works for DiGraph or Graph
    payoffs: PayoffMap,
    rng: np.random.Generator
) -> Dict[int, Strategy]:

    """
    Strategy update that respects signed trust edges.

    For each node u:
      1. Consider u itself and every neighbour v (out-neighbour on DiGraph,
         or neighbor on Graph) with attribute 'sign' (+1 trust, -1 distrust).
      2. Compute effective payoff:   s_uv * payoffs[v]
      3. Pick the candidate(s) with the highest effective payoff.
         Break ties uniformly at random.
      4. If the chosen candidate v* is:
           • u itself ........................... keep current strategy
           • a trusted neighbour (sign = +1) .... adopt v*'s strategy
           • a distrusted neighbour (sign = -1) . adopt the *opposite*
                                                  of v*'s strategy
    """
    new_strat: Dict[int, Strategy] = {}

    for u in G.nodes():
        # build candidate list: (node, sign)
        candidates = [(u, +1)]
        for v in G[u]:              # <- works for Graph and DiGraph
            s = G[u][v].get("sign", +1)
            candidates.append((v, s))

        # compute effective payoffs
        eff = [s * payoffs[v] for v, s in candidates]
        max_eff = max(eff)

        # tie‐break uniformly
        best_idxs = [i for i, val in enumerate(eff) if val == max_eff]
        idx = rng.choice(best_idxs)
        v_star, sign_uv = candidates[idx]

        # assign new strategy
        if v_star == u:
            new_strat[u] = G.nodes[u]["strategy"]
        else:
            neigh_s = G.nodes[v_star]["strategy"]
            new_strat[u] = neigh_s if sign_uv == +1 else 1 - neigh_s

    return new_strat

# Example usage
# G = nx.read_edgelist('facebook_combined.txt', nodetype=int)
# payoffs = {node: np.random.random() for node in G.nodes()}  # Dummy payoffs. Should be received from game logic.
# update_strategies(G, payoffs, imitate_best_neighbor, seed=42)
# print("Updated strategies:", nx.get_node_attributes(G, "strategy"))
