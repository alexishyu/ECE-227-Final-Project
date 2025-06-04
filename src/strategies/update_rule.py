'''
Module for flexible strategy-update mechanisms in an evolutionary-game simulation on networks.
Provides an abstraction for applying different update rules with the following strategies:
    - Imitate-best-neighbor
    - Trust-aware update
    - Fermi update
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
    new_strat: Dict[int, Strategy] = {}
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

def fermi_update(
    G: nx.Graph,
    payoffs: PayoffMap,
    rng: np.random.Generator
) -> Dict[int, Strategy]:
    """
    Fermi update rule: each agent u selects one random neighbor v and adopts v's strategy
    with probability given by the Fermi function
        P(u adopts v's strategy) = 1 / (1 + exp[(payoff_u - payoff_v) / K])
    where K > 0 is the "temperature" (noise) parameter. If u has no neighbors, it keeps its strategy.

    Args:
        G (nx.Graph) : The graph with strategy and payoff information.
        payoffs (Dict[int, float]) : Mapping of node ID to total payoff.
        rng (np.random.Generator) : Random generator for neighbor selection and probabilistic decision.

    Returns:
        Dict[int, Strategy] : Mapping from node ID to new strategy.
    """
    new_strat: Dict[int, Strategy] = {}
    K = 0.1  # Fermi noise parameter; adjust as needed

    for u in G.nodes():
        neighbors = list(G.neighbors(u))
        if not neighbors:
            # No neighbors: keep current strategy
            new_strat[u] = G.nodes[u]["strategy"]
            continue

        # Select a random neighbor v
        v = rng.choice(neighbors)
        payoff_u = payoffs[u]
        payoff_v = payoffs[v]

        # Compute adoption probability
        x = (payoff_v - payoff_u) / K
        # Numerically stable logistic
        if x >= 0:
            z = np.exp(-x)
            prob = 1.0 / (1.0 + z)
        else:
            z = np.exp(x)
            prob = z / (1.0 + z)


        # Decide whether to adopt v's strategy
        if rng.random() < prob:
            new_strat[u] = G.nodes[v]["strategy"]
        else:
            new_strat[u] = G.nodes[u]["strategy"]

    return new_strat

def all_neighbors_trust_aware_update(
    G: nx.Graph,               # works for DiGraph or Graph
    payoffs: PayoffMap,
    rng: np.random.Generator
) -> Dict[int, Strategy]:
    """
    Strategy update that considers all neighbors' payoffs weighted by trust relationships.
    
    For each node u:
    1. Calculate weighted sum of all neighbors' payoffs using trust signs as weights
       (includes node's own payoff with positive weight)
    2. If weighted sum is positive, adopt cooperative strategy (1)
    3. If weighted sum is negative, adopt defection strategy (0)
    4. If exactly zero, maintain current strategy with 50% probability or flip it
    
    Args:
        G (nx.Graph): The graph with strategy information and signed edges
        payoffs (Dict[int, float]): Mapping of node ID to total payoff
        rng (np.random.Generator): Random generator for tie-breaking
        
    Returns:
        Dict[int, Strategy]: Mapping from node ID to new strategy
    """
    new_strat = {}
    
    for u in G.nodes():
        # Start with node's own contribution
        weighted_payoff_sum = payoffs[u]
        
        # Add weighted payoffs from all neighbors
        for v in G[u]:
            sign = G[u][v].get("sign", +1)  # Default to trust if sign not specified
            weighted_payoff_sum += sign * payoffs[v]
        
        # Determine strategy based on weighted sum
        if weighted_payoff_sum > 0:
            new_strat[u] = 1  # Cooperate
        elif weighted_payoff_sum < 0:
            new_strat[u] = 0  # Defect
        else:  # weighted_payoff_sum == 0
            # Tie-breaking: keep current with 50% probability or flip it
            current_strat = G.nodes[u]["strategy"] 
            new_strat[u] = current_strat if rng.random() < 0.5 else 1 - current_strat
    
    return new_strat
# Example usage
# G = nx.read_edgelist('facebook_combined.txt', nodetype=int)
# payoffs = {node: np.random.random() for node in G.nodes()}  # Dummy payoffs. Should be received from game logic.
# update_strategies(G, payoffs, imitate_best_neighbor, seed=42)
