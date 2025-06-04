from typing import Callable, Dict, List, Tuple
import networkx as nx
import numpy as np

from src.strategies.initial_state import coin_flip_initializer, assign_strategies, Initializer
from src.strategies.update_rule import imitate_best_neighbor, update_strategies, UpdateRule

PAYOFF_MATRIX = {
    (1, 1): (3, 3),  # Both cooperate
    (1, 0): (0, 5),  # Cooperate vs. Defect
    (0, 1): (5, 0),  # Defect vs. Cooperate
    (0, 0): (1, 1),  # Both defect
}

def play_prisoners_dilemma(
    G: nx.Graph,
    seed: int = None
) -> Dict[Tuple[int, int], Tuple[int, int]]:
    """
    Plays the Prisoner's Dilemma game between all pairs of nodes in the graph and 
    returns the results of the game.

    Args:
        G (nx.Graph) : The graph whose nodes will play the game.
        seed (int) : Seed for random number generator for reproducibility.

    Returns:
        Dict[Tuple[int, int], Tuple[int, int]]: A dictionary mapping each pair of node IDs
                                                 to the outcome of their game (payoff for node 1, payoff for node 2).
    """
    rng = np.random.default_rng(seed)
    game_results = {}

    # Iterate over all pairs of nodes in the graph
    for u in G.nodes():
        for v in G.neighbors(u):
            if u < v:  # to ensure each pair is only played once (no duplication)
                strategy_u = G.nodes[u].get("strategy")
                strategy_v = G.nodes[v].get("strategy")
                
                # Get payoffs from the payoff matrix
                payoff_u, payoff_v = PAYOFF_MATRIX[(strategy_u, strategy_v)]
                
                # Store the result of the game
                game_results[(u, v)] = (payoff_u, payoff_v)
    
    return game_results

def play_with_trust_and_pd(
    G: nx.DiGraph,
    flip_prob: float = 0.7,
    seed: int = None
) -> Dict[int, float]:
    """
    For each undirected edge {u,v}:
      1) u and v possibly flip strategy based on trust(u->v) & trust(v->u)
      2) play PD with their new strategies
      3) accumulate and return payoffs[node]
    """
    rng = np.random.default_rng(seed)
    payoffs = {n: 0.0 for n in G.nodes()}

    for u in G.nodes():
        for v in G.neighbors(u):
            if u >= v:
                continue

            # --- 1) trust-based updates ---
            # u’s turn
            sign_uv = G.edges[u, v].get('sign', 0)
            if sign_uv ==  1 and rng.random() < flip_prob:
                G.nodes[u]['strategy'] = 1
            elif sign_uv == -1 and rng.random() < flip_prob:
                G.nodes[u]['strategy'] = 0

            # v’s turn (if reverse edge exists)
            if G.has_edge(v, u):
                sign_vu = G.edges[v, u].get('sign', 0)
                if sign_vu ==  1 and rng.random() < flip_prob:
                    G.nodes[v]['strategy'] = 1
                elif sign_vu == -1 and rng.random() < flip_prob:
                    G.nodes[v]['strategy'] = 0

            # --- 2) play PD with updated strategies ---
            su = G.nodes[u]['strategy']
            sv = G.nodes[v]['strategy']
            pu, pv = PAYOFF_MATRIX[(su, sv)]

            # --- 3) accumulate payoffs ---
            payoffs[u] += pu
            payoffs[v] += pv

    return payoffs
       

def game_round_trust(
    G: nx.DiGraph,
    update_rule: UpdateRule = imitate_best_neighbor,
    flip_prob: float = 0.7,
    seed: int = None
) -> None:
    """
    Conducts one round of the evolutionary game with trust-aware updates.
    This includes:
    1. Playing the Prisoner's Dilemma game with trust dependent game dynamics.
    2. Updating strategies based on the game outcomes and chosen update rule.
    Args:
        G (nx.DiGraph) : The directed graph to run the evolutionary game on.
        update_rule (UpdateRule) : The function used to update the strategies based on the payoffs.
        flip_prob (float) : Probability of flipping strategy based on trust edges.
        seed (int) : Seed for random number generation for reproducibility (default: None).
    """
    payoffs = play_with_trust_and_pd(G, flip_prob=flip_prob, seed=seed)
    update_strategies(G, payoffs, update_rule, seed=seed)

def evolutionary_game_round(
    G: nx.Graph,
    update_rule: UpdateRule,
    seed: int = None
) -> None:
    """
    Conducts one round of the evolutionary game, which includes:
    1. Setting up the network by assigning strategies to nodes.
    2. Playing the Prisoner's Dilemma game with every neighbor.
    3. Updating strategies based on the game outcomes and chosen update rule.

    Args:
        G (nx.Graph) : The network graph to run the evolutionary game on.
        initializer (Initializer) : The function used to initialize the strategies of the nodes.
        update_rule (UpdateRule) : The function used to update the strategies based on the payoffs.
        seed (int) : Seed for random number generation for reproducibility (default: None).
    """

    # Step 2: Play the Prisoner's Dilemma game with each pair of neighbors
    game_results = play_prisoners_dilemma(G, seed)
    
    # Step 3: Calculate accumulated payoffs for each node
    payoffs = {node: 0.0 for node in G.nodes()}
    
    for (u, v), (payoff_u, payoff_v) in game_results.items():
        payoffs[u] += payoff_u
        payoffs[v] += payoff_v

    # Step 4: Update the strategies based on the payoffs
    update_strategies(G, payoffs, update_rule, seed)

# Example usage
# G = nx.read_edgelist('data/facebook_combined.txt', nodetype=int)
# evolutionary_game_round(G, coin_flip_initializer, imitate_best_neighbor, seed=42)
# print("Updated strategies:", nx.get_node_attributes(G, "strategy"))