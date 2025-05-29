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
