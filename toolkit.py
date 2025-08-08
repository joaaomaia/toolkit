import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from itertools import combinations
from typing import Optional, Dict, Tuple
from statsmodels.tsa.stattools import grangercausalitytests
from tqdm.auto import tqdm

class ContagionGraph:
    """
    Construct a directed contagion graph between accounts based on *spike* events
    in daily transaction count or amount.

    Main features
    -------------
    • Two spike modes:
        * ``global``      – spike if value ≥ global Q-quantile (default).
        * ``individual``  – spike if value ≥ the account's own Q-quantile.
    • Edge weight decays with lag ( weight += 1/lag_days ).
    • Optional edge pruning by z-score (mean + z_thresh·σ).
    • Optional pairwise Granger causality filter (p < α).
    • Louvain community detection saved as ``community_id``.
    • Export helpers for centrality metrics and Granger p-values.
    • Quick interactive Plotly visualisation.

    Parameters
    ----------
    vol_q, amt_q : float, default=0.95
        Upper quantile used to define volume / amount spikes (0 < Q < 1).
    window_days : int, default=3
        Maximum lag (in days) for a target spike to be attributed to a source
        spike.
    min_edge_weight : float, default=0.5
        Edges with cumulative weight below this value are discarded.
    z_prune : bool, default=True
        If ``True`` prune edges whose weight z-score is below ``z_thresh``.
    z_thresh : float, default=1.5
        Z-score threshold used when ``z_prune`` is ``True``.
    granger_maxlag : int | None, default=None
        Maximum lag for the Granger test (set ``None`` to skip).
    granger_alpha : float, default=0.05
        Significance level for the Granger filter.

    Attributes
    ----------
    G : nx.DiGraph
        The contagion graph (after ``fit``).
    agg : pandas.DataFrame
        Daily aggregated data with spike flags.
    centrality_df : pandas.DataFrame
        Table of exported centrality metrics (populated after
        :py:meth:`export_centrality`).
    granger_df : pandas.DataFrame
        Table of Granger p-values (populated if Granger option is used).
    """

    # ------------------------------------------------------------------ #
    # Constructor
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        *,
        vol_q: float = 0.95,
        amt_q: float = 0.95,
        window_days: int = 3,
        min_edge_weight: float = 0.5,
        z_prune: bool = True,
        z_thresh: float = 1.5,
        granger_maxlag: Optional[int] = None,
        granger_alpha: float = 0.05,
        merge_centrality_feats: bool = True,
        merge_causality_feats: bool = True,
    ) -> None:
        # parameters (see original docstring for meaning)
        self.vol_q = vol_q
        self.amt_q = amt_q
        self.window_days = window_days
        self.min_edge_weight = min_edge_weight
        self.z_prune = z_prune
        self.z_thresh = z_thresh
        self.granger_maxlag = granger_maxlag
        self.granger_alpha = granger_alpha
        self.merge_centrality_feats = merge_centrality_feats
        self.merge_causality_feats = merge_causality_feats

        # artefacts
        self.agg: Optional[pd.DataFrame] = None
        self.G: Optional[nx.DiGraph] = None
        self.centrality_df: Optional[pd.DataFrame] = None
        self.granger_df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------ #
    # Internal helpers – spike labelling
    # ------------------------------------------------------------------ #
    @staticmethod
    def _aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
        """Return daily stats (count, amount) per ``account_id``."""
        out = (
            df.assign(order_date=pd.to_datetime(df["order_date"]))
            .groupby(["account_id", "order_date"], as_index=False)
            .agg(
                n_tx=("transaction_amount", "size"),
                tot_amt=("transaction_amount", "sum"),
            )
        )
        return out

    def _flag_spikes_global(self, agg: pd.DataFrame) -> pd.DataFrame:
        """Mark spikes versus the **global** distribution."""
        vol_cut = agg["n_tx"].quantile(self.vol_q)
        amt_cut = agg["tot_amt"].quantile(self.amt_q)
        agg["vol_spike"] = agg["n_tx"] >= vol_cut
        agg["amt_spike"] = agg["tot_amt"] >= amt_cut
        return agg

    def _flag_spikes_individual(self, agg: pd.DataFrame) -> pd.DataFrame:
        """Mark spikes versus each account's own historical distribution."""
        cuts = (
            agg.groupby("account_id")
            .agg(
                vol_cut=("n_tx", lambda x: x.quantile(self.vol_q)),
                amt_cut=("tot_amt", lambda x: x.quantile(self.amt_q)),
            )
            .reset_index()
        )
        agg = agg.merge(cuts, on="account_id", how="left")
        agg["vol_spike"] = agg["n_tx"] >= agg["vol_cut"]
        agg["amt_spike"] = agg["tot_amt"] >= agg["amt_cut"]
        return agg.drop(columns=["vol_cut", "amt_cut"])

    # ------------------------------------------------------------------ #
    # Fit – build the graph
    # ------------------------------------------------------------------ #
    def fit(self, df: pd.DataFrame, mode: str = "global", progress: bool = True) -> "ContagionGraph":
        """
        Detect spikes, build the contagion graph and assign Louvain communities.

        Parameters
        ----------
        df : pandas.DataFrame
            Must include at least ``['account_id','order_date','transaction_amount']``.
        mode : {"global","individual"}, default="global"
            Spike definition mode (see class docstring).
        progress : bool, default=True
            Whether to show tqdm progress bars.

        Returns
        -------
        self : ContagionGraph
            The fitted object (enables method chaining).
        """
        if mode not in {"global", "individual"}:
            raise ValueError("mode must be 'global' or 'individual'.")

        # 1) daily aggregation
        agg = self._aggregate_daily(df)
        agg = self._flag_spikes_global(agg) if mode == "global" else self._flag_spikes_individual(agg)
        self.agg = agg

        spikes = agg.loc[agg[["vol_spike", "amt_spike"]].any(axis=1)]
        spike_dict: Dict[str, list[pd.Timestamp]] = (
            spikes.groupby("account_id")["order_date"].apply(list).to_dict()
        )

        # 2) build weighted edges (weight decays with lag)
        edge_weights: Dict[Tuple[str, str], float] = {}
        iterator = tqdm(spike_dict.items(), desc="Scanning contagion") if progress else spike_dict.items()

        for src, days in iterator:
            for day in days:
                for lag in range(1, self.window_days + 1):
                    target_day = day + pd.Timedelta(days=lag)
                    targets = spikes.loc[
                        (spikes["order_date"] == target_day) & (spikes["account_id"] != src),
                        "account_id",
                    ].unique()
                    for tgt in targets:
                        edge_weights[(src, tgt)] = edge_weights.get((src, tgt), 0.0) + 1.0 / lag

        if not edge_weights:
            self.G = nx.DiGraph()
            return self

        edge_df = (
            pd.Series(edge_weights, name="weight")
            .reset_index()
            .rename(columns={"level_0": "src", "level_1": "tgt"})
        )

        edge_df = edge_df.loc[edge_df["weight"] >= self.min_edge_weight].copy()

        # 3) z-score pruning
        if self.z_prune and not edge_df.empty:
            mu, sigma = edge_df["weight"].mean(), edge_df["weight"].std(ddof=0)
            if sigma > 0:
                edge_df = edge_df.loc[((edge_df["weight"] - mu) / sigma) >= self.z_thresh]

        # 4) optional Granger filter
        if self.granger_maxlag is not None and not edge_df.empty:
            edge_df = self._apply_granger_filter(edge_df)

        # 5) build NetworkX graph
        G = nx.DiGraph()
        G.add_weighted_edges_from(edge_df[["src", "tgt", "weight"]].itertuples(index=False, name=None))
        self.G = G

        # 6) Louvain community detection
        self._detect_communities()
        return self

    # ------------------------------------------------------------------ #
    # Granger causality (pairwise)
    # ------------------------------------------------------------------ #
    def _apply_granger_filter(self, edge_df: pd.DataFrame) -> pd.DataFrame:
        """Filter edges by Granger causality (one-sided, p < alpha)."""
        ts = (
            self.agg.set_index("order_date")
            .pivot_table(values="n_tx", index="order_date", columns="account_id", fill_value=0.0)
            .astype(float)
            .sort_index()
        )

        keep_mask: list[bool] = []
        granger_records: list[Dict[str, float]] = []

        iterator = (
            tqdm(edge_df.itertuples(index=False), total=len(edge_df), desc="Granger")
            if len(edge_df) > 15
            else edge_df.itertuples(index=False)
        )

        for src, tgt, _ in iterator:
            arr = ts[[tgt, src]].dropna().values
            if arr.shape[0] < (self.granger_maxlag or 0) + 2:
                p_val = np.nan
            else:
                import contextlib, io
                with contextlib.redirect_stdout(io.StringIO()):
                    res = grangercausalitytests(arr, maxlag=self.granger_maxlag)
                p_val = min(res[lag][0]["ssr_ftest"][1] for lag in res)
            keep_mask.append(p_val < self.granger_alpha if not np.isnan(p_val) else False)
            granger_records.append({"src": src, "tgt": tgt, "granger_p": p_val})

        self.granger_df = pd.DataFrame(granger_records)
        return edge_df.loc[keep_mask]

    # ------------------------------------------------------------------ #
    # Community detection
    # ------------------------------------------------------------------ #
    def _detect_communities(self) -> None:
        """Attach a ``community_id`` attribute to each node."""
        if self.G is None or self.G.number_of_nodes() == 0:
            return

        mapping: Dict[str, int]

        if _LOUVAIN_BUILTIN:  # NetworkX built-in
            comms = louvain_communities(self.G, seed=42, weight="weight")  # type: ignore
            mapping = {node: cid for cid, community in enumerate(comms) for node in community}
        elif best_partition is not None:  # python-louvain fallback
            mapping = best_partition(self.G.to_undirected(), weight="weight", random_state=42)
        else:  # Louvain unavailable
            mapping = {node: -1 for node in self.G.nodes}

        nx.set_node_attributes(self.G, mapping, "community_id")

    # ------------------------------------------------------------------ #
    # Feature export
    # ------------------------------------------------------------------ #

    def merge_features(
        self,
        df_master: pd.DataFrame,
        *,
        account_col: str = "account_id",
        known_risk: Optional[set[str]] = None,
    ) -> pd.DataFrame:
        """
        Return ``df_master`` enriquecido com métricas de grafo e/ou causalidade.

        Strategy
        --------
        • Centrality metrics absent  → 0.0\n
        • community_id absent        → -1\n
        • Granger p absent           → 1.0 (i.e. no evidence)\n
        • Flags (`is_isolated`, `has_granger_pair`) são gerados.

        Parameters
        ----------
        df_master : pandas.DataFrame
            DataFrame principal contendo pelo menos a coluna ``account_id``.
        account_col : str, default="account_id"
            Nome da coluna de id.
        known_risk : set[str] | None
            Lista de contas de risco (opcional, repassa p/ centrality).

        Returns
        -------
        pandas.DataFrame
            Uma cópia de ``df_master`` com colunas extras.
        """
        if self.G is None:
            raise RuntimeError("Call `fit()` before merging features.")

        X = df_master.copy()

        # ----------------------------------------------------------------
        # Centrality
        # ----------------------------------------------------------------
        if self.merge_centrality_feats:
            cent = (
                self.export_centrality(known_risk=known_risk)
                .rename(columns={account_col: f"{account_col}"})
            )
            X = X.merge(cent, on=account_col, how="left")

            num_cols = ["out_strength", "in_strength",
                        "pagerank", "num_nodes_influenced"]
            X[num_cols] = X[num_cols].fillna(0.0)
            X["community_id"] = X["community_id"].fillna(-1).astype(int)
            X["is_isolated"] = (X["out_strength"] == 0).astype(int)

        # ----------------------------------------------------------------
        # Causality (Granger)
        # ----------------------------------------------------------------
        if self.merge_causality_feats and self.granger_maxlag is not None:
            if self.granger_df is None:
                raise RuntimeError("Granger tests were not executed.")
            gr = (
                self.granger_df.groupby("tgt")
                .agg(min_granger_p=("granger_p", "min"))
                .reset_index()
                .rename(columns={"tgt": account_col})
            )
            X = X.merge(gr, on=account_col, how="left")
            X["min_granger_p"] = X["min_granger_p"].fillna(1.0)
            X["has_granger_pair"] = (X["min_granger_p"] < 1.0).astype(int)

        return X

    def export_centrality(self, known_risk: Optional[set[str]] = None) -> pd.DataFrame:
        """
        Compute centrality metrics and return a DataFrame ready for ML models.

        Parameters
        ----------
        known_risk : set[str] | None, optional
            Black-list of risky accounts; used to flag nodes influenced by them.

        Returns
        -------
        pandas.DataFrame
            Centrality table (one row per account).
        """
        if self.G is None:
            raise RuntimeError("Call ``fit`` before exporting features.")

        out_strength = dict(self.G.out_degree(weight="weight"))
        in_strength = dict(self.G.in_degree(weight="weight"))
        pagerank = nx.pagerank(self.G, weight="weight") if self.G.number_of_edges() else {}
        community = nx.get_node_attributes(self.G, "community_id")

        df = pd.DataFrame(
            {
                "account_id": list(self.G.nodes),
                "out_strength": [out_strength.get(v, 0.0) for v in self.G.nodes],
                "in_strength": [in_strength.get(v, 0.0) for v in self.G.nodes],
                "pagerank": [pagerank.get(v, 0.0) for v in self.G.nodes],
                "num_nodes_influenced": [self.G.out_degree(v) for v in self.G.nodes],
                "community_id": [community.get(v, -1) for v in self.G.nodes],
            }
        )

        if known_risk is None:
            known_risk = set()

        df["is_influenced_by_known_risk"] = [
            int(any(pred in known_risk for pred in self.G.predecessors(v))) for v in self.G.nodes
        ]

        self.centrality_df = df
        return df

    def export_granger(self) -> pd.DataFrame:
        """
        Return the table of Granger p-values (available only if Granger was run).
        """
        if self.granger_df is None:
            raise RuntimeError("Granger tests were not executed.")
        return self.granger_df.copy()

    # ------------------------------------------------------------------ #
    # Plotly visualisation
    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    # Plotly visualisation (improved)
    # ------------------------------------------------------------------ #
    def plot(
        self,
        top_n: int = 50,
        k: float | None = None,
        size_factor: float = 40,
        edge_scale: float = 4,
        fig_size: tuple[int, int] = (900, 600),
        notebook: bool = True,
    ):
        """
        Interactive Plotly view of the contagion network.

        Parameters
        ----------
        top_n : int, default=50
            Show the top-N nodes by ``out_strength``.
        k : float | None
            Spring stiffness passed to ``nx.spring_layout``.  Smaller → tighter.
            ``None`` = automatic proportional to 1/√N.
        size_factor : float, default=40
            Multiplier applied to normalised node sizes.
        edge_scale : float, default=4
            Multiplier applied to normalised edge widths.
        fig_size : tuple[int,int], default=(900,600)
            Figure width × height in pixels.
        notebook : bool, default=True
            Return Figure (True) or call ``show`` (False).
        """
        if self.G is None or self.G.number_of_nodes() == 0:
            raise RuntimeError("Graph is empty – run ``fit`` first.")

        # -------- choose subgraph (top-N by out_strength) -----------------
        out_strength = dict(self.G.out_degree(weight="weight"))
        top_nodes = sorted(out_strength, key=out_strength.get, reverse=True)[:top_n]
        H = self.G.subgraph(top_nodes).copy()

        # -------- layout --------------------------------------------------
        if k is None:
            k = 1 / np.sqrt(len(H))
        pos = nx.spring_layout(H, k=k, seed=42, weight="weight")

        # -------- edge trace ---------------------------------------------
        w_arr = np.array([d["weight"] for _, _, d in H.edges(data=True)])
        if len(w_arr) == 0:
            w_arr = np.array([1.0])
        w_norm = (w_arr - w_arr.min()) / (w_arr.max() - w_arr.min() + 1e-9)

        edge_x, edge_y = [], []
        for (u, v, d), w in zip(H.edges(data=True), w_norm):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        # map weight → alpha (transparency) for visual hint
        avg_alpha = np.clip(w_norm.mean(), 0.25, 0.85)
        edge_rgba = f"rgba(120,120,120,{avg_alpha:.2f})"

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=1.2, color=edge_rgba),  # width must be scalar
            hoverinfo="none",
        )

        # -------- node trace ---------------------------------------------
        s_arr = np.array([out_strength.get(n, 0.0) for n in H.nodes()])
        s_norm = (s_arr - s_arr.min()) / (s_arr.max() - s_arr.min() + 1e-9)
        node_sizes = s_norm * size_factor + 8

        community = nx.get_node_attributes(H, "community_id")
        com_vals = np.array([community.get(n, -1) for n in H.nodes()])
        # map community to color
        unique_coms = {c: i for i, c in enumerate(np.unique(com_vals))}
        colors = [unique_coms[c] for c in com_vals]

        node_x, node_y = zip(*[pos[n] for n in H.nodes()])
        hover = [
            f"{n}<br>out_strength={out_strength.get(n,0):.2f}<br>community={community.get(n,-1)}"
            for n in H.nodes()
        ]

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            hoverinfo="text",
            text=hover,
            marker=dict(
                showscale=True,
                colorscale="Viridis",
                color=colors,
                size=node_sizes,
                line_width=1.5,
                colorbar=dict(title="Community<br>ID"),
            ),
        )

        # -------- figure --------------------------------------------------
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=f"Contagion network — top {top_n} nodes",
                width=fig_size[0],
                height=fig_size[1],
                showlegend=False,
                hovermode="closest",
                margin=dict(l=5, r=5, t=40, b=20),
                xaxis=dict(showgrid=False, zeroline=False, visible=False),
                yaxis=dict(showgrid=False, zeroline=False, visible=False),
            ),
        )
        return fig if notebook else fig.show()