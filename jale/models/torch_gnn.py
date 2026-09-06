"""PyTorch Geometric models for JA-LE.  COLAB ONLY — requires torch + torch_geometric.

This module is deliberately *not* imported by `scripts/run_v1.py`, so the sandbox
pipeline keeps working without torch installed. Importing it without torch raises
a clear message rather than an opaque ImportError deep in a notebook.

What is in here and why
-----------------------
GraphSAGE / GAT / GCN   Standard message-passing baselines via PyG's own layers.
                        They exist so the exotic models have something honest to
                        be compared against.

CAREGNNSimple           Implements the two mechanisms that the CARE-GNN paper
                        (Dou et al., CIKM 2020, DOI 10.1145/3340531.3411903)
                        identifies as necessary on fraud graphs: *feature
                        camouflage* means a fraudster's attributes look normal, so
                        averaging over neighbours drags the representation toward
                        normal; *relation camouflage* means most neighbours are
                        innocent, so the neighbourhood is mostly noise. The fix is
                        to select which neighbours to aggregate. We approximate
                        the paper's similarity-based selection with an attention
                        gate that learns to down-weight dissimilar neighbours.
                        This is a simplification of the published algorithm (the
                        original also does per-layer neighbour sampling and a
                        label-balanced loss); it is labelled as such.

BWGNN                   Band-pass graph filter from Tang et al., ICML 2022
                        (arXiv:2205.15508). Their core finding is that fraud
                        graphs are *heterophilous* -- connected nodes often have
                        different labels -- which shifts spectral energy to the
                        right, so low-pass filters (GCN, GraphSAGE) smooth the
                        signal away. A band-pass filter keeps mid frequencies.
                        The filter is a Beta kernel evaluated on the Laplacian
                        spectrum and applied through Chebyshev polynomials, which
                        avoids eigendecomposition.

IMPORTANT -- verification status
--------------------------------
NONE OF THIS FILE HAS BEEN EXECUTED. The sandbox has no torch and no
torch_geometric and 1 GB of RAM. It has been syntax-checked only (`py_compile`),
which catches typos but proves nothing about runtime behaviour, tensor shapes, or
whether the PyG API calls match the installed version. Treat every line as
unverified until the Colab notebook runs it, and expect to fix shape errors on
the first pass. The scipy/sklearn half of the project is the verified half.
"""
from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import scipy.sparse as sp
    from torch_geometric.data import Data
    from torch_geometric.nn import GATConv, GCNConv, SAGEConv
except ImportError as exc:  # pragma: no cover - sandbox path
    raise ImportError(
        "jale.models.torch_gnn requires torch and torch_geometric.\n"
        "This module runs on Colab, not in the sandbox. In a Colab cell run:\n"
        "    !pip install -q torch_geometric\n"
        "(torch itself is preinstalled on Colab GPU runtimes.)\n"
        f"original error: {exc}"
    ) from exc


# --------------------------------------------------------------------------
# scipy sparse -> PyG Data
# --------------------------------------------------------------------------
def to_pyg_data(A: sp.spmatrix, X: np.ndarray, y: np.ndarray,
                train_mask: np.ndarray, val_mask: np.ndarray,
                test_mask: np.ndarray, edge_weight: bool = False) -> Data:
    """Convert our scipy representation into a PyG `Data` object.

    `A` is the application-to-application affinity from
    `LenderGraph.cooccurrence_union()`. It is symmetric by construction (we set
    the diagonal to zero and never add asymmetric edges), which PyG requires for
    an undirected `edge_index`. We assert that rather than assume it, because a
    silently directed graph would quietly change every downstream result.
    """
    A = sp.coo_matrix(A)
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"affinity must be square, got {A.shape}")
    if (A - A.T).nnz != 0:
        raise ValueError("affinity must be symmetric; PyG would treat it as directed")

    edge_index = torch.from_numpy(np.vstack([A.row, A.col]).astype(np.int64))
    data = Data(
        x=torch.from_numpy(np.asarray(X, dtype=np.float32)),
        y=torch.from_numpy(np.asarray(y, dtype=np.int64)),
        edge_index=edge_index,
        train_mask=torch.from_numpy(np.asarray(train_mask, dtype=bool)),
        val_mask=torch.from_numpy(np.asarray(val_mask, dtype=bool)),
        test_mask=torch.from_numpy(np.asarray(test_mask, dtype=bool)),
    )
    if edge_weight:
        data.edge_weight = torch.from_numpy(A.data.astype(np.float32))
    data.num_classes = 2
    return data


# --------------------------------------------------------------------------
# Standard message-passing baselines
# --------------------------------------------------------------------------
class BaselineGNN(nn.Module):
    """GraphSAGE / GAT / GCN behind one interface.

    Two layers, ReLU, dropout. Deliberately small: on graphs this sparse a deep
    stack over-smooths and the extra depth buys nothing but variance.
    """

    def __init__(self, in_dim: int, hidden: int = 128, kind: str = "sage",
                 heads: int = 4, dropout: float = 0.5, num_classes: int = 2):
        super().__init__()
        self.kind = kind.lower()
        self.dropout = dropout
        if self.kind == "sage":
            self.c1 = SAGEConv(in_dim, hidden)
            self.c2 = SAGEConv(hidden, hidden)
        elif self.kind == "gat":
            # heads * out must equal `hidden` so the second layer's input matches
            self.c1 = GATConv(in_dim, hidden // heads, heads=heads)
            self.c2 = GATConv(hidden, hidden, heads=1)
        elif self.kind == "gcn":
            self.c1 = GCNConv(in_dim, hidden)
            self.c2 = GCNConv(hidden, hidden)
        else:
            raise ValueError(f"unknown kind {kind!r}; use sage/gat/gcn")
        self.lin = nn.Linear(hidden, num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.c1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.c2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin(x)


# --------------------------------------------------------------------------
# CARE-GNN (simplified)
# --------------------------------------------------------------------------
class CAREGNNSimple(nn.Module):
    """Neighbour-selection GNN, simplified from Dou et al. (CIKM 2020).

    The published algorithm estimates, per layer, which neighbours are likely
    camouflage and samples away from them, then balances the loss across classes.
    Here we replace the explicit sampling with a learned gate: each neighbour's
    contribution is scaled by a learned compatibility score, so dissimilar
    neighbours contribute less to the aggregate.

    This is a *deliberate simplification* and should be described that way in any
    write-up. It keeps the mechanism the paper argues matters (do not average
    blindly over a camouflaged neighbourhood) while staying cheap enough to run
    in a notebook. If CARE-GNN's exact behaviour is needed, use the authors'
    repository rather than this class.
    """

    def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.5,
                 num_classes: int = 2):
        super().__init__()
        self.dropout = dropout
        # gate maps the concatenation of (centre, neighbour) to a scalar weight
        self.gate1 = nn.Linear(2 * in_dim, 1)
        self.gate2 = nn.Linear(2 * hidden, 1)
        self.w1 = nn.Linear(in_dim, hidden)
        self.w2 = nn.Linear(hidden, hidden)
        self.lin = nn.Linear(hidden, num_classes)

    @staticmethod
    def _gated_agg(x, edge_index, gate, proj, dropout, training):
        row, col = edge_index[0], edge_index[1]          # row -> col
        # The gate scores each edge from the pair of raw endpoint features. It is
        # deliberately computed BEFORE projection so its input width is always
        # 2 * (width of x at this layer) -- which is what gate1/gate2 are sized for.
        pair = torch.cat([x[col], x[row]], dim=1)
        alpha = torch.sigmoid(gate(pair)).squeeze(-1)    # per-edge weight in (0,1)

        # Aggregate in the PROJECTED space, not the input space. Doing it the
        # other way round produces a [N, in_dim] aggregate that cannot be added
        # to the [N, hidden] projection -- that was a genuine shape bug, hit on
        # the first Colab run: 128 vs 86 at layer 1.
        h = proj(x)                                      # [N, hidden]
        msg = alpha.unsqueeze(-1) * h[row]               # [E, hidden]
        out = torch.zeros_like(h)
        out.index_add_(0, col, msg)
        deg = torch.zeros(h.size(0), device=h.device, dtype=h.dtype).index_add_(
            0, col, alpha).clamp(min=1e-9).unsqueeze(-1)
        out = out / deg                                  # weighted mean of neighbours
        out = F.relu(h + out)                            # residual: self + neighbours
        return F.dropout(out, p=dropout, training=training)

    def forward(self, x, edge_index):
        x = self._gated_agg(x, edge_index, self.gate1, self.w1,
                            self.dropout, self.training)
        x = self._gated_agg(x, edge_index, self.gate2, self.w2,
                            self.dropout, self.training)
        return self.lin(x)


# --------------------------------------------------------------------------
# BWGNN band-pass filter
# --------------------------------------------------------------------------
class BetaBandPass(nn.Module):
    """Beta-kernel band-pass filter over the Laplacian spectrum.

    g(lambda) = 1 - I_{lambda/2}(alpha, beta)

    where I is the regularised incomplete beta function and lambda in [0, 2] is a
    Laplacian eigenvalue. With alpha = beta = 1 this is a linear high-pass ramp;
    larger alpha shifts the pass-band right, which is the knob the paper tunes to
    match how far right the fraud graph's spectrum has shifted.

    We evaluate it at Chebyshev nodes to get polynomial coefficients, so the
    filter is applied by Chebyshev recursion instead of eigendecomposition --
    O(k * nnz) rather than O(n^3), which is the only reason it is usable at all.
    """

    def __init__(self, alpha: float = 2.0, beta: float = 2.0, order: int = 10):
        super().__init__()
        self.alpha, self.beta, self.order = alpha, beta, order

    def _freq_response(self, lam: torch.Tensor) -> torch.Tensor:
        x = (lam / 2.0).clamp(0.0, 1.0)
        return 1.0 - torch.special.betainc(
            torch.tensor(float(self.alpha), device=lam.device, dtype=lam.dtype),
            torch.tensor(float(self.beta), device=lam.device, dtype=lam.dtype),
            x,
        )

    def get_coefficients(self) -> torch.Tensor:
        """Chebyshev coefficients of the filter, via the discrete cosine transform."""
        k = self.order
        # Chebyshev nodes mapped into [0, 2]
        j = torch.arange(k + 1, dtype=torch.float32)
        lam = 1.0 - torch.cos(torch.pi * j / k)          # in [0, 2]
        gj = self._freq_response(lam)
        n = torch.arange(k + 1, dtype=torch.float32)
        coeffs = torch.zeros(k + 1)
        for m in range(k + 1):
            # DCT-II of the sampled response
            coeffs[m] = (gj * torch.cos(torch.pi * m * j / k)).sum() * (2.0 / k)
        coeffs[0] /= 2.0
        return coeffs


class BWGNN(nn.Module):
    """Band-pass GNN from Tang et al. (ICML 2022)."""

    def __init__(self, in_dim: int, hidden: int = 128, order: int = 10,
                 alpha: float = 2.0, beta: float = 2.0, dropout: float = 0.5,
                 num_classes: int = 2):
        super().__init__()
        self.order = order
        self.dropout = dropout
        self.filter = BetaBandPass(alpha, beta, order)
        self.lin1 = nn.Linear(in_dim, hidden)
        # one weight per Chebyshev order, per layer
        self.theta1 = nn.Parameter(torch.randn(order + 1, hidden, hidden) * 0.01)
        self.theta2 = nn.Parameter(torch.randn(order + 1, hidden, hidden) * 0.01)
        self.lin2 = nn.Linear(hidden, hidden)
        self.lin3 = nn.Linear(hidden, num_classes)

    @staticmethod
    def _norm_laplacian(edge_index, n):
        """Symmetric normalised Laplacian L = I - D^-1/2 A D^-1/2, as edge lists."""
        row, col = edge_index[0], edge_index[1]
        deg = torch.zeros(n, device=row.device).index_add_(
            0, row, torch.ones(row.size(0), device=row.device))
        dinv = deg.clamp(min=1).pow(-0.5)
        w = dinv[row] * dinv[col]
        return row, col, w

    def _apply_L(self, z, row, col, w):
        """L z for the symmetric normalised Laplacian L = I - D^-1/2 A D^-1/2."""
        az = torch.zeros_like(z)
        az.index_add_(0, col, z[row] * w.unsqueeze(-1))
        return z - az

    def _cheb(self, x, row, col, w, theta):
        """Apply sum_k theta_k T_k(L) to x by Chebyshev recursion.

        T_0 = x, T_1 = L x, T_k = 2 L T_{k-1} - T_{k-2}.

        An earlier version of this applied L to T_{k-2} instead of T_{k-1}, which
        does not raise -- it silently computes the wrong polynomials, so the
        filter weights no longer correspond to the intended band-pass response.
        The recursion is written out explicitly here to make the indexing
        impossible to get subtly wrong.
        """
        out = torch.einsum("nd,de->ne", x, theta[0])
        if self.order < 1:
            return out
        T_km2 = x                                  # T_0
        T_km1 = self._apply_L(x, row, col, w)      # T_1 = L T_0
        out = out + torch.einsum("nd,de->ne", T_km1, theta[1])
        for k in range(2, self.order + 1):
            T_k = 2.0 * self._apply_L(T_km1, row, col, w) - T_km2
            out = out + torch.einsum("nd,de->ne", T_k, theta[k])
            T_km2, T_km1 = T_km1, T_k
        return out

    def forward(self, x, edge_index):
        row, col, w = self._norm_laplacian(edge_index, x.size(0))
        x = F.relu(self.lin1(x))
        x = F.relu(self._cheb(x, row, col, w, self.theta1))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.relu(self._cheb(x, row, col, w, self.theta2))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin3(x)


# --------------------------------------------------------------------------
# Unified training loop
# --------------------------------------------------------------------------
def build_model(kind: str, in_dim: int, **kw):
    kind = kind.lower()
    if kind in ("sage", "gat", "gcn"):
        return BaselineGNN(in_dim, kind=kind, **kw)
    if kind == "caregnn":
        return CAREGNNSimple(in_dim, **kw)
    if kind == "bwgnn":
        return BWGNN(in_dim, **kw)
    raise ValueError(f"unknown model {kind!r}; use sage/gat/gcn/caregnn/bwgnn")


def train_and_eval(model, data, epochs: int = 200, lr: float = 1e-2,
                   weight_decay: float = 5e-4, pos_weight: float | None = None,
                   patience: int = 30, device: str = "cuda", verbose: bool = False):
    """Train on `train_mask`, early-stop on `val_mask`, score `test_mask`.

    Returns the raw test logits so the caller can compute whatever metric it
    likes. We do NOT pick the threshold on the test set -- that would be
    evaluation-time leakage and would invalidate the ring-disjoint protocol.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
    model = model.to(device)
    data = data.to(device)

    n_pos = int(data.y[data.train_mask].sum().item())
    n_neg = int((data.train_mask.sum() - n_pos).item())
    if pos_weight is None and n_pos > 0:
        pos_weight = n_neg / n_pos          # class imbalance correction
    pw = torch.tensor([1.0, float(pos_weight or 1.0)], device=device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val, best_state, bad = -1.0, None, 0

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask],
                               weight=pw)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            prob = torch.softmax(logits, dim=1)[:, 1]
            yt = data.y[data.val_mask].cpu().numpy()
            yv = prob[data.val_mask].cpu().numpy()
        val = average_precision_score(yt, yv) if 0 < yt.sum() < len(yt) else 0.0

        if val > best_val:
            best_val, bad = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
        if verbose and epoch % 20 == 0:
            print(f"  epoch {epoch:3d}  loss {loss.item():.4f}  val AUC-PR {val:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        prob = torch.softmax(logits, dim=1)[:, 1]

    out = prob.cpu().numpy()
    yt = data.y.cpu().numpy()
    te = data.test_mask.cpu().numpy()
    metrics = {
        "auc_pr": float(average_precision_score(yt[te], out[te])),
        "auc_roc": float(roc_auc_score(yt[te], out[te])),
        "best_val_auc_pr": float(best_val),
        "epochs_run": epoch + 1,
    }
    return out, metrics
