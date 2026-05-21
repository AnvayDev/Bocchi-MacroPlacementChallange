"""
ABPlace-Inspired Elliptical Placer (Team Bocchi–AP)
===================================

Core idea from ABPlace (ICCAD 2022):
  Reduce 2D placement (x,y) → 1D optimization (angle θ) by constraining
  macros to an ellipse inscribed in the canvas boundary. This naturally
  places macros on the periphery (like expert designers do) and makes
  optimization dramatically easier — N angles instead of 2N coordinates.
  1. Extract connectivity graph (weighted edges between hard macros)
  2.  Project initial macro positions → angles on boundary ellipse
  3. Gradient descent on angles: minimize wirelength + spacing penalty
  4. Convert optimized angles → (x,y) positions
  5. Legalize: resolve any remaining overlaps with minimum displacement
  6. SA refinement: fine-tune positions for wirelength/density/congestion

Usage:
    uv run evaluate submissions/abplace_placer.py -b ibm01
    uv run evaluate submissions/abplace_placer.py --all
"""

import math
import random
import torch
import numpy as np
from pathlib import Path
from macro_place.benchmark import Benchmark



# Connectivity extraction (reused from will_seed)


def _load_plc(name):
    """Load PlacementCost object for a benchmark (needed for connectivity)."""
    from macro_place.loader import load_benchmark_from_dir, load_benchmark
    root = Path("external/MacroPlacement/Testcases/ICCAD04") / name
    if root.exists():
        _, plc = load_benchmark_from_dir(str(root))
        return plc
    ng45 = {"ariane133_ng45": "ariane133", "ariane136_ng45": "ariane136",
            "nvdla_ng45": "nvdla", "mempool_tile_ng45": "mempool_tile"}
    d = ng45.get(name)
    if d:
        base = Path("external/MacroPlacement/Flows/NanGate45") / d / "netlist" / "output_CT_Grouping"
        if (base / "netlist.pb.txt").exists():
            _, plc = load_benchmark(str(base / "netlist.pb.txt"), str(base / "initial.plc"))
            return plc
    return None


def _extract_edges(benchmark, plc):
    """Build weighted edge list between hard macros from net hypergraph."""
    name_to_bidx = {}
    for bidx, idx in enumerate(plc.hard_macro_indices):
        name_to_bidx[plc.modules_w_pins[idx].get_name()] = bidx

    # Also map soft macros and ports for anchor forces
    name_to_soft = {}
    for bidx, idx in enumerate(plc.soft_macro_indices):
        name_to_soft[plc.modules_w_pins[idx].get_name()] = bidx

    edge_dict = {}
    # Anchor edges: hard macro ↔ fixed soft/port position (centroid of non-hard pins)
    anchor_dict = {}  # hard_macro_idx -> list of (x, y, weight)

    for driver, sinks in plc.nets.items():
        hard_macros = set()
        fixed_positions = []  # positions of soft macros / ports in this net

        for pin_name in [driver] + sinks:
            parent = pin_name.split("/")[0]
            if parent in name_to_bidx:
                hard_macros.add(name_to_bidx[parent])
            else:
                # It's a soft macro or port — get its position as anchor
                if parent in plc.mod_name_to_indices:
                    pidx = plc.mod_name_to_indices[parent]
                    px, py = plc.modules_w_pins[pidx].get_pos()
                    fixed_positions.append((px, py))

        # Hard-to-hard edges
        if len(hard_macros) >= 2:
            ml = sorted(hard_macros)
            w = 1.0 / (len(ml) - 1)
            for i in range(len(ml)):
                for j in range(i + 1, len(ml)):
                    pair = (ml[i], ml[j])
                    edge_dict[pair] = edge_dict.get(pair, 0) + w

        # Anchor forces: each hard macro in this net is attracted to centroid
        # of non-hard (soft + port) pins
        if hard_macros and fixed_positions:
            cx = sum(p[0] for p in fixed_positions) / len(fixed_positions)
            cy = sum(p[1] for p in fixed_positions) / len(fixed_positions)
            w = 1.0 / max(len(hard_macros), 1)
            for hidx in hard_macros:
                if hidx not in anchor_dict:
                    anchor_dict[hidx] = []
                anchor_dict[hidx].append((cx, cy, w))

    # Convert edge_dict
    if edge_dict:
        edges = torch.tensor(list(edge_dict.keys()), dtype=torch.long)
        edge_weights = torch.tensor([edge_dict[e] for e in edge_dict], dtype=torch.float32)
    else:
        edges = torch.zeros(0, 2, dtype=torch.long)
        edge_weights = torch.zeros(0)

    # Consolidate anchors: weighted average position per macro
    anchors = {}  # macro_idx -> (x, y, total_weight)
    for hidx, alist in anchor_dict.items():
        total_w = sum(a[2] for a in alist)
        ax = sum(a[0] * a[2] for a in alist) / total_w
        ay = sum(a[1] * a[2] for a in alist) / total_w
        anchors[hidx] = (ax, ay, total_w)

    return edges, edge_weights, anchors


# ABPlace: Elliptical angle optimization


def _abplace_optimize(benchmark, edges, edge_weights, anchors,
                      num_iters=800, lr=0.2):
    """
    ABPlace core: optimize macro angles on a boundary ellipse.

    1. Compute ellipse inscribed in canvas (with margin for macro sizes)
    2. Initialize angles from current positions
    3. Adam on angles to minimize wirelength + angular spacing penalty
    4. Return (x, y) positions on the ellipse
    """
    n_hard = benchmark.num_hard_macros
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    cx, cy = cw / 2, ch / 2

    sizes = benchmark.macro_sizes[:n_hard]
    movable = benchmark.get_movable_mask()[:n_hard]
    pos_init = benchmark.macro_positions[:n_hard].clone()

    # Ellipse semi-axes: canvas half-size minus margin for largest macros
    max_hw = sizes[:, 0].max().item() / 2  # max half-width
    max_hh = sizes[:, 1].max().item() / 2  # max half-height
    margin_x = max_hw + 0.5
    margin_y = max_hh + 0.5
    a = cw / 2 - margin_x  # semi-axis x
    b = ch / 2 - margin_y  # semi-axis y

    if a <= 0 or b <= 0:
        # Canvas too small for ellipse approach, fall back to initial
        return pos_init

    # Initialize angles from initial positions
    dx = pos_init[:, 0] - cx
    dy = pos_init[:, 1] - cy
    init_angles = torch.atan2(dy / b, dx / a)

    # Learnable angles (only for movable macros, but we optimize all then mask)
    theta = init_angles.clone().requires_grad_(True)

    # Precompute half-sizes for spacing
    half_w = sizes[:, 0] / 2
    half_h = sizes[:, 1] / 2

    # Arc-length spacing: minimum angular gap between adjacent macros
    # approximate by the macro's angular "width" on the ellipse
    perimeter_approx = math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))

    optimizer = torch.optim.Adam([theta], lr=lr)

    for iteration in range(num_iters):
        optimizer.zero_grad()

        # Convert angles to positions
        x = cx + a * torch.cos(theta)
        y = cy + b * torch.sin(theta)

        # ── Wirelength cost: L1 distance for each edge ──
        if len(edges) > 0:
            ei, ej = edges[:, 0], edges[:, 1]
            dx_e = torch.abs(x[ei] - x[ej])
            dy_e = torch.abs(y[ei] - y[ej])
            wl_cost = (edge_weights * (dx_e + dy_e)).sum()
        else:
            wl_cost = torch.tensor(0.0)

        # ── Anchor cost: attraction to soft macro / port centroids ──
        anchor_cost = torch.tensor(0.0)
        if anchors:
            for hidx, (ax, ay, aw) in anchors.items():
                anchor_cost = anchor_cost + aw * (
                        torch.abs(x[hidx] - ax) + torch.abs(y[hidx] - ay)
                )

        # ── Angular spacing penalty: prevent macros from bunching up ──
        # Sort angles and penalize pairs that are too close
        sorted_theta, sorted_idx = torch.sort(theta)
        # Circular differences
        diffs = sorted_theta[1:] - sorted_theta[:-1]
        # Wrap-around
        wrap_diff = (sorted_theta[0] + 2 * math.pi) - sorted_theta[-1]
        all_diffs = torch.cat([diffs, wrap_diff.unsqueeze(0)])
        # Minimum angular gap (heuristic: distribute evenly with some slack)
        min_gap = (2 * math.pi / n_hard) * 0.3
        # Penalize gaps smaller than min_gap
        spacing_violations = torch.relu(min_gap - all_diffs)
        spacing_cost = spacing_violations.sum()

        # ── Overlap penalty (soft, differentiable) ──
        # For all pairs within angular proximity, check spatial overlap
        overlap_cost = torch.tensor(0.0)
        if n_hard <= 600:  # tractable for all IBM benchmarks
            # Vectorized pairwise overlap using broadcasting
            px = x.unsqueeze(1)  # [N, 1]
            py = y.unsqueeze(1)
            dx_pair = torch.abs(px - px.t())  # [N, N]
            dy_pair = torch.abs(py - py.t())
            sep_x = (half_w.unsqueeze(1) + half_w.unsqueeze(0)) + 0.05  # gap
            sep_y = (half_h.unsqueeze(1) + half_h.unsqueeze(0)) + 0.05
            overlap_x = torch.relu(sep_x - dx_pair)
            overlap_y = torch.relu(sep_y - dy_pair)
            overlap_area = overlap_x * overlap_y
            # Zero out diagonal
            mask = 1 - torch.eye(n_hard)
            overlap_cost = (overlap_area * mask).sum() / 2

        # ── Total loss ──
        # Ramp up overlap penalty over iterations
        overlap_weight = 0.5 + 4.5 * (iteration / num_iters)
        anchor_weight = 0.3  # moderate anchor attraction
        spacing_weight = 2.0

        loss = (wl_cost
                + anchor_weight * anchor_cost
                + spacing_weight * spacing_cost
                + overlap_weight * overlap_cost)

        loss.backward()

        # Zero out gradients for fixed macros
        if not movable.all():
            with torch.no_grad():
                theta.grad[~movable] = 0.0

        optimizer.step()

    # Final positions from optimized angles
    with torch.no_grad():
        x_final = cx + a * torch.cos(theta)
        y_final = cy + b * torch.sin(theta)
        result = torch.stack([x_final, y_final], dim=1)

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Legalization: resolve overlaps with minimum displacement
# ═══════════════════════════════════════════════════════════════════════════

def _legalize(pos_np, movable, sizes_np, half_w, half_h, cw, ch, n_hard):
    """Greedy legalization: place largest macros first, spiral search for non-overlapping slot."""
    sep_x = (sizes_np[:, 0:1] + sizes_np[:, 0:1].T) / 2
    sep_y = (sizes_np[:, 1:2] + sizes_np[:, 1:2].T) / 2
    gap = 0.05  # safety gap for float precision

    # Place largest macros first (they're hardest to fit)
    order = sorted(range(n_hard), key=lambda i: -sizes_np[i, 0] * sizes_np[i, 1])
    placed = np.zeros(n_hard, dtype=bool)
    legal = pos_np.copy()

    for idx in order:
        if not movable[idx]:
            placed[idx] = True
            continue

        # Clamp to canvas bounds first
        legal[idx, 0] = np.clip(legal[idx, 0], half_w[idx], cw - half_w[idx])
        legal[idx, 1] = np.clip(legal[idx, 1], half_h[idx], ch - half_h[idx])

        # Check if current position is already legal
        if placed.any():
            dx = np.abs(legal[idx, 0] - legal[:, 0])
            dy = np.abs(legal[idx, 1] - legal[:, 1])
            conflicts = (dx < sep_x[idx] + gap) & (dy < sep_y[idx] + gap) & placed
            conflicts[idx] = False
            if not conflicts.any():
                placed[idx] = True
                continue

        # Spiral search for nearest legal position
        step = max(sizes_np[idx, 0], sizes_np[idx, 1]) * 0.25
        target_x, target_y = pos_np[idx, 0], pos_np[idx, 1]
        best_p = legal[idx].copy()
        best_d = float('inf')

        for r in range(1, 200):
            found = False
            for dxm in range(-r, r + 1):
                for dym in range(-r, r + 1):
                    if abs(dxm) != r and abs(dym) != r:
                        continue
                    cx_try = np.clip(target_x + dxm * step, half_w[idx], cw - half_w[idx])
                    cy_try = np.clip(target_y + dym * step, half_h[idx], ch - half_h[idx])
                    if placed.any():
                        dx = np.abs(cx_try - legal[:, 0])
                        dy = np.abs(cy_try - legal[:, 1])
                        conflicts = (dx < sep_x[idx] + gap) & (dy < sep_y[idx] + gap) & placed
                        conflicts[idx] = False
                        if conflicts.any():
                            continue
                    d = (cx_try - target_x) ** 2 + (cy_try - target_y) ** 2
                    if d < best_d:
                        best_d = d
                        best_p = np.array([cx_try, cy_try])
                        found = True
            if found:
                break

        legal[idx] = best_p
        placed[idx] = True

    return legal


# ═══════════════════════════════════════════════════════════════════════════
# SA Refinement: fine-tune positions post-legalization
# ═══════════════════════════════════════════════════════════════════════════

def _sa_refine(pos, edges_np, edge_weights_np, movable, sizes_np, half_w, half_h,
               cw, ch, n_hard, num_iters=5000, anchors=None):
    """SA with shift/swap/neighbor-attract moves + overlap rejection."""
    movable_idx = np.where(movable)[0]
    if len(movable_idx) == 0 or len(edges_np) == 0:
        return pos

    pos = pos.copy()
    sep_x = (sizes_np[:, 0:1] + sizes_np[:, 0:1].T) / 2
    sep_y = (sizes_np[:, 1:2] + sizes_np[:, 1:2].T) / 2

    # Build neighbor lists from edges
    neighbors = [[] for _ in range(n_hard)]
    for i, j in edges_np:
        neighbors[i].append(j)
        neighbors[j].append(i)

    # Precompute anchor pull for cost function
    anchor_positions = np.zeros((n_hard, 2))
    anchor_weights = np.zeros(n_hard)
    if anchors:
        for hidx, (ax, ay, aw) in anchors.items():
            if hidx < n_hard:
                anchor_positions[hidx] = [ax, ay]
                anchor_weights[hidx] = aw

    def wl_cost():
        dx = np.abs(pos[edges_np[:, 0], 0] - pos[edges_np[:, 1], 0])
        dy = np.abs(pos[edges_np[:, 0], 1] - pos[edges_np[:, 1], 1])
        c = (edge_weights_np * (dx + dy)).sum()
        # Add anchor cost
        if anchors:
            adx = np.abs(pos[:, 0] - anchor_positions[:, 0])
            ady = np.abs(pos[:, 1] - anchor_positions[:, 1])
            c += 0.3 * (anchor_weights * (adx + ady)).sum()
        return c

    def check_overlap(idx):
        gap = 0.05
        dx = np.abs(pos[idx, 0] - pos[:, 0])
        dy = np.abs(pos[idx, 1] - pos[:, 1])
        overlaps = (dx < sep_x[idx] + gap) & (dy < sep_y[idx] + gap)
        overlaps[idx] = False
        return overlaps.any()

    current_cost = wl_cost()
    best_pos = pos.copy()
    best_cost = current_cost

    T_start = max(cw, ch) * 0.15
    T_end = max(cw, ch) * 0.001

    for step in range(num_iters):
        frac = step / num_iters
        T = T_start * (T_end / T_start) ** frac

        move = random.random()
        i = random.choice(movable_idx)
        old_x, old_y = pos[i, 0], pos[i, 1]

        if move < 0.5:
            # SHIFT
            shift = T * (0.3 + 0.7 * (1 - frac))
            pos[i, 0] = np.clip(pos[i, 0] + random.gauss(0, shift), half_w[i], cw - half_w[i])
            pos[i, 1] = np.clip(pos[i, 1] + random.gauss(0, shift), half_h[i], ch - half_h[i])
        elif move < 0.8:
            # SWAP with connected or random macro
            if neighbors[i] and random.random() < 0.7:
                cands = [j for j in neighbors[i] if movable[j]]
                j = random.choice(cands) if cands else random.choice(movable_idx)
            else:
                j = random.choice(movable_idx)
            if i != j:
                old_jx, old_jy = pos[j, 0], pos[j, 1]
                pos[i, 0] = np.clip(old_jx, half_w[i], cw - half_w[i])
                pos[i, 1] = np.clip(old_jy, half_h[i], ch - half_h[i])
                pos[j, 0] = np.clip(old_x, half_w[j], cw - half_w[j])
                pos[j, 1] = np.clip(old_y, half_h[j], ch - half_h[j])
                if check_overlap(i) or check_overlap(j):
                    pos[i, 0] = old_x; pos[i, 1] = old_y
                    pos[j, 0] = old_jx; pos[j, 1] = old_jy
                    continue
                new_cost = wl_cost()
                delta = new_cost - current_cost
                if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost; best_pos = pos.copy()
                else:
                    pos[i, 0] = old_x; pos[i, 1] = old_y
                    pos[j, 0] = old_jx; pos[j, 1] = old_jy
                continue
        else:
            # MOVE TOWARD connected neighbor
            if neighbors[i]:
                j = random.choice(neighbors[i])
                alpha = random.uniform(0.05, 0.3)
                pos[i, 0] = np.clip(pos[i, 0] + alpha * (pos[j, 0] - pos[i, 0]),
                                    half_w[i], cw - half_w[i])
                pos[i, 1] = np.clip(pos[i, 1] + alpha * (pos[j, 1] - pos[i, 1]),
                                    half_h[i], ch - half_h[i])

        if check_overlap(i):
            pos[i, 0] = old_x; pos[i, 1] = old_y
            continue

        new_cost = wl_cost()
        delta = new_cost - current_cost
        if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost; best_pos = pos.copy()
        else:
            pos[i, 0] = old_x; pos[i, 1] = old_y

    return best_pos
# ═══════════════════════════════════════════════════════════════════════════
# Main Placer Class
# ═══════════════════════════════════════════════════════════════════════════

class ABPlacePlacer:
    """
    ABPlace-inspired elliptical placer with SA refinement.

    Pipeline:
      1. Extract connectivity (hard macro edges + anchor forces)
      2. ABPlace: optimize angles on boundary ellipse (gradient descent)
      3. Legalize: remove overlaps with min displacement
      4. SA refinement: fine-tune with shift/swap/attract moves
    """

    def __init__(self, seed=42, abplace_iters=3000, abplace_lr=0.05,
                 sa_iters=500000):
        self.seed = seed
        self.abplace_iters = abplace_iters
        self.abplace_lr = abplace_lr
        self.sa_iters = sa_iters

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

        n_hard = benchmark.num_hard_macros
        sizes_np = benchmark.macro_sizes[:n_hard].numpy().astype(np.float64)
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        half_w = sizes_np[:, 0] / 2
        half_h = sizes_np[:, 1] / 2
        movable = benchmark.get_movable_mask()[:n_hard].numpy()

        # ── Step 1: Extract connectivity ──────────────────────────────
        plc = _load_plc(benchmark.name)
        if plc is not None:
            edges, edge_weights, anchors = _extract_edges(benchmark, plc)
        else:
            edges = torch.zeros(0, 2, dtype=torch.long)
            edge_weights = torch.zeros(0)
            anchors = {}

        # ── Step 2: ABPlace — elliptical angle optimization ───────────
        print(f"    [ABPlace] Optimizing {n_hard} macro angles on ellipse "
              f"({self.abplace_iters} iters)...", end="", flush=True)
        ab_pos = _abplace_optimize(
            benchmark, edges, edge_weights, anchors,
            num_iters=self.abplace_iters, lr=self.abplace_lr,
        )
        print(" done")

        # ── Step 3: Legalize ──────────────────────────────────────────
        print("    [Legalize] Resolving overlaps...", end="", flush=True)
        pos_np = ab_pos.numpy().astype(np.float64)
        pos_np = _legalize(pos_np, movable, sizes_np, half_w, half_h, cw, ch, n_hard)
        print(" done")

        # ── Step 4: SA refinement ─────────────────────────────────────
        if len(edges) > 0 and self.sa_iters > 0:
            print(f"    [SA] Refining ({self.sa_iters} iters)...", end="", flush=True)
            pos_np = _sa_refine(
                pos_np, edges.numpy(), edge_weights.numpy(),
                movable, sizes_np, half_w, half_h, cw, ch, n_hard,
                num_iters=self.sa_iters, anchors=anchors,
            )
            print(" done")

        # ── Step 5: Assemble full placement ───────────────────────────
        full_pos = benchmark.macro_positions.clone()
        full_pos[:n_hard] = torch.tensor(pos_np, dtype=torch.float32)
        # Soft macros stay at initial positions because every time
        

        return full_pos
