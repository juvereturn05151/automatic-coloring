"""
File Name:    graph_matcher.py
Author(s):    Ju-ve Chankasemporn (with ChatGPT assist)
Description:  Enhanced shape-aware graph matching for region correspondence
              between reference and target images.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import cv2


@dataclass
class RegionNode:
    """Single node in a Region Adjacency Graph (RAG)."""
    idx: int
    area: float
    centroid: Tuple[float, float]
    bbox: Tuple[int, int, int, int]
    degree: int
    neighbors: List[int]
    shape_signature: np.ndarray


class GraphMatcher:
    """
    Enhanced Shape-Aware Graph Matcher using region adjacency graphs.
    """

    def __init__(
        self,
        nbins_shape: int = 16,
        pos_weight: float = 2.0,
        area_weight: float = 1.0,
        degree_weight: float = 1.0,
        shape_weight: float = 3.0,
        max_cost: float = 1e3,
        debug: bool = False,
    ):
        self.nbins_shape = nbins_shape
        self.pos_weight = pos_weight
        self.area_weight = area_weight
        self.degree_weight = degree_weight
        self.shape_weight = shape_weight
        self.max_cost = max_cost
        self.debug = debug

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------
    def match(
        self,
        ref_objects: List[dict],
        tgt_objects: List[dict],
    ) -> Dict[int, Optional[int]]:
        """
        Returns mapping[target_index] = reference_index or None.
        """
        if self.debug:
            print("\n[GraphMatcher] Building graphs...")

        if not ref_objects or not tgt_objects:
            return {}

        img_shape = ref_objects[0]["mask"].shape
        h, w = img_shape[:2]

        ref_nodes = self._build_graph(ref_objects, w, h, "Reference")
        tgt_nodes = self._build_graph(tgt_objects, w, h, "Target")

        if self.debug:
            print(f"[GraphMatcher] Reference nodes: {len(ref_nodes)}")
            print(f"[GraphMatcher] Target nodes:    {len(tgt_nodes)}")

        cost_matrix = self._compute_cost_matrix(ref_nodes, tgt_nodes)

        if self.debug:
            print("\n[GraphMatcher] Cost matrix shape:", cost_matrix.shape)
            print("[GraphMatcher] Raw (unpadded) cost matrix:")
            for i in range(len(tgt_nodes)):
                row = ", ".join(f"{cost_matrix[i, j]:.3f}" for j in range(len(ref_nodes)))
                print(f"  T{i}: [{row}]")

        row_ind, col_ind = self._hungarian(cost_matrix)

        n_tgt = len(tgt_nodes)
        n_ref = len(ref_nodes)
        mapping: Dict[int, Optional[int]] = {i: None for i in range(n_tgt)}

        for r, c in zip(row_ind, col_ind):
            if r >= n_tgt or c >= n_ref:
                continue

            cost_val = cost_matrix[r, c]
            if cost_val >= self.max_cost * 0.5:
                continue

            tgt_idx = tgt_nodes[r].idx
            ref_idx = ref_nodes[c].idx
            mapping[tgt_idx] = ref_idx

            if self.debug:
                print(
                    f"  Match T{tgt_idx} -> R{ref_idx} "
                    f"(row {r}, col {c}, cost {cost_val:.3f})"
                )

        return mapping

    # --------------------------------------------------------
    # Graph building
    # --------------------------------------------------------
    def _build_graph(
        self,
        objects: List[dict],
        width: int,
        height: int,
        label: str = "",
    ) -> List[RegionNode]:
        nodes: List[RegionNode] = []

        areas = []
        centroids = []
        bboxes = []
        masks = []

        for idx, obj in enumerate(objects):
            mask = obj["mask"]
            masks.append(mask)

            area_px = float(np.count_nonzero(mask))
            areas.append(area_px)

            ys, xs = np.nonzero(mask)
            if len(xs) == 0:
                cx, cy = 0.5, 0.5
            else:
                cx = float(xs.mean()) / max(width - 1, 1)
                cy = float(ys.mean()) / max(height - 1, 1)
            centroids.append((cx, cy))

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                x, y, w_box, h_box = cv2.boundingRect(contours[0])
            else:
                x = y = w_box = h_box = 0
            bboxes.append((x, y, w_box, h_box))

        total_area = max(sum(areas), 1.0)
        norm_areas = [a / total_area for a in areas]

        kernel = np.ones((3, 3), np.uint8)
        dilated_masks = [cv2.dilate(m, kernel, iterations=1) for m in masks]

        neighbors_list: List[List[int]] = [[] for _ in range(len(objects))]

        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                if not self._bboxes_might_touch(bboxes[i], bboxes[j]):
                    continue

                overlap = cv2.bitwise_and(dilated_masks[i], dilated_masks[j])
                if np.any(overlap):
                    neighbors_list[i].append(j)
                    neighbors_list[j].append(i)

        for idx in range(len(objects)):
            signature = self._compute_shape_signature(
                masks[idx], centroids[idx], self.nbins_shape
            )
            node = RegionNode(
                idx=idx,
                area=norm_areas[idx],
                centroid=centroids[idx],
                bbox=bboxes[idx],
                degree=len(neighbors_list[idx]),
                neighbors=neighbors_list[idx],
                shape_signature=signature,
            )
            nodes.append(node)

        if self.debug:
            print(f"[GraphMatcher] Built graph for {label}:")
            for n in nodes:
                print(
                    f"  Node {n.idx}: area={n.area:.3f}, "
                    f"centroid=({n.centroid[0]:.3f},{n.centroid[1]:.3f}), "
                    f"degree={n.degree}"
                )

        return nodes

    @staticmethod
    def _bboxes_might_touch(b1, b2, margin: int = 2) -> bool:
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2

        if x1 > x2 + w2 + margin:
            return False
        if x2 > x1 + w1 + margin:
            return False
        if y1 > y2 + h2 + margin:
            return False
        if y2 > y1 + h1 + margin:
            return False
        return True

    def _compute_shape_signature(
        self,
        mask: np.ndarray,
        centroid: Tuple[float, float],
        nbins: int,
    ) -> np.ndarray:
        h, w = mask.shape[:2]
        cy = centroid[1] * (h - 1)
        cx = centroid[0] * (w - 1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return np.zeros(nbins, dtype=np.float32)

        contour = max(contours, key=cv2.contourArea)
        pts = contour.reshape(-1, 2)

        xs = pts[:, 0].astype(np.float32)
        ys = pts[:, 1].astype(np.float32)
        dists = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
        if len(dists) == 0:
            return np.zeros(nbins, dtype=np.float32)

        max_dist = float(dists.max())
        if max_dist < 1e-4:
            return np.zeros(nbins, dtype=np.float32)

        dists_norm = dists / max_dist

        hist, _ = np.histogram(
            dists_norm, bins=nbins, range=(0.0, 1.0), density=True
        )
        hist = hist.astype(np.float32)
        norm = np.linalg.norm(hist)
        if norm > 1e-6:
            hist /= norm
        return hist

    # --------------------------------------------------------
    # Cost matrix
    # --------------------------------------------------------
    def _compute_cost_matrix(
        self,
        ref_nodes: List[RegionNode],
        tgt_nodes: List[RegionNode],
    ) -> np.ndarray:
        n_ref = len(ref_nodes)
        n_tgt = len(tgt_nodes)
        n = max(n_ref, n_tgt)

        cost = np.full((n, n), self.max_cost, dtype=np.float32)

        max_degree = max(
            [n.degree for n in ref_nodes] + [n.degree for n in tgt_nodes] + [1]
        )

        for i_t, t in enumerate(tgt_nodes):
            for j_r, r in enumerate(ref_nodes):
                pos_dist = np.sqrt(
                    (t.centroid[0] - r.centroid[0]) ** 2
                    + (t.centroid[1] - r.centroid[1]) ** 2
                )
                area_dist = abs(t.area - r.area)
                deg_dist = abs(t.degree - r.degree) / max_degree
                shape_dist = np.linalg.norm(
                    t.shape_signature - r.shape_signature
                )

                total = (
                    self.pos_weight * pos_dist
                    + self.area_weight * area_dist
                    + self.degree_weight * deg_dist
                    + self.shape_weight * shape_dist
                )

                cost[i_t, j_r] = total

        return cost

    # --------------------------------------------------------
    # Hungarian algorithm (stable)
    # --------------------------------------------------------
    def _hungarian(self, cost):
        """
        Stable Hungarian algorithm for square matrices.
        Returns row_ind, col_ind minimizing total cost.
        """
        C = cost.copy()
        n = C.shape[0]

        # Row + column reduction
        C -= C.min(axis=1, keepdims=True)
        C -= C.min(axis=0, keepdims=True)

        star = np.zeros((n, n), dtype=bool)
        prime = np.zeros((n, n), dtype=bool)
        row_cover = np.zeros(n, dtype=bool)
        col_cover = np.zeros(n, dtype=bool)

        # Initial starring
        for i in range(n):
            for j in range(n):
                if C[i, j] == 0 and not row_cover[i] and not col_cover[j]:
                    star[i, j] = True
                    row_cover[i] = True
                    col_cover[j] = True

        row_cover[:] = False
        col_cover[:] = False

        def cover_columns_with_stars():
            for j in range(n):
                if star[:, j].any():
                    col_cover[j] = True

        cover_columns_with_stars()

        while col_cover.sum() < n:
            # Find uncovered zero
            found_zero = False
            z_row = z_col = None
            for i in range(n):
                if row_cover[i]:
                    continue
                for j in range(n):
                    if col_cover[j]:
                        continue
                    if C[i, j] == 0:
                        prime[i, j] = True
                        found_zero = True
                        z_row, z_col = i, j
                        break
                if found_zero:
                    break

            if not found_zero:
                # Adjust matrix
                uncovered = C[~row_cover][:, ~col_cover]
                min_val = uncovered.min()
                C[row_cover[:, None] & col_cover[None, :]] += min_val
                C[~row_cover[:, None] & ~col_cover[None, :]] -= min_val
                continue

            star_col = np.where(star[z_row])[0]
            if star_col.size == 1:
                # Cover row, uncover column
                row_cover[z_row] = True
                col_cover[star_col[0]] = False
            else:
                self._augment_path(star, prime, z_row, z_col)
                prime[:, :] = False
                row_cover[:] = False
                col_cover[:] = False
                cover_columns_with_stars()

        row_ind = []
        col_ind = []
        for i in range(n):
            j_star = np.where(star[i])[0]
            if j_star.size == 1:
                row_ind.append(i)
                col_ind.append(j_star[0])

        return np.array(row_ind), np.array(col_ind)

    def _augment_path(self, star, prime, row, col):
        """
        Build and flip an augmenting path starting at the primed zero (row, col).
        """
        path = [(row, col)]

        while True:
            r, c = path[-1]

            # Starred zero in same column?
            r_star = np.where(star[:, c])[0]
            if r_star.size == 1:
                r2 = r_star[0]
                path.append((r2, c))
            else:
                break

            # Primed zero in that row
            c_prime = np.where(prime[r2])[0][0]
            path.append((r2, c_prime))

        # Flip stars along the path
        for r, c in path:
            star[r, c] = not star[r, c]
