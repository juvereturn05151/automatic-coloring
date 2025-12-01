"""
File Name:    shape_matcher.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.
"""

import os
import cv2
import numpy as np
from skimage.morphology import skeletonize

from contour_extractor import ContourExtractor


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def centroid(contour):
    """Centroid with bbox fallback."""
    M = cv2.moments(contour)
    if M["m00"] != 0:
        return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    x, y, w, h = cv2.boundingRect(contour)
    return (x + w // 2, y + h // 2)


def aspect_ratio(contour):
    """Width / Height of bounding box."""
    x, y, w, h = cv2.boundingRect(contour)
    return w / float(h + 1e-6)


def extent(contour):
    """Extent = contour_area / bounding_box_area."""
    a = cv2.contourArea(contour)
    x, y, w, h = cv2.boundingRect(contour)
    return a / float(w * h + 1e-6)


def circularity(contour):
    """
    4 * pi * area / perimeter^2
    ~1.0 for a perfect circle, smaller for rectangles / stars.
    """
    area = cv2.contourArea(contour)
    peri = cv2.arcLength(contour, True)
    if peri <= 1e-6:
        return 0.0
    return float(4.0 * np.pi * area / (peri * peri))


def best_rotational_match(tgt_contour, ref_contour, step_deg=10):
    """
    Try several rotations of the target contour and keep the best matchShapes score.
    Rotation is around (0, 0) which is fine because matchShapes is translation invariant.
    """
    best_score = float("inf")
    for angle in range(0, 360, step_deg):
        M = cv2.getRotationMatrix2D((0, 0), angle, 1.0)
        rotated = cv2.transform(tgt_contour, M)
        score = cv2.matchShapes(rotated, ref_contour, cv2.CONTOURS_MATCH_I1, 0.0)
        if score < best_score:
            best_score = score
    return best_score


def skeleton_midpoint(mask):
    """
    Compute a robust shape center using skeleton midpoint.
    mask: 2D uint8 array (0 or 255)
    """
    # Normalize mask for skeletonize (expects 0/1)
    binary = (mask > 0).astype(np.uint8)

    # Compute skeleton (thin 1-pixel width)
    skel = skeletonize(binary).astype(np.uint8)

    # Get all skeleton coordinates
    ys, xs = np.where(skel > 0)

    if len(xs) == 0:
        # Fallback to bounding box center
        y, x, w, h = cv2.boundingRect(mask)
        return (x + w // 2, y + h // 2)

    # Average of skeleton coordinates
    cx = int(np.mean(xs))
    cy = int(np.mean(ys))

    return (cx, cy)


# ------------------------------------------------------------
# Debug drawing helpers (image-based)
# ------------------------------------------------------------

def _ensure_debug_dir():
    """Create debug directory if missing."""
    debug_dir = "debug"
    os.makedirs(debug_dir, exist_ok=True)
    return debug_dir


def draw_labeled_regions(image_rgb, objects, id_prefix, out_name):
    """
    Draw each region contour with an index label at its skeleton midpoint.

    image_rgb : original RGB image
    objects   : list of dicts from ContourExtractor (must have "contour" and "mask")
    id_prefix : "R" for reference or "T" for target
    out_name  : filename inside debug/ folder
    """
    if image_rgb is None or not objects:
        return

    debug_dir = _ensure_debug_dir()

    # Convert RGB → BGR for OpenCV drawing
    img_bgr = cv2.cvtColor(image_rgb.copy(), cv2.COLOR_RGB2BGR)

    for idx, obj in enumerate(objects):
        cnt = obj["contour"]
        mask = obj.get("mask", None)

        if mask is not None:
            cx, cy = skeleton_midpoint(mask)
        else:
            cx, cy = centroid(cnt)

        # Random-ish color per index (just for visualization)
        color = (
            int(50 + (idx * 70) % 205),
            int(80 + (idx * 40) % 175),
            int(120 + (idx * 90) % 135),
        )

        # Draw contour
        cv2.drawContours(img_bgr, [cnt], -1, color, 2)

        # Draw center
        cv2.circle(img_bgr, (cx, cy), 4, (0, 255, 255), -1)

        # Put label
        label = f"{id_prefix}{idx}"
        cv2.putText(
            img_bgr,
            label,
            (cx + 5, cy - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
        )

    out_path = os.path.join(debug_dir, out_name)
    cv2.imwrite(out_path, img_bgr)
    print(f"[Debug] Saved labeled {id_prefix} regions → {out_path}")


def draw_matching_lines(ref_img, tgt_img, matches, out_name="matching_lines.png"):
    """
    Draw ref image on the left, target on the right,
    and lines connecting matched centers (ref ↔ tgt).

    'matches' is a list of dicts:
        {
          "tgt_idx": int,         # original index in tgt_objs
          "ref_idx": int,         # original index in ref_objs
          "tgt_center": (tx, ty),
          "ref_center": (rx, ry),
        }
    """
    if ref_img is None or tgt_img is None or not matches:
        return

    debug_dir = _ensure_debug_dir()

    ref_h, ref_w = ref_img.shape[:2]
    tgt_h, tgt_w = tgt_img.shape[:2]
    canvas_h = max(ref_h, tgt_h)
    gap = 40

    # Create canvas RGB
    canvas_w = ref_w + gap + tgt_w
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

    # Blit images
    canvas[0:ref_h, 0:ref_w] = ref_img
    canvas[0:tgt_h, ref_w + gap: ref_w + gap + tgt_w] = tgt_img

    # Convert to BGR for drawing
    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    for idx, m in enumerate(matches):
        (rx, ry) = m["ref_center"]
        (tx, ty) = m["tgt_center"]

        # offset target x by left width + gap
        tx_canvas = ref_w + gap + tx
        ty_canvas = ty

        color = (
            int(50 + (idx * 70) % 205),
            int(80 + (idx * 40) % 175),
            int(120 + (idx * 90) % 135),
        )

        # Draw circles
        cv2.circle(canvas_bgr, (rx, ry), 5, color, -1)
        cv2.circle(canvas_bgr, (tx_canvas, ty_canvas), 5, color, -1)

        # Draw line
        cv2.line(canvas_bgr, (rx, ry), (tx_canvas, ty_canvas), color, 2)

        # Put label near target side: "T# → R#"
        label = f"T{m['tgt_idx']}→R{m['ref_idx']}"
        cv2.putText(
            canvas_bgr,
            label,
            (tx_canvas + 5, ty_canvas - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
        )

    out_path = os.path.join(debug_dir, out_name)
    cv2.imwrite(out_path, canvas_bgr)
    print(f"[Debug] Saved matching lines → {out_path}")


# ------------------------------------------------------------
# Main matcher
# ------------------------------------------------------------

class ShapeMatcher:
    def __init__(self, contour_extractor=None):
        self._extractor = contour_extractor or ContourExtractor()

        # Tunable thresholds
        self.AREA_RATIO_MIN = 0.3
        self.AREA_RATIO_MAX = 3.0
        self.ASPECT_TOL     = 0.5   # max |aspect_tgt - aspect_ref|
        self.EXTENT_TOL     = 0.5   # max |extent_tgt - extent_ref|
        self.CIRC_TOL       = 0.20  # max |circ_tgt - circ_ref|

    def match_and_colorize(self, reference_path, target_path):
        """
        Extract shapes from reference and target, match them,
        and transfer colors from reference to target.
        """

        # ----------------------------------------------------
        # Extract objects (contours + masks + mean colors)
        # ----------------------------------------------------
        ref_img, ref_gray, _, ref_objs = self._extractor.extract(reference_path)
        tgt_img, tgt_gray, _, tgt_objs = self._extractor.extract(target_path)

        colorized_match = tgt_img.copy()
        print(f"[ShapeMatcher] Reference objects: {len(ref_objs)} | Target objects: {len(tgt_objs)}")

        if not ref_objs or not tgt_objs:
            return ref_img, tgt_img, colorized_match

        h, w = tgt_gray.shape

        # ----------------------------------------------------
        # Precompute simple geometric descriptors for refs
        # ----------------------------------------------------
        ref_info = []
        for idx, ref in enumerate(ref_objs):
            cnt = ref["contour"]
            area = cv2.contourArea(cnt)
            if area <= 0:
                continue

            ar   = aspect_ratio(cnt)
            ext  = extent(cnt)
            circ = circularity(cnt)
            cx, cy = centroid(cnt)

            ref_info.append({
                "obj": ref,
                "area": area,
                "aspect": ar,
                "extent": ext,
                "circ": circ,
                "centroid": (cx, cy),
            })

        if not ref_info:
            return ref_img, tgt_img, colorized_match

        # ----------------------------------------------------
        # DEBUG: labeled images with IDs on shapes
        # ----------------------------------------------------
        draw_labeled_regions(ref_img, ref_objs, "R", "reference_labeled.png")
        draw_labeled_regions(tgt_img, tgt_objs, "T", "target_labeled.png")

        # ----------------------------------------------------
        # Sort targets from largest → smallest (stable painting)
        # BUT keep original index for debug IDs (T0, T1, ...)
        # ----------------------------------------------------
        tgt_objs_sorted = sorted(
            list(enumerate(tgt_objs)),   # (orig_idx, obj)
            key=lambda pair: cv2.contourArea(pair[1]["contour"]),
            reverse=True
        )

        # For matching-lines debug
        match_pairs = []

        # ----------------------------------------------------
        # Matching loop (greedy, per target region)
        # ----------------------------------------------------
        for sort_idx, (orig_tgt_idx, tgt) in enumerate(tgt_objs_sorted):

            print("\n====================================")
            print(f"### TARGET REGION {sort_idx + 1} (T{orig_tgt_idx}) ###")
            print("====================================\n")

            tgt_cnt = tgt["contour"]
            tgt_mask = tgt["mask"]  # from ContourExtractor
            tgt_area = cv2.contourArea(tgt_cnt)

            if tgt_area <= 0:
                print("  Target area <= 0, skipping.")
                continue

            # Target descriptors
            t_ar   = aspect_ratio(tgt_cnt)
            t_ext  = extent(tgt_cnt)
            t_circ = circularity(tgt_cnt)
            tx, ty = skeleton_midpoint(tgt_mask)

            print(f"Target area      = {tgt_area:.2f}")
            print(f"Target aspect    = {t_ar:.3f}")
            print(f"Target extent    = {t_ext:.3f}")
            print(f"Target circular. = {t_circ:.3f}")
            print(f"Target center    = (tx={tx}, ty={ty})\n")

            best_cost = float("inf")
            best_color = None
            best_ref_id = None
            best_ref_center = None

            ranking_debug = []

            # ---------- loop over references ----------
            for rid, info in enumerate(ref_info):

                r_obj = info["obj"]
                r_cnt = r_obj["contour"]

                r_area = info["area"]
                r_ar   = info["aspect"]
                r_ext  = info["extent"]
                r_circ = info["circ"]
                rx, ry = skeleton_midpoint(r_obj["mask"])

                print(f"  >> Testing REF {rid} color={r_obj['color']}")
                print(f"     ref_area   = {r_area:.2f}")
                print(f"     ref_aspect = {r_ar:.3f}")
                print(f"     ref_extent = {r_ext:.3f}")
                print(f"     ref_circ   = {r_circ:.3f}")
                print(f"     ref_center = (rx={rx}, ry={ry})")

                # -----------------------------
                # Quick geometric filters
                # -----------------------------
                area_ratio  = tgt_area / (r_area + 1e-6)
                aspect_diff = abs(t_ar - r_ar)
                extent_diff = abs(t_ext - r_ext)
                circ_diff   = abs(t_circ - r_circ)

                print("     Filters:")
                print(f"       area_ratio     = {area_ratio:.3f}  (range {self.AREA_RATIO_MIN}-{self.AREA_RATIO_MAX})")
                print(f"       |aspect diff|  = {aspect_diff:.3f} (tol {self.ASPECT_TOL})")
                print(f"       |extent diff|  = {extent_diff:.3f} (tol {self.EXTENT_TOL})")
                print(f"       |circ diff|    = {circ_diff:.3f} (tol {self.CIRC_TOL})")

                # Filter checks
                if not (self.AREA_RATIO_MIN <= area_ratio <= self.AREA_RATIO_MAX):
                    print("     ❌ rejected: area ratio\n")
                    continue
                #if aspect_diff > self.ASPECT_TOL:
                #    print("     ❌ rejected: aspect\n")
                #    continue
                if extent_diff > self.EXTENT_TOL:
                    print("     ❌ rejected: extent\n")
                    continue
                if circ_diff > self.CIRC_TOL:
                    print("     ❌ rejected: circularity\n")
                    continue

                # -----------------------------
                # Shape similarity (with rotation)
                # -----------------------------
                shape_score = best_rotational_match(tgt_cnt, r_cnt)

                # -----------------------------
                # Centroid distance (tie breaker)
                # -----------------------------
                dist = abs(tx - rx) + abs(ty - ry)

                # Combined cost (weights chosen empirically)
                # shape_score dominates, big weight on distance
                cost = (
                    shape_score +
                    0.2 * abs(area_ratio - 1.0) +
                    1.0 * dist
                )

                print(f"     shape_score   = {shape_score:.6f}")
                print(f"     centroid_dist = {dist:.3f}")
                print(f"     FINAL COST    = {cost:.6f}\n")

                ranking_debug.append((cost, rid, r_obj["color"]))

                if cost < best_cost:
                    best_cost = cost
                    best_color = r_obj["color"]
                    best_ref_id = rid
                    best_ref_center = (rx, ry)

            # ---------- show ranking for this target ----------
            ranking_debug.sort()
            print("=== FINAL RANKINGS FOR THIS TARGET ===")
            for rank, (cost, rid, col) in enumerate(ranking_debug):
                print(f"   #{rank+1}: REF {rid} color={col} cost={cost:.6f}")
            if best_color is not None:
                print(f"\n### WINNER REF {best_ref_id} → color={best_color} ###\n")
            else:
                print("\n### NO REF PASSED FILTERS → FALLBACK COLOR ###\n")

            # ------------------------------------------------
            # If no match passed filters, fall back to sampling
            # reference color at the target center
            # ------------------------------------------------
            if best_color is None:
                cx = int(np.clip(tx, 0, w - 1))
                cy = int(np.clip(ty, 0, h - 1))
                sampled = ref_img[cy, cx]  # ref_img is RGB
                best_color = (int(sampled[0]), int(sampled[1]), int(sampled[2]))
                print(f"[Match] Target {sort_idx+1} (T{orig_tgt_idx}): fallback sampled color {best_color}")
            else:
                print(
                    f"[Match] Target {sort_idx+1} (T{orig_tgt_idx}): matched color {best_color}, "
                    f"best_cost={best_cost:.6f}, t_circ={t_circ:.3f}"
                )
                # Save match pair for line visualization
                if best_ref_center is not None:
                    match_pairs.append({
                        "tgt_idx": orig_tgt_idx,         # index used in T#
                        "ref_idx": best_ref_id,         # index used in R#
                        "tgt_center": (tx, ty),
                        "ref_center": best_ref_center,
                    })

            # ------------------------------------------------
            # Paint using the TRUE region mask
            # ------------------------------------------------
            colorized_match[tgt_mask == 255] = best_color

        # ----------------------------------------------------
        # DEBUG: matching-lines visualization
        # ----------------------------------------------------
        if match_pairs:
            draw_matching_lines(
                ref_img,
                colorized_match,  # show final colored target on the right
                match_pairs,
                out_name="matching_lines.png",
            )

        return ref_img, tgt_img, colorized_match
