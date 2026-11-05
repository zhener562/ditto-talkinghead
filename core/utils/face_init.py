"""Find a face without InsightFace, for the first crop of `landmark203`.

InsightFace's detection and landmark models are for non-commercial research
only. They were used for one thing here: placing the first rough crop that
`landmark203` then refines. The final crop comes from `landmark203` alone.

BlazeFace's box is reliable on illustrated faces, but its eye and mouth points
are not (they set the crop's scale, and were off by more than half a crop on
some characters). So only the box is used, and `landmark203` is applied until
its result stops moving.

`landmark203` settles on a face-like shape even on a car, and BlazeFace's score
overlaps between faces and objects, so neither can say whether there is a face.
Checks on the aligned crop decide that instead: FaceMesh's presence output,
FaceMesh and `landmark203` agreeing on where the eyes are, and BlazeFace finding
a box of the right size there with a good score. On a real face that box fits the
landmarks; a false accept is a magnified bit of background and the box is far too
large. Presence alone is not enough: taking the best of many starts lets some
start pass by chance.
"""

import cv2
import numpy as np

from .crop import crop_image

WINDOW_FRACTIONS = (1.0, 0.66, 0.45, 0.3)
MIN_WINDOW_PX = 96
START_SCALES = (1.0, 1.5, 0.7)
ROTATIONS = (45, -45, 90, -90)
MAX_BOXES = 4
SETTLE_STEPS = 5
CONVERGED = 0.01  # mean landmark movement / interocular distance
MIN_PRESENCE = 0.0  # FaceMesh logit
MAX_EYE_DISAGREEMENT = 0.15  # distance between the two models / interocular distance
MIN_RECHECK_SCORE = 0.56  # BlazeFace on the aligned crop
MAX_RECHECK_SIZE = 1.28  # BlazeFace box width / landmark extent on the aligned crop


def _pad_square(img):
    h, w = img.shape[:2]
    n = max(h, w)
    out = np.zeros((n, n, 3), img.dtype)
    y, x = (n - h) // 2, (n - w) // 2
    out[y : y + h, x : x + w] = img
    return out, x, y


def _windows(img):
    h, w = img.shape[:2]
    short = min(h, w)
    for fraction in WINDOW_FRACTIONS:
        size = int(short * fraction)
        if size < MIN_WINDOW_PX:
            continue
        step = max(1, size // 2)
        for y in range(0, max(1, h - size + 1), step):
            for x in range(0, max(1, w - size + 1), step):
                yield x, y, size


def _iou(a, b):
    ix = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    iy = max(0.0, min(a[4], b[4]) - max(a[2], b[2]))
    inter = ix * iy
    union = (a[3] - a[1]) * (a[4] - a[2]) + (b[3] - b[1]) * (b[4] - b[2]) - inter
    return inter / union if union > 0 else 0.0


def detect_boxes(blaze, img):
    """[(score, x1, y1, x2, y2)] in image pixels, best first, overlaps removed."""
    found = []

    def collect(detections, side, dx, dy):
        scale = side / 128.0
        for d in detections:
            found.append(
                (
                    float(d[-1]),
                    d[0] * scale + dx,
                    d[1] * scale + dy,
                    d[2] * scale + dx,
                    d[3] * scale + dy,
                )
            )

    square, ox, oy = _pad_square(img)
    collect(blaze(square), square.shape[0], -ox, -oy)
    for x, y, size in _windows(img):
        collect(blaze(img[y : y + size, x : x + size]), size, x, y)

    found.sort(key=lambda box: -box[0])
    kept = []
    for box in found:
        if all(_iou(box, other) <= 0.3 for other in kept):
            kept.append(box)
        if len(kept) >= MAX_BOXES:
            break
    return kept


def rough_points(box, scale=1.0):
    """Five upright points for `crop_image`. Only their position and size matter."""
    _, x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    h = (y2 - y1) * scale
    eye = np.array([cx, cy - 0.18 * h], np.float32)
    lip = np.array([cx, cy + 0.26 * h], np.float32)
    half = np.array([0.2 * h, 0], np.float32)
    return np.stack(
        [eye - half, eye + half, np.array([cx, cy], np.float32), lip - half / 2, lip + half / 2]
    )


def landmark_movement(a, b):
    interocular = np.linalg.norm(a[[0, 6, 12, 18]].mean(0) - a[[24, 30, 36, 42]].mean(0))
    return float(np.linalg.norm(a - b, axis=1).mean() / max(interocular, 1e-3))


class FaceInit:
    def __init__(self, blaze, face_mesh, landmark203, min_presence=MIN_PRESENCE):
        self.blaze = blaze
        self.face_mesh = face_mesh
        self.landmark203 = landmark203
        self.min_presence = min_presence

    def _check(self, img, lmk):
        crop = crop_image(img, lmk, dsize=512, scale=2.3, vy_ratio=-0.125, pt_crop_flag=True)
        image, points = crop["img_crop"], crop["pt_crop"]
        cx, cy = points.mean(0)
        size = float(np.ptp(points, axis=0).max()) * 1.5
        mesh = self.face_mesh(image.copy(), np.array([cx, cy, size, size, 0.0]))[:, :2]
        presence = self.face_mesh.last_presence

        eye_a, eye_b = points[[0, 6, 12, 18]].mean(0), points[[24, 30, 36, 42]].mean(0)
        mesh_a, mesh_b = mesh[[33, 133]].mean(0), mesh[[362, 263]].mean(0)
        interocular = max(np.linalg.norm(eye_a - eye_b), 1e-3)
        disagreement = min(
            np.linalg.norm(mesh_a - eye_a) + np.linalg.norm(mesh_b - eye_b),
            np.linalg.norm(mesh_a - eye_b) + np.linalg.norm(mesh_b - eye_a),
        ) / (2 * interocular)
        boxes = self.blaze(image)
        recheck = max(boxes, key=lambda box: box[-1]) if len(boxes) else None
        accepted = bool(
            presence > self.min_presence
            and disagreement <= MAX_EYE_DISAGREEMENT
            and recheck is not None
            and recheck[-1] >= MIN_RECHECK_SCORE
            and (recheck[2] - recheck[0]) * 512 / 128.0 / np.ptp(points, axis=0).max()
            <= MAX_RECHECK_SIZE
        )
        return presence, float(disagreement), accepted

    def _refine(self, img, points):
        crop = crop_image(
            img, points, dsize=self.landmark203.dsize, scale=1.5, vy_ratio=-0.1, pt_crop_flag=False
        )
        return self.landmark203(crop["img_crop"], crop["M_c2o"])

    def _settle(self, img, points):
        lmk = self._refine(img, points)
        movement = float("inf")
        for _ in range(SETTLE_STEPS):
            refined = self._refine(img, lmk)
            movement = landmark_movement(lmk, refined)
            lmk = refined
            if movement < CONVERGED:
                break
        return lmk, movement

    def _best_from(self, img, view, to_original=None):
        best = None
        for box in detect_boxes(self.blaze, view):
            for scale in START_SCALES:
                points = rough_points(box, scale)
                if to_original is not None:
                    points = cv2.transform(points[None], to_original)[0]
                lmk, movement = self._settle(img, points)
                presence, disagreement, accepted = self._check(img, lmk)
                key = (not accepted, -presence)
                if best is None or key < best["key"]:
                    best = {
                        "key": key,
                        "lmk": lmk,
                        "score": box[0],
                        "movement": movement,
                        "presence": presence,
                        "disagreement": disagreement,
                        "accepted": accepted,
                    }
                if movement < CONVERGED and accepted:
                    break
        return best

    def candidate(self, img):
        """The best guess with its score and movement, whether or not it is accepted."""
        best = self._best_from(img, img)
        if best is not None and best["accepted"]:
            return best
        h, w = img.shape[:2]
        for angle in ROTATIONS:
            forward = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            cos, sin = abs(forward[0, 0]), abs(forward[0, 1])
            nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
            forward[0, 2] += nw / 2 - w / 2
            forward[1, 2] += nh / 2 - h / 2
            view = cv2.warpAffine(img, forward, (nw, nh), borderMode=cv2.BORDER_REPLICATE)
            found = self._best_from(img, view, cv2.invertAffineTransform(forward))
            if found is not None and (best is None or found["key"] < best["key"]):
                best = found
            if best is not None and best["accepted"]:
                break
        return best

    def __call__(self, img):
        """lmk203 of the most reliable face, or None when nothing looks like one."""
        best = self.candidate(img)
        if best is None or not best["accepted"]:
            return None
        return best["lmk"]
