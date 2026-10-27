# -*- coding: utf-8 -*-
"""
Простое сопровождение лиц между кадрами.

На вебинаре плитки участников почти не двигаются, поэтому достаточно
жадного сопоставления: сначала по IoU рамок, а если IoU мал (лицо
сместилось или кадр пропущен) — по расстоянию между центрами.
Каждому новому лицу присваивается ID участника P1, P2, ...

Треки не удаляются до конца занятия: если слушатель отошёл от камеры,
его плитка остаётся на месте, и при возвращении лицо снова получит
тот же ID.
"""

from dataclasses import dataclass


def iou(a, b):
    """IoU двух рамок (x, y, w, h)."""
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2, bx2, by2 = ax1 + aw, ay1 + ah, bx1 + bw, by1 + bh
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def center_distance(a, b):
    """Расстояние между центрами рамок в долях среднего размера лица."""
    acx, acy = a[0] + a[2] / 2, a[1] + a[3] / 2
    bcx, bcy = b[0] + b[2] / 2, b[1] + b[3] / 2
    size = (a[2] + a[3] + b[2] + b[3]) / 4
    return ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5 / max(size, 1.0)


@dataclass
class Track:
    track_id: str
    box: tuple
    first_seen: float
    last_seen: float
    hits: int = 1


class FaceTracker:
    def __init__(self, min_iou=0.3, max_dist=0.6):
        self.min_iou = min_iou
        self.max_dist = max_dist
        self.tracks: list[Track] = []
        self._next_id = 1

    def update(self, boxes, timestamp):
        """Сопоставляет рамки текущего кадра с треками.

        Возвращает список ID в том же порядке, что и boxes.
        """
        # Все возможные пары (трек, лицо) с оценкой «похожести»
        pairs = []
        for ti, track in enumerate(self.tracks):
            for bi, box in enumerate(boxes):
                ov = iou(track.box, box)
                dist = center_distance(track.box, box)
                if ov >= self.min_iou or dist <= self.max_dist:
                    # IoU важнее, расстояние — запасной критерий
                    pairs.append((ov - 0.1 * dist, ti, bi))
        pairs.sort(reverse=True)

        ids = [None] * len(boxes)
        used_tracks = set()
        for _, ti, bi in pairs:
            if ti in used_tracks or ids[bi] is not None:
                continue
            track = self.tracks[ti]
            track.box = tuple(boxes[bi])
            track.last_seen = timestamp
            track.hits += 1
            ids[bi] = track.track_id
            used_tracks.add(ti)

        # Несопоставленные лица — новые участники
        for bi, box in enumerate(boxes):
            if ids[bi] is None:
                track = Track(f"P{self._next_id}", tuple(box), timestamp, timestamp)
                self._next_id += 1
                self.tracks.append(track)
                ids[bi] = track.track_id
        return ids
