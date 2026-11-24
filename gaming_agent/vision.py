from collections import deque
from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image


COLORS = ("red", "green", "blue", "yellow", "orange", "purple", "cyan", "white")


@dataclass(frozen=True, slots=True)
class Blob:
    x: int
    y: int
    color: str
    area: int
    motion: float


def thumbnail(jpeg, size=112):
    try:
        with Image.open(BytesIO(jpeg)) as image:
            if image.width * image.height > 8_400_000:
                return None
            image.draft("RGB", (size * 2, size * 2))
            image.thumbnail((size, size), Image.Resampling.BILINEAR)
            return np.asarray(image.convert("RGB"))
    except (OSError, TypeError, ValueError, Image.DecompressionBombError):
        return None


def _labels(image):
    rgb = image.astype(np.float32)
    red, green, blue = rgb.transpose(2, 0, 1)
    high, low = rgb.max(axis=2), rgb.min(axis=2)
    chroma = high - low
    delta = np.maximum(chroma, 1)
    hue = np.where(high == red, (green - blue) / delta,
                   np.where(high == green, 2 + (blue - red) / delta, 4 + (red - green) / delta))
    hue %= 6
    label = np.select((hue >= 5.5, hue >= 4.45, hue >= 3.45, hue >= 2.65,
                       hue >= 1.5, hue >= 0.75, hue >= 0.25), (1, 6, 3, 7, 2, 4, 5), 1)
    colorful = (high >= 80) & (chroma >= 45) & (chroma >= high * 0.38)
    return np.where(colorful, label, np.where((low > 185) & (chroma < 40), 8, 0))


def analyze(image, previous, width, height):
    labels = _labels(image)
    h, w = labels.shape
    motion = np.zeros((h, w), dtype=bool)
    if previous is not None and previous.shape == image.shape:
        motion = np.max(np.abs(image.astype(np.int16) - previous), axis=2) > 28
    # Flat indices with a sentinel border keep component traversal independent of edge checks.
    board = np.pad(labels, 1).ravel().tolist()
    stride = w + 2
    moved = np.pad(motion, 1).ravel().tolist()
    blobs = []
    for seed, label in enumerate(board):
        if not label:
            continue
        board[seed] = 0
        stack = [seed]
        n = sx = sy = nmoved = 0
        while stack:
            i = stack.pop()
            y, x = divmod(i, stride)
            n += 1
            sx += x - 1
            sy += y - 1
            nmoved += moved[i]
            for j in (i - 1, i + 1, i - stride, i + stride):
                if board[j] == label:
                    board[j] = 0
                    stack.append(j)
        if 3 <= n <= min(w * h * 0.18, 1600):
            blobs.append(Blob(int((sx / n + .5) * width / w),
                              int((sy / n + .5) * height / h),
                              COLORS[label - 1], n, nmoved / n))
    return blobs


def hud_signature(image):
    if image is not None:
        h, w = image.shape[:2]
        return (image[:max(2, h // 10), :max(2, w // 3)].astype(np.int16) // 48)


def route(image, player, target, width, height):
    h, w = image.shape[:2]
    step = max(1, max(h, w) // 28)
    rgb = image[::step, ::step].astype(np.int16)
    rows, cols = rgb.shape[:2]
    background = np.median(rgb.reshape(-1, 3), axis=0)
    raw = (np.ptp(rgb, axis=2) < 50) & (np.max(np.abs(rgb - background), axis=2) > 33)
    padded = np.pad(raw, 1)
    raw &= (padded[:-2, 1:-1] | padded[2:, 1:-1] | padded[1:-1, :-2] | padded[1:-1, 2:])
    raw[:max(1, rows // 10)] = True
    if not raw[max(1, rows // 10):].any():
        return None
    buffer = np.pad(raw, 1)
    blocked = np.logical_or.reduce([buffer[y:y + rows, x:x + cols]
                                    for y in range(3) for x in range(3)])
    def cell(blob):
        return (min(cols - 1, round(blob.x * w / width / step)),
                min(rows - 1, round(blob.y * h / height / step)))

    start, end = cell(player), cell(target)
    queue = deque([start])
    visited = {start: None}
    dx, dy = end[0] - start[0], end[1] - start[1]
    span = max(abs(dx), abs(dy))
    if not any(blocked[round(start[1] + i * dy / span), round(start[0] + i * dx / span)]
               for i in range(1, span + 1)):
        return None
    while queue:
        x, y = queue.popleft()
        if (x, y) == end:
            return visited[end]
        neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        neighbors.sort(key=lambda point: abs(point[0] - end[0]) + abs(point[1] - end[1]))
        for nx, ny in neighbors:
            if (nx, ny) not in visited and 0 <= nx < cols and 0 <= ny < rows and not blocked[ny, nx]:
                visited[nx, ny] = (visited[x, y] or ((nx - x) * width / cols, (ny - y) * height / rows))
                queue.append((nx, ny))
