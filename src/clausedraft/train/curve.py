"""Draw a loss curve as SVG. No plotting dependency."""

from __future__ import annotations

from pathlib import Path

_WIDTH = 720
_HEIGHT = 360
_LEFT = 58
_RIGHT = 18
_TOP = 32
_BOTTOM = 40


def loss_curve_svg(rows: list[dict]) -> str:
    train = _points(rows, "loss")
    evals = _points(rows, "eval_loss")
    if not train and not evals:
        raise ValueError("loss history has no points")

    xs = [x for x, _ in train + evals]
    ys = [y for _, y in train + evals]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmax == xmin:
        xmax = xmin + 1
    span = ymax - ymin
    pad = span * 0.08 if span else max(abs(ymin) * 0.05, 0.05)
    ymin -= pad
    ymax += pad

    plot_w = _WIDTH - _LEFT - _RIGHT
    plot_h = _HEIGHT - _TOP - _BOTTOM

    def project(x: float, y: float) -> tuple[float, float]:
        px = _LEFT + (x - xmin) / (xmax - xmin) * plot_w
        py = _TOP + (ymax - y) / (ymax - ymin) * plot_h
        return px, py

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_WIDTH}" height="{_HEIGHT}" '
        f'viewBox="0 0 {_WIDTH} {_HEIGHT}">',
        '<rect width="100%" height="100%" fill="#f7f5f2"/>',
        f'<text x="{_LEFT}" y="20" font-family="Georgia, serif" font-size="14" fill="#1c1a17">'
        "Training loss</text>",
        _axis(xmin, xmax, ymin, ymax, project),
        _series(train, project, "#1f4e79"),
        _series(evals, project, "#b85c38"),
        _legend(train, evals),
        "</svg>",
    ]
    return "\n".join(parts) + "\n"


def write_loss_curve(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(loss_curve_svg(rows), encoding="utf-8")


def _points(rows: list[dict], key: str) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for row in rows:
        if key not in row or "step" not in row:
            continue
        points.append((float(row["step"]), float(row[key])))
    return points


def _series(points: list[tuple[float, float]], project, color: str) -> str:
    if not points:
        return ""
    if len(points) == 1:
        px, py = project(*points[0])
        return f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.5" fill="{color}"/>'
    coords = " ".join(f"{project(x, y)[0]:.1f},{project(x, y)[1]:.1f}" for x, y in points)
    return f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{coords}"/>'


def _axis(xmin, xmax, ymin, ymax, project) -> str:
    x0, y0 = project(xmin, ymin)
    x1, y_top = project(xmin, ymax)
    _, y_base = project(xmax, ymin)
    x_end, _ = project(xmax, ymin)
    label = (
        f'<text x="{_LEFT}" y="{_HEIGHT - 8}" font-family="Consolas, monospace" '
        f'font-size="11" fill="#5c574e">step {xmin:g} to {xmax:g}</text>'
        f'<text x="14" y="{y_top + 12:.0f}" font-family="Consolas, monospace" '
        f'font-size="11" fill="#5c574e">{ymax:.2f}</text>'
        f'<text x="14" y="{y0:.0f}" font-family="Consolas, monospace" '
        f'font-size="11" fill="#5c574e">{ymin:.2f}</text>'
    )
    return (
        f'<line x1="{x0:.1f}" y1="{y_top:.1f}" x2="{x0:.1f}" y2="{y_base:.1f}" stroke="#c4bfb6"/>'
        f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x_end:.1f}" y2="{y0:.1f}" stroke="#c4bfb6"/>'
        + label
    )


def _legend(train, evals) -> str:
    items = []
    x = _WIDTH - 168
    y = 18
    if train:
        items.append(("train", "#1f4e79"))
    if evals:
        items.append(("eval", "#b85c38"))
    chunks = []
    for label, color in items:
        chunks.append(
            f'<line x1="{x}" y1="{y}" x2="{x + 16}" y2="{y}" stroke="{color}" stroke-width="2"/>'
            f'<text x="{x + 22}" y="{y + 4}" font-family="Consolas, monospace" '
            f'font-size="11" fill="#1c1a17">{label}</text>'
        )
        x += 78
    return "".join(chunks)
