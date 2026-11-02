import matplotlib.pyplot as plt
import numpy as np

from shiftguard.viz.style import _overlap, apply_style, label_points, spread_positions


def test_spread_positions_keeps_order_gap_and_bounds():
    vals = [0.50, 0.51, 0.52, 0.10, 0.97]
    out = spread_positions(vals, gap=0.05, lo=0.0, hi=1.0)
    order = np.argsort(vals)
    assert list(np.argsort(out)) == list(order)  # same vertical order as the points
    assert np.all(np.diff(np.sort(out)) >= 0.05 - 1e-12)
    assert out.min() >= 0.0 and out.max() <= 1.0 + 1e-12
    assert out[3] == 0.10  # an isolated point keeps its own height


def test_label_points_avoids_overlapping_labels():
    apply_style()
    fig, ax = plt.subplots(figsize=(3, 3))
    xs = [0.50, 0.51, 0.52, 0.50, 0.20]
    ys = [0.50, 0.51, 0.50, 0.52, 0.80]
    names = ["ALPHA", "BRAVO", "CHARLIE", "DELTA", "ECHO"]
    ax.scatter(xs, ys)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    label_points(ax, xs, ys, names)
    renderer = fig.canvas.get_renderer()
    boxes = [t.get_window_extent(renderer).extents for t in ax.texts]
    assert [t.get_text() for t in ax.texts] == names
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            assert _overlap(tuple(boxes[i]), tuple(boxes[j])) == 0
    plt.close(fig)
