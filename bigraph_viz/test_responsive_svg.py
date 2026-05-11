"""Tests for ResponsiveGraph SVG post-processing.

The critical invariant: the viewBox in the emitted SVG must cover ALL
rendered geometry. Graphviz natively emits a viewBox sized to the
UN-scaled extent while applying an inner transform="scale(s) ..." that
pushes the post-scale extent past those bounds — causing right/bottom
clipping in browsers and image viewers.
"""
import re

from bigraph_viz import plot_bigraph


def _make_responsive_svg() -> str:
    state = {
        'A': {'_type': 'process', 'inputs': {'in': ['store_a']}, 'outputs': {'out': ['store_b']}},
        'B': {'_type': 'process', 'inputs': {'in': ['store_b']}, 'outputs': {'out': ['store_c']}},
        'store_a': 1.0,
        'store_b': 0.0,
        'store_c': 0.0,
    }
    rg = plot_bigraph(state)
    return rg._make_responsive_svg()


def test_responsive_svg_is_width_100pct():
    svg = _make_responsive_svg()
    assert 'width="100%"' in svg
    assert 'height="auto"' in svg


def test_responsive_svg_viewbox_matches_scaled_extent():
    """viewBox must equal the SVG's natural pt dimensions so all
    transform-scaled content stays inside the rendered box.
    """
    svg = _make_responsive_svg()
    # The post-rewrite SVG has width="100%" — we still want to confirm
    # the viewBox matches the geometry. Re-pipe a fresh graphviz SVG
    # to recover the unmodified width/height in pt for comparison.
    from bigraph_viz import plot_bigraph as _pb
    rg = _pb({'A': {'_type': 'process', 'inputs': {'in': ['s']}}, 's': 0.0})
    raw = rg._graph.pipe(format='svg').decode()
    wm = re.search(r'<svg[^>]*\bwidth="([0-9.]+)pt"', raw)
    hm = re.search(r'<svg[^>]*\bheight="([0-9.]+)pt"', raw)
    assert wm and hm, "raw graphviz SVG should have width=Npt height=Npt"
    nat_w, nat_h = float(wm.group(1)), float(hm.group(1))

    # Now check the responsive SVG's viewBox covers that extent.
    fixed = rg._make_responsive_svg()
    vbm = re.search(r'viewBox="([0-9.\-]+)\s+([0-9.\-]+)\s+([0-9.]+)\s+([0-9.]+)"', fixed)
    assert vbm, "responsive SVG must keep a viewBox"
    vb_w, vb_h = float(vbm.group(3)), float(vbm.group(4))
    assert vb_w >= nat_w - 1e-3, f"viewBox width {vb_w} < scaled extent {nat_w}"
    assert vb_h >= nat_h - 1e-3, f"viewBox height {vb_h} < scaled extent {nat_h}"
