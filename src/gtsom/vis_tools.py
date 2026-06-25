"""
vis_tools.py
------------
Plotnine-based visualization utilities, ported from R/VisTools.

Public API
----------
vis_embedding_discrete(x, y, ...)
vis_embedding_continuous(x, y, z, ...)
parse_ctab(labels, colors)
build_ctab(labels, seed=None)
theme_minimal_bold(text_color, **kwargs)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotnine import (
    ggplot, aes,
    geom_point, geom_segment,
    scale_color_manual, scale_color_continuous,
    scale_size_continuous,
    scale_x_continuous, scale_x_reverse,
    scale_y_continuous, scale_y_reverse,
    labs, guides, guide_legend, guide_colorbar,
    theme_minimal, theme,
    element_text, element_line, element_blank,
)


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _is_missing(v):
    """Return True if v is None or a scalar NaN."""
    if v is None:
        return True
    try:
        return np.isnan(v)
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

def theme_minimal_bold(text_color="black", **kwargs):
    """
    Plotnine equivalent of theme_minimal_bold() from R/VisTools.

    Parameters
    ----------
    text_color : str
        Color applied to all axis text, titles, and legend labels.
    **kwargs
        Additional theme() keyword arguments forwarded verbatim,
        allowing the caller to override or extend any theme element.

    Returns
    -------
    plotnine.theme
        A composable theme object; add it to any ggplot with +.
    """
    return (
        theme_minimal() +
        theme(
            axis_title   = element_text(family="DejaVu Sans", weight="bold", color=text_color),
            plot_title   = element_text(family="DejaVu Sans", ha="left", weight="bold", color=text_color),
            axis_line    = element_line(color=text_color),
            axis_text    = element_text(family="DejaVu Sans", color=text_color),
            legend_title = element_text(family="DejaVu Sans", weight="bold", color=text_color),
            legend_text  = element_text(family="DejaVu Sans", color=text_color),
            panel_border = element_blank(),
            **kwargs,
        )
    )


# ---------------------------------------------------------------------------
# Color table helpers
# ---------------------------------------------------------------------------

def parse_ctab(labels, colors):
    """
    Build a label->color mapping dict, accepting colors as hex strings
    or as an RGB array.

    Parameters
    ----------
    labels : array-like
        Category labels.
    colors : array-like
        Either a list of hex strings (e.g. ["#FF0000", "#00FF00"]) or
        an (N, 3) array of RGB values. RGB values are auto-detected as
        uint8 (0-255) if any value exceeds 1, otherwise treated as
        float (0-1).

    Returns
    -------
    dict
        {label: hex_color_string}

    Raises
    ------
    ValueError
        If inputs are inconsistent or malformed.
    """
    labels = list(labels)
    n = len(labels)

    # Strip None/NaN from labels before any processing so that callers
    # don't have to pre-filter. None keys in the dict would cause silent
    # mismatches downstream.
    labels = [l for l in labels if not _is_missing(l)]
    n = len(labels)
    if n == 0:
        raise ValueError("labels contains only None/NaN values.")

    # --- colors is a list / 1-D array: must be hex strings ----------------
    if isinstance(colors, (list, tuple)) or (
        isinstance(colors, np.ndarray) and colors.ndim == 1
    ):
        colors = list(colors)
        if not all(isinstance(c, str) for c in colors):
            raise ValueError("If colors is 1-D, it must contain hex strings.")
        if len(colors) < n:
            raise ValueError("len(colors) must be >= len(labels).")
        return dict(zip(labels, colors[:n]))

    # --- colors is a 2-D array: treat as RGB ------------------------------
    colors = np.asarray(colors)
    if colors.ndim != 2 or colors.shape[1] != 3:
        raise ValueError("If colors is 2-D, it must have shape (N, 3) for RGB.")
    if colors.shape[0] < n:
        raise ValueError("colors must have at least as many rows as labels.")

    colors = colors[:n].astype(float)
    if colors.max() > 1.0:
        colors = colors / 255.0
    colors = np.clip(colors, 0, 1)

    hex_colors = [
        "#{:02X}{:02X}{:02X}".format(int(r * 255), int(g * 255), int(b * 255))
        for r, g, b in colors
    ]
    return dict(zip(labels, hex_colors))


def build_ctab(labels, seed=None):
    """
    Automatically generate a label->color mapping for a set of category labels.

    Uses matplotlib's Tab10 palette for the first 10 labels (perceptually
    well-tuned for categorical data), then extends with distinctipy for any
    additional labels, ensuring new colors are visually distinct from the
    Tab10 colors already assigned.

    Labels are sorted before color assignment so that the mapping is
    deterministic and stable across calls with the same label set.

    Parameters
    ----------
    labels : array-like
        Category labels. Duplicates are ignored; labels are sorted before
        color assignment.
    seed : int or None
        Random seed passed to distinctipy for reproducibility when more
        than 10 labels are present. If None, results may vary between calls.

    Returns
    -------
    dict
        {label: hex_color_string}

    Raises
    ------
    ImportError
        If more than 10 labels are provided and distinctipy is not installed.

    Notes
    -----
    distinctipy is an optional dependency. Install with:
        pip install distinctipy
    """
    # Strip None/NaN before sorting — sorted() can't compare None to str
    sorted_labels = sorted(set(l for l in labels if not _is_missing(l)))
    n = len(sorted_labels)
    if n == 0:
        raise ValueError("labels contains only None/NaN values.")

    # Pull Tab10 colors as (r, g, b) tuples in [0, 1]
    tab10 = plt.get_cmap("tab10")
    tab10_rgb = [tab10(i)[:3] for i in range(10)]   # list of (r,g,b)

    if n <= 10:
        chosen_rgb = tab10_rgb[:n]
    else:
        try:
            import distinctipy
        except ImportError:
            raise ImportError(
                "More than 10 labels requires distinctipy. "
                "Install with: pip install distinctipy"
            )
        n_extra = n - 10
        extra_rgb = distinctipy.get_colors(
            n_colors=n_extra,
            exclude_colors=tab10_rgb,
            rng=seed,
        )
        chosen_rgb = tab10_rgb + list(extra_rgb)

    hex_colors = [
        "#{:02X}{:02X}{:02X}".format(int(r * 255), int(g * 255), int(b * 255))
        for r, g, b in chosen_rgb
    ]
    return dict(zip(sorted_labels, hex_colors))


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _build_graph_layer(x, y, graph, edge_size, edge_color):
    """
    Convert a square adjacency matrix into a geom_segment layer.

    Accepts either a dense array-like or any scipy.sparse matrix.
    Only the upper triangle is used to avoid duplicate edges.
    """
    try:
        import scipy.sparse as sp
        is_sparse = sp.issparse(graph)
    except ImportError:
        is_sparse = False

    x = np.asarray(x)
    y = np.asarray(y)

    if graph.shape[0] != len(x):
        raise ValueError("graph must be square with nrows == len(x).")

    if is_sparse:
        # Use sparse-native upper triangle — never densifies the matrix
        upper = sp.triu(graph, k=1)
        rows, cols = upper.nonzero()
    else:
        graph = np.asarray(graph, dtype=float)
        rows, cols = np.where(np.triu(graph, k=1) > 0)

    if len(rows) == 0:
        return None

    edge_df = pd.DataFrame({
        "x":    x[rows],
        "y":    y[rows],
        "xend": x[cols],
        "yend": y[cols],
    })
    return geom_segment(
        data=edge_df,
        mapping=aes(x="x", y="y", xend="xend", yend="yend"),
        size=edge_size,
        color=edge_color,
    )


def _apply_axis_scales(g, xrev, yrev, xlim, ylim, xlab, ylab):
    """Add x and y continuous/reverse scales to a plotnine object."""
    x_scale = (
        scale_x_reverse(name=xlab, limits=xlim)
        if xrev
        else scale_x_continuous(name=xlab, limits=xlim)
    )
    y_scale = (
        scale_y_reverse(name=ylab, limits=ylim)
        if yrev
        else scale_y_continuous(name=ylab, limits=ylim)
    )
    return g + x_scale + y_scale


def _apply_size_scale(g, point_size, point_size_wts_trans):
    """Add size scale that suppresses the size legend."""
    return g + scale_size_continuous(
        range=(point_size * 0.1, point_size * 1.0),
        trans=point_size_wts_trans,
    ) + guides(size=False)


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def vis_embedding_discrete(
    x, y,
    z=None, ctab=None,
    point_size=2, point_color="black",
    point_size_wts=None, point_size_wts_trans="identity",
    point_shape="o",
    graph=None, edge_size=0.25, edge_color="antiquewhite3",
    xrev=False, yrev=False, xlim=None, ylim=None,
    xlab="x", ylab="y",
    title=None, subtitle=None, caption=None,
    legend_pos="none", legend_title=None,
    legend_nlines=1, legend_key_size=5,
    text_color="black", show_grid=True,
    seed=None,
):
    """
    Scatter / embedding plot with a discrete (categorical) color scale.

    Parameters
    ----------
    x, y : array-like
        Point coordinates.
    z : array-like or None
        Categorical labels for each point. If None, all points are
        drawn in ``point_color`` with no legend.
    ctab : dict or None
        Label->hex-color mapping (as returned by ``parse_ctab`` or
        ``build_ctab``). If z is provided but ctab is None, colors are
        generated automatically via ``build_ctab``.
    point_size : float
        Maximum point size. The actual size is scaled by ``point_size_wts``
        in the range [point_size * 0.1, point_size].
    point_color : str
        Fallback color when both z and ctab are None.
    point_size_wts : array-like or None
        Per-point size weights. If None all points get the same size.
    point_size_wts_trans : str
        Scale transformation applied to size weights (e.g. "identity",
        "sqrt", "log10").
    point_shape : str
        Matplotlib marker string (e.g. "o", "s", "^").
    graph : array-like (N, N) or None
        Adjacency matrix. When provided, edges are drawn before points.
    edge_size : float
        Line width for graph edges.
    edge_color : str
        Color for graph edges.
    xrev, yrev : bool
        Reverse the x or y axis.
    xlim, ylim : tuple or None
        Axis limits (min, max).
    xlab, ylab : str
        Axis labels.
    title, subtitle, caption : str or None
        Plot annotations.
    legend_pos : str
        "none", "right", or "bottom".
    legend_title : str or None
        Legend title.
    legend_nlines : int
        Number of columns (right legend) or rows (bottom legend).
    legend_key_size : float
        Override size of legend color keys.
    text_color : str
        Color for all text elements (passed to ``theme_minimal_bold``).
    show_grid : bool
        If False, panel grid lines are removed.
    seed : int or None
        Random seed for ``build_ctab`` when auto-generating colors
        (only used when ctab is None and n_labels > 10).

    Returns
    -------
    plotnine.ggplot
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")

    # ---- resolve z / ctab ------------------------------------------------
    if z is None and ctab is None:
        # No labels: single color, no legend
        z = np.ones(len(x), dtype=int)
        ctab = parse_ctab([1], [point_color])
        legend_pos = "none"
    elif z is not None and ctab is None:
        # Labels given but no color table: auto-generate one (ignoring None)
        ctab = build_ctab(z, seed=seed)
    elif z is not None and ctab is not None:
        # Both given: validate that all non-None labels are covered
        z = np.asarray(z, dtype=object)
        missing = set(z[~pd.isnull(z)]) - set(ctab.keys())
        if missing:
            raise ValueError(f"z contains labels not found in ctab: {missing}")

    z = np.asarray(z, dtype=object)

    # ---- mask out None/NaN points ----------------------------------------
    # Points with no label are excluded from both the point layer and any
    # graph edges, matching R/ggplot2 behavior of silently dropping NA rows.
    valid = np.array([not _is_missing(v) for v in z])
    x_plot     = x[valid]
    y_plot     = y[valid]
    z_plot     = z[valid]
    valid_idx  = np.where(valid)[0]

    # ---- build plot -------------------------------------------------------
    g = ggplot()

    if graph is not None:
        # Subset graph to valid points only
        try:
            import scipy.sparse as sp
            if sp.issparse(graph):
                graph_sub = graph[np.ix_(valid_idx, valid_idx)]
            else:
                graph_sub = np.asarray(graph)[np.ix_(valid_idx, valid_idx)]
        except ImportError:
            graph_sub = np.asarray(graph)[np.ix_(valid_idx, valid_idx)]
        seg = _build_graph_layer(x_plot, y_plot, graph_sub, edge_size, edge_color)
        if seg is not None:
            g = g + seg

    n_plot = len(x_plot)
    size_wts = (
        np.ones(len(x)) if point_size_wts is None else np.asarray(point_size_wts)
    )
    if len(size_wts) != len(x):
        raise ValueError("point_size_wts must have the same length as x.")
    size_wts_plot = size_wts[valid]

    plot_df = pd.DataFrame({
        "x":    x_plot,
        "y":    y_plot,
        "z":    pd.Categorical(z_plot, categories=list(ctab.keys())),
        "size": size_wts_plot,
    })

    g = g + geom_point(
        data=plot_df,
        mapping=aes(x="x", y="y", color="z", size="size"),
        shape=point_shape,
    )
    g = g + scale_color_manual(
        name=legend_title,
        values=ctab,
        breaks=list(ctab.keys()),
    )
    g = _apply_size_scale(g, point_size, point_size_wts_trans)

    # ---- axes, labels ----------------------------------------------------
    g = _apply_axis_scales(g, xrev, yrev, xlim, ylim, xlab, ylab)
    g = g + labs(title=title, subtitle=subtitle, caption=caption)

    # ---- legend guides (before theme) ------------------------------------
    if legend_pos == "right":
        g = g + guides(
            color=guide_legend(
                title=legend_title,
                nrow=None, ncol=legend_nlines,
                byrow=False,
                override_aes={"shape": "s", "size": legend_key_size},
            )
        )
    elif legend_pos == "bottom":
        g = g + guides(
            color=guide_legend(
                title=legend_title,
                nrow=legend_nlines, ncol=None,
                byrow=False,
                override_aes={"shape": "s", "size": legend_key_size},
            )
        )

    # ---- single consolidated theme call ----------------------------------
    # All theme elements in one call so nothing overrides the bold settings.
    legend_kwargs = {}
    if legend_pos == "none":
        legend_kwargs["legend_position"] = "none"
    elif legend_pos == "right":
        legend_kwargs["legend_position"] = "right"
        legend_kwargs["legend_direction"] = "vertical"
    elif legend_pos == "bottom":
        legend_kwargs["legend_position"] = "bottom"
        legend_kwargs["legend_direction"] = "horizontal"

    grid_kwargs = {}
    if not show_grid:
        grid_kwargs["panel_grid_major"] = element_blank()
        grid_kwargs["panel_grid_minor"] = element_blank()

    g = g + theme_minimal_bold(text_color=text_color, **legend_kwargs, **grid_kwargs)

    return g


def vis_embedding_continuous(
    x, y, z,
    cmap="viridis",
    point_size=2,
    point_size_wts=None, point_size_wts_trans="identity",
    point_shape="o",
    graph=None, edge_size=0.25, edge_color="antiquewhite3",
    xrev=False, yrev=False, xlim=None, ylim=None,
    xlab="x", ylab="y",
    title=None, subtitle=None, caption=None,
    legend_pos="none", legend_title=None,
    text_color="black", show_grid=True,
):
    """
    Scatter / embedding plot with a continuous color scale.

    Parameters
    ----------
    x, y : array-like
        Point coordinates.
    z : array-like
        Continuous values mapped to color.
    cmap : str
        Colormap name (any matplotlib-compatible string, e.g. "viridis",
        "plasma", "magma").
    point_size : float
        Maximum point size.
    point_size_wts : array-like or None
        Per-point size weights. If None all points get the same size.
    point_size_wts_trans : str
        Scale transformation for size weights.
    point_shape : str
        Matplotlib marker string.
    graph : array-like (N, N) or None
        Adjacency matrix for optional edge layer.
    edge_size : float
        Line width for graph edges.
    edge_color : str
        Color for graph edges.
    xrev, yrev : bool
        Reverse x or y axis.
    xlim, ylim : tuple or None
        Axis limits.
    xlab, ylab : str
        Axis labels.
    title, subtitle, caption : str or None
        Plot annotations.
    legend_pos : str
        "none", "right", or "bottom".
    legend_title : str or None
        Colorbar title.
    text_color : str
        Color for all text elements.
    show_grid : bool
        If False, panel grid lines are removed.

    Returns
    -------
    plotnine.ggplot
    """
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z, dtype=float)
    if not (len(x) == len(y) == len(z)):
        raise ValueError("x, y, and z must all have the same length.")

    # ---- build plot -------------------------------------------------------
    g = ggplot()

    # ---- mask out NaN points ---------------------------------------------
    valid = ~np.isnan(z)
    x_plot        = x[valid]
    y_plot        = y[valid]
    z_plot        = z[valid]
    valid_idx     = np.where(valid)[0]

    if graph is not None:
        try:
            import scipy.sparse as sp
            if sp.issparse(graph):
                graph_plot = graph[np.ix_(valid_idx, valid_idx)]
            else:
                graph_plot = np.asarray(graph)[np.ix_(valid_idx, valid_idx)]
        except ImportError:
            graph_plot = np.asarray(graph)[np.ix_(valid_idx, valid_idx)]
        seg = _build_graph_layer(x_plot, y_plot, graph_plot, edge_size, edge_color)
        if seg is not None:
            g = g + seg

    size_wts = (
        np.ones(len(x)) if point_size_wts is None else np.asarray(point_size_wts)
    )
    if len(size_wts) != len(x):
        raise ValueError("point_size_wts must have the same length as x.")
    size_wts_plot = size_wts[valid]

    plot_df = pd.DataFrame({
        "x":    x_plot,
        "y":    y_plot,
        "z":    z_plot,
        "size": size_wts_plot,
    })

    g = g + geom_point(
        data=plot_df,
        mapping=aes(x="x", y="y", color="z", size="size"),
        shape=point_shape,
    )
    g = g + scale_color_continuous(cmap_name=cmap, name=legend_title)
    g = _apply_size_scale(g, point_size, point_size_wts_trans)

    # ---- axes, labels ----------------------------------------------------
    g = _apply_axis_scales(g, xrev, yrev, xlim, ylim, xlab, ylab)
    g = g + labs(title=title, subtitle=subtitle, caption=caption)

    # ---- legend guides (before theme) ------------------------------------
    if legend_pos == "right":
        g = g + guides(color=guide_colorbar(title=legend_title))
    elif legend_pos == "bottom":
        g = g + guides(color=guide_colorbar(title=legend_title))

    # ---- single consolidated theme call ----------------------------------
    legend_kwargs = {}
    if legend_pos == "none":
        legend_kwargs["legend_position"] = "none"
    elif legend_pos == "right":
        legend_kwargs["legend_position"] = "right"
        legend_kwargs["legend_direction"] = "vertical"
    elif legend_pos == "bottom":
        legend_kwargs["legend_position"] = "bottom"
        legend_kwargs["legend_direction"] = "horizontal"

    grid_kwargs = {}
    if not show_grid:
        grid_kwargs["panel_grid_major"] = element_blank()
        grid_kwargs["panel_grid_minor"] = element_blank()

    g = g + theme_minimal_bold(text_color=text_color, **legend_kwargs, **grid_kwargs)

    return g