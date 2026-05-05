import re

import numpy as np


def _get_plt():
    import matplotlib.pyplot as plt

    return plt


def detect_composition_columns(df):
    """Return composition column names such as x_A, x_B, preserving CSV order."""
    return [col for col in df.columns if col.startswith("x_")]


def detect_s0_columns(df):
    """Return available split ids and S0 pair names from split{n}_{pair}_S0 columns."""
    pattern = re.compile(r"^split(\d+)_(.+)_S0$")
    splits = []
    pairs = []

    for col in df.columns:
        match = pattern.match(col)
        if match:
            split = int(match.group(1))
            pair = match.group(2)
            if split not in splits:
                splits.append(split)
            if pair not in pairs:
                pairs.append(pair)

    return sorted(splits), _order_pairs_from_components(pairs, detect_composition_columns(df))


def _order_pairs_from_components(pairs, composition_cols):
    components = [col[2:] for col in composition_cols]
    if not components:
        return list(pairs)

    rank = {}
    for i, a in enumerate(components):
        for j, b in enumerate(components[i:], start=i):
            rank[f"{a}{b}"] = len(rank)
            rank[f"{a}_{b}"] = rank[f"{a}{b}"]

    return sorted(pairs, key=lambda pair: rank.get(pair, len(rank) + pairs.index(pair)))


def _resolve_s0_selection(df, pairs=None, splits=None):
    available_splits, available_pairs = detect_s0_columns(df)
    if not available_splits or not available_pairs:
        raise ValueError("No S0 columns found. Expected columns like split0_AB_S0.")

    if splits is None:
        splits = available_splits
    else:
        splits = list(splits)

    if pairs is None:
        pairs = available_pairs
    else:
        pairs = list(pairs)

    return pairs, splits


def _plot_ordered_frame(df, sort_by=None):
    if sort_by is None:
        sort_by = ("row_id",) if "row_id" in df.columns else ()
    sort_cols = [col for col in sort_by if col in df.columns]
    plot_df = df.sort_values(sort_cols).reset_index(drop=True) if sort_cols else df.copy()
    plot_df["_plot_index"] = np.arange(len(plot_df))
    return plot_df


def _x_values(plot_df, x_col=None):
    if x_col is None:
        x_col = "row_id" if "row_id" in plot_df.columns else "_plot_index"

    if x_col not in plot_df.columns:
        raise ValueError(f"x_col={x_col!r} is not in the dataframe.")

    return plot_df[x_col].to_numpy(), x_col


def _subplot_grid(n_panels, max_cols=3):
    plt = _get_plt()
    ncols = min(max_cols, max(1, n_panels))
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.5 * ncols, 3.1 * nrows),
        sharex=True,
        squeeze=False,
    )
    return fig, axes.ravel()


def _robust_limits(values, quantile=0.995, pad_fraction=0.05):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None

    low_q = (1.0 - quantile) / 2.0
    high_q = 1.0 - low_q
    lo, hi = np.nanquantile(finite, [low_q, high_q])

    if np.isclose(lo, hi):
        delta = 1.0 if np.isclose(lo, 0.0) else abs(lo) * pad_fraction
        return lo - delta, hi + delta

    pad = (hi - lo) * pad_fraction
    return lo - pad, hi + pad


def _mark_clipped(ax, x, y, ylim, color):
    lo, hi = ylim
    y = np.asarray(y, dtype=float)
    high = np.isfinite(y) & (y > hi)
    low = np.isfinite(y) & (y < lo)

    if np.any(high):
        ax.scatter(
            np.asarray(x)[high],
            np.full(np.count_nonzero(high), hi),
            marker="^",
            s=16,
            color=color,
            edgecolor="none",
            clip_on=False,
            zorder=5,
        )
    if np.any(low):
        ax.scatter(
            np.asarray(x)[low],
            np.full(np.count_nonzero(low), lo),
            marker="v",
            s=16,
            color=color,
            edgecolor="none",
            clip_on=False,
            zorder=5,
        )

    return int(np.count_nonzero(high) + np.count_nonzero(low))


def _visible_yerr(y, yerr, ylim):
    if ylim is None:
        return y, yerr

    lo, hi = ylim
    y = np.asarray(y, dtype=float)
    visible = np.isfinite(y) & (y >= lo) & (y <= hi)

    y_plot = y.copy()
    y_plot[~visible] = np.nan

    if yerr is None:
        return y_plot, None

    yerr_plot = np.asarray(yerr, dtype=float).copy()
    yerr_plot[~visible] = np.nan
    return y_plot, yerr_plot


def find_s0_split_outlier_rows(
    df,
    pairs=None,
    splits=None,
    max_diff=10.0,
    threshold=None,
    percent_min_diff=0.5,
    percent_threshold=1.0,
    mean_eps=1e-12,
    use_abs=True,
):
    """
    Find rows where split S0 values disagree strongly.

    For each composition and pair, compute the mean S0 over splits, then the
    maximum split deviation from that mean. A row is flagged if any pair has
    max deviation >= max_diff, or if it satisfies both relative-spread checks:

        max deviation > percent_min_diff
        (S0_max - S0_min) / abs(mean S0) > percent_threshold
    """
    if threshold is not None:
        max_diff = threshold

    pairs, splits = _resolve_s0_selection(df, pairs=pairs, splits=splits)

    bad_row_mask = np.zeros(len(df), dtype=bool)
    records = []

    for pair in pairs:
        split_cols = [f"split{split}_{pair}_S0" for split in splits]
        split_cols = [col for col in split_cols if col in df.columns]
        if not split_cols:
            continue

        values = df[split_cols].to_numpy(dtype=float)
        mean = np.nanmean(values, axis=1)
        deviations = values - mean[:, None]
        scores = np.abs(deviations) if use_abs else deviations
        max_deviation = np.nanmax(scores, axis=1)
        spread = np.nanmax(values, axis=1) - np.nanmin(values, axis=1)
        denom = np.abs(mean)
        relative_spread = np.where(
            denom > mean_eps,
            spread / denom,
            np.where(spread > mean_eps, np.inf, 0.0),
        )

        hard_bad = max_deviation >= max_diff
        percent_bad = (
            (max_deviation > percent_min_diff)
            & (relative_spread > percent_threshold)
        )
        local_bad = hard_bad | percent_bad
        bad_row_mask |= local_bad

        for row_index in np.where(local_bad)[0]:
            worst_split_idx = int(np.nanargmax(scores[row_index]))
            worst_col = split_cols[worst_split_idx]
            reasons = []
            if hard_bad[row_index]:
                reasons.append("max_diff")
            if percent_bad[row_index]:
                reasons.append("percent")
            records.append(
                {
                    "csv_line": int(row_index + 2),
                    "row_index": int(row_index),
                    "row_id": df.iloc[row_index]["row_id"]
                    if "row_id" in df.columns
                    else row_index,
                    "pair": pair,
                    "worst_column": worst_col,
                    "mean_s0": mean[row_index],
                    "worst_value": values[row_index, worst_split_idx],
                    "deviation": deviations[row_index, worst_split_idx],
                    "max_deviation": max_deviation[row_index],
                    "spread": spread[row_index],
                    "relative_spread": relative_spread[row_index],
                    "reason": "+".join(reasons),
                }
            )

    return bad_row_mask, records


def clean_s0_splits_csv(
    csv_path,
    output_path=None,
    max_diff=10.0,
    threshold=None,
    percent_min_diff=0.5,
    percent_threshold=1.0,
    mean_eps=1e-12,
    pairs=None,
    splits=None,
    use_abs=True,
):
    """
    Remove compositions with inconsistent S0 split values and write a new CSV.

    Rows are kept only when, for every S0 pair,

        max(abs(S0_split - mean(S0_splits))) < max_diff

    and when the relative-spread rule does not trigger:

        max(abs(S0_split - mean(S0_splits))) > percent_min_diff
        and (S0_max - S0_min) / abs(mean(S0_splits)) > percent_threshold

    using the available split columns for that pair.
    """
    import pandas as pd

    if threshold is not None:
        max_diff = threshold

    csv_path = str(csv_path)
    if output_path is None:
        stem = csv_path[:-4] if csv_path.endswith(".csv") else csv_path
        output_path = f"{stem}-cleaned.csv"

    df = pd.read_csv(csv_path)
    bad_row_mask, records = find_s0_split_outlier_rows(
        df,
        pairs=pairs,
        splits=splits,
        max_diff=max_diff,
        percent_min_diff=percent_min_diff,
        percent_threshold=percent_threshold,
        mean_eps=mean_eps,
        use_abs=use_abs,
    )
    cleaned = df.loc[~bad_row_mask].copy()
    cleaned.to_csv(output_path, index=False)

    return cleaned, records, output_path


def plot_s0_splits(
    df,
    pairs=None,
    splits=None,
    x_col=None,
    sort_by=None,
    show_errors=True,
    clip=True,
    robust_ylim=True,
    robust_quantile=0.98,
):
    """
    Plot split S0 values for each detected pair.

    Use this directly after:

        CSV_PATH = "my-s0-splits.csv"
        df = pd.read_csv(CSV_PATH)
    """
    pairs, splits = _resolve_s0_selection(df, pairs=pairs, splits=splits)

    plot_df = _plot_ordered_frame(df, sort_by=sort_by)
    x, x_label = _x_values(plot_df, x_col=x_col)

    plt = _get_plt()
    fig, axes = _subplot_grid(len(pairs))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for ax, pair in zip(axes, pairs):
        values = []
        for split in splits:
            col = f"split{split}_{pair}_S0"
            if col in plot_df.columns:
                values.append(plot_df[col].to_numpy())

        ylim = None
        if clip and robust_ylim and values:
            ylim = _robust_limits(np.concatenate(values), quantile=robust_quantile)

        n_clipped = 0
        for i, split in enumerate(splits):
            col = f"split{split}_{pair}_S0"
            err_col = f"split{split}_{pair}_S0_error"
            if col not in plot_df.columns:
                continue

            color = colors[i % len(colors)]
            y = plot_df[col].to_numpy()
            yerr = plot_df[err_col].to_numpy() if show_errors and err_col in plot_df.columns else None
            y_plot, yerr_plot = _visible_yerr(y, yerr, ylim)

            ax.errorbar(
                x,
                y_plot,
                yerr=yerr_plot,
                fmt=".",
                ms=2.5,
                lw=0.8,
                elinewidth=0.45,
                capsize=0,
                alpha=0.78,
                color=color,
                label=f"split {split}",
            )

            if ylim is not None:
                n_clipped += _mark_clipped(ax, x, y, ylim, color)

        if ylim is not None:
            ax.set_ylim(*ylim)

        title = pair if n_clipped == 0 else f"{pair} ({n_clipped} clipped)"
        ax.set_title(title)
        ax.grid(True, alpha=0.22)
        ax.set_ylabel(r"$S^0$")

    for ax in axes[len(pairs):]:
        ax.set_visible(False)

    for ax in axes[-3:]:
        if ax.get_visible():
            ax.set_xlabel(x_label if x_label != "_plot_index" else "composition index")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=max(1, len(labels)),
        frameon=False,
        fontsize=9,
    )
    fig.suptitle("S0 Split Consistency", y=0.945, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return fig, axes


def plot_s0_split_residuals(
    df,
    pairs=None,
    splits=None,
    x_col=None,
    sort_by=None,
    clip=True,
    robust_ylim=True,
    robust_quantile=0.98,
    center="median",
):
    """Plot split residuals for a more direct consistency check."""
    pairs, splits = _resolve_s0_selection(df, pairs=pairs, splits=splits)

    plot_df = _plot_ordered_frame(df, sort_by=sort_by)
    x, x_label = _x_values(plot_df, x_col=x_col)

    plt = _get_plt()
    fig, axes = _subplot_grid(len(pairs))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for ax, pair in zip(axes, pairs):
        split_cols = [f"split{split}_{pair}_S0" for split in splits]
        split_cols = [col for col in split_cols if col in plot_df.columns]
        if not split_cols:
            continue

        y_values = plot_df[split_cols].to_numpy()
        if center == "mean":
            y_center = np.nanmean(y_values, axis=1)
            center_label = r"\overline{S^0}"
        elif center == "median":
            y_center = np.nanmedian(y_values, axis=1)
            center_label = r"\mathrm{median}(S^0)"
        else:
            raise ValueError("center must be either 'median' or 'mean'.")
        residuals = y_values - y_center[:, None]

        ylim = None
        if clip and robust_ylim:
            ylim = _robust_limits(residuals.ravel(), quantile=robust_quantile)

        n_clipped = 0
        for i, col in enumerate(split_cols):
            split = col.split("_", 1)[0].replace("split", "")
            color = colors[i % len(colors)]
            y = residuals[:, i]
            y_plot, _ = _visible_yerr(y, None, ylim)
            ax.plot(
                x,
                y_plot,
                ".",
                ms=2.5,
                alpha=0.78,
                color=color,
                label=f"split {split}",
            )
            if ylim is not None:
                n_clipped += _mark_clipped(ax, x, y, ylim, color)

        if ylim is not None:
            ax.set_ylim(*ylim)

        ax.axhline(0.0, color="0.2", lw=0.8)
        title = pair if n_clipped == 0 else f"{pair} ({n_clipped} clipped)"
        ax.set_title(title)
        ax.grid(True, alpha=0.22)
        ax.set_ylabel(rf"$S^0_\mathrm{{split}} - {center_label}$")

    for ax in axes[len(pairs):]:
        ax.set_visible(False)

    for ax in axes[-3:]:
        if ax.get_visible():
            ax.set_xlabel(x_label if x_label != "_plot_index" else "composition index")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=max(1, len(labels)),
        frameon=False,
        fontsize=9,
    )
    fig.suptitle("S0 Split Residuals", y=0.945, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return fig, axes


def main():
    import argparse
    import matplotlib
    import pandas as pd

    matplotlib.use("Agg")
    plt = _get_plt()

    parser = argparse.ArgumentParser(description="Plot S0 consistency across CSV splits.")
    parser.add_argument("csv")
    parser.add_argument("--values-out", default="s0_split_values.png")
    parser.add_argument("--residuals-out", default="s0_split_residuals.png")
    parser.add_argument("--clean-out", default=None)
    parser.add_argument("--max-diff", type=float, default=10.0)
    parser.add_argument("--percent-min-diff", type=float, default=0.5)
    parser.add_argument("--percent-threshold", type=float, default=1.0)
    parser.add_argument("--clean-threshold", type=float, default=None)
    parser.add_argument("--no-errors", action="store_true")
    parser.add_argument("--no-clip", action="store_true")
    parser.add_argument("--robust-quantile", type=float, default=0.98)
    parser.add_argument("--residual-center", choices=("median", "mean"), default="median")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    fig, _ = plot_s0_splits(
        df,
        show_errors=not args.no_errors,
        clip=not args.no_clip,
        robust_quantile=args.robust_quantile,
    )
    fig.savefig(args.values_out, dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, _ = plot_s0_split_residuals(
        df,
        clip=not args.no_clip,
        robust_quantile=args.robust_quantile,
        center=args.residual_center,
    )
    fig.savefig(args.residuals_out, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {args.values_out}")
    print(f"Saved {args.residuals_out}")

    if args.clean_out is not None:
        cleaned, records, output_path = clean_s0_splits_csv(
            args.csv,
            output_path=args.clean_out,
            max_diff=args.max_diff,
            threshold=args.clean_threshold,
            percent_min_diff=args.percent_min_diff,
            percent_threshold=args.percent_threshold,
        )
        print(f"Saved {output_path}")
        print(f"Kept {len(cleaned)} rows; removed {len(set(r['row_index'] for r in records))} rows")
        if records:
            print("Removed row details:")
            for record in records:
                print(
                    "csv_line={csv_line} row_id={row_id} pair={pair} "
                    "worst_column={worst_column} value={worst_value:.6g} "
                    "mean={mean_s0:.6g} deviation={deviation:.6g} "
                    "relative_spread={relative_spread:.6g} reason={reason}".format(
                        **record
                    )
                )


if __name__ == "__main__":
    main()
