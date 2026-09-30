"""Project the local KTZ *second moments* before pair normalisation.

Unlike a transfer applied to an already normalised, separation-only curve,
these responses retain the midpoint rate/template and the actual annulus of
each endpoint. Every valid pixel pair, including separations outside the fit,
enters the annular averages. The connected local-source approximation is
unchanged; this is not an exact solution of the global diffusion equation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.signal import fftconvolve

VERSION = 2


def annular_project(values, valid, radius, annulus_width):
    result = np.where(valid, values, 0.0).astype(float)
    labels = np.floor(radius / annulus_width).astype(int)
    for label in np.unique(labels[valid]):
        selected = valid & (labels == label)
        result[selected] -= np.mean(result[selected])
    return result


def mean_pair_moments(values, valid, pixel_size, edges):
    """Exhaustive ordered-pair numerator and arithmetic endpoint moment."""
    height, width = valid.shape
    ly, lx = np.meshgrid(
        np.arange(1-height, height)*pixel_size,
        np.arange(1-width, width)*pixel_size, indexing="ij",
    )
    distance = np.hypot(lx, ly).ravel()
    bins = np.searchsorted(edges, distance, side="right") - 1
    nb = len(edges)-1
    selected = (bins >= 0) & (bins < nb) & (distance > 0)
    mask = valid.astype(float)
    field = np.where(valid, values, 0.0)
    numerator = fftconvolve(field, field[::-1, ::-1], mode="full")
    endpoint = fftconvolve(field**2, mask[::-1, ::-1], mode="full")
    counts = fftconvolve(mask, mask[::-1, ::-1], mode="full")
    reduce = lambda a: np.bincount(
        bins[selected], weights=a.ravel()[selected], minlength=nb,
    )
    count = reduce(counts)
    return reduce(numerator)/count, reduce(endpoint)/count, count


def build_window_responses(
    valid, x_grid, y_grid, modulation, pixel_size, edges,
    annulus_width, reference_radius, rate_log_slope_grid, block_size=32,
    resume=None, checkpoint=None,
):
    """Exact P C P^T responses for the adopted midpoint local covariance.

    C_ij = f(midpoint_ij) K(|x_i-x_j|). f is lambda_0,
    lambda_0*g, or lambda_0**2. K is linearly interpolated on a
    full-window radial grid. Responses return average raw moments, not xi.
    No dense N_pix by N_pix covariance is stored or sampled.
    """
    valid = np.asarray(valid, bool)
    iy, ix = np.nonzero(valid)
    x, y = x_grid[valid], y_grid[valid]
    radius = np.hypot(x_grid, y_grid)
    label_values, ann = np.unique(
        np.floor(radius[valid]/annulus_width).astype(int), return_inverse=True,
    )
    npix, na, nb = len(x), len(label_values), len(edges)-1
    ann_counts = np.bincount(ann, minlength=na)
    # The bounding diagonal exceeds every actual pixel-pair separation.
    maximum_distance = np.hypot(np.ptp(x), np.ptp(y))
    # Resolve the narrow injection kernel without making the whole response
    # tensor uniformly fine out to the disc diameter.
    kernel_radius = np.concatenate((
        np.arange(0.0, 0.600001, 0.01),
        np.arange(0.66, maximum_distance+0.12, 0.06),
    ))
    nk = len(kernel_radius)
    half_y, half_x = np.meshgrid(
        np.arange(2*valid.shape[0]-1)/2,
        np.arange(2*valid.shape[1]-1)/2, indexing="ij",
    )
    half_modulation = map_coordinates(
        modulation, [half_y, half_x], order=1, mode="nearest",
    )

    def block_geometry(start, stop):
        distance = np.hypot(
            x[start:stop, None]-x[None, :],
            y[start:stop, None]-y[None, :],
        )
        midpoint_radius = np.hypot(
            0.5*(x[start:stop, None]+x[None, :]),
            0.5*(y[start:stop, None]+y[None, :]),
        )
        midpoint_modulation = half_modulation[
            iy[start:stop, None]+iy[None, :],
            ix[start:stop, None]+ix[None, :],
        ]
        lower = np.searchsorted(kernel_radius, distance, side="right")-1
        lower = np.clip(lower, 0, nk-2)
        fraction = (distance-kernel_radius[lower])/(kernel_radius[lower+1]-kernel_radius[lower])
        output_bin = np.searchsorted(edges, distance, side="right")-1
        fitted = (distance > 0) & (output_bin >= 0) & (output_bin < nb)
        return midpoint_radius, midpoint_modulation, lower, fraction, output_bin, fitted

    # S[b,i,a] is the number of partners in annulus a for endpoint i and bin b.
    partner = np.zeros((npix, na, nb), dtype=np.float64)
    for start in range(0, npix, block_size):
        stop = min(start+block_size, npix)
        *_, bins, selected = block_geometry(start, stop)
        row_ann = np.arange(stop-start)[:, None]*na + ann[None, :]
        indices = row_ann*nb + bins
        partner[start:stop] = np.bincount(
            indices[selected], minlength=(stop-start)*na*nb,
        ).reshape(stop-start, na, nb)
    ordered_counts = partner.sum(axis=(0, 1))
    partner_flat = np.ascontiguousarray(partner.reshape(npix*na, nb).T)
    endpoint_weight = partner.sum(axis=1).T
    ann_pair_counts = np.asarray([
        partner[ann == a].sum(axis=0) for a in range(na)
    ])  # [endpoint annulus, partner annulus, bin]
    response = np.full((len(rate_log_slope_grid), 3, 2, nb, nk), np.nan, dtype=float)
    mean_response = np.full((len(rate_log_slope_grid), 3, 2, nb), np.nan, dtype=float)
    completed = np.zeros(len(rate_log_slope_grid), bool)
    if resume is not None:
        response[:], mean_response[:], completed[:] = resume

    for ei, rate_log_slope in enumerate(rate_log_slope_grid):
        if completed[ei]:
            print(f"Resumed completed rate_log_slope node {ei+1}",flush=True)
            continue
        print(f"Window response rate_log_slope {ei+1}/{len(rate_log_slope_grid)}: {rate_log_slope:+.4f}", flush=True)
        rate_grid = np.exp(rate_log_slope*(radius-reference_radius))
        mean0 = annular_project(rate_grid, valid, radius, annulus_width)
        mean1 = annular_project(rate_grid*modulation, valid, radius, annulus_width)
        n0, d0, _ = mean_pair_moments(mean0, valid, pixel_size, edges)
        n1, d1, _ = mean_pair_moments(mean1, valid, pixel_size, edges)
        nt, dt, _ = mean_pair_moments(mean0+mean1, valid, pixel_size, edges)
        mean_response[ei] = np.asarray([
            [n0, d0], [nt-n0-n1, dt-d0-d1], [n1, d1],
        ])
        for component in range(3):
            row_mean = np.zeros((npix, na, nk), dtype=float)
            raw_numerator = np.zeros((nb, nk), dtype=float)
            for start in range(0, npix, block_size):
                stop = min(start+block_size, npix)
                mr, gm, lo, fraction, bins, selected = block_geometry(start, stop)
                rate = np.exp(rate_log_slope*(mr-reference_radius))
                factor = rate if component == 0 else rate*gm if component == 1 else rate**2
                row_ann = np.arange(stop-start)[:, None]*na + ann[None, :]
                left_weight, right_weight = factor*(1-fraction), factor*fraction
                histogram = np.bincount(
                    (row_ann*nk+lo).ravel(), weights=left_weight.ravel(),
                    minlength=(stop-start)*na*nk,
                ) + np.bincount(
                    (row_ann*nk+lo+1).ravel(), weights=right_weight.ravel(),
                    minlength=(stop-start)*na*nk,
                )
                row_mean[start:stop] = (
                    histogram.reshape(stop-start, na, nk)/ann_counts[None, :, None]
                )
                raw_numerator += (
                    np.bincount(
                        (bins*nk+lo)[selected], weights=left_weight[selected],
                        minlength=nb*nk,
                    ) + np.bincount(
                        (bins*nk+lo+1)[selected], weights=right_weight[selected],
                        minlength=nb*nk,
                    )
                ).reshape(nb, nk)
            # A[a,c,k] averages row_mean(i,a,k) over endpoints i in annulus c.
            ann_mean = np.asarray([
                row_mean[ann == a].mean(axis=0) for a in range(na)
            ])
            numerator = (
                raw_numerator - 2*(partner_flat @ row_mean.reshape(npix*na, nk))
                + ann_pair_counts.reshape(na*na, nb).T @ ann_mean.reshape(na*na, nk)
            )
            diagonal = -2*row_mean[np.arange(npix), ann] + ann_mean[ann, ann]
            factor_diag = rate_grid[valid]
            if component == 1:
                factor_diag = factor_diag*modulation[valid]
            elif component == 2:
                factor_diag = factor_diag**2
            diagonal[:, 0] += factor_diag
            denominator = endpoint_weight @ diagonal
            response[ei, component] = np.asarray([numerator, denominator])/ordered_counts[None, :, None]
            print(f"  projected component {component+1}/3", flush=True)
        completed[ei] = True
        if checkpoint is not None:
            checkpoint(response, mean_response, completed)
    return {
        "kernel_radius": kernel_radius, "rate_log_slope_grid": np.asarray(rate_log_slope_grid),
        "response": response, "mean_response": mean_response,
        "ordered_pair_counts": ordered_counts,
    }


def load_or_build_window_responses(path, **kwargs):
    digest = hashlib.sha256(str(VERSION).encode())
    for key, value in sorted(kwargs.items()):
        if key == "block_size":
            continue
        digest.update(key.encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    signature = digest.hexdigest()
    path = Path(path)
    if path.exists():
        with np.load(path, allow_pickle=False) as archive:
            if str(archive["signature"].item()) == signature:
                return {k: archive[k] for k in archive.files if k != "signature"}
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path=path.with_name(path.stem+"_checkpoint.npz")
    resume=None
    if checkpoint_path.exists():
        with np.load(checkpoint_path,allow_pickle=False) as a:
            if str(a["signature"].item())==signature:
                resume=(a["response"],a["mean_response"],a["completed"])
    def checkpoint(response, mean_response, completed):
        np.savez_compressed(checkpoint_path,signature=signature,response=response,
                            mean_response=mean_response,completed=completed)
        progress={"stage":"full-window response precomputation",
                  "completed_rate_log_slope_nodes":int(completed.sum()),"total_rate_log_slope_nodes":len(completed)}
        (path.parent/"run_progress.json").write_text(json.dumps(progress,indent=2)+"\n")
    (path.parent/"run_progress.json").write_text(json.dumps({"stage":"full-window response precomputation","completed_rate_log_slope_nodes":0 if resume is None else int(resume[2].sum()),"total_rate_log_slope_nodes":len(kwargs["rate_log_slope_grid"])},indent=2)+"\n")
    result = build_window_responses(**kwargs,resume=resume,checkpoint=checkpoint)
    np.savez_compressed(path, signature=signature, **result)
    return result
