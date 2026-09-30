"""MUSE pointing (observing-block) tiles for the PHANGS--MUSE mosaics.

Each PHANGS--MUSE mosaic is a grid of ~1'x1' pointings observed and flux-calibrated
separately.  The tile boundaries are read off the H-beta flux-error plane: the noise
floor of each pointing is a constant that jumps at the mosaic seams, so rectangles in
the (rotated) MUSE pixel frame reproduce the observing-block footprint without any
external metadata.  ``RECT`` records the rectangles adopted per galaxy (rotation angle
of the mosaic and (u1, u2, v1, v2) pixel ranges); ``noise_floor_tiles`` and
``edge_profiles`` are the diagnostics used to read them off.

The face-on <-> sky mapping replicates the stage-1 notebooks (affine fit of the
catalogue deprojection).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from scipy.ndimage import median_filter

GALAXIES = {
    "NGC628":  dict(distance_mpc=9.8,  r25_arcmin=4.9, maps="NGC0628_MAPS_copt_0.92asec.fits", catalog="NGC0628_catalog.fits"),
    "NGC1566": dict(distance_mpc=17.7, r25_arcmin=3.6, maps="NGC1566/NGC1566_MAPS_copt_0.80asec.fits", catalog="NGC1566/NGC1566_catalog.fits"),
    "NGC4254": dict(distance_mpc=13.1, r25_arcmin=2.5, maps="NGC4254/NGC4254_MAPS_copt_0.89asec.fits", catalog="NGC4254/NGC4254_catalog.fits"),
    "NGC4303": dict(distance_mpc=17.0, r25_arcmin=3.4, maps="NGC4303/NGC4303_MAPS_copt_0.78asec.fits", catalog="NGC4303/NGC4303_catalog.fits"),
}
# rectangles (u1, u2, v1, v2) in MUSE pixel coordinates rotated by ``angle`` [deg]; read off the H-beta error floor
RECT = {
    "NGC628": dict(angle=45.0, rects=[(u1, u2, v1, v2) for u1, u2 in zip([440, 744, 1016, 1296], [744, 1016, 1296, 1576])
                                      for v1, v2 in zip([-412, -140, 172], [-140, 172, 468])]),
    "NGC1566": dict(angle=0.0, rects=[(190, 490, 0, 295), (490, 790, 0, 295), (190, 490, 600, 890), (490, 790, 600, 890), (190, 490, 890, 1180),
                                      (20, 320, 295, 600), (320, 610, 295, 600), (610, 895, 295, 600)]),
    "NGC4254": dict(angle=0.0, rects=[(u1, u2, v1, v2) for u1, u2 in zip([20, 313, 607], [313, 607, 900]) for v1, v2 in zip([300, 590, 880], [590, 880, 1180])]
                    + [(120, 413, 0, 300), (413, 707, 0, 300), (707, 1000, 0, 300)]),
    "NGC4303": dict(angle=0.0, rects=[(u1, u2, v1, v2) for u1, u2 in zip([0, 315, 605], [315, 605, 900]) for v1, v2 in zip([0, 292, 580], [292, 580, 884])]),
}
TILE_PITCH_PIX = 295.0   # ~1' MUSE field at 0.2"/pix


class Geometry:
    """Face-on <-> sky affine transform replicated from the stage-1 notebooks."""

    def __init__(self, data_root, galaxy):
        g = GALAXIES[galaxy]; self.distance_mpc = g["distance_mpc"]
        r25_kpc = g["distance_mpc"]*1e3*np.tan(np.deg2rad(g["r25_arcmin"]/60))
        cat = Table.read(Path(data_root)/g["catalog"])
        num = lambda n: np.asarray(np.ma.asarray(cat[n]).filled(np.nan), float)
        r = num("DEPROJ_R_R25")*r25_kpc; phi = num("DEPROJ_PHI")
        target = np.column_stack((r*np.cos(phi), r*np.sin(phi)))
        ra, dec = num("RAJ2000"), num("DECJ2000")
        self.ra0, self.dec0 = float(np.nanmedian(ra)), float(np.nanmedian(dec))
        self.a2k = np.pi/180*self.distance_mpc*1e3
        Dm = self.design(ra, dec)
        inl = np.all(np.isfinite(Dm), axis=1) & np.all(np.isfinite(target), axis=1)
        for _ in range(6):
            A = np.linalg.lstsq(Dm[inl], target[inl], rcond=None)[0]
            res = np.hypot(*(Dm@A-target).T)
            med = np.median(res[inl]); mad = np.median(np.abs(res[inl]-med))
            inl = np.isfinite(res) & (res < max(0.10, med+10*1.4826*mad))
        self.A = A; self.fit_residual_kpc = float(np.median(res[inl]))
        self.wcs = WCS(fits.getheader(Path(data_root)/g["maps"], "HA6562_FLUX"))

    def design(self, ra, dec):
        east = (ra-self.ra0)*np.cos(np.deg2rad(self.dec0))*self.a2k; north = (dec-self.dec0)*self.a2k
        return np.column_stack((east, north, np.ones_like(east)))

    def face_to_sky(self, x, y):
        xy = np.column_stack((np.ravel(x), np.ravel(y))); en = (xy-self.A[2]) @ np.linalg.inv(self.A[:2])
        return self.ra0+en[:, 0]/(self.a2k*np.cos(np.deg2rad(self.dec0))), self.dec0+en[:, 1]/self.a2k

    def face_to_skypix(self, x, y):
        ra, dec = self.face_to_sky(x, y); return self.wcs.all_world2pix(ra, dec, 0)


def noise_floor_tiles(err_map, angle_deg, cell=8):
    """10th-percentile log-error per cell of the (rotated) pixel frame: exposes the mosaic seams."""
    e = np.log10(err_map.astype(float)); ny, nx = e.shape; yy, xx = np.mgrid[:ny, :nx]; ok = np.isfinite(e)
    th = np.deg2rad(angle_deg); u = xx*np.cos(th)+yy*np.sin(th); v = -xx*np.sin(th)+yy*np.cos(th)
    u0, v0 = np.floor(u[ok].min()), np.floor(v[ok].min())
    ui = ((u-u0)//cell).astype(int); vi = ((v-v0)//cell).astype(int); nu, nv = ui.max()+1, vi.max()+1
    idx = (vi*nu+ui)[ok]; order = np.argsort(idx); vals = e[ok][order]; ids = idx[order]
    bounds = np.searchsorted(ids, np.arange(nv*nu+1)); out = np.full(nv*nu, np.nan)
    for k in range(nv*nu):
        sl = vals[bounds[k]:bounds[k+1]]
        if len(sl) > 20:
            out[k] = np.percentile(sl, 10)
    return out.reshape(nv, nu), u0, v0, cell


def edge_profiles(tile_map):
    du = np.nanmedian(np.abs(np.diff(tile_map, axis=1)), axis=0); dv = np.nanmedian(np.abs(np.diff(tile_map, axis=0)), axis=1)
    return np.nan_to_num(du), np.nan_to_num(dv)


def rotation_angle(err_map):
    e = np.log10(err_map.astype(float)); ok = np.isfinite(e)
    m = np.nan_to_num(median_filter(np.where(ok, e, np.nan), size=9))
    gy, gx = np.gradient(m); g = np.hypot(gx, gy); g[~ok] = 0
    sel = g > np.percentile(g[ok], 99)
    ang = np.mod(np.degrees(np.arctan2(gy[sel], gx[sel])), 90)
    h, ed = np.histogram(ang, bins=90, range=(0, 90), weights=g[sel])
    return 0.5*(ed[np.argmax(h)]+ed[np.argmax(h)+1])


def labels_from_rects(u, v, rects, du=0.0, dv=0.0):
    lab = np.full(u.shape, -1); best = np.full(u.shape, np.inf)
    for k, (u1, u2, v1, v2) in enumerate(rects):
        u1, u2, v1, v2 = u1+du, u2+du, v1+dv, v2+dv
        inside = (u >= u1) & (u < u2) & (v >= v1) & (v < v2)
        d = np.hypot(np.clip(u1-u, 0, None)+np.clip(u-u2, 0, None), np.clip(v1-v, 0, None)+np.clip(v-v2, 0, None))
        better = (lab < 0) & (d < best); best[better] = d[better]
        lab = np.where(inside, k, lab)
    if (lab < 0).any():   # cells outside every rectangle -> nearest rectangle
        for k, (u1, u2, v1, v2) in enumerate(rects):
            d = np.hypot(np.clip(u1+du-u, 0, None)+np.clip(u-u2-du, 0, None), np.clip(v1+dv-v, 0, None)+np.clip(v-v2-dv, 0, None))
            take = (lab < 0) & (d <= best); lab[take] = k
    return lab


def pointing_labels(data_root, galaxy, pixel_map, shift=(0.0, 0.0)):
    """Pointing label of every face-on cell (integer grid the shape of the pixel map)."""
    geom = Geometry(data_root, galaxy)
    X, Y = pixel_map["x_grid_kpc"], pixel_map["y_grid_kpc"]
    px, py = geom.face_to_skypix(X, Y); th = np.deg2rad(RECT[galaxy]["angle"])
    u = (px*np.cos(th)+py*np.sin(th)).reshape(X.shape); v = (-px*np.sin(th)+py*np.cos(th)).reshape(X.shape)
    labels = labels_from_rects(u, v, RECT[galaxy]["rects"], *shift)
    return dict(pointing=labels, u=u, v=v, rects=np.array(RECT[galaxy]["rects"]), angle=np.array(RECT[galaxy]["angle"]),
                geometry_fit_residual_kpc=np.array(geom.fit_residual_kpc))


def shifted_tiling_labels(labels_info, rng, pitch=TILE_PITCH_PIX, margin=25.0):
    """Randomly shifted, periodically extended tiling (null test for the same/cross split)."""
    u, v, rects = labels_info["u"], labels_info["v"], [tuple(r) for r in labels_info["rects"]]
    du, dv = rng.uniform(margin, pitch-margin), rng.uniform(margin, pitch-margin)
    ext = [(u1+a*pitch, u2+a*pitch, v1+b*pitch, v2+b*pitch) for (u1, u2, v1, v2) in rects for a in (-1, 0) for b in (-1, 0)]
    return labels_from_rects(u, v, ext, du, dv)
