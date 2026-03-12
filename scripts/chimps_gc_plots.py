"""
chimps_gc_plots.py
==================
Two figures from the CHIMPS 12CO(3-2) data:

1. chimps_gcregion_arms.png
   Four-panel grayscale spatial image of the inner GC: |l| < 3°, |b| < 0.5°
   One panel per arm, using pre-computed background-subtracted mosaics.

2. chimps_pv.png
   Position-velocity (l–v) diagram along b ~ 0
   with labelled arm tracks:
     • Near 3 kpc arm   v(l) = 4l − 50  km/s
     • 18 km/s arm      v(l) = 4l + 18  km/s
     • Local arm        v(l) = 0         km/s
     • Norma arm        v(l) = (50/9)l − 95/3  km/s

Usage:
    python chimps_gc_plots.py
"""

import os
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
import astropy.units as u
from reproject import reproject_interp

warnings.filterwarnings('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', message='.*RADECSYS.*')
warnings.filterwarnings('ignore', message='.*obsfix.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*empty slice.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*All-NaN.*')

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CHIMPS_DIR = '/orange/adamginsburg/cmz/CHIMPS'
CHIMPS_FILES = [
    os.path.join(CHIMPS_DIR, '12CO_GC_357-358_mosaic.fits'),
    os.path.join(CHIMPS_DIR, '12CO_GC_359-000_mosaic.fits'),
    os.path.join(CHIMPS_DIR, '12CO_GC_001-002_mosaic.fits'),
    os.path.join(CHIMPS_DIR, '12CO_GC_003-005_mosaic.fits'),
]

# Pre-computed per-arm background-subtracted 2-D mosaics
ARM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHIMPS_BGSUB = {
    'Near 3 kpc arm': os.path.join(ARM_DIR, '3kpc_arm',   'CHIMPS_arm_bgsub_mosaic.fits'),
    '18 km/s arm':    os.path.join(ARM_DIR, '18kms_arm',  'CHIMPS_18kms_arm_backgroundsub_mosaic.fits'),
    'Local arm':      os.path.join(ARM_DIR, 'local_arm',  'CHIMPS_arm_bgsub_mosaic.fits'),
    'Norma arm':      os.path.join(ARM_DIR, 'norma_arm',  'CHIMPS_arm_bgsub_mosaic.fits'),
}

OUTDIR = os.path.join(ARM_DIR, 'gcregion_plots')
os.makedirs(OUTDIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Arm velocity functions  (all return km/s, ell in deg)
# ---------------------------------------------------------------------------
ell_line = np.linspace(-3.2, 3.2, 200)

ARMS = [
    {
        'name':   'Near 3 kpc arm',
        'v_fn':   lambda l: 4.0 * l - 50.0,
        'color':  'steelblue',
        'ls':     '-',
        'lw':     2.0,
        'label_l': -2.6,   # longitude where the label is placed
        'label_va': 'bottom',
    },
    {
        'name':   '18 km/s arm',
        'v_fn':   lambda l: 4.0 * l + 18.0,
        'color':  'limegreen',
        'ls':     '-',
        'lw':     2.0,
        'label_l': 1.8,
        'label_va': 'bottom',
    },
    {
        'name':   'Local arm',
        'v_fn':   lambda l: np.zeros_like(l),
        'color':  'orange',
        'ls':     '--',
        'lw':     1.8,
        'label_l': -2.6,
        'label_va': 'top',
    },
    {
        'name':   'Norma arm',
        'v_fn':   lambda l: (50.0 / 9.0) * l - 95.0 / 3.0,
        'color':  'tomato',
        'ls':     '-.',
        'lw':     2.0,
        'label_l': 2.2,
        'label_va': 'top',
    },
]

# ---------------------------------------------------------------------------
# Target WCS for the spatial image
# ---------------------------------------------------------------------------
LON_MIN, LON_MAX = -3.05, 3.05   # °   (small margin)
LAT_MIN, LAT_MAX = -0.55, 0.55   # °
PIX_SCALE = 0.003                 # °/pix  ≈ CHIMPS native ~0.001°/pix but 3× to save memory

def make_spatial_wcs():
    nx = int(round((LON_MAX - LON_MIN) / PIX_SCALE)) + 1
    ny = int(round((LAT_MAX - LAT_MIN) / PIX_SCALE)) + 1
    w = WCS(naxis=2)
    w.wcs.ctype = ['GLON-CAR', 'GLAT-CAR']
    w.wcs.crval = [0.0, 0.0]
    w.wcs.cdelt = [-PIX_SCALE, PIX_SCALE]
    w.wcs.crpix = [(nx + 1) / 2.0, (ny + 1) / 2.0]
    w.wcs.set()
    return w, (ny, nx)

SPATIAL_WCS, SPATIAL_SHAPE = make_spatial_wcs()

# ---------------------------------------------------------------------------
# Target grid for the PV diagram:  l × v
# ---------------------------------------------------------------------------
PV_LON_MIN, PV_LON_MAX = -3.1,  3.1    # deg
PV_VEL_MIN, PV_VEL_MAX = -200., 200.   # km/s
PV_LON_STEP = 0.005                      # deg/pix
PV_VEL_STEP = 2.0                        # km/s / pix

PV_NL = int(round((PV_LON_MAX - PV_LON_MIN) / PV_LON_STEP)) + 1
PV_NV = int(round((PV_VEL_MAX - PV_VEL_MIN) / PV_VEL_STEP)) + 1

# Output lon/vel arrays for the PV grid
pv_lons = np.linspace(PV_LON_MAX, PV_LON_MIN, PV_NL)   # decreasing (l increases right to left)
pv_vels = np.linspace(PV_VEL_MIN, PV_VEL_MAX, PV_NV)

# ---------------------------------------------------------------------------
# Helper: collapse along velocity and reproject to spatial grid
# ---------------------------------------------------------------------------
def load_moment0(fits_path, v_min_kms=-200, v_max_kms=200):
    """Integrate cube over v_min..v_max km/s, return 2-D reprojected array."""
    print(f'  Loading {os.path.basename(fits_path)} ...')
    hdul = fits.open(fits_path)
    cube = hdul[0].data.astype(float)   # (nv, ny, nx)  or (1, nv, ny, nx)
    hdr  = hdul[0].header.copy()
    hdul.close()

    while cube.ndim > 3:
        cube = cube[0]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        w = WCS(hdr)

    nv, ny, nx = cube.shape

    # Find velocity axis (axis index 2 in 0-based, axis 3 in FITS)
    crval3 = hdr.get('CRVAL3', 0.0)
    cdelt3 = hdr.get('CDELT3', 1000.0)
    crpix3 = hdr.get('CRPIX3', 1.0)
    vels_ms = crval3 + cdelt3 * (np.arange(nv) - (crpix3 - 1))
    vels_kms = vels_ms / 1000.0

    vel_mask = (vels_kms >= v_min_kms) & (vels_kms <= v_max_kms)
    if not vel_mask.any():
        return None

    # Mean (not plain sum) so different velocity resolutions are comparable
    slc = cube[vel_mask]
    mom0 = np.nanmean(slc, axis=0)   # (ny, nx)

    # Build clean 2-D header for reprojection
    w2 = WCS(hdr, naxis=2)
    hdr2 = w2.to_header()
    hdr2['NAXIS']  = 2
    hdr2['NAXIS1'] = nx
    hdr2['NAXIS2'] = ny

    out, _ = reproject_interp(
        fits.PrimaryHDU(data=mom0, header=hdr2),
        SPATIAL_WCS, shape_out=SPATIAL_SHAPE)
    return out


# ---------------------------------------------------------------------------
# Helper: load a pre-computed 2-D bgsub mosaic and reproject to spatial grid
# ---------------------------------------------------------------------------
def load_bgsub_mosaic(fits_path):
    """Load and reproject a 2-D background-subtracted mosaic to SPATIAL_WCS."""
    print(f'  Loading {os.path.basename(fits_path)} ...')
    hdul = fits.open(fits_path)
    data = hdul[0].data.astype(float)
    hdr  = hdul[0].header.copy()
    hdul.close()

    while data.ndim > 2:
        data = data[0]

    out, _ = reproject_interp(
        fits.PrimaryHDU(data=data, header=hdr),
        SPATIAL_WCS, shape_out=SPATIAL_SHAPE)
    return out


# ---------------------------------------------------------------------------
# Helper: extract b-averaged PV slice and resample to common l-v grid
# ---------------------------------------------------------------------------
def load_pv_slice(fits_path, b_range=(-0.25, 0.25), v_min_kms=-200, v_max_kms=200):
    """
    Average over |b| < b_range then resample onto the common (PV_NV, PV_NL) grid.
    Returns (PV_NV, PV_NL) float array.
    """
    print(f'  PV from {os.path.basename(fits_path)} ...')
    hdul = fits.open(fits_path)
    cube = hdul[0].data.astype(float)
    hdr  = hdul[0].header.copy()
    hdul.close()

    while cube.ndim > 3:
        cube = cube[0]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        w3 = WCS(hdr)

    nv, ny, nx = cube.shape

    # Velocity array (km/s)
    crval3 = hdr.get('CRVAL3', 0.0)
    cdelt3 = hdr.get('CDELT3', 1000.0)
    crpix3 = hdr.get('CRPIX3', 1.0)
    vels_kms = (crval3 + cdelt3 * (np.arange(nv) - (crpix3 - 1))) / 1000.0

    # Latitude array (deg) at centre-column  
    w2 = WCS(hdr, naxis=2)
    _, lats_all = w2.all_pix2world(np.full(ny, nx // 2), np.arange(ny), 0)

    # Wrap lons to (-180, 180)
    lons_all, _ = w2.all_pix2world(np.arange(nx), np.full(nx, ny // 2), 0)
    lons_all = np.where(lons_all > 180, lons_all - 360, lons_all)

    # Spatial masks
    b_lo, b_hi = b_range
    lat_mask = (lats_all >= b_lo) & (lats_all <= b_hi)
    lon_mask  = (lons_all >= PV_LON_MIN) & (lons_all <= PV_LON_MAX)
    vel_mask  = (vels_kms >= v_min_kms)  & (vels_kms <= v_max_kms)

    if not (lat_mask.any() and lon_mask.any() and vel_mask.any()):
        return None

    # Average over latitude  → shape (nv, nx)
    pv_native = np.nanmean(cube[:, lat_mask, :], axis=1)   # (nv, nx)

    # Resample onto common grid using simple nearest-value interpolation
    # Build output array
    out = np.full((PV_NV, PV_NL), np.nan)

    # For each native velocity/longitude pair, find output pixel
    lons_crop = lons_all[lon_mask]
    pv_crop   = pv_native[np.ix_(vel_mask, lon_mask)]   # (nv_crop, nx_crop)
    vels_crop = vels_kms[vel_mask]

    # Target pixel indices via broadcasting
    il = np.round((PV_LON_MAX - lons_crop) / PV_LON_STEP).astype(int)   # (nx_crop,)
    iv = np.round((vels_crop   - PV_VEL_MIN) / PV_VEL_STEP).astype(int) # (nv_crop,)

    valid_l = (il >= 0) & (il < PV_NL)
    valid_v = (iv >= 0) & (iv < PV_NV)

    # Build flat index arrays for every (v, l) pair at once
    iv_grid, il_grid = np.meshgrid(iv, il, indexing='ij')    # (nv_crop, nx_crop)
    valid_grid = (np.meshgrid(valid_v, valid_l, indexing='ij')[0] &
                  np.meshgrid(valid_v, valid_l, indexing='ij')[1])
    finite_grid = np.isfinite(pv_crop) & valid_grid

    iv_flat = iv_grid[finite_grid]
    il_flat = il_grid[finite_grid]
    vals    = pv_crop[finite_grid]

    flat_idx = iv_flat * PV_NL + il_flat
    accum = np.bincount(flat_idx, weights=vals,     minlength=PV_NV * PV_NL)
    count = np.bincount(flat_idx, minlength=PV_NV * PV_NL).astype(float)

    mask = count > 0
    out_flat = np.full(PV_NV * PV_NL, np.nan)
    out_flat[mask] = accum[mask] / count[mask]
    out = out_flat.reshape(PV_NV, PV_NL)
    return out


# ---------------------------------------------------------------------------
# Figure 1: Four-panel per-arm spatial image (grayscale, bgsub mosaics)
# ---------------------------------------------------------------------------
def make_spatial_figure():
    print('\n=== Spatial image (per-arm, bgsub) ===')

    arm_order = ['Near 3 kpc arm', '18 km/s arm', 'Local arm', 'Norma arm']
    extent = [LON_MAX, LON_MIN, LAT_MIN, LAT_MAX]

    fig, axes = plt.subplots(2, 2, figsize=(14, 7),
                             constrained_layout=True)
    axes = axes.flatten()

    for idx, arm_name in enumerate(arm_order):
        ax = axes[idx]
        fits_path = CHIMPS_BGSUB[arm_name]
        arr = load_bgsub_mosaic(fits_path)

        ax.set_facecolor('black')
        if arr is not None:
            finite = arr[np.isfinite(arr)]
            if finite.size > 0:
                vlo = float(np.nanpercentile(finite, 5))
                vhi = float(np.nanpercentile(finite, 99.5))
                vlo = max(vlo, 0.0)
                im = ax.imshow(arr, origin='lower', aspect='auto',
                               extent=extent, cmap='gray_r',
                               vmin=vlo, vmax=vhi, interpolation='nearest')
                cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
                cb.set_label(r'$T_{\rm A}^*$ (K)', fontsize=7)
                cb.ax.tick_params(labelsize=7)

        # Light grid
        for gl in np.arange(-3, 4):
            ax.axvline(gl, color='gray', lw=0.3, alpha=0.4)
        for gb in [-0.5, -0.25, 0, 0.25, 0.5]:
            ax.axhline(gb, color='gray', lw=0.3, alpha=0.4)

        ax.set_xlim(LON_MAX, LON_MIN)
        ax.set_ylim(-0.5, 0.5)
        ax.set_title(arm_name, fontsize=10)
        ax.set_xlabel(r'$\ell$ (deg)', fontsize=9)
        ax.set_ylabel(r'$b$ (deg)', fontsize=9)
        ax.tick_params(labelsize=8)

    fig.suptitle(r'CHIMPS $^{12}$CO(3–2) — background-subtracted arm mosaics'
                 r'  ($|\ell|<3°$, $|b|<0.5°$)',
                 fontsize=11)

    outpath = os.path.join(OUTDIR, 'chimps_gcregion_arms.png')
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {outpath}')


# ---------------------------------------------------------------------------
# Figure 2: PV diagram  (v_lim controls the velocity axis half-range in km/s)
# ---------------------------------------------------------------------------
def make_pv_figure(v_lim=200):
    print(f'\n=== PV diagram (v_lim=±{v_lim} km/s) ===')
    v_min, v_max = -v_lim, v_lim

    pvs = []
    for f in CHIMPS_FILES:
        arr = load_pv_slice(f, b_range=(-0.25, 0.25),
                            v_min_kms=v_min, v_max_kms=v_max)
        if arr is not None:
            pvs.append(arr)

    if not pvs:
        print('  No PV data!'); return

    # Coadd with NaN-mean; crop full PV grid to requested velocity range
    pv_full = np.nanmean(np.array(pvs, dtype=float), axis=0)
    iv_lo = max(int(round((v_min - PV_VEL_MIN) / PV_VEL_STEP)), 0)
    iv_hi = min(int(round((v_max - PV_VEL_MIN) / PV_VEL_STEP)) + 1, PV_NV)
    pv_mosaic = pv_full[iv_lo:iv_hi, :]

    # --- Plot ---
    aspect_ratio = (v_max - v_min) / (PV_LON_MAX - PV_LON_MIN)
    fig_h = max(3.0, min(7.0, 12 * aspect_ratio * 0.55))
    fig, ax = plt.subplots(figsize=(12, fig_h))
    ax.set_facecolor('black')

    finite = pv_mosaic[np.isfinite(pv_mosaic)]
    if finite.size > 0:
        pos = finite[finite > 0]
        _, vhi = 0.0, (float(np.nanpercentile(pos, 99)) if pos.size > 3 else 1.0)

        display = np.arcsinh(pv_mosaic / max(vhi * 0.05, 1e-9))
        dlo = np.arcsinh(0)
        dhi = np.arcsinh(1 / 0.05)

        ax.imshow(display, origin='lower', aspect='auto',
                  extent=[PV_LON_MAX, PV_LON_MIN, v_min, v_max],
                  cmap='gray_r', vmin=dlo, vmax=dhi,
                  interpolation='nearest')

    # --- Arm tracks ------------------------------------------------------
    stroke = [pe.Stroke(linewidth=3.5, foreground='black'), pe.Normal()]
    ell = ell_line
    for arm in ARMS:
        v_track = arm['v_fn'](ell)
        vis = (ell >= PV_LON_MIN) & (ell <= PV_LON_MAX) & \
              (v_track >= v_min) & (v_track <= v_max)
        if not vis.any():
            continue
        ax.plot(ell[vis], v_track[vis],
                color=arm['color'], ls=arm['ls'], lw=arm['lw'],
                path_effects=stroke, zorder=9)

        ll = arm['label_l']
        vl = arm['v_fn'](np.array([ll]))[0]
        pad_v = (v_max - v_min) * 0.025
        # If the default label position is outside the display range, fall back
        # to the midpoint of the visible track segment
        if not (v_min + pad_v < vl < v_max - pad_v and PV_LON_MIN < ll < PV_LON_MAX):
            vis_ell = ell[vis]
            if vis_ell.size > 0:
                ll = float(vis_ell[len(vis_ell) // 2])
                vl = arm['v_fn'](np.array([ll]))[0]
        if v_min + pad_v < vl < v_max - pad_v and PV_LON_MIN < ll < PV_LON_MAX:
            va = arm['label_va']
            pad = +pad_v if va == 'bottom' else -pad_v
            ax.text(ll, vl + pad, arm['name'],
                    color=arm['color'], fontsize=9, ha='center', va=va,
                    fontweight='bold',
                    path_effects=[pe.Stroke(linewidth=2.5, foreground='black'),
                                  pe.Normal()],
                    zorder=10)

    # --- Reference lines -------------------------------------------------
    ax.axhline(0, color='white', lw=0.5, ls=':', alpha=0.4)
    ax.axvline(0, color='white', lw=0.5, ls=':', alpha=0.4)

    for gl in np.arange(-3, 4):
        ax.axvline(gl, color='white', lw=0.3, alpha=0.25)
    tick_step = 20 if v_lim <= 60 else 50
    for gv in np.arange(v_min, v_max + 1, tick_step):
        ax.axhline(gv, color='white', lw=0.3, alpha=0.25)

    ax.set_xlim(PV_LON_MAX, PV_LON_MIN)
    ax.set_ylim(v_min, v_max)
    ax.set_xlabel('Galactic longitude (deg)', fontsize=12)
    ax.set_ylabel('LSR velocity (km/s)', fontsize=12)
    ax.set_title('CHIMPS 12CO(3–2) position–velocity diagram  '
                 r'($|b| < 0.25°$,  $|\ell| < 3°$)', fontsize=12)

    suffix = f'_v{v_lim}' if v_lim != 200 else ''
    outpath = os.path.join(OUTDIR, f'chimps_pv{suffix}.png')
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {outpath}')


# ---------------------------------------------------------------------------
# Standalone single-arm figures (one PNG per arm, saved to per_arm/ subdir)
# ---------------------------------------------------------------------------
def make_standalone_arm_figures():
    print('\n=== Standalone per-arm CHIMPS figures ===')
    subdir = os.path.join(OUTDIR, 'per_arm')
    os.makedirs(subdir, exist_ok=True)

    extent = [LON_MAX, LON_MIN, LAT_MIN, LAT_MAX]
    arm_slugs = {
        'Near 3 kpc arm': 'near3kpc',
        '18 km/s arm':    '18kms',
        'Local arm':      'local',
        'Norma arm':      'norma',
    }

    for arm_name, slug in arm_slugs.items():
        fits_path = CHIMPS_BGSUB[arm_name]
        arr = load_bgsub_mosaic(fits_path)

        fig, ax = plt.subplots(figsize=(14, 5.0), constrained_layout=True)
        ax.set_facecolor('black')

        if arr is not None:
            finite = arr[np.isfinite(arr)]
            if finite.size > 0:
                vlo = max(float(np.nanpercentile(finite, 5)), 0.0)
                vhi = float(np.nanpercentile(finite, 99.5))
                im = ax.imshow(arr, origin='lower', aspect='auto',
                               extent=extent, cmap='gray_r',
                               vmin=vlo, vmax=vhi, interpolation='nearest')
                cb = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.025)
                cb.set_label(r'$T_{\rm A}^*$ (K)', fontsize=14)
                cb.ax.tick_params(labelsize=12)

        for gl in np.arange(-3, 4):
            ax.axvline(gl, color='gray', lw=0.3, alpha=0.4)
        for gb in [-0.5, -0.25, 0, 0.25, 0.5]:
            ax.axhline(gb, color='gray', lw=0.3, alpha=0.4)

        ax.set_xlim(LON_MAX, LON_MIN)
        ax.set_ylim(-0.5, 0.5)
        ax.set_title(f'CHIMPS ¹²CO(3–2) — {arm_name} (background-subtracted)',
                     fontsize=17)
        ax.set_xlabel(r'$\ell$ (deg)', fontsize=15)
        ax.set_ylabel(r'$b$ (deg)', fontsize=15)
        ax.tick_params(labelsize=14)

        outpath = os.path.join(subdir, f'chimps_{slug}_arm.png')
        fig.savefig(outpath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved: {outpath}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    make_spatial_figure()
    make_pv_figure(v_lim=200)
    make_pv_figure(v_lim=60)
    make_standalone_arm_figures()
    print('\nDone.')
