"""
sedigism_gc_plots.py
====================
Two figures from SEDIGISM 13CO(2-1) data:

1. sedigism_gcregion_arms.png
   Four-panel grayscale spatial image of the inner GC: |l| < 3°, |b| < 0.5°
   One panel per arm, using pre-computed background-subtracted mosaics.

2. sedigism_pv.png
   Position-velocity (l–v) diagram along b ~ 0 (coadded from per-arm PV mosaics)
   with labelled arm tracks:
     • Near 3 kpc arm   v(l) = 4l − 50  km/s
     • 18 km/s arm      v(l) = 4l + 18  km/s
     • Local arm        v(l) = 0         km/s
     • Norma arm        v(l) = (50/9)l − 95/3  km/s

Usage:
    python sedigism_gc_plots.py
"""

import os
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from reproject import reproject_interp

warnings.filterwarnings('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', message='.*RADECSYS.*')
warnings.filterwarnings('ignore', message='.*obsfix.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*empty slice.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*All-NaN.*')

# ---------------------------------------------------------------------------
# Approximate WFI footprint boxes (Galactic l, b) [deg]
# ---------------------------------------------------------------------------
GBTDS_BOX = dict(l_min=-2.5, l_max=+2.5, b_min=-0.5, b_max=+0.3)
RGPS_BOX  = dict(l_min=-0.9, l_max=+1.6, b_min=-1.6, b_max=-0.8)


def _draw_roman_boxes(ax):
    """Overlay approximate GBTDS (blue) and RGPS (orange) footprint boxes."""
    for box, color in [(GBTDS_BOX, 'dodgerblue'), (RGPS_BOX, 'darkorange')]:
        xs = [box['l_max'], box['l_min'], box['l_min'], box['l_max'], box['l_max']]
        ys = [box['b_min'], box['b_min'], box['b_max'], box['b_max'], box['b_min']]
        ax.plot(xs, ys, color=color, lw=2.0, ls='--', alpha=0.9, zorder=9,
                solid_capstyle='butt')

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ARM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SEDIGISM_BGSUB = {
    'Near 3 kpc arm': os.path.join(ARM_DIR, '3kpc_arm',   'SEDIGISM_arm_bgsub_mosaic.fits'),
    '18 km/s arm':    os.path.join(ARM_DIR, '18kms_arm',  'SEDIGISM_18kms_arm_backgroundsub_mosaic.fits'),
    'Local arm':      os.path.join(ARM_DIR, 'local_arm',  'SEDIGISM_arm_bgsub_mosaic.fits'),
    'Norma arm':      os.path.join(ARM_DIR, 'norma_arm',  'SEDIGISM_arm_bgsub_mosaic.fits'),
}

# Per-arm PV mosaics (GLON-CAR × VELO-LSR, full velocity range)
SEDIGISM_PV_FILES = [
    os.path.join(ARM_DIR, '3kpc_arm',  'SEDIGISM_pv_mosaic.fits'),
    os.path.join(ARM_DIR, 'local_arm', 'SEDIGISM_pv_mosaic.fits'),
    os.path.join(ARM_DIR, 'norma_arm', 'SEDIGISM_pv_mosaic.fits'),
]

OUTDIR = os.path.join(ARM_DIR, 'gcregion_plots')
os.makedirs(OUTDIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Arm velocity functions  (all return km/s, ell in deg)
# ---------------------------------------------------------------------------
ell_line = np.linspace(-3.2, 3.2, 200)

ARMS = [
    {
        'name':     'Near 3 kpc arm',
        'v_fn':     lambda l: 4.0 * l - 50.0,
        'color':    'steelblue',
        'ls':       '-',
        'lw':       2.0,
        'label_l':  -2.6,
        'label_va': 'bottom',
    },
    {
        'name':     '18 km/s arm',
        'v_fn':     lambda l: 4.0 * l + 18.0,
        'color':    'limegreen',
        'ls':       '-',
        'lw':       2.0,
        'label_l':  1.8,
        'label_va': 'bottom',
    },
    {
        'name':     'Local arm',
        'v_fn':     lambda l: np.zeros_like(l),
        'color':    'orange',
        'ls':       '--',
        'lw':       1.8,
        'label_l':  -2.6,
        'label_va': 'top',
    },
    {
        'name':     'Norma arm',
        'v_fn':     lambda l: (50.0 / 9.0) * l - 95.0 / 3.0,
        'color':    'tomato',
        'ls':       '-.',
        'lw':       2.0,
        'label_l':  2.2,
        'label_va': 'top',
    },
]

# ---------------------------------------------------------------------------
# Target WCS for the spatial image
# ---------------------------------------------------------------------------
LON_MIN, LON_MAX = -3.05, 3.05   # °
LAT_MIN, LAT_MAX = -0.55, 0.55   # °
PIX_SCALE = 0.003                 # °/pix

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
PV_VEL_STEP = 0.5                        # km/s/pix  (finer than CHIMPS; SEDIGISM is 0.25 km/s)

PV_NL = int(round((PV_LON_MAX - PV_LON_MIN) / PV_LON_STEP)) + 1
PV_NV = int(round((PV_VEL_MAX - PV_VEL_MIN) / PV_VEL_STEP)) + 1

pv_lons = np.linspace(PV_LON_MAX, PV_LON_MIN, PV_NL)
pv_vels = np.linspace(PV_VEL_MIN, PV_VEL_MAX, PV_NV)

# ---------------------------------------------------------------------------
# Helper: load a pre-computed 2-D bgsub mosaic and reproject to spatial grid
# ---------------------------------------------------------------------------
def load_bgsub_mosaic(fits_path):
    """Load a 2-D background-subtracted mosaic and reproject to SPATIAL_WCS."""
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
# Helper: load a SEDIGISM PV mosaic (GLON-CAR × VELO-LSR) and resample
#         onto the common PV grid (PV_NV, PV_NL)
# ---------------------------------------------------------------------------
def load_sedigism_pv(fits_path):
    """Resample a SEDIGISM 2-D PV mosaic onto the common l–v output grid."""
    print(f'  PV from {os.path.basename(fits_path)} ...')
    hdul = fits.open(fits_path)
    data = hdul[0].data.astype(float)  # shape (NAXIS2=v, NAXIS1=l)
    hdr  = hdul[0].header.copy()
    hdul.close()

    while data.ndim > 2:
        data = data[0]

    nv, nl = data.shape

    # Build l and v coordinate arrays from header keywords
    crpix1 = hdr.get('CRPIX1', 1.0)
    crval1 = hdr.get('CRVAL1', 0.0)
    cdelt1 = hdr.get('CDELT1', -0.0026)
    lons = crval1 + (np.arange(nl) - (crpix1 - 1)) * cdelt1  # degrees, possibly >180

    crpix2 = hdr.get('CRPIX2', 1.0)
    crval2 = hdr.get('CRVAL2', 0.0)
    cdelt2 = hdr.get('CDELT2', 0.25)
    vels = crval2 + (np.arange(nv) - (crpix2 - 1)) * cdelt2  # km/s

    # Wrap longitudes to (-180, 180)
    lons = np.where(lons > 180, lons - 360, lons)

    # Clip to target range
    lon_mask = (lons >= PV_LON_MIN) & (lons <= PV_LON_MAX)
    vel_mask = (vels >= PV_VEL_MIN) & (vels <= PV_VEL_MAX)

    if not (lon_mask.any() and vel_mask.any()):
        print(f'    No data in target range for {os.path.basename(fits_path)}')
        return None

    lons_crop = lons[lon_mask]
    vels_crop = vels[vel_mask]
    data_crop = data[np.ix_(vel_mask, lon_mask)]

    # Map native l,v onto output pixel indices
    il = np.round((PV_LON_MAX - lons_crop) / PV_LON_STEP).astype(int)
    iv = np.round((vels_crop   - PV_VEL_MIN) / PV_VEL_STEP).astype(int)

    valid_l = (il >= 0) & (il < PV_NL)
    valid_v = (iv >= 0) & (iv < PV_NV)

    # Broadcast to full 2-D grid
    iv_grid, il_grid = np.meshgrid(iv, il, indexing='ij')    # (nv_crop, nl_crop)
    vv_valid, ll_valid = np.meshgrid(valid_v, valid_l, indexing='ij')
    finite_grid = np.isfinite(data_crop) & vv_valid & ll_valid

    iv_flat  = iv_grid[finite_grid]
    il_flat  = il_grid[finite_grid]
    vals     = data_crop[finite_grid]

    flat_idx = iv_flat * PV_NL + il_flat
    accum    = np.bincount(flat_idx, weights=vals, minlength=PV_NV * PV_NL)
    count    = np.bincount(flat_idx, minlength=PV_NV * PV_NL).astype(float)

    mask     = count > 0
    out_flat = np.full(PV_NV * PV_NL, np.nan)
    out_flat[mask] = accum[mask] / count[mask]
    return out_flat.reshape(PV_NV, PV_NL)


# ---------------------------------------------------------------------------
# Figure 1: Four-panel per-arm spatial image (grayscale, bgsub mosaics)
# ---------------------------------------------------------------------------
def make_spatial_figure():
    print('\n=== SEDIGISM spatial image (per-arm, bgsub) ===')

    arm_order = ['Near 3 kpc arm', '18 km/s arm', 'Local arm', 'Norma arm']
    extent = [LON_MAX, LON_MIN, LAT_MIN, LAT_MAX]

    fig, axes = plt.subplots(2, 2, figsize=(14, 7),
                             constrained_layout=True)
    axes = axes.flatten()

    for idx, arm_name in enumerate(arm_order):
        ax = axes[idx]
        fits_path = SEDIGISM_BGSUB[arm_name]
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
                cb.set_label(r'$T_{\rm mb}$ (K)', fontsize=12)
                cb.ax.tick_params(labelsize=12)

        # Light grid
        for gl in np.arange(-3, 4):
            ax.axvline(gl, color='gray', lw=0.3, alpha=0.4)
        for gb in [-0.5, -0.25, 0, 0.25, 0.5]:
            ax.axhline(gb, color='gray', lw=0.3, alpha=0.4)

        ax.set_xlim(LON_MAX, LON_MIN)
        ax.set_ylim(-0.5, 0.5)
        ax.set_title(arm_name, fontsize=16)
        ax.set_xlabel(r'$\ell$ (deg)', fontsize=14)
        ax.set_ylabel(r'$b$ (deg)', fontsize=14)
        ax.tick_params(labelsize=14)

    fig.suptitle(r'SEDIGISM $^{13}$CO(2\u20131) \u2014 background-subtracted arm mosaics'
                 r'  ($|\ell|<3\degree$, $|b|<0.5\degree$)',
                 fontsize=13)

    # Save plain version (no Roman boxes)
    outpath_plain = os.path.join(OUTDIR, 'sedigism_gcregion_arms.png')
    fig.savefig(outpath_plain, dpi=150, bbox_inches='tight')
    print(f'  Saved: {outpath_plain}')

    # Now add Roman footprint boxes and save the roman version
    from matplotlib.lines import Line2D
    for ax in axes:
        _draw_roman_boxes(ax)
    legend_handles = [
        Line2D([0], [0], color='dodgerblue', lw=2.0, ls='--', label='Roman GBTDS'),
        Line2D([0], [0], color='darkorange',  lw=2.0, ls='--', label='Roman RGPS'),
    ]
    axes[-1].legend(handles=legend_handles, loc='upper right', fontsize=11,
                    framealpha=0.75)
    outpath_roman = os.path.join(OUTDIR, 'sedigism_gcregion_arms_roman.png')
    fig.savefig(outpath_roman, dpi=150, bbox_inches='tight')
    print(f'  Saved: {outpath_roman}')
    plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 2: PV diagram  (v_lim controls the velocity axis half-range in km/s)
# ---------------------------------------------------------------------------
def make_pv_figure(v_lim=200):
    print(f'\n=== SEDIGISM PV diagram (v_lim=±{v_lim} km/s) ===')
    v_min, v_max = -v_lim, v_lim

    pvs = []
    for f in SEDIGISM_PV_FILES:
        arr = load_sedigism_pv(f)
        if arr is not None:
            pvs.append(arr)

    if not pvs:
        print('  No PV data found!'); return

    # Coadd; crop to requested velocity range
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
        vhi = float(np.nanpercentile(pos, 99)) if pos.size > 3 else 1.0

        display = np.arcsinh(pv_mosaic / max(vhi * 0.05, 1e-9))
        dlo = np.arcsinh(0)
        dhi = np.arcsinh(1 / 0.05)

        ax.imshow(display, origin='lower', aspect='auto',
                  extent=[PV_LON_MAX, PV_LON_MIN, v_min, v_max],
                  cmap='gray_r', vmin=dlo, vmax=dhi,
                  interpolation='nearest')

    # --- Arm tracks ---
    stroke = [pe.Stroke(linewidth=3.5, foreground='black'), pe.Normal()]
    ell = ell_line
    for arm in ARMS:
        v_track = arm['v_fn'](ell)
        vis = ((ell >= PV_LON_MIN) & (ell <= PV_LON_MAX) &
               (v_track >= v_min) & (v_track <= v_max))
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
            va  = arm['label_va']
            pad = +pad_v if va == 'bottom' else -pad_v
            ax.text(ll, vl + pad, arm['name'],
                    color=arm['color'], fontsize=14, ha='center', va=va,
                    fontweight='bold',
                    path_effects=[pe.Stroke(linewidth=2.5, foreground='black'),
                                  pe.Normal()],
                    zorder=10)

    # --- Reference lines ---
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
    ax.set_title(r'SEDIGISM $^{13}$CO(2–1) position–velocity diagram'
                 r'  ($|\ell|<3°$)', fontsize=12)

    suffix = f'_v{v_lim}' if v_lim != 200 else ''
    outpath = os.path.join(OUTDIR, f'sedigism_pv{suffix}.png')
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {outpath}')


# ---------------------------------------------------------------------------
# Standalone single-arm figures (one PNG per arm, saved to per_arm/ subdir)
# ---------------------------------------------------------------------------
def make_standalone_arm_figures():
    print('\n=== Standalone per-arm SEDIGISM figures ===')
    subdir = os.path.join(OUTDIR, 'per_arm')
    os.makedirs(subdir, exist_ok=True)

    # Wider latitude coverage for standalone SEDIGISM plots
    SA_LAT_LIM = 1.05   # °  → ylim will be ±1°
    nx = int(round((LON_MAX - LON_MIN) / PIX_SCALE)) + 1
    ny = int(round(2 * SA_LAT_LIM / PIX_SCALE)) + 1
    wide_wcs = WCS(naxis=2)
    wide_wcs.wcs.ctype = ['GLON-CAR', 'GLAT-CAR']
    wide_wcs.wcs.crval = [0.0, 0.0]
    wide_wcs.wcs.cdelt = [-PIX_SCALE, PIX_SCALE]
    wide_wcs.wcs.crpix = [(nx + 1) / 2.0, (ny + 1) / 2.0]
    wide_wcs.wcs.set()
    wide_shape = (ny, nx)

    extent = [LON_MAX, LON_MIN, -SA_LAT_LIM, SA_LAT_LIM]

    arm_slugs = {
        'Near 3 kpc arm': 'near3kpc',
        '18 km/s arm':    '18kms',
        'Local arm':      'local',
        'Norma arm':      'norma',
    }

    for arm_name, slug in arm_slugs.items():
        fits_path = SEDIGISM_BGSUB[arm_name]
        print(f'  Loading {os.path.basename(fits_path)} ...')
        hdul = fits.open(fits_path)
        data = hdul[0].data.astype(float)
        hdr  = hdul[0].header.copy()
        hdul.close()
        while data.ndim > 2:
            data = data[0]
        arr, _ = reproject_interp(
            fits.PrimaryHDU(data=data, header=hdr),
            wide_wcs, shape_out=wide_shape)

        fig, ax = plt.subplots(figsize=(14, 6.5), constrained_layout=True)
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
                cb.set_label(r'$T_{\rm mb}$ (K)', fontsize=14)
                cb.ax.tick_params(labelsize=14)

        for gl in np.arange(-3, 4):
            ax.axvline(gl, color='gray', lw=0.3, alpha=0.4)
        for gb in [-1.0, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0]:
            ax.axhline(gb, color='gray', lw=0.3, alpha=0.4)

        ax.set_xlim(LON_MAX, LON_MIN)
        ax.set_ylim(-1.0, 1.0)
        ax.set_title(f'SEDIGISM \u00b9\u00b3CO(2\u20131) \u2014 {arm_name} (background-subtracted)',
                     fontsize=17)
        ax.set_xlabel(r'$\ell$ (deg)', fontsize=15)
        ax.set_ylabel(r'$b$ (deg)', fontsize=15)
        ax.tick_params(labelsize=14)

        outpath = os.path.join(subdir, f'sedigism_{slug}_arm.png')
        fig.savefig(outpath, dpi=150, bbox_inches='tight')
        print(f'  Saved: {outpath}')

        # Save roman variant with footprint boxes
        _draw_roman_boxes(ax)
        from matplotlib.lines import Line2D
        legend_handles = [
            Line2D([0], [0], color='dodgerblue', lw=2.0, ls='--', label='Roman GBTDS'),
            Line2D([0], [0], color='darkorange',  lw=2.0, ls='--', label='Roman RGPS'),
        ]
        ax.legend(handles=legend_handles, loc='upper right', fontsize=13,
                  framealpha=0.75)
        outpath_roman = os.path.join(subdir, f'sedigism_{slug}_arm_roman.png')
        fig.savefig(outpath_roman, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved: {outpath_roman}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    make_spatial_figure()
    make_pv_figure(v_lim=200)
    make_pv_figure(v_lim=60)
    make_standalone_arm_figures()
    print('\nDone.')

