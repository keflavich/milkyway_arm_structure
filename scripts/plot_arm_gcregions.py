"""
plot_arm_gcregions.py
=====================
Plots of the inner Galactic Center region (|l| < 3°, -2° < b < +1°) for
each spiral arm, using:
  - Dame CO (DHT02) velocity-slab as a grayscale background
  - CHIMPS 12CO(3-2) arm-extracted mosaic as a higher-resolution colour overlay

Two output products per arm (plus combined multi-panel figures):
  1. {arm}_gcregion.png         – Dame + CHIMPS only
  2. {arm}_gcregion_roman.png   – same + Roman GBTDS & RGPS footprints

Also produces multi-panel figures combining all arms:
  arm_gcregion_all.png
  arm_gcregion_roman_all.png

Usage:
    python plot_arm_gcregions.py [--outdir /path/to/output]
"""

import os
import glob
import argparse
import warnings
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from astropy.coordinates import SkyCoord, Galactic, FK5
import astropy.units as u
from reproject import reproject_interp

warnings.filterwarnings('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', message='.*RADECSYS.*')

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ARMS_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROMAN_DIR      = '/orange/adamginsburg/galactic_plane_surveys/roman'
ROMAN_GBTDS    = os.path.join(ROMAN_DIR, 'roman_gbtds_footprint.reg')
ROMAN_RGPS     = os.path.join(ROMAN_DIR, 'roman_rgps_footprint.reg')

# Arm definitions  --------------------------------------------------------
#   dame    : 2D Dame CO arm-extracted FITS (velocityslab = no bg sub)
#   chimps  : CHIMPS 12CO(3-2) arm-extracted mosaic
#   label   : display label
#   vel     : short velocity description for subtitle
#   color   : matplotlib colormap for CHIMPS overlay
ARMS = {
    '18kms': {
        'dame':   None,   # no pre-extracted Dame image; will use velocityslab from 3kpc as colour context
        'chimps': os.path.join(ARMS_DIR, '18kms_arm', 'CHIMPS_18kms_arm_backgroundsub_mosaic.fits'),
        'label':  '18 km/s arm',
        'vel':    r'$v \approx +18$ km/s',
        'color':  'Greens',
    },
    '3kpc': {
        'dame':   os.path.join(ARMS_DIR, '3kpc_arm', 'DameCO_DHT02_3kpc_velocityslab.fits'),
        'chimps': os.path.join(ARMS_DIR, '3kpc_arm', 'CHIMPS_arm_bgsub_mosaic.fits'),
        'label':  'Near 3 kpc arm',
        'vel':    r'$v(l\!=\!0) \approx -50$ km/s',
        'color':  'Blues',
    },
    'local': {
        'dame':   os.path.join(ARMS_DIR, 'local_arm', 'DameCO_DHT02_local_velocityslab.fits'),
        'chimps': os.path.join(ARMS_DIR, 'local_arm', 'CHIMPS_arm_bgsub_mosaic.fits'),
        'label':  'Local arm',
        'vel':    r'$v \approx 0$ km/s',
        'color':  'Oranges',
    },
    'norma': {
        'dame':   os.path.join(ARMS_DIR, 'norma_arm', 'DameCO_DHT02_norma_velocityslab.fits'),
        'chimps': os.path.join(ARMS_DIR, 'norma_arm', 'CHIMPS_arm_bgsub_mosaic.fits'),
        'label':  'Norma arm',
        'vel':    r'$v(l\!=\!0) \approx -32$ km/s',
        'color':  'Reds',
    },
}

# ---------------------------------------------------------------------------
# Target WCS: |l| < 3°, -2° < b < +1°  at 0.01 °/pix
# ---------------------------------------------------------------------------
LON_MIN, LON_MAX = -3.0, +3.0    # Galactic longitude limits
LAT_MIN, LAT_MAX = -2.0, +1.0    # Galactic latitude limits
PIX_SCALE = 0.01                  # deg/pix

def make_target_wcs():
    """Return (WCS, (ny, nx)) for the target sky region."""
    nx = int(round((LON_MAX - LON_MIN) / PIX_SCALE)) + 1   # 601
    ny = int(round((LAT_MAX - LAT_MIN) / PIX_SCALE)) + 1   # 301
    w = WCS(naxis=2)
    w.wcs.ctype  = ['GLON-CAR', 'GLAT-CAR']
    w.wcs.crval  = [0.0, (LAT_MIN + LAT_MAX) / 2.0]        # (0, -0.5)
    w.wcs.cdelt  = [-PIX_SCALE, PIX_SCALE]
    # Place CRVAL at the centre pixel
    w.wcs.crpix  = [(nx + 1) / 2.0, (ny + 1) / 2.0]
    w.wcs.set()
    return w, (ny, nx)


TARGET_WCS, TARGET_SHAPE = make_target_wcs()


def reproject_to_target(fits_path):
    """Reproject a 2-D FITS file onto the target WCS.  Returns array or None."""
    if fits_path is None or not os.path.exists(fits_path):
        return None
    hdu = fits.open(fits_path)[0]
    data = hdu.data.squeeze().astype(float)
    hdr  = hdu.header.copy()
    # Strip degenerate axes if any
    while data.ndim > 2:
        data = data[0]
    # Rebuild clean 2-D header
    w2 = WCS(hdr, naxis=2)
    hdr2 = w2.to_header()
    hdr2['NAXIS']  = 2
    hdr2['NAXIS1'] = hdu.header['NAXIS1']
    hdr2['NAXIS2'] = hdu.header['NAXIS2']
    hdu2 = fits.PrimaryHDU(data=data, header=hdr2)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out, _ = reproject_interp(hdu2, TARGET_WCS, shape_out=TARGET_SHAPE)
    return out


# ---------------------------------------------------------------------------
# Roman footprint parsing
# ---------------------------------------------------------------------------

def parse_ds9_polygons(reg_path, sky_in='fk5'):
    """
    Parse a DS9 region file and return a list of (tag, galactic_vertices) tuples.
    Each galactic_vertices is an (N, 2) array of (lon, lat) in Galactic degrees.
    """
    if not os.path.exists(reg_path):
        print(f'  [warn] Region file not found: {reg_path}')
        return []

    results = []
    # Longitude margin: some polygons near the wrap boundary
    LON_MARGIN = 1.0
    LAT_MARGIN = 0.5
    coord_system = sky_in   # default from file header
    with open(reg_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                # Check for coordinate system directive
                if line.lower() in ('fk5', 'icrs', 'galactic', 'j2000'):
                    coord_system = line.lower()
                continue
            if line.lower() in ('fk5', 'icrs', 'galactic', 'j2000'):
                coord_system = line.lower()
                continue
            m = re.match(r'polygon\(([^)]+)\)', line, re.IGNORECASE)
            if not m:
                continue
            # DS9 polygons can be "x1,y1 x2,y2..." or "x1,y1,x2,y2,..."
            raw = m.group(1)
            if ',' in raw and ' ' in raw:
                # space-separated pairs, comma within each pair
                pairs = raw.split()
                coords = []
                for p in pairs:
                    coords.extend([float(v) for v in p.split(',')])
            else:
                # all comma-separated
                coords = [float(v) for v in raw.split(',')]
            if len(coords) % 2 != 0:
                continue
            ra_arr  = np.array(coords[0::2])
            dec_arr = np.array(coords[1::2])

            # Extract tag if present
            tag_m = re.search(r'tag=\{([^}]+)\}', line)
            tag = tag_m.group(1) if tag_m else ''

            if coord_system in ('fk5', 'icrs', 'j2000'):
                sc = SkyCoord(ra=ra_arr*u.deg, dec=dec_arr*u.deg,
                              frame='icrs')
                gc = sc.galactic
                lon = gc.l.wrap_at(180*u.deg).deg
                lat = gc.b.deg
            else:
                lon = ra_arr
                lat = dec_arr

            verts = np.column_stack([lon, lat])
            # Pre-filter: skip polygons entirely outside the target region
            if (np.all(lon > LON_MAX + LON_MARGIN) or
                    np.all(lon < LON_MIN - LON_MARGIN) or
                    np.all(lat > LAT_MAX + LAT_MARGIN) or
                    np.all(lat < LAT_MIN - LAT_MARGIN)):
                continue
            results.append((tag, verts))
    return results


# Approximate WFI footprint boxes (Galactic l, b) [deg]
# GBTDS = Roman Galactic Bulge Time Domain Survey
GBTDS_BOX = dict(l_min=-2.5, l_max=+2.5, b_min=-0.5, b_max=+0.3)
# RGPS = Roman Galactic Center Time Domain extension
RGPS_BOX  = dict(l_min=-0.9, l_max=+1.6, b_min=-1.6, b_max=-0.8)

# Load region data once
print('Loading Roman footprint regions...')
_gbtds_polys = parse_ds9_polygons(ROMAN_GBTDS)
_rgps_polys  = parse_ds9_polygons(ROMAN_RGPS)
print(f'  GBTDS: {len(_gbtds_polys)} polygons')
print(f'  RGPS:  {len(_rgps_polys)} polygons')


def get_roman_patches(wcs, shape):
    """
    Convert Roman polygons to matplotlib Patch objects in pixel coordinates.
    Returns (gbtds_patches, rgps_patches).
    """
    ny, nx = shape

    def polys_to_patches(polys):
        patches = []
        for _tag, verts in polys:
            # Transform galactic (l, b) → pixel
            try:
                px, py = wcs.all_world2pix(verts[:, 0], verts[:, 1], 0)
            except Exception:
                continue
            # Skip if entirely outside the plot region
            if (np.all(px < 0) or np.all(px > nx) or
                    np.all(py < 0) or np.all(py > ny)):
                continue
            patches.append(MplPolygon(np.column_stack([px, py]), closed=True))
        return patches

    return polys_to_patches(_gbtds_polys), polys_to_patches(_rgps_polys)


# ---------------------------------------------------------------------------
# Core single-panel plot
# ---------------------------------------------------------------------------

def plot_gcregion_ax(ax, dame_data, chimps_data, arm_cfg, show_roman=False):
    """
    Fill *ax* with the Dame background + CHIMPS overlay for one arm.
    Dame and CHIMPS arrays are already reprojected to TARGET_WCS / TARGET_SHAPE.
    """
    ny, nx = TARGET_SHAPE
    lon_extent = [LON_MAX, LON_MIN]
    lat_extent = [LAT_MIN, LAT_MAX]
    extent = [LON_MAX, LON_MIN, LAT_MIN, LAT_MAX]  # left, right, bottom, top

    # --- Dame background -------------------------------------------------
    if dame_data is not None:
        finite = dame_data[np.isfinite(dame_data)]
        if finite.size > 0:
            vlo = float(np.nanpercentile(finite, 5))
            vhi = float(np.nanpercentile(finite, 99))
            ax.imshow(dame_data, origin='lower', aspect='auto',
                      extent=extent, cmap='gray_r',
                      vmin=vlo, vmax=vhi, interpolation='nearest')
    else:
        # Grey background when no Dame data
        ax.set_facecolor('#888888')

    # --- CHIMPS overlay ---------------------------------------------------
    if chimps_data is not None:
        finite = chimps_data[np.isfinite(chimps_data)]
        if finite.size > 0:
            # Use positive values only (background-subtracted can be negative)
            pos = finite[finite > 0]
            if pos.size > 3:
                vlo = 0.0
                vhi = float(np.nanpercentile(pos, 99))
            else:
                vlo, vhi = float(np.nanmin(finite)), float(np.nanmax(finite))
            # Transparent where ≤ 0 / NaN
            cmap = matplotlib.colormaps[arm_cfg['color']].copy()
            cmap.set_under(alpha=0)
            cmap.set_bad(alpha=0)
            ax.imshow(chimps_data, origin='lower', aspect='auto',
                      extent=extent, cmap=cmap,
                      vmin=max(vlo, vhi * 0.05), vmax=vhi,
                      alpha=0.85, interpolation='nearest')

    # --- Roman footprints ------------------------------------------------
    if show_roman:
        # Approximate boxes from nominal survey footprint coordinates
        def _draw_box(box, color, lw=2.0, ls='--', alpha=0.9):
            l0, l1 = box['l_max'], box['l_min']   # note: lon axis flipped
            b0, b1 = box['b_min'], box['b_max']
            xs = [l0, l1, l1, l0, l0]
            ys = [b0, b0, b1, b1, b0]
            ax.plot(xs, ys, color=color, lw=lw, ls=ls, alpha=alpha,
                    transform=ax.transData, zorder=9, solid_capstyle='butt')

        _draw_box(GBTDS_BOX, color='dodgerblue')
        _draw_box(RGPS_BOX,  color='darkorange')

        # Optionally also overlay DS9-file polygons if they loaded
        gbtds_patches, rgps_patches = get_roman_patches(TARGET_WCS, TARGET_SHAPE)
        if gbtds_patches:
            col = PatchCollection(gbtds_patches, linewidths=0.8,
                                  edgecolors='dodgerblue', facecolors='none',
                                  transform=ax.transData, zorder=8,
                                  label='Roman GBTDS')
            ax.add_collection(col)
        if rgps_patches:
            col = PatchCollection(rgps_patches, linewidths=0.8,
                                  edgecolors='darkorange', facecolors='none',
                                  transform=ax.transData, zorder=8,
                                  label='Roman RGPS')
            ax.add_collection(col)

    # --- Axes formatting -------------------------------------------------
    ax.set_xlim(LON_MAX, LON_MIN)     # Galactic longitude increases to the left
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_xlabel('Galactic longitude (deg)', fontsize=13)
    ax.set_ylabel('Galactic latitude (deg)', fontsize=13)
    ax.tick_params(labelsize=11)

    # Gridlines at integer degrees
    for gl in range(-3, 4):
        ax.axvline(x=gl, color='white', lw=0.4, alpha=0.4)
    for gb in range(-2, 2):
        ax.axhline(y=gb, color='white', lw=0.4, alpha=0.4)

    ax.set_title(f"{arm_cfg['label']}\n{arm_cfg['vel']}", fontsize=13)


# ---------------------------------------------------------------------------
# High-level: produce figures
# ---------------------------------------------------------------------------

def make_arm_figure(arm_names, show_roman=False, outpath=None, outdir='.'):
    """
    Create a multi-panel figure, one panel per arm.
    """
    # Pre-load reprojected data
    panels = []
    for name in arm_names:
        cfg = ARMS[name]
        print(f'  [{name}] reprojecting Dame...')
        dame   = reproject_to_target(cfg['dame'])
        print(f'  [{name}] reprojecting CHIMPS...')
        chimps = reproject_to_target(cfg['chimps'])
        panels.append((cfg, dame, chimps))

    n  = len(panels)
    # Aspect ratio of one panel: width / height = (LON_MAX-LON_MIN)/(LAT_MAX-LAT_MIN) = 6/3 = 2
    fig_w = 5.0 * n
    fig_h = 5.0 * (LAT_MAX - LAT_MIN) / (LON_MAX - LON_MIN)  # 2.5 in
    fig_h = max(fig_h + 1.5, 4.0)   # add room for title/axes

    fig, axes = plt.subplots(1, n, figsize=(fig_w, fig_h),
                             constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, (cfg, dame, chimps) in zip(axes, panels):
        plot_gcregion_ax(ax, dame, chimps, cfg, show_roman=show_roman)

    # Legend for Roman on last axis
    if show_roman:
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], color='dodgerblue', lw=2.0, ls='--', label='Roman GBTDS'),
            Line2D([0], [0], color='darkorange',  lw=2.0, ls='--', label='Roman RGPS'),
        ]
        axes[-1].legend(handles=handles, loc='lower right', fontsize=11,
                        framealpha=0.7)

    tag = '_roman' if show_roman else ''
    if outpath is None:
        outpath = os.path.join(outdir, f'arm_gcregion{tag}_all.png')
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {outpath}')


def make_individual_figures(outdir='.', show_roman=False):
    """Save one PNG per arm."""
    tag = '_roman' if show_roman else ''
    for name, cfg in ARMS.items():
        print(f'\n  [{name}] individual figure...')
        dame   = reproject_to_target(cfg['dame'])
        chimps = reproject_to_target(cfg['chimps'])

        fig_w = 8.0
        fig_h = fig_w * (LAT_MAX - LAT_MIN) / (LON_MAX - LON_MIN) + 1.2
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        plot_gcregion_ax(ax, dame, chimps, cfg, show_roman=show_roman)

        if show_roman:
            from matplotlib.lines import Line2D
            handles = [
                Line2D([0], [0], color='dodgerblue', lw=2.0, ls='--', label='Roman GBTDS'),
                Line2D([0], [0], color='darkorange',  lw=2.0, ls='--', label='Roman RGPS'),
            ]
            ax.legend(handles=handles, loc='lower right', fontsize=11,
                      framealpha=0.7)

        fig.tight_layout()
        outpath = os.path.join(outdir, f'{name}_gcregion{tag}.png')
        fig.savefig(outpath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'    Saved: {outpath}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', default=os.path.join(ARMS_DIR, 'gcregion_plots'),
                        help='Output directory for PNGs')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    print(f'Output directory: {args.outdir}')

    ARM_ORDER = ['18kms', '3kpc', 'local', 'norma']

    print('\n=== Individual figures (no footprints) ===')
    make_individual_figures(outdir=args.outdir, show_roman=False)

    print('\n=== Individual figures (with Roman footprints) ===')
    make_individual_figures(outdir=args.outdir, show_roman=True)

    print('\n=== Combined multi-panel (no footprints) ===')
    make_arm_figure(ARM_ORDER, show_roman=False,
                    outpath=os.path.join(args.outdir, 'arm_gcregion_all.png'))

    print('\n=== Combined multi-panel (with Roman footprints) ===')
    make_arm_figure(ARM_ORDER, show_roman=True,
                    outpath=os.path.join(args.outdir, 'arm_gcregion_roman_all.png'))

    print('\nDone.')
