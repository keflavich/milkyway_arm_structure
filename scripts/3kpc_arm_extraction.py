"""
Near 3 kpc Arm Extraction
=========================
Extract moment maps (integrated intensity) for the near 3 kpc arm from all
available CO / HI data cubes on disk.

Velocity function (near 3 kpc arm, Dame & Thaddeus 2008 geometry):
  v(l) = 4*l - 50   km/s
  Anchors:  l = +12.5 deg -> v =   0 km/s
            l = -12.5 deg -> v = -100 km/s

Extraction parameters:
  velo_halfwidth = 10 km/s  (half-width of arm window per longitude)
  bg_halfwidth   = 30 km/s  (outer edge of background window)

Datasets processed:
  1. Dame CO (DHT02)       - l ~ -12 to +13 deg  (full arm coverage)
  2. SEDIGISM 13CO(2-1)    - tiles G348-G017 (l ~ -12 to +17 deg)
  3. CHIMPS 12CO(3-2)      - four GC mosaics    (l ~ -2 to +5 deg)
  4. Nobeyama BEARS 12CO   - l ~ -1 to +1 deg
  5. Nobeyama S115Q 12CO   - l ~ -2 to +4 deg
  6. Nobeyama FOREST 13CO  - l ~ -1 to +1 deg
  7. Nobeyama S115Q 13CO   - l ~ -2 to +4 deg
  8. HI (McClure-Griffiths)- l ~ -5 to +5 deg
  9. ACES (multiple lines) - l ~ CMZ (for inner-arm check)

Outputs go to /orange/adamginsburg/cmz/arms/
"""

import os
import glob
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse

from astropy.io import fits
from astropy import units as u
from astropy.wcs import WCS
from spectral_cube import SpectralCube
from reproject.mosaicking import reproject_and_coadd, find_optimal_celestial_wcs
from reproject import reproject_interp

warnings.filterwarnings('ignore')

# =============================================================================
# Command-line arguments
# =============================================================================
_parser = argparse.ArgumentParser(description='Extract near 3 kpc arm from CO/HI cubes.')
_parser.add_argument('--velo-hw', type=float, default=10.0,
                     help='Half-width of arm velocity window [km/s] (default 10.0)')
_parser.add_argument('--bg-hw', type=float, default=10.0,
                     help='Outer half-width of background window [km/s] (default 10.0)')
args = _parser.parse_args()

# Output root -- per-arm subdirectories live directly under arms/
OUTDIR_BASE = '/orange/adamginsburg/cmz/arms'
OUTDIR = OUTDIR_BASE
os.makedirs(OUTDIR, exist_ok=True)

# ?? Velocity function & extraction parameters ?????????????????????????????????
# near 3 kpc arm: 0 km/s at l=+12.5, -100 km/s at l=-12.5  ->  slope=4, offset=-50
def v_3kpc(ell):
    """Near 3 kpc arm: v(l) = 4*l - 50 km/s  (0 km/s at l=+12.5)."""
    if hasattr(ell, 'unit'):
        ell = ell.to(u.deg).value
    return (4.0 * np.asarray(ell, dtype=float) - 50.0) * u.km / u.s

def v_local(ell):
    """Local arm: flat rotation curve at v ~ 0 km/s."""
    if hasattr(ell, 'unit'):
        ell = ell.to(u.deg).value
    return np.zeros_like(np.asarray(ell, dtype=float)) * u.km / u.s

def v_norma(ell):
    """Norma arm: v = -15 km/s at l=+3, v = -40 km/s at l=-1.5 (Oka 2012).
    Linear fit: slope = (-40-(-15))/(-1.5-3) = 50/9 km/s/deg.
    """
    if hasattr(ell, 'unit'):
        ell = ell.to(u.deg).value
    return ((50.0 / 9.0) * np.asarray(ell, dtype=float) - 95.0 / 3.0) * u.km / u.s

VELO_HW = args.velo_hw * u.km / u.s   # half-width of arm extraction window
BG_HW   = args.bg_hw   * u.km / u.s   # outer half-width of background window

# Longitude range of the near 3kpc arm (used to filter SEDIGISM tiles, ACES)
L_ARM_MIN = -13.0   # deg (allow a little margin beyond -12.5)
L_ARM_MAX =  18.0   # deg

plt.rcParams.update({'figure.facecolor': 'w', 'font.size': 10,
'dpi': 150,
                     'image.origin': 'lower', 'image.interpolation': 'none'})


# ?????????????????????????????????????????????????????????????????????????????
# Core extraction function
# ?????????????????????????????????????????????????????????????????????????????
def _imshow_extent(h):
    """Return (extent_2d, lon_range, lat_range) from a 2-D FITS header."""
    ny = h['NAXIS2']; nx = h['NAXIS1']
    crpix1 = h.get('CRPIX1', nx / 2.0); crval1 = h.get('CRVAL1', 0.0); cdelt1 = h.get('CDELT1', 1.0)
    crpix2 = h.get('CRPIX2', ny / 2.0); crval2 = h.get('CRVAL2', 0.0); cdelt2 = h.get('CDELT2', 1.0)
    lon0 = crval1 + (0      - (crpix1 - 1)) * cdelt1
    lon1 = crval1 + (nx - 1 - (crpix1 - 1)) * cdelt1
    lat0 = crval2 + (0      - (crpix2 - 1)) * cdelt2
    lat1 = crval2 + (ny - 1 - (crpix2 - 1)) * cdelt2
    return [lon0, lon1, lat0, lat1], abs(lon1 - lon0), abs(lat1 - lat0)


def _save_single_panel(data, extent, title, png, dpi=120):
    """Save a standalone single-panel image sized to match data aspect ratio."""
    lon_range = abs(extent[1] - extent[0])
    lat_range = abs(extent[3] - extent[2])
    fig_w = 14.0
    # Panel height scaled so pixels appear square (aspect='auto' + correct fig size)
    panel_h = max(fig_w * lat_range / lon_range, 0.5)
    fig_h   = panel_h + 0.6   # room for xlabel/ylabel/title
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    finite = data[np.isfinite(data)]
    vlo = float(np.nanpercentile(finite, 1))  if finite.size > 0 else 0.0
    vhi = float(np.nanpercentile(finite, 99)) if finite.size > 0 else 1.0
    ax.imshow(data, origin='lower', aspect='auto', extent=extent,
              vmin=vlo, vmax=vhi, cmap='gray_r')
    ax.set_title(title, fontsize=9)
    ax.set_xlabel('Galactic Longitude (deg)')
    ax.set_ylabel('Galactic Latitude (deg)')
    fig.subplots_adjust(left=0.06, right=0.99, top=0.93, bottom=0.12)
    fig.savefig(png, dpi=dpi)
    plt.close(fig)


def plot_arm_png(bgsub_file, velslab_file, outbase, label, arm_slug='3kpc'):
    """Generate arm PNGs from already-saved FITS files.

    Saves:
      {outbase}_{arm_slug}_arm_bgsub.png   – background-subtracted alone
      {outbase}_{arm_slug}_arm_velslab.png – velocity-slab mean alone (if available)
      {outbase}_{arm_slug}_arm.png         – both panels stacked, no gap between them

    Always called even when extraction is skipped, so PNGs are always fresh.
    velslab_file may be None.
    """
    b_data = fits.getdata(bgsub_file).astype(float)
    v_data = fits.getdata(velslab_file).astype(float) if velslab_file else None
    h = fits.getheader(bgsub_file)
    extent_2d, lon_range, lat_range = _imshow_extent(h)

    # --- Standalone PNGs -------------------------------------------------
    _save_single_panel(b_data, extent_2d,
                       f'{label} – background-subtracted',
                       f'{outbase}_{arm_slug}_arm_bgsub.png')
    if v_data is not None:
        _save_single_panel(v_data, extent_2d,
                           f'{label} – velocity slab mean',
                           f'{outbase}_{arm_slug}_arm_velslab.png')

    # --- Combined stacked PNG (no whitespace between panels) -------------
    panels = [b_data] if v_data is None else [b_data, v_data]
    titles = (['Background-subtracted']
              if v_data is None
              else ['Background-subtracted', 'Velocity slab mean'])
    n = len(panels)
    fig_w = 14.0
    # Each panel height chosen so pixels are square; add small fixed margin for
    # tick labels (0.35 in) and title (0.25 in) per panel.
    label_h = 0.35 + 0.25          # inches per panel for labels/title
    panel_h = max(fig_w * lat_range / lon_range, 0.4)
    fig_h   = n * (panel_h + label_h) + 0.1   # 0.1 fudge at bottom
    fig, axes = plt.subplots(n, 1, figsize=(fig_w, fig_h), squeeze=False)
    axes = axes[:, 0]
    for ax, data, title in zip(axes, panels, titles):
        finite = data[np.isfinite(data)]
        vlo = float(np.nanpercentile(finite, 1))  if finite.size > 0 else 0.0
        vhi = float(np.nanpercentile(finite, 99)) if finite.size > 0 else 1.0
        ax.imshow(data, origin='lower', aspect='auto', extent=extent_2d,
                  vmin=vlo, vmax=vhi, cmap='gray_r')
        ax.set_title(f'{label} – {title}', fontsize=9)
        ax.set_xlabel('Galactic Longitude (deg)')
        ax.set_ylabel('Galactic Latitude (deg)')
    # Tight no-gap layout: allocate equal height to every row, keeping each
    # image at panel_h inches with label_h for margins.
    row_frac  = panel_h / fig_h        # fraction of fig height per image row
    marg_frac = label_h / fig_h / 2.0  # top/bottom margin fraction per panel
    fig.subplots_adjust(
        left=0.06, right=0.99,
        bottom=marg_frac,
        top=1.0 - marg_frac,
        hspace=(label_h * 2) / panel_h,   # gap = label space relative to panel
    )
    png = f'{outbase}_{arm_slug}_arm.png'
    fig.savefig(png, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  Arm PNG: {png}')


def extract_arm(cube, label, outbase, vel_fn=None, use_dask=True, arm_slug='3kpc'):
    """
    Extract the arm defined by *vel_fn* from *cube* and save outputs.

    Returns (bgsub_file, velslab_file) paths, or (None, None) on failure.
    """
    if vel_fn is None:
        vel_fn = v_3kpc
    bgsub_file   = f'{outbase}_{arm_slug}_arm_backgroundsub.fits'
    velslab_file = f'{outbase}_{arm_slug}_velocityslab.fits'
    pv_file      = f'{outbase}_{arm_slug}_pv_b0.fits'

    if os.path.exists(bgsub_file) and os.path.exists(velslab_file):
        print(f'  [skip] {label}: outputs already exist')
        return bgsub_file, velslab_file

    print(f'  Shape: {cube.shape}')
    print(f'  Spectral: {cube.spectral_axis[0].to(u.km/u.s):.1f} '
          f'to {cube.spectral_axis[-1].to(u.km/u.s):.1f}')

    # Per-column longitude array (pixel centre, bottom row)
    ells = (cube.wcs.celestial
               .pixel_to_world(np.arange(cube.shape[2]), np.zeros(cube.shape[2]))
               .l.wrap_at(180 * u.deg).deg)

    # ?? Arm mask (l-dependent velocity window) ????????????????????????????????
    arm_mask = np.array([
        (cube.spectral_axis > vel_fn(l) - VELO_HW) &
        (cube.spectral_axis < vel_fn(l) + VELO_HW)
        for l in ells])   # shape (nx, nv)

    bg_mask = np.array([
        ((cube.spectral_axis > vel_fn(l) - BG_HW) &
         (cube.spectral_axis < vel_fn(l) - VELO_HW)) |
        ((cube.spectral_axis > vel_fn(l) + VELO_HW) &
         (cube.spectral_axis < vel_fn(l) + BG_HW))
        for l in ells])

    # Transpose from (nx, nv) -> (nv, 1, nx) to match cube (vel, lat, lon)
    arm_3d = arm_mask.T[:, None, :]
    bg_3d  = bg_mask .T[:, None, :]

    cube.allow_huge_operations = True
    arm_mean = cube.with_mask(arm_3d).mean(axis=0)
    bg_mean  = cube.with_mask(bg_3d ).mean(axis=0)

    bgsub = arm_mean - bg_mean

    # ?? Save background-subtracted map ????????????????????????????????????????
    hdr = bgsub.wcs.to_header()
    hdr['BUNIT']   = str(cube.unit)
    hdr['COMMENT'] = f'Near 3kpc arm: mean in v_arm(l)+/-{VELO_HW.value:.1f} km/s minus bg'
    hdr['V3KPC_A']  = (4.0,  'Arm vel slope (km/s/deg)')
    hdr['V3KPC_B']  = (-50., 'Arm vel offset at l=0 (km/s)')
    hdr['VELOHW']   = (VELO_HW.value, 'Arm half-width (km/s)')
    hdr['BGHW']     = (BG_HW.value,   'Background outer half-width (km/s)')
    fits.PrimaryHDU(data=bgsub.hdu.data, header=hdr).writeto(bgsub_file, overwrite=True)
    print(f'  Saved: {(bgsub_file)}')

    # ?? Simple velocity-slab map (arm window, no bg sub) ?????????????????????
    arm_sum = cube.with_mask(arm_3d).mean(axis=0)
    hdr2 = arm_sum.wcs.to_header()
    hdr2['BUNIT'] = str(cube.unit)
    hdr2['COMMENT'] = f'Near 3kpc arm: mean in v_arm(l)+/-{VELO_HW.value:.1f} km/s (no bg sub)'
    fits.PrimaryHDU(data=arm_sum.hdu.data, header=hdr2).writeto(velslab_file, overwrite=True)
    print(f'  Saved: {(velslab_file)}')

    # PV diagram at b ~ 0
    lataxis = cube.world[0, :, 0][1].to(u.deg).value
    b0_idx  = int(np.argmin(np.abs(lataxis)))
    pv_data = cube[:, b0_idx, :].value          # (nv, nx)
    vels    = cube.spectral_axis.to(u.km/u.s).value
    pv_hdr  = fits.Header()
    pv_hdr['NAXIS']  = 2
    pv_hdr['NAXIS1'] = pv_data.shape[1]
    pv_hdr['NAXIS2'] = pv_data.shape[0]
    pv_hdr['CTYPE1'] = 'GLON-CAR'; pv_hdr['CRPIX1'] = cube.shape[2]//2
    pv_hdr['CRVAL1'] = float(np.median(ells))
    pv_hdr['CDELT1'] = float(np.diff(ells).mean()) if len(ells)>1 else 1.0
    pv_hdr['CTYPE2'] = 'VELO-LSR'; pv_hdr['CRPIX2'] = len(vels)//2
    pv_hdr['CRVAL2'] = float(np.median(vels)); pv_hdr['CDELT2'] = float(np.diff(vels).mean())
    pv_hdr['CUNIT2'] = 'km/s'
    fits.PrimaryHDU(data=pv_data, header=pv_hdr).writeto(pv_file, overwrite=True)

    plot_arm_png(bgsub_file, velslab_file, outbase, label, arm_slug=arm_slug)
    return bgsub_file, velslab_file


def load_and_extract(path, label, outbase, vel_fn=None, use_dask=True, arm_slug='3kpc'):
    """Load cube from *path* and call extract_arm."""
    if vel_fn is None:
        vel_fn = v_3kpc
    bgsub_file   = f'{outbase}_{arm_slug}_arm_backgroundsub.fits'
    velslab_file = f'{outbase}_{arm_slug}_velocityslab.fits'
    if os.path.exists(bgsub_file) and os.path.exists(velslab_file):
        print(f'  [skip] {label}: outputs already exist')
        plot_arm_png(bgsub_file, velslab_file, outbase, label, arm_slug=arm_slug)  # always remake PNG
        return bgsub_file, velslab_file
    print(f'  Loading {os.path.basename(path)} ...')
    cube = SpectralCube.read(path, format='fits', use_dask=use_dask)
    if cube.wcs.wcs.restfrq > 0:
        cube = cube.with_spectral_unit(
            u.km / u.s, velocity_convention='radio',
            rest_value=cube.wcs.wcs.restfrq * u.Hz)
    else:
        cube = cube.with_spectral_unit(
            u.km / u.s, velocity_convention='radio')
    return extract_arm(cube, label, outbase, vel_fn=vel_fn, use_dask=use_dask, arm_slug=arm_slug)


def cleanup_tile_fits(file_list, reason=''):
    """Delete per-tile FITS files after mosaicking to save disk space."""
    removed = 0
    for f in file_list:
        if f and os.path.exists(f):
            os.remove(f)
            removed += 1
    if removed:
        msg = f' ({reason})' if reason else ''
        print(f'  Cleaned up {removed} tile FITS files{msg}')


def cleanup_tile_pngs(outbase_list, arm_slug, reason=''):
    """Delete per-tile PNG files after mosaicking (keep only mosaic PNGs)."""
    removed = 0
    for base in outbase_list:
        if not base:
            continue
        for suffix in (f'_{arm_slug}_arm_bgsub.png', f'_{arm_slug}_arm_velslab.png',
                       f'_{arm_slug}_arm.png', f'_{arm_slug}_pv.png'):
            f = base + suffix
            if os.path.exists(f):
                os.remove(f)
                removed += 1
    if removed:
        msg = f' ({reason})' if reason else ''
        print(f'  Cleaned up {removed} tile PNG files{msg}')


def mosaic_files(file_list, outfile):
    """Reproject-and-coadd a list of 2D FITS images into one mosaic."""
    valid = [f for f in file_list if f and os.path.exists(f)]
    if len(valid) == 0:
        print('  No valid files to mosaic.')
        return
    hdus = [fits.open(f)[0] for f in valid]
    wcs_out, shape_out = find_optimal_celestial_wcs(hdus)
    mosaic, _ = reproject_and_coadd(hdus, wcs_out, shape_out=shape_out,
                                    reproject_function=reproject_interp)
    hdr = wcs_out.to_header()
    fits.PrimaryHDU(data=mosaic, header=hdr).writeto(outfile, overwrite=True)
    print(f'  Mosaic saved: {(outfile)}')


def mosaic_pv_files(pv_list, outfile):
    """Stitch per-tile l-v PV FITS files into one mosaicked PV image.

    Each input has CTYPE1='GLON-CAR', CTYPE2='VELO-LSR'.  wcslib refuses to
    construct a WCS object for these because GLON has no matching celestial
    counterpart, so we read CRPIX/CRVAL/CDELT directly from the header and
    do the resampling with scipy.ndimage.map_coordinates.
    """
    from scipy.ndimage import map_coordinates

    valid = [f for f in pv_list if f and os.path.exists(f)]
    if len(valid) == 0:
        print('  No valid PV files to mosaic.')
        return None

    # --- read header metadata without invoking WCS() -------------------------
    tiles = []
    for f in valid:
        h      = fits.getheader(f)
        nv     = h['NAXIS2'];  nx     = h['NAXIS1']
        crpix1 = h['CRPIX1'];  crpix2 = h['CRPIX2']
        crval1 = h['CRVAL1'];  crval2 = h['CRVAL2']
        cdelt1 = h['CDELT1'];  cdelt2 = h['CDELT2']
        # pixel 0-indexed: world(i) = crval + (i - (crpix-1)) * cdelt
        lon0 = crval1 + (0     - (crpix1 - 1)) * cdelt1
        lon1 = crval1 + (nx-1  - (crpix1 - 1)) * cdelt1
        vel0 = crval2 + (0     - (crpix2 - 1)) * cdelt2
        vel1 = crval2 + (nv-1  - (crpix2 - 1)) * cdelt2
        tiles.append(dict(f=f, nv=nv, nx=nx,
                          crpix1=crpix1, crpix2=crpix2,
                          crval1=crval1, crval2=crval2,
                          cdelt1=cdelt1, cdelt2=cdelt2,
                          lon0=lon0, lon1=lon1,
                          vel0=vel0, vel1=vel1, h=h))

    # --- build output grid ---------------------------------------------------
    cdelt1_out = float(np.median([t['cdelt1'] for t in tiles]))
    cdelt2_out = float(np.median([t['cdelt2'] for t in tiles]))
    lon_lo = min(min(t['lon0'], t['lon1']) for t in tiles)
    lon_hi = max(max(t['lon0'], t['lon1']) for t in tiles)
    vel_lo = min(min(t['vel0'], t['vel1']) for t in tiles)
    vel_hi = max(max(t['vel0'], t['vel1']) for t in tiles)

    out_nx = max(int(round(abs(lon_hi - lon_lo) / abs(cdelt1_out))) + 1, 2)
    out_nv = max(int(round(abs(vel_hi - vel_lo) / abs(cdelt2_out))) + 1, 2)

    # output world coordinates (row = vel, col = lon)
    # When cdelt1<0 (longitude decreasing with pixel), start from lon_hi so
    # that arange * cdelt1 steps toward lon_lo (not away from it).
    out_lon_start = lon_hi if cdelt1_out < 0 else lon_lo
    out_vel_start = vel_hi if cdelt2_out < 0 else vel_lo
    out_lons = out_lon_start + np.arange(out_nx) * cdelt1_out
    out_vels = out_vel_start + np.arange(out_nv) * cdelt2_out

    # 2-D grids of lon and vel for every output pixel
    out_lon_grid, out_vel_grid = np.meshgrid(out_lons, out_vels)  # (out_nv, out_nx)

    # --- resample each tile onto output grid and nanmean ----------------------
    stack = []
    for t in tiles:
        data  = fits.getdata(t['f']).astype(float)
        # fractional input pixel coordinates corresponding to each output world coord
        i_in = (out_lon_grid - t['crval1']) / t['cdelt1'] + (t['crpix1'] - 1)
        j_in = (out_vel_grid - t['crval2']) / t['cdelt2'] + (t['crpix2'] - 1)
        # mask output pixels that fall outside this tile
        outside = ((i_in < -0.5) | (i_in > t['nx'] - 0.5) |
                   (j_in < -0.5) | (j_in > t['nv'] - 0.5))
        i_in_clipped = np.clip(i_in, 0, t['nx'] - 1)
        j_in_clipped = np.clip(j_in, 0, t['nv'] - 1)
        # map_coordinates expects (row, col) = (vel_axis, lon_axis)
        reproj = map_coordinates(data, [j_in_clipped, i_in_clipped],
                                 order=1, mode='nearest')
        reproj[outside] = np.nan
        stack.append(reproj)

    mosaic = np.nanmean(stack, axis=0).astype(np.float32)

    # --- write output with simple linear WCS header --------------------------
    out_hdr = fits.Header()
    out_hdr['NAXIS']  = 2
    out_hdr['NAXIS1'] = out_nx
    out_hdr['NAXIS2'] = out_nv
    out_hdr['CTYPE1'] = 'GLON-CAR'
    out_hdr['CRPIX1'] = 1.0
    out_hdr['CRVAL1'] = float(out_lons[0])
    out_hdr['CDELT1'] = cdelt1_out
    out_hdr['CUNIT1'] = 'deg'
    out_hdr['CTYPE2'] = 'VELO-LSR'
    out_hdr['CRPIX2'] = 1.0
    out_hdr['CRVAL2'] = float(out_vels[0])
    out_hdr['CDELT2'] = cdelt2_out
    out_hdr['CUNIT2'] = 'km/s'
    fits.PrimaryHDU(data=mosaic, header=out_hdr).writeto(outfile, overwrite=True)
    print(f'  PV mosaic saved: {(outfile)}')
    return outfile


def plot_pv_from_fits(pv_fits, label, outbase, vel_fn=None):
    """Plot a 2-D l-v FITS file (CTYPE1=GLON, CTYPE2=VELO) with the arm track.

    Reads the coordinate axes directly from CRPIX/CRVAL/CDELT so that wcslib
    does not need to parse the non-celestial GLON+VELO axis combination.
    """
    if vel_fn is None:
        vel_fn = v_3kpc
    png = f'{outbase}_3kpc_pv.png'
    hdu  = fits.open(pv_fits)[0]
    h    = hdu.header
    nv, nx = hdu.data.shape
    # pixel-to-world: world(i) = crval + (i - (crpix-1)) * cdelt
    lons = h['CRVAL1'] + (np.arange(nx) - (h['CRPIX1'] - 1)) * h['CDELT1']
    vels = h['CRVAL2'] + (np.arange(nv) - (h['CRPIX2'] - 1)) * h['CDELT2']

    extent_pv = [lons[0], lons[-1], vels[0], vels[-1]]
    fig, ax = plt.subplots(figsize=(14, 6))
    d  = hdu.data
    finite_vals = d[np.isfinite(d)]
    if finite_vals.size > 0:
        vlo = float(np.nanpercentile(finite_vals, 2))
        vhi = float(np.nanpercentile(finite_vals, 98))
    else:
        vlo, vhi = 0.0, 1.0
    ax.imshow(d, origin='lower', aspect='equal', extent=extent_pv,
              vmin=vlo, vmax=vhi, cmap='gray_r')
    ll = np.linspace(min(lons[0], lons[-1]), max(lons[0], lons[-1]), 500)
    vv = vel_fn(ll).to(u.km / u.s).value
    ax.plot(ll, vv,                    'r-',  lw=2,   label='Arm model')
    ax.plot(ll, vv - VELO_HW.value,    'r--', lw=1,   alpha=0.6)
    ax.plot(ll, vv + VELO_HW.value,    'r--', lw=1,   alpha=0.6)
    ax.set_xlabel('Galactic Longitude (deg)')
    ax.set_ylabel('Velocity (km/s)')
    ax.set_title(f'{label} - l-v diagram at b~0')
    ax.set_ylim(-250, 150)
    ax.legend()
    fig.tight_layout()
    fig.savefig(png, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  PV PNG: {(png)}')


def plot_pv_with_arm(cube_path, label, outbase, vel_fn=None, l_range=None):
    """Make a PV diagram PNG with the 3kpc arm track overlaid."""
    png = f'{outbase}_3kpc_pv.png'
    cube = SpectralCube.read(cube_path, format='fits', use_dask=True)
    if cube.wcs.wcs.restfrq > 0:
        cube = cube.with_spectral_unit(
            u.km / u.s, velocity_convention='radio',
            rest_value=cube.wcs.wcs.restfrq * u.Hz)
    else:
        cube = cube.with_spectral_unit(
            u.km / u.s, velocity_convention='radio')
    vels = cube.spectral_axis.to(u.km / u.s).value
    ells = (cube.wcs.celestial
                .pixel_to_world(np.arange(cube.shape[2]),
                                np.zeros(cube.shape[2]))
                .l.wrap_at(180 * u.deg).deg)
    lat_axis = cube.world[0, :, 0][1].to(u.deg).value
    b0 = int(np.argmin(np.abs(lat_axis)))
    cube.allow_huge_operations = True
    pv = cube[:, b0, :].value

    # Use per-pixel coordinates as extent so array displays with correct orientation
    if vel_fn is None:
        vel_fn = v_3kpc
    extent_pv = [ells[0], ells[-1], vels[0], vels[-1]]
    fig, ax = plt.subplots(figsize=(14, 6))
    finite_vals = pv[np.isfinite(pv)]
    if finite_vals.size > 0:
        vlo = float(np.nanpercentile(finite_vals, 2))
        vhi = float(np.nanpercentile(finite_vals, 98))
    else:
        vlo, vhi = 0.0, 1.0
    ax.imshow(pv, origin='lower', aspect='auto',
              extent=extent_pv,
              vmin=vlo, vmax=vhi, cmap='gray_r')
    # Arm track spanning the cube's longitude range
    ll = np.linspace(min(ells[0], ells[-1]), max(ells[0], ells[-1]), 300)
    vv = vel_fn(ll).to(u.km / u.s).value
    ax.plot(ll, vv, 'r-',  lw=2,   label='Arm model')
    ax.plot(ll, vv - VELO_HW.value, 'r--', lw=1, alpha=0.6)
    ax.plot(ll, vv + VELO_HW.value, 'r--', lw=1, alpha=0.6)
    ax.set_xlabel('Galactic Longitude (deg)')
    ax.set_ylabel('Velocity (km/s)')
    ax.set_title(f'{label} - PV at b~0')
    ax.set_ylim(-250, 150)
    ax.legend()
    fig.tight_layout()
    fig.savefig(png, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  PV PNG: {png}')


# ===========================================================================
# Dataset paths (discovered once at module load; shared across arm pipelines)
# ===========================================================================
DAME_CUBE    = '/orange/adamginsburg/cmz/dameCO/DHT02_Center_interp_spectralcube.fits'
SEDIGISM_DIR = '/orange/adamginsburg/galactic_plane_surveys/sedigism'
CHIMPS_DIR   = '/orange/adamginsburg/cmz/CHIMPS'
NOB_DIR      = '/orange/adamginsburg/cmz/nobeyama'
HI_CUBE      = '/orange/adamginsburg/cmz/hi/mcluregriffiths/GC.hi.tb.allgal.fits'
ACES_CUBES   = sorted(glob.glob(
    '/orange/adamginsburg/ACES/mosaics/cubes/*downsampled9.fits'))

all_sedigism = sorted(glob.glob(os.path.join(SEDIGISM_DIR, 'G*_13CO21_Tmb_DR1.fits')))
chimps_cubes = sorted([
    f for f in glob.glob(os.path.join(CHIMPS_DIR, '12CO_GC_*_mosaic.fits'))
    if '_arm' not in f and '_velocityslab' not in f
    and '_rewrite' not in f and '_hack' not in f and '_spectralcube' not in f
])
nobeyama_cubes = [
    ('12CO-2.BEARS.FITS',  'BEARS_12CO'),
    ('12CO-2.S115Q.FITS',  'S115Q_12CO'),
    ('13CO-2.FOREST.FITS', 'FOREST_13CO'),
    ('13CO-2.S115Q.FITS',  'S115Q_13CO'),
]


def tile_glon(fn):
    """Return Galactic longitude of a SEDIGISM tile filename like G352_*.fits."""
    g = int(os.path.basename(fn).split('_')[0][1:])
    return g - 360 if g >= 180 else g   # wrap to (-180, 180)


def _make_summary_figure(mosaic_list, arm_label, outdir):
    """Save a stacked summary PNG of background-subtracted mosaics."""
    from matplotlib.gridspec import GridSpec
    from astropy.wcs.utils import proj_plane_pixel_scales

    fig_width = 22.0
    panels = []
    for fn in mosaic_list:
        hdu    = fits.open(fn)[0]
        wcs_2d = WCS(hdu.header)
        ny, nx = hdu.data.shape
        scales    = proj_plane_pixel_scales(wcs_2d)
        lon_range = float(abs(scales[0]) * nx)
        lat_range = float(abs(scales[1]) * ny)
        title = (os.path.basename(fn)
                 .replace('_arm_bgsub_mosaic.fits', '')
                 .replace('_3kpc_arm_backgroundsub.fits', '')
                 .replace('_3kpc_arm_bgsub_mosaic.fits', ''))
        panels.append((hdu, wcs_2d, lon_range, lat_range, title))

    panel_heights = [max(fig_width * p[3] / p[2], 0.6) for p in panels]
    n = len(panels)
    fig = plt.figure(figsize=(fig_width, sum(panel_heights) + 1.0))
    gs  = GridSpec(n, 1, figure=fig, height_ratios=panel_heights, hspace=0.6)

    for i, (hdu, wcs_2d, lon_range, lat_range, title) in enumerate(panels):
        ax = fig.add_subplot(gs[i], projection=wcs_2d)
        d  = hdu.data
        finite = d[np.isfinite(d)]
        vlo = float(np.nanpercentile(finite, 2))  if finite.size > 0 else 0.0
        vhi = float(np.nanpercentile(finite, 99)) if finite.size > 0 else 1.0
        vhi = max(vhi, vlo + 1e-6)
        ax.imshow(d, origin='lower', aspect='auto', interpolation='nearest',
                  vmin=vlo, vmax=vhi, cmap='gray_r')
        lon_coord = ax.coords[0]
        lat_coord = ax.coords[1]
        lon_coord.set_axislabel('Galactic Longitude (deg)')
        lat_coord.set_axislabel('Galactic Latitude (deg)')
        lon_coord.set_major_formatter('d.d')
        lat_coord.set_major_formatter('d.dd')
        lon_coord.set_ticks(spacing=2.0 * u.deg)
        lat_coord.set_ticks(spacing=0.25 * u.deg)
        ax.coords.grid(color='white', alpha=0.25, linestyle='--', linewidth=0.5)
        ax.set_title(title, fontsize=10, pad=4)

    fig.suptitle(f'{arm_label} Arm - Background-Subtracted Maps', fontsize=14)
    outpng = os.path.join(outdir, f'{arm_label}_summary.png')
    fig.savefig(outpng, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'\n  Summary figure: {outpng}')


def run_arm_pipeline(arm_label, vel_fn, l_min, l_max):
    """Run the full extraction + mosaicking + plotting pipeline for one arm.

    arm_label : short name used for subdirectory and file prefixes
    vel_fn    : function l [deg] -> velocity Quantity
    l_min/max : Galactic longitude range for tile selection
    """
    arm_outdir = os.path.join(OUTDIR, f'{arm_label}_arm')
    os.makedirs(arm_outdir, exist_ok=True)

    print(f'\n{"=" * 72}')
    print(f'Arm pipeline: {arm_label}')
    print(f'  vel at l=0: {vel_fn(0.0):.1f}  |  l=+10: {vel_fn(10.0):.1f}'
          f'  |  l=-10: {vel_fn(-10.0):.1f}')
    print(f'  Window: +/-{VELO_HW}  Background: +/-{BG_HW}')
    print(f'  Output: {arm_outdir}')
    print('=' * 72)

    arm_bgsub_files = {}

    # ------------------------------------------------------------------
    # 1. Dame CO
    # ------------------------------------------------------------------
    print('\n-- 1. Dame CO (DHT02) --')
    ob = os.path.join(arm_outdir, 'DameCO_DHT02')
    b, _ = load_and_extract(DAME_CUBE, f'DameCO_DHT02 [{arm_label}]', ob,
                            vel_fn=vel_fn, use_dask=False, arm_slug=arm_label)
    arm_bgsub_files['DameCO'] = [b] if b else []
    if os.path.exists(DAME_CUBE):
        plot_pv_with_arm(DAME_CUBE, f'DameCO DHT02 [{arm_label}]', ob, vel_fn=vel_fn)

    # ------------------------------------------------------------------
    # 2. SEDIGISM 13CO(2-1)
    # ------------------------------------------------------------------
    print('\n-- 2. SEDIGISM 13CO(2-1) --')
    sedigism_files = [f for f in all_sedigism
                      if l_min - 1 <= tile_glon(f) <= l_max + 1]
    print(f'  {len(sedigism_files)} SEDIGISM tiles in l=[{l_min}, {l_max}]')
    sedigism_bs, sedigism_vs, sedigism_pvs = [], [], []
    for fn in sedigism_files:
        gname = os.path.basename(fn).replace('_13CO21_Tmb_DR1.fits', '')
        ob    = os.path.join(arm_outdir, f'{gname}_SEDIGISM')
        b, v2 = load_and_extract(fn, f'SEDIGISM {gname} [{arm_label}]', ob,
                                 vel_fn=vel_fn, arm_slug=arm_label)
        if b:  sedigism_bs.append(b)
        if v2: sedigism_vs.append(v2)
        sedigism_pvs.append(f'{ob}_{arm_label}_pv_b0.fits')
    arm_bgsub_files['SEDIGISM'] = sedigism_bs

    seg_bgsub_mosaic   = os.path.join(arm_outdir, 'SEDIGISM_arm_bgsub_mosaic.fits')
    seg_velslab_mosaic = os.path.join(arm_outdir, 'SEDIGISM_velslab_mosaic.fits')
    seg_pv_mosaic      = os.path.join(arm_outdir, 'SEDIGISM_pv_mosaic.fits')
    print(f'  Mosaicing {len(sedigism_bs)} SEDIGISM arm maps ...')
    mosaic_files(sedigism_bs, seg_bgsub_mosaic)
    mosaic_files(sedigism_vs, seg_velslab_mosaic)
    if os.path.exists(seg_bgsub_mosaic):
        plot_arm_png(seg_bgsub_mosaic, seg_velslab_mosaic if os.path.exists(seg_velslab_mosaic) else None,
                    os.path.join(arm_outdir, 'SEDIGISM_mosaic'), f'SEDIGISM mosaic [{arm_label}]',
                    arm_slug=arm_label)
    mosaic_pv_files(sedigism_pvs, seg_pv_mosaic)
    if os.path.exists(seg_pv_mosaic):
        plot_pv_from_fits(seg_pv_mosaic, f'SEDIGISM [{arm_label}]',
                          os.path.join(arm_outdir, 'SEDIGISM'), vel_fn=vel_fn)
    cleanup_tile_fits(sedigism_bs + sedigism_vs
                      + [f for f in sedigism_pvs if os.path.exists(f)],
                      'SEDIGISM tiles')
    sedigism_outbases = [os.path.join(arm_outdir, f'{os.path.basename(fn).replace("_13CO21_Tmb_DR1.fits", "")}_SEDIGISM')
                         for fn in sedigism_files]
    cleanup_tile_pngs(sedigism_outbases, arm_label, 'SEDIGISM tiles')

    # ------------------------------------------------------------------
    # 3. CHIMPS 12CO(3-2)
    # ------------------------------------------------------------------
    print(f'\n-- 3. CHIMPS 12CO(3-2)  ({len(chimps_cubes)} cubes) --')
    chimps_bs, chimps_vs, chimps_pvs = [], [], []
    for fn in chimps_cubes:
        name  = os.path.basename(fn).replace('.fits', '')
        ob    = os.path.join(arm_outdir, name)
        b, v2 = load_and_extract(fn, f'CHIMPS {name} [{arm_label}]', ob,
                                 vel_fn=vel_fn, arm_slug=arm_label)
        if b:  chimps_bs.append(b)
        if v2: chimps_vs.append(v2)
        chimps_pvs.append(f'{ob}_{arm_label}_pv_b0.fits')
    arm_bgsub_files['CHIMPS'] = chimps_bs

    if chimps_bs:
        chimps_bgsub_mosaic   = os.path.join(arm_outdir, 'CHIMPS_arm_bgsub_mosaic.fits')
        chimps_velslab_mosaic = os.path.join(arm_outdir, 'CHIMPS_velslab_mosaic.fits')
        chimps_pv_mosaic      = os.path.join(arm_outdir, 'CHIMPS_pv_mosaic.fits')
        mosaic_files(chimps_bs, chimps_bgsub_mosaic)
        mosaic_files(chimps_vs, chimps_velslab_mosaic)
        valid_pvs = [f for f in chimps_pvs if os.path.exists(f)]
        if valid_pvs:
            mosaic_pv_files(valid_pvs, chimps_pv_mosaic)
    else:
        chimps_bgsub_mosaic   = os.path.join(arm_outdir, 'CHIMPS_arm_bgsub_mosaic.fits')
        chimps_velslab_mosaic = os.path.join(arm_outdir, 'CHIMPS_velslab_mosaic.fits')
        chimps_pv_mosaic      = os.path.join(arm_outdir, 'CHIMPS_pv_mosaic.fits')
    if os.path.exists(chimps_bgsub_mosaic):
        plot_arm_png(chimps_bgsub_mosaic,
                     chimps_velslab_mosaic if os.path.exists(chimps_velslab_mosaic) else None,
                     os.path.join(arm_outdir, 'CHIMPS_mosaic'), f'CHIMPS mosaic [{arm_label}]',
                     arm_slug=arm_label)
    if os.path.exists(chimps_pv_mosaic):
        plot_pv_from_fits(chimps_pv_mosaic, f'CHIMPS mosaic [{arm_label}]',
                          os.path.join(arm_outdir, 'CHIMPS_mosaic'), vel_fn=vel_fn)
    for fn, pv_file in zip(chimps_cubes, chimps_pvs):
        name = os.path.basename(fn).replace('.fits', '')
        ob   = os.path.join(arm_outdir, name)
        if os.path.exists(pv_file):
            plot_pv_from_fits(pv_file, f'CHIMPS {name} [{arm_label}]', ob,
                              vel_fn=vel_fn)
        else:
            plot_pv_with_arm(fn, f'CHIMPS {name} [{arm_label}]', ob, vel_fn=vel_fn)
    cleanup_tile_fits(chimps_bs + chimps_vs
                      + [f for f in chimps_pvs if os.path.exists(f)],
                      'CHIMPS tiles')
    chimps_outbases = [os.path.join(arm_outdir, os.path.basename(fn).replace('.fits', ''))
                       for fn in chimps_cubes]
    cleanup_tile_pngs(chimps_outbases, arm_label, 'CHIMPS tiles')

    # ------------------------------------------------------------------
    # 4-7. Nobeyama
    # ------------------------------------------------------------------
    print('\n-- 4-7. Nobeyama --')
    nob_bs = []
    for fname, lbl in nobeyama_cubes:
        path = os.path.join(NOB_DIR, fname)
        if not os.path.exists(path):
            print(f'  Skipping {lbl}: file not found')
            continue
        ob    = os.path.join(arm_outdir, f'Nobeyama_{lbl}')
        b, v2 = load_and_extract(path, f'Nobeyama {lbl} [{arm_label}]', ob,
                                 vel_fn=vel_fn, use_dask=False, arm_slug=arm_label)
        if b: nob_bs.append(b)
        plot_pv_with_arm(path, f'Nobeyama {lbl} [{arm_label}]', ob, vel_fn=vel_fn)
    arm_bgsub_files['Nobeyama'] = nob_bs

    # ------------------------------------------------------------------
    # 8. HI
    # ------------------------------------------------------------------
    print('\n-- 8. HI (McClure-Griffiths) --')
    ob    = os.path.join(arm_outdir, 'HI_MCG')
    b, v2 = load_and_extract(HI_CUBE, f'HI_MCG [{arm_label}]', ob,
                             vel_fn=vel_fn, use_dask=False, arm_slug=arm_label)
    arm_bgsub_files['HI'] = [b] if b else []
    if os.path.exists(HI_CUBE):
        plot_pv_with_arm(HI_CUBE, f'HI MCG [{arm_label}]', ob, vel_fn=vel_fn)

    # ------------------------------------------------------------------
    # 9. ACES
    # ------------------------------------------------------------------
    print(f'\n-- 9. ACES  ({len(ACES_CUBES)} cubes) --')
    aces_bs, aces_vs = [], []
    for fn in ACES_CUBES:
        name  = (os.path.basename(fn)
                 .replace('_downsampled9.fits', '')
                 .replace('_CubeMosaic', ''))
        ob    = os.path.join(arm_outdir, f'ACES_{name}')
        b, v2 = load_and_extract(fn, f'ACES {name} [{arm_label}]', ob,
                                 vel_fn=vel_fn, use_dask=True, arm_slug=arm_label)
        if b:  aces_bs.append(b)
        if v2: aces_vs.append(v2)
    arm_bgsub_files['ACES'] = aces_bs

    if aces_bs:
        aces_mosaic = os.path.join(arm_outdir, 'ACES_arm_bgsub_mosaic.fits')
        mosaic_files(aces_bs, aces_mosaic)
        cleanup_tile_fits(aces_bs + aces_vs, 'ACES tiles')
    else:
        aces_mosaic = os.path.join(arm_outdir, 'ACES_arm_bgsub_mosaic.fits')
    if os.path.exists(aces_mosaic):
        plot_arm_png(aces_mosaic, None,
                     os.path.join(arm_outdir, 'ACES_mosaic'), f'ACES mosaic [{arm_label}]',
                     arm_slug=arm_label)

    # ------------------------------------------------------------------
    # Per-arm summary
    # ------------------------------------------------------------------
    print(f'\n{"=" * 72}')
    print(f'{arm_label} Arm Extraction - Summary')
    print('=' * 72)
    total = 0
    for survey, flist in arm_bgsub_files.items():
        good = [f for f in flist if f and os.path.exists(f)]
        print(f'  {survey:12s}  {len(good):3d} maps')
        total += len(good)
    print(f'  {"TOTAL":12s}  {total:3d} maps')

    arm_mosaics = sorted(glob.glob(os.path.join(arm_outdir, '*_arm_bgsub_mosaic.fits')))
    arm_singles = [os.path.join(arm_outdir, f'{s}_{arm_label}_arm_backgroundsub.fits')
                   for s in ['DameCO_DHT02', 'HI_MCG']
                   if os.path.exists(os.path.join(arm_outdir,
                                                  f'{s}_{arm_label}_arm_backgroundsub.fits'))]
    all_for_summary = arm_mosaics + arm_singles
    if all_for_summary:
        _make_summary_figure(all_for_summary, arm_label, arm_outdir)

    return arm_bgsub_files


# ===========================================================================
# Main: run pipeline for all arms
# ===========================================================================
print('=' * 72)
print('Arm Extraction Pipeline')
print(f'Output base: {OUTDIR}')
print(f'Extraction half-width: +/-{VELO_HW}  Background: +/-{BG_HW}')
print('=' * 72)

run_arm_pipeline('3kpc',  v_3kpc,  L_ARM_MIN, L_ARM_MAX)
run_arm_pipeline('local', v_local, L_ARM_MIN, L_ARM_MAX)
run_arm_pipeline('norma', v_norma, -5.0, 7.0)

print('\nAll arms done.')
