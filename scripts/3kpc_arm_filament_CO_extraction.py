"""
3kpc Arm Filament CO Extraction
================================
Extract moment maps and position-velocity diagrams for the near-3kpc arm
filament in a 30' × 30' region centred on (l, b) = (0.34°, +0.024°) in
Galactic coordinates.

Velocity window (signal):   -59 to -50 km/s  (centre −55 km/s)
Background windows:         -75 to -61 km/s  and  -48 to -35 km/s

Two background-subtraction methods are applied to every dataset:

  1. SIMPLE MEAN
     The mean brightness of all background channels on each side of the
     signal window is averaged together into one per-pixel constant.  That
     constant is subtracted from every signal channel before integrating.

  2. LINEAR BASELINE
     A linear (order-1) polynomial is fit, per spatial pixel, through the
     brightness temperatures at all background channels on both sides of the
     signal window simultaneously (least-squares).  The fitted line is
     evaluated at each signal-window channel and subtracted before
     integrating.

Datasets processed
------------------
  • CHIMPS    12CO(3-2)    – 15" beam, ~0.5 km/s   (GC 359-000 mosaic)
  • SEDIGISM  13CO(2-1)    – 28" beam, 0.5 km/s    (G000 tile)
  • Dame CO   12CO(1-0)    – ~0.5° beam, 1.3 km/s  (DHT02 interpolated)
  • Nobeyama  12CO(2-1)    – 73" beam, 0.08 km/s   (BEARS)
  • Nobeyama  13CO(2-1)    – 73" beam, 0.08 km/s   (FOREST)

Outputs  →  /orange/adamginsburg/cmz/arms/3kpc_arm_filament/
"""

import os
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord

from spectral_cube import SpectralCube

warnings.filterwarnings('ignore')

# =============================================================================
# Configuration – edit here
# =============================================================================

# Target region
L_CENTER  =  0.34   # deg  Galactic longitude
B_CENTER  =  0.024  # deg  Galactic latitude
HALF_SIZE =  0.25   # deg  half-width of square (= 15', giving a 30' side)

# Signal velocity window  [km/s]
V_SIG_LOW  = -59.0
V_SIG_HIGH = -50.0

# Background windows on each side of the signal  [km/s]
# Low-velocity side:
BG_LO_LOW  = -75.0
BG_LO_HIGH = -61.0
# High-velocity side:
BG_HI_LOW  = -48.0
BG_HI_HIGH = -35.0

# Label for output filenames / plot titles
LABEL = '3kpc_arm_filament'

# Output directory
OUTDIR = '/orange/adamginsburg/cmz/arms/3kpc_arm_filament'

# Data paths
_CHIMPS_DIR   = '/orange/adamginsburg/cmz/CHIMPS'
_SEDIGISM_DIR = '/orange/adamginsburg/galactic_plane_surveys/sedigism'
_NOB_DIR      = '/orange/adamginsburg/cmz/nobeyama'
_DAME_CUBE    = '/orange/adamginsburg/cmz/dameCO/DHT02_Center_interp_spectralcube.fits'

# (name, path, spectral-line label, survey label)
DATASETS = [
    ('CHIMPS_12CO32',
     os.path.join(_CHIMPS_DIR, '12CO_GC_359-000_mosaic.fits'),
     r'$^{12}$CO(3–2)', 'CHIMPS'),

    ('SEDIGISM_13CO21',
     os.path.join(_SEDIGISM_DIR, 'G000_13CO21_Tmb_DR1.fits'),
     r'$^{13}$CO(2–1)', 'SEDIGISM'),

    ('Dame_12CO10',
     _DAME_CUBE,
     r'$^{12}$CO(1–0)', 'Dame'),

    ('Nobeyama_12CO21',
     os.path.join(_NOB_DIR, '12CO-2.BEARS.FITS'),
     r'$^{12}$CO(2–1)', 'Nobeyama BEARS'),

    ('Nobeyama_13CO21',
     os.path.join(_NOB_DIR, '13CO-2.FOREST.FITS'),
     r'$^{13}$CO(2–1)', 'Nobeyama FOREST'),
]

# =============================================================================
# Helpers
# =============================================================================

def spatial_cutout(cube):
    """Return the portion of *cube* that overlaps the target rectangle.

    Uses the 2-D celestial WCS to map the four corners of the target box to
    pixel coordinates, then slices.  Handles Galactic cubes whose longitude
    axis wraps near 0°/360°.

    Returns None if the cube does not overlap the target region at all.
    """
    wcs2d = cube.wcs.celestial

    # Work out the expected pixel positions of the four box corners.
    l_lo = L_CENTER - HALF_SIZE
    l_hi = L_CENTER + HALF_SIZE
    b_lo = B_CENTER - HALF_SIZE
    b_hi = B_CENTER + HALF_SIZE

    corners_sky = SkyCoord(
        l=[l_lo, l_lo, l_hi, l_hi] * u.deg,
        b=[b_lo, b_hi, b_lo, b_hi] * u.deg,
        frame='galactic',
    )

    # pixel_to_world expectation: input (x, y) in pixel, output sky coords
    # Use world_to_pixel which returns (x, y) with origin=0
    try:
        px, py = wcs2d.world_to_pixel(corners_sky)
    except Exception:
        # Fall back to FITS convention (origin=1)
        px, py = wcs2d.world_to_pixel(corners_sky)

    px = np.round(px).astype(int)
    py = np.round(py).astype(int)

    xmin, xmax = int(np.nanmin(px)), int(np.nanmax(px))
    ymin, ymax = int(np.nanmin(py)), int(np.nanmax(py))

    # Clip to cube extent
    nx, ny = cube.shape[2], cube.shape[1]
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(nx - 1, xmax)
    ymax = min(ny - 1, ymax)

    if xmin >= xmax or ymin >= ymax:
        return None  # no overlap

    return cube[:, ymin:ymax + 1, xmin:xmax + 1]


def load_subcube(path, name):
    """Load *path* as a SpectralCube in km/s and cut to target region + velocity range.

    Returns the sub-cube containing all signal + background channels, or None
    if the file is missing / does not overlap.
    """
    if not os.path.exists(path):
        print(f'    [skip] file not found: {path}')
        return None

    print(f'  Loading {name} …')
    cube = (SpectralCube.read(path)
            .with_spectral_unit(u.km / u.s, velocity_convention='radio'))

    # Spatial cutout first (cheap: tiny pixel slice)
    cube = spatial_cutout(cube)
    if cube is None:
        print(f'    [skip] cube does not overlap target region')
        return None

    # Velocity slab: all channels needed for signal + both bg windows
    v_total_lo = (BG_LO_LOW  - 5) * u.km / u.s
    v_total_hi = (BG_HI_HIGH + 5) * u.km / u.s
    v_lo_edge  = min(v_total_lo.value, cube.spectral_axis.to(u.km/u.s).value.min())
    v_hi_edge  = max(v_total_hi.value, cube.spectral_axis.to(u.km/u.s).value.max())

    # Clamp to actual cube range
    v_lo = max(BG_LO_LOW  - 5, cube.spectral_axis.to(u.km/u.s).value.min()) * u.km/u.s
    v_hi = min(BG_HI_HIGH + 5, cube.spectral_axis.to(u.km/u.s).value.max()) * u.km/u.s

    cube = cube.spectral_slab(v_lo, v_hi)

    if cube.shape[0] < 2:
        print(f'    [skip] insufficient spectral coverage after velocity cut')
        return None

    print(f'    shape after cut: {cube.shape}  '
          f'v=[{cube.spectral_axis.to(u.km/u.s).value.min():.1f}, '
          f'{cube.spectral_axis.to(u.km/u.s).value.max():.1f}] km/s')
    return cube


# =============================================================================
# Background subtraction
# =============================================================================

def _bg_mask(spectral_axis_kms):
    """Boolean mask: True for channels in either background window."""
    v = spectral_axis_kms
    return ((v >= BG_LO_LOW) & (v <= BG_LO_HIGH)) | \
           ((v >= BG_HI_LOW) & (v <= BG_HI_HIGH))


def _sig_mask(spectral_axis_kms):
    """Boolean mask: True for channels in the signal window."""
    v = spectral_axis_kms
    return (v >= V_SIG_LOW) & (v <= V_SIG_HIGH)


def subtract_mean_bg(cube):
    """Method 1 – Simple mean background subtraction.

    Parameters
    ----------
    cube : SpectralCube
        Sub-cube containing at least the signal and both background windows.

    Returns
    -------
    moment0 : 2-D array  [K km/s]
        Background-subtracted integrated intensity (moment 0).
    bg_map : 2-D array  [K]
        Per-pixel background level that was subtracted.
    sig_cube_data : 3-D array  [K]
        Background-subtracted signal-window data cube.
    header : fits.Header
        2-D WCS header for saving moment0 as a FITS image.
    """
    vax = cube.spectral_axis.to(u.km / u.s).value

    bg_sel  = _bg_mask(vax)
    sig_sel = _sig_mask(vax)

    data = cube.filled_data[:].value          # (nchan, ny, nx)  [K or Jy/bm]

    # Mean bg: average over all background channels simultaneously
    bg_map = np.nanmean(data[bg_sel], axis=0)  # (ny, nx)

    # Subtract constant bg from every signal channel
    sig_data = data[sig_sel] - bg_map[np.newaxis]   # (n_sig, ny, nx)

    # Moment 0 = Σ T_bgsub * dv  (trapezoid integration)
    dv = np.abs(np.median(np.diff(vax)))           # km/s per channel
    moment0 = np.nansum(sig_data, axis=0) * dv      # K km/s

    # 2-D spatial WCS header
    header = cube.wcs.celestial.to_header()
    header['BUNIT'] = 'K km/s'

    return moment0, bg_map, sig_data, header


def subtract_linear_bg(cube):
    """Method 2 – Linear baseline background subtraction.

    A first-order polynomial (slope + intercept) is fit, per spatial pixel,
    through the brightness-temperature values at all background channels on
    both sides of the signal window.  The fitted baseline is then evaluated
    at every signal-window channel and subtracted before integrating.

    Returns
    -------
    moment0 : 2-D array  [K km/s]
    slope   : 2-D array  [K / (km/s)]   – fitted slope per pixel
    intercept : 2-D array  [K]           – fitted intercept per pixel
    sig_cube_data : 3-D array  [K]
    header  : fits.Header
    """
    vax = cube.spectral_axis.to(u.km / u.s).value

    bg_sel  = _bg_mask(vax)
    sig_sel = _sig_mask(vax)

    data = cube.filled_data[:].value          # (nchan, ny, nx)

    v_bg   = vax[bg_sel]                      # (n_bg,)
    T_bg   = data[bg_sel]                     # (n_bg, ny, nx)

    ny, nx = data.shape[1], data.shape[2]
    n_bg   = v_bg.shape[0]

    # Design matrix A of shape (n_bg, 2): columns are [v, 1]
    A = np.column_stack([v_bg, np.ones(n_bg)])   # (n_bg, 2)

    # Reshape to (n_bg, n_pix) for batch least-squares
    T_flat = T_bg.reshape(n_bg, ny * nx)

    # Replace NaN pixels with zeros for the fit; track fully-NaN pixels
    nan_pix = np.all(np.isnan(T_flat), axis=0)  # (n_pix,)
    T_flat_safe = np.where(np.isnan(T_flat), 0.0, T_flat)

    # Batch least-squares  →  coeffs shape (2, n_pix)
    coeffs, _, _, _ = np.linalg.lstsq(A, T_flat_safe, rcond=None)

    slope     = coeffs[0].reshape(ny, nx)    # [K / (km/s)]
    intercept = coeffs[1].reshape(ny, nx)    # [K]

    # Restore NaN for fully-masked pixels
    slope[nan_pix.reshape(ny, nx)]     = np.nan
    intercept[nan_pix.reshape(ny, nx)] = np.nan

    # Evaluate baseline at every signal channel and subtract
    v_sig = vax[sig_sel]                     # (n_sig,)
    # baseline shape: (n_sig, ny, nx)
    baseline = (slope[np.newaxis]  * v_sig[:, np.newaxis, np.newaxis]
                + intercept[np.newaxis])

    sig_data = data[sig_sel] - baseline      # (n_sig, ny, nx)

    # Moment 0
    dv = np.abs(np.median(np.diff(vax)))
    moment0 = np.nansum(sig_data, axis=0) * dv

    header = cube.wcs.celestial.to_header()
    header['BUNIT'] = 'K km/s'

    return moment0, slope, intercept, sig_data, header


# =============================================================================
# PV diagram  (b-collapsed, l-v)
# =============================================================================

def make_pv(sig_data_bgsub, cube, label, method):
    """Collapse the background-subtracted signal cube along latitude (axis 1).

    Returns
    -------
    pv : 2-D array  (n_sig_chan × n_lon)
    pv_header : fits.Header  with CTYPE1=GLON-CAR, CTYPE2=VELO-LSR
    """
    pv = np.nanmean(sig_data_bgsub, axis=1)  # average along b  → (n_sig, nx)

    # Build a simple 2-D FITS header: axis1=longitude, axis2=velocity
    wcs2d = cube.wcs.celestial
    h2d   = wcs2d.to_header()

    vax   = cube.spectral_axis.to(u.km / u.s).value
    sig_v = vax[_sig_mask(vax)]
    dv    = float(np.median(np.diff(sig_v))) if len(sig_v) > 1 else 1.0

    pv_header = fits.Header()
    pv_header['NAXIS']  = 2
    pv_header['NAXIS1'] = pv.shape[1]          # longitude axis
    pv_header['NAXIS2'] = pv.shape[0]          # velocity axis
    pv_header['CTYPE1'] = h2d.get('CTYPE1', 'GLON-CAR')
    pv_header['CRPIX1'] = h2d.get('CRPIX1', 1)
    pv_header['CRVAL1'] = h2d.get('CRVAL1', L_CENTER)
    pv_header['CDELT1'] = h2d.get('CDELT1', -0.01)
    pv_header['CUNIT1'] = 'deg'
    pv_header['CTYPE2'] = 'VELO-LSR'
    pv_header['CRPIX2'] = 1
    pv_header['CRVAL2'] = sig_v[0] * 1e3       # m/s
    pv_header['CDELT2'] = dv * 1e3             # m/s per channel
    pv_header['CUNIT2'] = 'm/s'
    pv_header['BUNIT']  = 'K'

    return pv, pv_header


# =============================================================================
# Plotting
# =============================================================================

def plot_comparison(name, line_label, survey_label,
                    mom0_simple,  bg_map,
                    mom0_linear,  slope, intercept,
                    pv_simple,    pv_linear,
                    cube, outdir):
    """Four-panel figure comparing the two background-subtraction methods."""
    vax  = cube.spectral_axis.to(u.km / u.s).value
    sig  = _sig_mask(vax)
    v_sig = vax[sig]

    wcs2d = cube.wcs.celestial

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f'{survey_label}  {line_label}  –  {LABEL}\n'
                 f'l={L_CENTER:.2f}°  b={B_CENTER:+.3f}°  '
                 f'v=[{V_SIG_LOW:.0f}, {V_SIG_HIGH:.0f}] km/s',
                 fontsize=12)

    gs = GridSpec(2, 3, figure=fig, wspace=0.35, hspace=0.4)

    # Shared colour scale for moment 0 maps
    vmax = np.nanpercentile(np.concatenate([mom0_simple[np.isfinite(mom0_simple)],
                                             mom0_linear[np.isfinite(mom0_linear)]]),
                            99)
    vmin = np.nanpercentile(np.concatenate([mom0_simple[np.isfinite(mom0_simple)],
                                             mom0_linear[np.isfinite(mom0_linear)]]),
                            1)

    # --- Row 0: moment-0 maps ---
    for col, (mom0, method) in enumerate([(mom0_simple, 'Simple mean bg'),
                                           (mom0_linear, 'Linear baseline bg')]):
        ax = fig.add_subplot(gs[0, col], projection=wcs2d)
        im = ax.imshow(mom0, origin='lower', aspect='auto',
                       vmin=vmin, vmax=vmax, cmap='inferno')
        ax.set_xlabel('Galactic Longitude', fontsize=8)
        ax.set_ylabel('Galactic Latitude', fontsize=8)
        ax.set_title(f'Moment 0  [{method}]', fontsize=9)
        plt.colorbar(im, ax=ax, label='K km/s', fraction=0.046)

    # Difference map (linear − simple)
    ax = fig.add_subplot(gs[0, 2], projection=wcs2d)
    diff = mom0_linear - mom0_simple
    sym  = np.nanpercentile(np.abs(diff[np.isfinite(diff)]), 95)
    im   = ax.imshow(diff, origin='lower', aspect='auto',
                     vmin=-sym, vmax=sym, cmap='RdBu_r')
    ax.set_xlabel('Galactic Longitude', fontsize=8)
    ax.set_ylabel('Galactic Latitude', fontsize=8)
    ax.set_title('Difference  (linear − simple)', fontsize=9)
    plt.colorbar(im, ax=ax, label='K km/s', fraction=0.046)

    # --- Row 1: PV diagrams ---
    # Build a simple longitude axis from the cube header
    h = wcs2d.to_header()
    nx  = pv_simple.shape[1]
    lon = (h.get('CRVAL1', L_CENTER)
           + (np.arange(nx) - (h.get('CRPIX1', nx / 2) - 1))
           * h.get('CDELT1', -HALF_SIZE * 2 / nx))

    extent = [lon[-1], lon[0], v_sig[0], v_sig[-1]]

    for col, (pv, method) in enumerate([(pv_simple,  'Simple mean bg'),
                                         (pv_linear,  'Linear baseline bg')]):
        ax = fig.add_subplot(gs[1, col])
        pvmax = np.nanpercentile(pv[np.isfinite(pv)], 99)
        ax.imshow(pv, origin='lower', aspect='auto',
                  extent=extent, vmin=0, vmax=pvmax, cmap='inferno')
        ax.set_xlabel('Galactic Longitude (deg)', fontsize=8)
        ax.set_ylabel('v$_{LSR}$ (km/s)', fontsize=8)
        ax.set_title(f'PV (b-averaged)  [{method}]', fontsize=9)
        ax.axhline(V_SIG_LOW,  color='w', lw=0.7, ls='--')
        ax.axhline(V_SIG_HIGH, color='w', lw=0.7, ls='--')

    # Background level map (simple method)
    ax = fig.add_subplot(gs[1, 2], projection=wcs2d)
    bmax = np.nanpercentile(np.abs(bg_map[np.isfinite(bg_map)]), 95)
    im   = ax.imshow(bg_map, origin='lower', aspect='auto',
                     vmin=-bmax, vmax=bmax, cmap='RdBu_r')
    ax.set_xlabel('Galactic Longitude', fontsize=8)
    ax.set_ylabel('Galactic Latitude', fontsize=8)
    ax.set_title('Mean bg level  (simple method)  [K]', fontsize=9)
    plt.colorbar(im, ax=ax, label='K', fraction=0.046)

    outpath = os.path.join(outdir, f'{LABEL}_{name}_comparison.png')
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'    Plot: {outpath}')


def plot_spectra(name, line_label, survey_label, cube, outdir):
    """Mean spectrum of the region with background windows marked."""
    vax  = cube.spectral_axis.to(u.km / u.s).value
    spec = np.nanmean(cube.filled_data[:].value, axis=(1, 2))

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(vax, spec, 'k-', lw=0.9, label='Mean spectrum')
    ax.axvspan(BG_LO_LOW,  BG_LO_HIGH, alpha=0.15, color='steelblue',
               label='Background (low)')
    ax.axvspan(BG_HI_LOW,  BG_HI_HIGH, alpha=0.15, color='steelblue',
               label='Background (high)')
    ax.axvspan(V_SIG_LOW,  V_SIG_HIGH, alpha=0.20, color='tomato',
               label='Signal window')
    ax.axhline(0, color='gray', lw=0.5, ls='--')
    ax.set_xlabel('v$_{LSR}$ (km/s)')
    ax.set_ylabel('Mean T$_B$ (K)')
    ax.set_title(f'{survey_label}  {line_label}  –  mean spectrum\n'
                 f'(l={L_CENTER:.2f}°, b={B_CENTER:+.3f}°, '
                 f'{HALF_SIZE*2*60:.0f}′×{HALF_SIZE*2*60:.0f}′)')
    ax.legend(fontsize=8)
    outpath = os.path.join(outdir, f'{LABEL}_{name}_spectrum.png')
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'    Spectrum: {outpath}')


# =============================================================================
# Save FITS helpers
# =============================================================================

def save_fits(data, header, path):
    hdu = fits.PrimaryHDU(data.astype(np.float32), header=header)
    hdu.writeto(path, overwrite=True)
    print(f'    FITS: {path}')


# =============================================================================
# Main pipeline
# =============================================================================

def run_dataset(name, path, line_label, survey_label):
    print(f'\n{"=" * 60}')
    print(f'{survey_label}  {line_label}  ({name})')
    print(f'{"=" * 60}')

    # 1. Load
    cube = load_subcube(path, name)
    if cube is None:
        return

    # 2. Diagnostic spectrum
    plot_spectra(name, line_label, survey_label, cube, OUTDIR)

    # 3. Simple mean background subtraction
    print('  Method 1: simple mean background …')
    mom0_simple, bg_map, sig_simple, hdr2d = subtract_mean_bg(cube)

    save_fits(mom0_simple, hdr2d,
              os.path.join(OUTDIR, f'{LABEL}_{name}_mom0_simplebg.fits'))
    save_fits(bg_map, hdr2d,
              os.path.join(OUTDIR, f'{LABEL}_{name}_bgmap_simplebg.fits'))

    # 4. Linear baseline background subtraction
    print('  Method 2: linear baseline …')
    mom0_linear, slope, intercept, sig_linear, hdr2d = subtract_linear_bg(cube)

    save_fits(mom0_linear, hdr2d,
              os.path.join(OUTDIR, f'{LABEL}_{name}_mom0_linearbg.fits'))
    save_fits(slope, hdr2d,
              os.path.join(OUTDIR, f'{LABEL}_{name}_bgslope_linearbg.fits'))
    save_fits(intercept, hdr2d,
              os.path.join(OUTDIR, f'{LABEL}_{name}_bgintercept_linearbg.fits'))

    # 5. PV diagrams
    pv_simple, pv_hdr_s = make_pv(sig_simple,  cube, name, 'simplebg')
    pv_linear, pv_hdr_l = make_pv(sig_linear,  cube, name, 'linearbg')

    save_fits(pv_simple, pv_hdr_s,
              os.path.join(OUTDIR, f'{LABEL}_{name}_pv_simplebg.fits'))
    save_fits(pv_linear, pv_hdr_l,
              os.path.join(OUTDIR, f'{LABEL}_{name}_pv_linearbg.fits'))

    # 6. Comparison figure
    print('  Plotting …')
    plot_comparison(name, line_label, survey_label,
                    mom0_simple, bg_map,
                    mom0_linear, slope, intercept,
                    pv_simple, pv_linear,
                    cube, OUTDIR)

    print(f'  Done: {name}')


if __name__ == '__main__':
    os.makedirs(OUTDIR, exist_ok=True)

    print('=' * 60)
    print(f'3kpc Arm Filament CO Extraction  –  {LABEL}')
    print(f'Centre:  l={L_CENTER:.3f}°  b={B_CENTER:+.4f}°')
    print(f'Size:    {HALF_SIZE*2*60:.0f}′ × {HALF_SIZE*2*60:.0f}′')
    print(f'Signal:  v ∈ [{V_SIG_LOW:.0f}, {V_SIG_HIGH:.0f}] km/s')
    print(f'Bg low:  v ∈ [{BG_LO_LOW:.0f}, {BG_LO_HIGH:.0f}] km/s')
    print(f'Bg high: v ∈ [{BG_HI_LOW:.0f}, {BG_HI_HIGH:.0f}] km/s')
    print(f'Output:  {OUTDIR}')
    print('=' * 60)

    for name, path, line_label, survey_label in DATASETS:
        try:
            run_dataset(name, path, line_label, survey_label)
        except Exception as exc:
            print(f'  ERROR processing {name}: {exc}')
            import traceback
            traceback.print_exc()

    print('\nAll datasets done.')
