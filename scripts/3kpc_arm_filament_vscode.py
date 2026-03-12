"""
3kpc Arm Filament Extraction — targeted small-region script
============================================================
Region  : l = 0.34 deg, b = +0.024 deg, ~30' × 30' (0.5 deg per side)
Velocity: arm window  -59 to -50 km/s  (centre -55 km/s)
          background  -69 to -59 km/s  (low side)
                      -50 to -40 km/s  (high side)

Two background-subtraction approaches are saved independently:
  *_simplebg.fits   – mean of bg channels on both sides subtracted from
                      mean of arm channels  (conventional approach)
  *_linearbg.fits   – per-pixel linear baseline fit through the bg channels
                      as a function of velocity, evaluated & subtracted at
                      each arm channel before the arm mean is taken

Outputs go to  /orange/adamginsburg/cmz/arms/3kpc_arm_filament_vscode/

Datasets tried (skipped gracefully if not found or outside region):
  ACES  – all *_downsampled9.fits cubes
  CHIMPS 12CO(3-2) – GC mosaics
  Nobeyama 12CO / 13CO – BEARS/S115Q/FOREST cubes
  SEDIGISM 13CO(2-1)  – G000 tile
  Dame CO  – DHT02 whole-galaxy cube
"""

import os
import glob
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy import units as u
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, Galactic
import astropy.units as au
from spectral_cube import SpectralCube

warnings.filterwarnings('ignore')

# =============================================================================
# Output directory
# =============================================================================
SLUG    = '3kpc_arm_filament_vscode'
OUTDIR  = f'/orange/adamginsburg/cmz/arms/{SLUG}'
os.makedirs(OUTDIR, exist_ok=True)

# =============================================================================
# Spatial region (30' × 30' centred on l=0.34, b=+0.024)
# =============================================================================
L_CENTER  =  0.34   # deg
B_CENTER  =  0.024  # deg
HALF_SIZE =  0.25   # deg  (30' / 2 = 0.25 deg)

L_LO = L_CENTER - HALF_SIZE   #  0.09 deg
L_HI = L_CENTER + HALF_SIZE   #  0.59 deg
B_LO = B_CENTER - HALF_SIZE   # -0.226 deg
B_HI = B_CENTER + HALF_SIZE   #  0.274 deg

# =============================================================================
# Velocity settings
# =============================================================================
V_ARM_LO  = -59.0  * u.km / u.s   # arm window lower edge
V_ARM_HI  = -50.0  * u.km / u.s   # arm window upper edge
V_ARM_CEN = -55.0  * u.km / u.s   # nominal centre

BG_WIDTH  =  10.0  * u.km / u.s   # extent of each BG sideband

V_BG_LO_LO = V_ARM_LO - BG_WIDTH  # -69 km/s
V_BG_LO_HI = V_ARM_LO             # -59 km/s
V_BG_HI_LO = V_ARM_HI             # -50 km/s
V_BG_HI_HI = V_ARM_HI + BG_WIDTH  # -40 km/s

# Wide spectral slice needed to encompass arm + both BG sidebands
V_WIDE_LO = V_BG_LO_LO - 2.0 * u.km / u.s   # a little margin
V_WIDE_HI = V_BG_HI_HI + 2.0 * u.km / u.s

plt.rcParams.update({'figure.facecolor': 'w', 'font.size': 16,
                     'image.origin': 'lower', 'image.interpolation': 'none'})

# =============================================================================
# Helper — save a quick PNG of a 2-D map
# =============================================================================
def save_png(data2d, header, title, outpng, vmin_pct=1, vmax_pct=99):
    h   = header
    ny  = h.get('NAXIS2', data2d.shape[0])
    nx  = h.get('NAXIS1', data2d.shape[1])
    crpix1 = h.get('CRPIX1', nx / 2.)
    crval1 = h.get('CRVAL1', L_CENTER)
    cdelt1 = h.get('CDELT1', 1.)
    crpix2 = h.get('CRPIX2', ny / 2.)
    crval2 = h.get('CRVAL2', B_CENTER)
    cdelt2 = h.get('CDELT2', 1.)
    lon0 = crval1 + (0      - (crpix1 - 1)) * cdelt1
    lon1 = crval1 + (nx - 1 - (crpix1 - 1)) * cdelt1
    lat0 = crval2 + (0      - (crpix2 - 1)) * cdelt2
    lat1 = crval2 + (ny - 1 - (crpix2 - 1)) * cdelt2
    extent = [lon0, lon1, lat0, lat1]

    finite = data2d[np.isfinite(data2d)]
    if finite.size < 2:
        print(f'    [skip PNG — no finite data]: {outpng}')
        return
    vlo = float(np.nanpercentile(finite, vmin_pct))
    vhi = float(np.nanpercentile(finite, vmax_pct))

    lon_range = abs(lon1 - lon0) or 0.5
    lat_range = abs(lat1 - lat0) or 0.5
    fig_w = 10.
    fig_h = max(fig_w * lat_range / lon_range, 0.5) + 0.8
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(data2d, origin='lower', aspect='auto', extent=extent,
              vmin=vlo, vmax=vhi, cmap='gray_r')
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Galactic Longitude (deg)')
    ax.set_ylabel('Galactic Latitude (deg)')
    # mark the target centre
    ax.axvline(L_CENTER, color='r', lw=0.7, ls='--', alpha=0.6)
    ax.axhline(B_CENTER, color='r', lw=0.7, ls='--', alpha=0.6)
    fig.tight_layout()
    fig.savefig(outpng, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'    PNG: {outpng}')


# =============================================================================
# Core: given a spectral cube (already in km/s), perform the extraction
# =============================================================================
def extract_filament(cube, label, outbase):
    """
    Extract the filament signal with two background-subtraction methods.

    Saves:
      {outbase}_velslab.fits      – raw arm-window mean (no BG sub)
      {outbase}_simplebg.fits     – simple mean BG subtracted
      {outbase}_linearbg.fits     – linear-fit BG subtracted
      Corresponding PNGs.

    Returns (simplebg_file, linearbg_file) or (None, None) on failure.
    """
    velslab_file  = f'{outbase}_velslab.fits'
    simplebg_file = f'{outbase}_simplebg.fits'
    linearbg_file = f'{outbase}_linearbg.fits'

    if (os.path.exists(simplebg_file) and os.path.exists(linearbg_file)):
        print(f'  [skip] {label}: outputs already exist')
        # Remake PNGs even if FITS already exist
        for fn, title_sfx in [
                (velslab_file,  'vel-slab (no BG sub)'),
                (simplebg_file, 'simple mean BG sub'),
                (linearbg_file, 'linear-fit BG sub')]:
            if os.path.exists(fn):
                h = fits.getheader(fn)
                save_png(fits.getdata(fn).astype(float), h,
                         f'{label} — {title_sfx}',
                         fn.replace('.fits', '.png'))
        return simplebg_file, linearbg_file

    print(f'  Shape: {cube.shape}')
    spec_lo = cube.spectral_axis.to(u.km/u.s).min()
    spec_hi = cube.spectral_axis.to(u.km/u.s).max()
    print(f'  Spectral: {spec_lo:.1f} to {spec_hi:.1f}')

    # ---- spectral masks -------------------------------------------------------
    vels = cube.spectral_axis.to(u.km / u.s)

    arm_mask = (vels >= V_ARM_LO ) & (vels <= V_ARM_HI )
    bg_mask  = (((vels >= V_BG_LO_LO) & (vels <= V_BG_LO_HI)) |
                ((vels >= V_BG_HI_LO) & (vels <= V_BG_HI_HI)))

    n_arm = int(arm_mask.sum())
    n_bg  = int(bg_mask.sum())
    print(f'  Arm channels: {n_arm}   BG channels: {n_bg}')
    if n_arm == 0:
        print(f'  [skip] {label}: no arm channels found in spectral range')
        return None, None
    if n_bg < 2:
        print(f'  [warn] {label}: only {n_bg} BG channels — BG subtraction may be poor')

    cube.allow_huge_operations = True

    # ---- cube data as numpy (vel, lat, lon) ------------------------------------
    # Subcube to the relevant spectral range and pull values once
    wide_mask = arm_mask | bg_mask
    wide_cube = cube.with_mask(
        np.broadcast_to(wide_mask[:, None, None], cube.shape))
    # Use spectral slabs to avoid pulling the entire cube
    v_lo_req = min(V_BG_LO_LO, V_ARM_LO).to(u.km/u.s).value
    v_hi_req = max(V_BG_HI_HI, V_ARM_HI).to(u.km/u.s).value
    try:
        subcube = cube.spectral_slab(v_lo_req * u.km/u.s,
                                     v_hi_req * u.km/u.s)
    except Exception as exc:
        print(f'  [warn] spectral_slab failed ({exc}); using full cube')
        subcube = cube

    vels_sub  = subcube.spectral_axis.to(u.km / u.s).value   # (nv_sub,)
    arm_idx   = np.where((vels_sub >= V_ARM_LO.value) &
                         (vels_sub <= V_ARM_HI.value))[0]
    bg_idx    = np.where(((vels_sub >= V_BG_LO_LO.value) & (vels_sub <= V_BG_LO_HI.value)) |
                         ((vels_sub >= V_BG_HI_LO.value) & (vels_sub <= V_BG_HI_HI.value)))[0]

    n_arm2 = len(arm_idx)
    n_bg2  = len(bg_idx)
    print(f'  Subcube arm channels: {n_arm2}   BG channels: {n_bg2}')
    if n_arm2 == 0:
        print(f'  [skip] {label}: subcube has no arm channels')
        return None, None

    # Pull data to numpy (this is the only large memory operation)
    print(f'  Reading cube data ...')
    data = subcube.filled_data[:].to_value()    # (nv_sub, ny, nx)
    if data.ndim != 3:
        print(f'  [skip] {label}: unexpected cube dimensionality {data.ndim}')
        return None, None

    nv_sub, ny, nx = data.shape

    # ---- arm mean (no BG sub) --------------------------------------------------
    arm_data  = data[arm_idx]                                    # (n_arm, ny, nx)
    arm_mean  = np.nanmean(arm_data, axis=0).astype(np.float32) # (ny, nx)

    # ---- METHOD 1: simple mean of BG channels on both sides -------------------
    if n_bg2 >= 1:
        bg_data   = data[bg_idx]                                  # (n_bg, ny, nx)
        bg_simple = np.nanmean(bg_data, axis=0).astype(np.float32)
    else:
        bg_simple = np.zeros_like(arm_mean)

    simplebg_map = (arm_mean - bg_simple).astype(np.float32)

    # ---- METHOD 2: linear fit per spatial pixel --------------------------------
    # For every (lat, lon) pixel, fit a line: value = a * velocity + b
    # using the bg_idx channels, then subtract the fit evaluated at arm_idx
    # channels before taking the mean.
    if n_bg2 >= 2:
        vels_bg  = vels_sub[bg_idx]                  # (n_bg,)
        vels_arm = vels_sub[arm_idx]                  # (n_arm,)
        # Flatten spatial dims for vectorised polyfit
        data_flat = data.reshape(nv_sub, ny * nx)     # (nv, npix)
        bg_flat   = data_flat[bg_idx, :]              # (n_bg, npix)
        # Replace NaN with pixel mean so polyfit doesn't propagate NaN everywhere
        col_means = np.nanmean(bg_flat, axis=0, keepdims=True)
        bg_flat_filled = np.where(np.isfinite(bg_flat), bg_flat, col_means)
        # Fit: rows=samples, cols=pixels; use polyfit along axis 0
        # vels_bg normalised to centre ~0 for numerical stability
        v_ref   = np.mean(vels_bg)
        v_bg_n  = vels_bg - v_ref
        v_arm_n = vels_arm - v_ref
        # Least-squares slope and intercept for all pixels at once
        # X = [v_bg_n, ones],  shape (n_bg, 2)
        X = np.stack([v_bg_n,
                      np.ones(len(v_bg_n))], axis=1)       # (n_bg, 2)
        # Solve  X @ coeff = bg_flat_filled  for coeff (2, npix)
        coeff, _, _, _ = np.linalg.lstsq(X, bg_flat_filled,
                                          rcond=None)       # (2, npix)
        # Evaluate baseline at arm channels and subtract
        X_arm = np.stack([v_arm_n,
                          np.ones(len(v_arm_n))], axis=1)  # (n_arm, 2)
        baseline_at_arm = X_arm @ coeff                    # (n_arm, npix)
        arm_flat  = data_flat[arm_idx, :]                  # (n_arm, npix)
        arm_bgsub_flat = arm_flat - baseline_at_arm        # (n_arm, npix)
        # NaN out pixels that had all-NaN in the arm
        all_nan = ~np.any(np.isfinite(arm_flat), axis=0)   # (npix,)
        linearbg_flat  = np.nanmean(arm_bgsub_flat, axis=0).astype(np.float32)
        linearbg_flat[all_nan] = np.nan
        linearbg_map = linearbg_flat.reshape(ny, nx)
    elif n_bg2 == 1:
        # Only one BG channel: fall back to simple subtraction, warn
        print(f'  [warn] {label}: only 1 BG channel; linear-bg == simple-bg')
        linearbg_map = simplebg_map.copy()
    else:
        print(f'  [warn] {label}: no BG channels; linear-bg == arm mean (no sub)')
        linearbg_map = arm_mean.copy()

    # ---- WCS for 2-D output ---------------------------------------------------
    wcs_2d = subcube.wcs.celestial
    hdr = wcs_2d.to_header()
    hdr['BUNIT']    = str(subcube.unit)
    hdr['RESTFREQ'] = subcube.wcs.wcs.restfrq
    hdr['OBJECT']   = f'l={L_CENTER:.3f} b={B_CENTER:.3f} filament'
    hdr['V_ARM_LO'] = (V_ARM_LO.value,  'Arm velocity lower bound (km/s)')
    hdr['V_ARM_HI'] = (V_ARM_HI.value,  'Arm velocity upper bound (km/s)')
    hdr['V_BG_LL']  = (V_BG_LO_LO.value, 'BG lower sideband lo (km/s)')
    hdr['V_BG_LH']  = (V_BG_LO_HI.value, 'BG lower sideband hi (km/s)')
    hdr['V_BG_HL']  = (V_BG_HI_LO.value, 'BG upper sideband lo (km/s)')
    hdr['V_BG_HH']  = (V_BG_HI_HI.value, 'BG upper sideband hi (km/s)')
    hdr['NARM_CH']  = (n_arm2, 'Number of arm velocity channels')
    hdr['NBG_CH']   = (n_bg2,  'Number of BG velocity channels')

    # ---- Save FITS ------------------------------------------------------------
    hdr_vs = hdr.copy(); hdr_vs['COMMENT'] = 'Arm window mean (no BG sub)'
    hdr_sb = hdr.copy(); hdr_sb['COMMENT'] = 'Simple mean BG subtracted'
    hdr_lb = hdr.copy(); hdr_lb['COMMENT'] = 'Linear-fit BG subtracted per pixel'

    fits.PrimaryHDU(data=arm_mean,    header=hdr_vs).writeto(velslab_file,  overwrite=True)
    fits.PrimaryHDU(data=simplebg_map, header=hdr_sb).writeto(simplebg_file, overwrite=True)
    fits.PrimaryHDU(data=linearbg_map, header=hdr_lb).writeto(linearbg_file, overwrite=True)
    print(f'  Saved: {velslab_file}')
    print(f'  Saved: {simplebg_file}')
    print(f'  Saved: {linearbg_file}')

    # ---- PNGs -----------------------------------------------------------------
    for fn, data2d, title_sfx in [
            (velslab_file,  arm_mean,     'vel-slab (no BG sub)'),
            (simplebg_file, simplebg_map, 'simple mean BG sub'),
            (linearbg_file, linearbg_map, 'linear-fit BG sub')]:
        save_png(data2d, hdr, f'{label} — {title_sfx}',
                 fn.replace('.fits', '.png'))

    # ---- Comparison PNG (side-by-side of the two BG methods) -----------------
    comp_png = f'{outbase}_bgsub_compare.png'
    finite_s = simplebg_map[np.isfinite(simplebg_map)]
    finite_l = linearbg_map[np.isfinite(linearbg_map)]
    if finite_s.size > 1 and finite_l.size > 1:
        lon_range = abs(L_HI - L_LO) or 0.5
        lat_range = abs(B_HI - B_LO) or 0.5
        fig_w = 12.
        panel_h = max(fig_w / 2 * lat_range / lon_range, 0.4)
        fig_h   = panel_h + 1.2
        fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h))
        for ax, d2, title_sfx in zip(
                axes,
                [simplebg_map, linearbg_map],
                ['Simple mean BG', 'Linear-fit BG']):
            finite = d2[np.isfinite(d2)]
            vlo = float(np.nanpercentile(finite, 1))
            vhi = float(np.nanpercentile(finite, 99))
            h_tmp  = fits.getheader(simplebg_file)
            ny_t   = h_tmp.get('NAXIS2', d2.shape[0])
            nx_t   = h_tmp.get('NAXIS1', d2.shape[1])
            crpix1 = h_tmp.get('CRPIX1', nx_t / 2.)
            crval1 = h_tmp.get('CRVAL1', L_CENTER)
            cdelt1 = h_tmp.get('CDELT1', 1.)
            crpix2 = h_tmp.get('CRPIX2', ny_t / 2.)
            crval2 = h_tmp.get('CRVAL2', B_CENTER)
            cdelt2 = h_tmp.get('CDELT2', 1.)
            lon0_e = crval1 + (0       - (crpix1 - 1)) * cdelt1
            lon1_e = crval1 + (nx_t-1  - (crpix1 - 1)) * cdelt1
            lat0_e = crval2 + (0       - (crpix2 - 1)) * cdelt2
            lat1_e = crval2 + (ny_t-1  - (crpix2 - 1)) * cdelt2
            ext    = [lon0_e, lon1_e, lat0_e, lat1_e]
            ax.imshow(d2, origin='lower', aspect='auto', extent=ext,
                      vmin=vlo, vmax=vhi, cmap='gray_r')
            ax.set_title(f'{label}\n{title_sfx}', fontsize=15)
            ax.set_xlabel('Galactic Longitude (deg)')
            ax.set_ylabel('Galactic Latitude (deg)')
            ax.axvline(L_CENTER, color='r', lw=0.7, ls='--', alpha=0.6)
            ax.axhline(B_CENTER, color='r', lw=0.7, ls='--', alpha=0.6)
        fig.suptitle(f'{label} — BG subtraction comparison\n'
                     f'v_arm = [{V_ARM_LO.value:.0f}, {V_ARM_HI.value:.0f}] km/s', fontsize=15)
        fig.tight_layout()
        fig.savefig(comp_png, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f'  Comparison PNG: {comp_png}')

    return simplebg_file, linearbg_file


# =============================================================================
# Load a cube, convert to km/s, spatially & spectrally subset, then extract
# =============================================================================
def load_and_extract_region(path, label, outbase):
    """
    Load *path*, cut to the filament spatial region and wide velocity range,
    then call extract_filament().
    """
    # Check existing outputs to avoid reloading the (often large) cube
    simplebg_file = f'{outbase}_simplebg.fits'
    linearbg_file = f'{outbase}_linearbg.fits'
    if os.path.exists(simplebg_file) and os.path.exists(linearbg_file):
        print(f'  [skip] {label}: FITS outputs already exist')
        for fn, sfx in [(simplebg_file, 'simple mean BG sub'),
                        (linearbg_file, 'linear-fit BG sub')]:
            png = fn.replace('.fits', '.png')
            if not os.path.exists(png):
                h = fits.getheader(fn)
                save_png(fits.getdata(fn).astype(float), h,
                         f'{label} — {sfx}', png)
        return simplebg_file, linearbg_file

    if not os.path.exists(path):
        print(f'  [skip] {label}: file not found — {path}')
        return None, None

    print(f'\n  Loading {os.path.basename(path)} ...')
    try:
        cube = SpectralCube.read(path, format='fits', use_dask=True)
    except Exception as exc:
        print(f'  [fail] {label}: read error: {exc}')
        return None, None

    # Convert spectral axis to km/s
    try:
        if cube.wcs.wcs.restfrq > 0:
            cube = cube.with_spectral_unit(
                u.km / u.s, velocity_convention='radio',
                rest_value=cube.wcs.wcs.restfrq * u.Hz)
        else:
            cube = cube.with_spectral_unit(
                u.km / u.s, velocity_convention='radio')
    except Exception as exc:
        print(f'  [warn] spectral conversion failed ({exc}); assuming already km/s')

    vmin_c = cube.spectral_axis.to(u.km/u.s).min().value
    vmax_c = cube.spectral_axis.to(u.km/u.s).max().value

    # Check spectral overlap
    if vmax_c < V_WIDE_LO.value or vmin_c > V_WIDE_HI.value:
        print(f'  [skip] {label}: spectral range [{vmin_c:.0f}, {vmax_c:.0f}] km/s '
              f'does not overlap extraction window '
              f'[{V_WIDE_LO.value:.0f}, {V_WIDE_HI.value:.0f}] km/s')
        return None, None

    # ---- Spectral slab (include BG region on both sides) ----------------------
    try:
        cube = cube.spectral_slab(V_WIDE_LO, V_WIDE_HI)
    except Exception as exc:
        print(f'  [warn] spectral_slab failed: {exc}')

    # ---- Spatial subcube to 30' region ----------------------------------------
    try:
        from astropy.coordinates import SkyCoord, Galactic
        centre = SkyCoord(l=L_CENTER * u.deg, b=B_CENTER * u.deg,
                          frame=Galactic)
        width  = (2 * HALF_SIZE) * u.deg
        cube = cube.subcube_from_regions(
            [centre.directional_offset_by(0 * u.deg, 0 * u.deg)],
        )
    except Exception:
        pass  # will try pixel-based crop below

    # Pixel-based spatial crop (more robust fallback)
    try:
        wcs_cel = cube.wcs.celestial
        ny_c, nx_c = cube.shape[1], cube.shape[2]
        # sample pixel coordinates for a grid
        yi, xi = np.mgrid[0:ny_c, 0:nx_c]
        sky = wcs_cel.pixel_to_world(xi.ravel(), yi.ravel())
        try:
            l_pix = sky.galactic.l.wrap_at(180 * u.deg).deg
            b_pix = sky.galactic.b.deg
        except AttributeError:
            l_pix = sky.l.wrap_at(180 * u.deg).deg
            b_pix = sky.b.deg
        in_region = ((l_pix >= L_LO) & (l_pix <= L_HI) &
                     (b_pix >= B_LO) & (b_pix <= B_HI))
        in_region_2d = in_region.reshape(ny_c, nx_c)
        rows  = np.where(in_region_2d.any(axis=1))[0]
        cols  = np.where(in_region_2d.any(axis=0))[0]
        if rows.size == 0 or cols.size == 0:
            print(f'  [skip] {label}: region l=[{L_LO:.2f},{L_HI:.2f}], '
                  f'b=[{B_LO:.2f},{B_HI:.2f}] not in cube footprint')
            return None, None
        r0, r1 = int(rows[0]),  int(rows[-1])
        c0, c1 = int(cols[0]),  int(cols[-1])
        print(f'  Spatial crop: rows {r0}:{r1}  cols {c0}:{c1}  '
              f'(from {ny_c}×{nx_c})')
        cube = cube[:, r0:r1+1, c0:c1+1]
    except Exception as exc:
        print(f'  [warn] spatial crop failed: {exc}; using full spatial extent')

    if cube.shape[1] == 0 or cube.shape[2] == 0:
        print(f'  [skip] {label}: empty spatial subcube after crop')
        return None, None

    return extract_filament(cube, label, outbase)


# =============================================================================
# Dataset inventory
# =============================================================================
ACES_CUBES    = sorted(glob.glob(
    '/orange/adamginsburg/ACES/mosaics/cubes/*_downsampled9.fits'))
CHIMPS_DIR    = '/orange/adamginsburg/cmz/CHIMPS'
chimps_cubes  = sorted([
    f for f in glob.glob(os.path.join(CHIMPS_DIR, '12CO_GC_*_mosaic.fits'))
    if '_arm' not in f and '_velocityslab' not in f
    and '_rewrite' not in f and '_hack' not in f and 'spectralcube' not in f
])
NOB_DIR       = '/orange/adamginsburg/cmz/nobeyama'
NOBEYAMA_CUBES = [
    ('12CO-2.BEARS.FITS',  'Nobeyama_BEARS_12CO'),
    ('12CO-2.S115Q.FITS',  'Nobeyama_S115Q_12CO'),
    ('13CO-2.FOREST.FITS', 'Nobeyama_FOREST_13CO'),
    ('13CO-2.S115Q.FITS',  'Nobeyama_S115Q_13CO'),
]
SEDIGISM_DIR  = '/orange/adamginsburg/galactic_plane_surveys/sedigism'
# G000 SEDIGISM tile covers l = -0.5 to 0.5 (approx)
SEDIGISM_G000 = os.path.join(SEDIGISM_DIR, 'G000_13CO21_Tmb_DR1.fits')
DAME_CUBE     = '/orange/adamginsburg/cmz/dameCO/DHT02_Center_interp_spectralcube.fits'

# =============================================================================
# Main
# =============================================================================
print('=' * 72)
print(f'{SLUG}')
print(f'Region : l=[{L_LO:.3f}, {L_HI:.3f}] deg, b=[{B_LO:.3f}, {B_HI:.3f}] deg')
print(f'Arm vel: [{V_ARM_LO:.0f}, {V_ARM_HI:.0f}]  '
      f'BG low: [{V_BG_LO_LO:.0f}, {V_BG_LO_HI:.0f}]  '
      f'BG high: [{V_BG_HI_LO:.0f}, {V_BG_HI_HI:.0f}]')
print(f'Output : {OUTDIR}')
print('=' * 72)

results = {}   # label -> (simplebg_file, linearbg_file)

# ------------------------------------------------------------------
# 1. Dame CO
# ------------------------------------------------------------------
print('\n--- 1. Dame CO (DHT02) ---')
ob = os.path.join(OUTDIR, 'DameCO_DHT02')
sb, lb = load_and_extract_region(DAME_CUBE, 'DameCO_DHT02', ob)
results['DameCO'] = (sb, lb)

# ------------------------------------------------------------------
# 2. SEDIGISM G000 13CO(2-1)
# ------------------------------------------------------------------
print('\n--- 2. SEDIGISM G000 13CO(2-1) ---')
ob = os.path.join(OUTDIR, 'SEDIGISM_G000')
sb, lb = load_and_extract_region(SEDIGISM_G000, 'SEDIGISM_G000', ob)
results['SEDIGISM_G000'] = (sb, lb)

# ------------------------------------------------------------------
# 3. CHIMPS 12CO(3-2)
# ------------------------------------------------------------------
print(f'\n--- 3. CHIMPS 12CO(3-2) ({len(chimps_cubes)} cubes) ---')
for fn in chimps_cubes:
    name = os.path.basename(fn).replace('.fits', '')
    ob   = os.path.join(OUTDIR, name)
    sb, lb = load_and_extract_region(fn, f'CHIMPS_{name}', ob)
    results[f'CHIMPS_{name}'] = (sb, lb)

# ------------------------------------------------------------------
# 4. Nobeyama
# ------------------------------------------------------------------
print('\n--- 4. Nobeyama ---')
for fname, lbl in NOBEYAMA_CUBES:
    path = os.path.join(NOB_DIR, fname)
    ob   = os.path.join(OUTDIR, lbl)
    sb, lb = load_and_extract_region(path, lbl, ob)
    results[lbl] = (sb, lb)

# ------------------------------------------------------------------
# 5. ACES cubes
# ------------------------------------------------------------------
print(f'\n--- 5. ACES ({len(ACES_CUBES)} cubes) ---')
for fn in ACES_CUBES:
    name = (os.path.basename(fn)
            .replace('_downsampled9.fits', '')
            .replace('_CubeMosaic', ''))
    ob   = os.path.join(OUTDIR, f'ACES_{name}')
    sb, lb = load_and_extract_region(fn, f'ACES_{name}', ob)
    results[f'ACES_{name}'] = (sb, lb)

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
print('\n' + '=' * 72)
print(f'Summary — {SLUG}')
print('=' * 72)
ok_simple  = [(k, v[0]) for k, v in results.items() if v[0] and os.path.exists(v[0])]
ok_linear  = [(k, v[1]) for k, v in results.items() if v[1] and os.path.exists(v[1])]
print(f'  Successful simple-BG maps : {len(ok_simple)}')
print(f'  Successful linear-BG maps : {len(ok_linear)}')
for k, f in ok_simple:
    print(f'    {k:40s}  {f}')

# ------------------------------------------------------------------
# Combined comparison figure (all datasets, 3 columns: velslab, simple, linear)
# ------------------------------------------------------------------
all_simplebg = [(k, v[0]) for k, v in results.items()
                if v[0] and os.path.exists(v[0])]
all_linearbg = [(k, v[1]) for k, v in results.items()
                if v[1] and os.path.exists(v[1])]

if all_simplebg:
    n_panels = len(all_simplebg)
    fig_w = 18.
    panel_h = max(fig_w / 2. * (B_HI - B_LO) / (L_HI - L_LO), 0.4)
    row_h   = panel_h + 0.5
    fig_h   = n_panels * row_h + 1.
    fig, axes = plt.subplots(n_panels, 2,
                             figsize=(fig_w, fig_h), squeeze=False)
    for row_idx, (k, s_file) in enumerate(all_simplebg):
        l_file = dict(all_linearbg).get(k)
        for col_idx, (fn, method) in enumerate(
                [(s_file, 'Simple mean BG'),
                 (l_file,  'Linear-fit BG')]):
            ax = axes[row_idx][col_idx]
            if fn is None or not os.path.exists(fn):
                ax.set_visible(False)
                continue
            h = fits.getheader(fn)
            d = fits.getdata(fn).astype(float)
            ny_t = h.get('NAXIS2', d.shape[0])
            nx_t = h.get('NAXIS1', d.shape[1])
            crpix1 = h.get('CRPIX1', nx_t / 2.)
            crval1 = h.get('CRVAL1', L_CENTER)
            cdelt1 = h.get('CDELT1', 1.)
            crpix2 = h.get('CRPIX2', ny_t / 2.)
            crval2 = h.get('CRVAL2', B_CENTER)
            cdelt2 = h.get('CDELT2', 1.)
            lon0_e = crval1 + (0      - (crpix1 - 1)) * cdelt1
            lon1_e = crval1 + (nx_t-1 - (crpix1 - 1)) * cdelt1
            lat0_e = crval2 + (0      - (crpix2 - 1)) * cdelt2
            lat1_e = crval2 + (ny_t-1 - (crpix2 - 1)) * cdelt2
            ext    = [lon0_e, lon1_e, lat0_e, lat1_e]
            finite = d[np.isfinite(d)]
            vlo = float(np.nanpercentile(finite, 1)) if finite.size > 1 else 0.
            vhi = float(np.nanpercentile(finite, 99)) if finite.size > 1 else 1.
            ax.imshow(d, origin='lower', aspect='auto', extent=ext,
                      vmin=vlo, vmax=vhi, cmap='gray_r')
            ax.set_title(f'{k}\n{method}', fontsize=14)
            ax.set_xlabel('l (deg)', fontsize=14)
            ax.set_ylabel('b (deg)', fontsize=14)
            ax.tick_params(labelsize=6)
            ax.axvline(L_CENTER, color='r', lw=0.5, ls='--', alpha=0.5)
            ax.axhline(B_CENTER, color='r', lw=0.5, ls='--', alpha=0.5)
    fig.suptitle(f'{SLUG}\n'
                 f'v=[{V_ARM_LO.value:.0f},{V_ARM_HI.value:.0f}] km/s  '
                 f'l=[{L_LO:.2f},{L_HI:.2f}]  b=[{B_LO:.2f},{B_HI:.2f}]',
                 fontsize=14)
    fig.tight_layout()
    summary_png = os.path.join(OUTDIR, f'{SLUG}_summary.png')
    fig.savefig(summary_png, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'\n  Summary PNG: {summary_png}')

print('\nDone.')
