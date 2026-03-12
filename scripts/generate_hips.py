"""
generate_hips.py
================
Convert CHIMPS and SEDIGISM arm background-subtracted FITS mosaics into
local HiPS tile sets for use with Aladin Lite.

Uses reproject.hips.reproject_to_hips for correct HiPS tile generation.
HiPS tile sets are written to:
    /orange/adamginsburg/cmz/arms/hips/<survey_name>/

Usage:
    python generate_hips.py                          # all surveys
    python generate_hips.py CHIMPS_near3kpc          # specific survey(s)
    python generate_hips.py --force CHIMPS_near3kpc  # delete and regenerate
"""

import os
import sys
import shutil
import warnings
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from reproject import reproject_interp
from reproject.hips import reproject_to_hips

warnings.filterwarnings('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', message='.*RADECSYS.*')
warnings.filterwarnings('ignore', message='.*obsfix.*')
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Level 7 → sub-pixel ≈ 8 arcsec; SEDIGISM native ≈ 9 arcsec
MAX_ORDER = 7

ARM_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIPS_ROOT = os.path.join(ARM_DIR, 'hips')

# Ordered list of surveys to generate
SURVEYS = [
    {
        'name':  'CHIMPS_near3kpc',
        'title': 'CHIMPS 12CO(3-2) Near-3kpc Arm',
        'fits':  os.path.join(ARM_DIR, '3kpc_arm',  'CHIMPS_arm_bgsub_mosaic.fits'),
    },
    {
        'name':  'CHIMPS_18kms',
        'title': 'CHIMPS 12CO(3-2) 18 km/s Arm',
        'fits':  os.path.join(ARM_DIR, '18kms_arm', 'CHIMPS_18kms_arm_backgroundsub_mosaic.fits'),
    },
    {
        'name':  'CHIMPS_local',
        'title': 'CHIMPS 12CO(3-2) Local Arm',
        'fits':  os.path.join(ARM_DIR, 'local_arm', 'CHIMPS_arm_bgsub_mosaic.fits'),
    },
    {
        'name':  'CHIMPS_norma',
        'title': 'CHIMPS 12CO(3-2) Norma Arm',
        'fits':  os.path.join(ARM_DIR, 'norma_arm', 'CHIMPS_arm_bgsub_mosaic.fits'),
    },
    {
        'name':  'SEDIGISM_near3kpc',
        'title': 'SEDIGISM 13CO(2-1) Near-3kpc Arm',
        'fits':  os.path.join(ARM_DIR, '3kpc_arm',  'SEDIGISM_arm_bgsub_mosaic.fits'),
    },
    {
        'name':  'SEDIGISM_18kms',
        'title': 'SEDIGISM 13CO(2-1) 18 km/s Arm',
        'fits':  os.path.join(ARM_DIR, '18kms_arm', 'SEDIGISM_18kms_arm_backgroundsub_mosaic.fits'),
    },
    {
        'name':  'SEDIGISM_local',
        'title': 'SEDIGISM 13CO(2-1) Local Arm',
        'fits':  os.path.join(ARM_DIR, 'local_arm', 'SEDIGISM_arm_bgsub_mosaic.fits'),
    },
    {
        'name':  'SEDIGISM_norma',
        'title': 'SEDIGISM 13CO(2-1) Norma Arm',
        'fits':  os.path.join(ARM_DIR, 'norma_arm', 'SEDIGISM_arm_bgsub_mosaic.fits'),
    },
]


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------
def generate_hips(survey, force=False):
    name      = survey['name']
    title     = survey['title']
    fits_path = survey['fits']
    hips_dir  = os.path.join(HIPS_ROOT, name)

    print(f'\n=== {name} ===')

    if not os.path.exists(fits_path):
        print(f'  FITS file missing: {fits_path}')
        return

    # Skip if already done (unless --force)
    if os.path.isdir(hips_dir):
        if not force:
            print(f'  Already exists – skipping (use --force to regenerate)')
            return
        print(f'  Removing existing directory: {hips_dir}')
        shutil.rmtree(hips_dir)

    # --- Read FITS ---
    with fits.open(fits_path) as hdul:
        data = hdul[0].data.copy()
        hdr  = hdul[0].header.copy()

    # Collapse degenerate axes
    while data.ndim > 2:
        data = data[0]

    data = data.astype(np.float32)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        wcs_in = WCS(hdr, naxis=2)

    data_hdu = fits.PrimaryHDU(data=data, header=wcs_in.to_header())
    print(f'  Data shape: {data.shape}')

    # --- Generate HiPS using reproject ---
    reproject_to_hips(
        data_hdu,
        output_directory=hips_dir,
        coord_system_out='galactic',
        reproject_function=reproject_interp,
        level=MAX_ORDER,
        threads=True,
        properties={
            'obs_title':       title,
            'creator_did':     f'ivo://cmz.arms/hips/{name}',
            'obs_description': (
                f'Background-subtracted galactic arm mosaic from {title}'
            ),
        },
    )

    print(f'  Done → {hips_dir}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    args = sys.argv[1:]
    force = '--force' in args
    names_requested = [a for a in args if not a.startswith('--')]
    names_requested = set(names_requested) if names_requested else None

    for survey in SURVEYS:
        if names_requested and survey['name'] not in names_requested:
            continue
        generate_hips(survey, force=force)

    print('\nAll HiPS surveys generated.')
    print(f'Tiles written to: {HIPS_ROOT}')
    print()
    print('To view, start a local HTTP server:')
    print(f'  cd {ARM_DIR} && python -m http.server 8765')
    print('Then open: http://localhost:8765/index.html')

