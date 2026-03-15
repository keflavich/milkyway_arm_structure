"""
3kpc Arm Filament - instrument + JWST AV-map zoom-in inset figure
==================================================================
Layout
------
  Main panel : moment map (background-subtracted), 15' x 15' centered
               on l=0.34 deg, b=+0.024 deg (Galactic), with Galactic
               WCS axes.

  Right inset: JWST AV map (filament_av_map.fits) shown at full
               resolution with RA/Dec WCS axes, placed off to the right
               as a zoom-in off to the side.

               A rotated dashed polygon on the main panel marks the
               AV-map footprint; two ConnectionPatch lines connect the
               polygon corners to the inset axes.

Style follows keflavich/SgrB2_ALMA_3mm_Mosaic and
keflavich/OrionNotebooks inset_figures.py:
  inset_axes  -> mpl_toolkits.axes_grid1.inset_locator.inset_axes
  connector   -> matplotlib.patches.ConnectionPatch
  WCS inset   -> astropy.visualization.wcsaxes.WCSAxes

Produces figures for:
  CHIMPS 12CO(3-2)  - simple and linear BG sub
  Nobeyama BEARS 12CO - simple and linear BG sub
  Nobeyama FOREST 13CO - simple and linear BG sub
    ACES CS(2-1), SO(3_2-2_1), HNCO, H13CO+ - simple and linear BG sub

Outputs
-------
  chimps_av_inset_simple.png
  chimps_av_inset_linear.png
  nobeyama_bears_av_inset_simple.png
  ...etc
"""

import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import ConnectionPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord
import astropy.visualization.wcsaxes as wcsaxes_mod

warnings.filterwarnings("ignore")

BIG_FONTSIZE = 20
MIDDLE_FONTSIZE = 18
SMALL_FONTSIZE = 16

plt.rcParams.update({
    "figure.facecolor":   "w",
    "font.size":           18,
    "axes.labelsize":      18,
    "axes.titlesize":      17,
    "xtick.labelsize":     16,
    "ytick.labelsize":     16,
    "legend.fontsize":     16,
    "image.origin":       "lower",
    "image.interpolation": "nearest",
})

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
FILAMENT_DIR  = "/orange/adamginsburg/cmz/arms/3kpc_arm_filament_vscode/"
CHIMPS_SIMPLE = FILAMENT_DIR + "12CO_GC_359-000_mosaic_simplebg.fits"
CHIMPS_LINEAR = FILAMENT_DIR + "12CO_GC_359-000_mosaic_linearbg.fits"
BEARS_SIMPLE  = FILAMENT_DIR + "Nobeyama_BEARS_12CO_simplebg.fits"
BEARS_LINEAR  = FILAMENT_DIR + "Nobeyama_BEARS_12CO_linearbg.fits"
FOREST_SIMPLE = FILAMENT_DIR + "Nobeyama_FOREST_13CO_simplebg.fits"
FOREST_LINEAR = FILAMENT_DIR + "Nobeyama_FOREST_13CO_linearbg.fits"
ACES_CS_SIMPLE = FILAMENT_DIR + "ACES_CS21_simplebg.fits"
ACES_CS_LINEAR = FILAMENT_DIR + "ACES_CS21_linearbg.fits"
ACES_SO32_SIMPLE = FILAMENT_DIR + "ACES_SO32_simplebg.fits"
ACES_SO32_LINEAR = FILAMENT_DIR + "ACES_SO32_linearbg.fits"
ACES_HNCO_SIMPLE = FILAMENT_DIR + "ACES_HNCO_7m12mTP_simplebg.fits"
ACES_HNCO_LINEAR = FILAMENT_DIR + "ACES_HNCO_7m12mTP_linearbg.fits"
ACES_H13COP_SIMPLE = FILAMENT_DIR + "ACES_H13COp_simplebg.fits"
ACES_H13COP_LINEAR = FILAMENT_DIR + "ACES_H13COp_linearbg.fits"
AV_MAP        = "/orange/adamginsburg/jwst/cloudc/images/filament_av_map.fits"
OUTDIR        = FILAMENT_DIR

MAIN_VMAX_PERCENTILE = 99.90

# ---------------------------------------------------------------------------
# Region
# ---------------------------------------------------------------------------
CENTER_L  = 0.340   # Galactic longitude [deg]
CENTER_B  = 0.024   # Galactic latitude  [deg]
HALF_SIZE = 0.125   # half-size = 7.5 arcmin -> 15' total

# ---------------------------------------------------------------------------
# Load AV map (shared across all figures)
# ---------------------------------------------------------------------------
av_hdu  = fits.open(AV_MAP)[0]
av_data = av_hdu.data.squeeze().astype(float)
av_wcs  = WCS(av_hdu.header).celestial

# Pre-compute AV footprint polygon in Galactic sky coords (reused for all
# instruments; only the *pixel* projection into each instrument WCS changes)
av_ny, av_nx = av_data.shape
# 5 points: BL, BR, TR, TL, BL (closed polygon)
_corner_pix = np.array([[0, 0], [av_nx-1, 0], [av_nx-1, av_ny-1],
                         [0, av_ny-1], [0, 0]])   # shape (5, 2)
_av_sky = av_wcs.pixel_to_world(_corner_pix[:, 0], _corner_pix[:, 1])
AV_GAL  = _av_sky.galactic                          # SkyCoord (5,)
AV_L    = AV_GAL.l.wrap_at(180 * u.deg).deg         # (5,) GLON
AV_B    = AV_GAL.b.deg                               # (5,) GLAT


def load_and_crop(fits_path, half_size_deg=HALF_SIZE):
    """Load a 2-D moment FITS, crop to half_size_deg around the region
    centre, and return (cropped_data, cropped_wcs, original_header)."""
    hdu  = fits.open(fits_path)[0]
    data = hdu.data.squeeze().astype(float)
    wcs_full = WCS(hdu.header).celestial
    cen  = SkyCoord(l=CENTER_L * u.deg, b=CENTER_B * u.deg, frame="galactic")
    cpx  = wcs_full.world_to_pixel(cen)
    hp   = half_size_deg / abs(hdu.header["CDELT1"])
    x0 = max(int(round(float(cpx[0]) - hp)), 0)
    x1 = min(int(round(float(cpx[0]) + hp)), data.shape[1] - 1)
    y0 = max(int(round(float(cpx[1]) - hp)), 0)
    y1 = min(int(round(float(cpx[1]) + hp)), data.shape[0] - 1)
    return data[y0:y1+1, x0:x1+1], wcs_full[y0:y1+1, x0:x1+1], hdu.header


def make_figure(main_crop, main_wcs, main_header,
                instrument_label, line_label, bg_method_label, outbase,
                contour_data=None, contour_wcs=None, contour_levels=None):
    """
    Build and save one instrument + AV-inset figure.

    Parameters
    ----------
    main_crop        : 2-D ndarray  (background-subtracted moment map, cropped)
    main_wcs         : WCS          (celestial WCS for cropped image)
    main_header      : fits.Header  (original header for BUNIT etc.)
    instrument_label : str          e.g. 'CHIMPS', 'Nobeyama BEARS'
    line_label       : str          e.g. r'$^{12}$CO(3-2)'
    bg_method_label  : str          e.g. 'simple mean BG subtraction'
    outbase          : str          output path prefix (no extension)
    """
    # AV footprint in this instrument's WCS pixel space
    av_px_inst = np.array(main_wcs.world_to_pixel(AV_GAL)).T   # (5, 2)

    fig = plt.figure(figsize=(14, 7))

    # ---- main axes --------------------------------------------------------
    main_rect = [0.08, 0.12, 0.50, 0.80]
    ax = fig.add_axes(main_rect, projection=main_wcs)

    finite = main_crop[np.isfinite(main_crop)]
    vlo = float(np.nanpercentile(finite,  1)) if finite.size else 0.
    vhi = float(np.nanpercentile(finite, MAIN_VMAX_PERCENTILE)) if finite.size else 1.
    vhi = max(vhi, vlo + 1e-6)

    im = ax.imshow(main_crop,
                   origin="lower", aspect="equal",
                   interpolation="nearest",
                   vmin=vlo, vmax=vhi, cmap="gray_r")

    if contour_data is not None and contour_wcs is not None and contour_levels is not None:
        axlims = ax.axis()
        ax.contour(
            contour_data,
            levels=contour_levels,
            colors="cyan",
            linewidths=1.0,
            linestyle='--',
            alpha=0.9,
            transform=ax.get_transform(contour_wcs),
            zorder=8,
        )
        ax.axis(axlims)

    # Galactic WCS axis labels and ticks
    lon_c = ax.coords["glon"]
    lat_c = ax.coords["glat"]
    lon_c.set_axislabel("Galactic Longitude", fontsize=BIG_FONTSIZE)
    lat_c.set_axislabel("Galactic Latitude",  fontsize=BIG_FONTSIZE)
    lon_c.set_major_formatter("d.dd")
    lat_c.set_major_formatter("d.ddd")
    lon_c.set_ticks(spacing=4 * u.arcmin)
    lat_c.set_ticks(spacing=4 * u.arcmin)
    lon_c.display_minor_ticks(True)
    lat_c.display_minor_ticks(True)
    lon_c.set_ticklabel(exclude_overlapping=True)
    lat_c.set_ticklabel(exclude_overlapping=True)
    lon_c.ticklabels.set_fontsize(MIDDLE_FONTSIZE)
    lat_c.ticklabels.set_fontsize(MIDDLE_FONTSIZE)
    #ax.coords.grid(color="white", alpha=0.15, linestyle="--", linewidth=0.4)

    bunit = main_header.get("BUNIT", "K")
    ax.set_title(
        rf"{instrument_label} {line_label}",
        # r"  $|$  v = [$-$59, $-$50] km s$^{{-1}}$" + "\n"
        # + rf"{bg_method_label}  $|$  "
        # + rf"$l={CENTER_L:.3f}^\circ,\; b={CENTER_B:.3f}^\circ$"
        # + "  (\u00b17.5\u2032 FOV)",
        fontsize=BIG_FONTSIZE, pad=6)

    # Colorbar flush against the right edge of the main axes
    cb_gap  = 0.001   # gap between axes right edge and colorbar left edge
    cb_wid  = 0.018
    cax_m_rect = [main_rect[0] + main_rect[2] - cb_gap,
                  main_rect[1], cb_wid, main_rect[3]]
    # cax_m = fig.add_axes(cax_m_rect)
    # cb_m  = fig.colorbar(im, cax=cax_m)
    # cb_m.set_label(rf"Mean $T_{{\rm mb}}$ [{bunit}]", fontsize=MIDDLE_FONTSIZE)
    # cb_m.ax.tick_params(labelsize=SMALL_FONTSIZE)

    # ---- AV footprint polygon on main panel --------------------------------
    # ax.get_transform("world") maps (GLON, GLAT) -> display coords
    poly_tr = ax.get_transform("world")
    poly = mpatches.Polygon(
        np.column_stack([AV_L, AV_B]),
        closed=True,
        transform=poly_tr,
        fill=False,
        edgecolor="C1", linewidth=2.0, linestyle="--",
        zorder=5,
    )
    ax.add_patch(poly)

    # Label near the box
    top_b  = float(np.max(AV_B[:4]))
    cen_l  = float(np.mean(AV_L[:4]))
    # ax.annotate(
    #     "JWST $A_V$",
    #     xy=(cen_l, top_b),
    #     xycoords=poly_tr,
    #     fontsize=14, color="C1",
    #     ha="center", va="bottom",
    #     xytext=(0, 5), textcoords="offset points",
    # )

    # ---- inset axes (right side) -------------------------------------------
    # Place tight against the main panel, and keep the inset colorbar
    # immediately adjacent to the inset instead of at a fixed far-right slot.
    ins_left = cax_m_rect[0] + cax_m_rect[2] - 0.23
    cb_gap_av = 0.003
    inset_right = 0.965
    ins_width = inset_right - ins_left - cb_gap_av - cb_wid
    inset_rect = [ins_left, main_rect[1] + 0.015, ins_width, main_rect[3] + 0.008]

    axins = inset_axes(
        ax,
        width="100%", height="100%",
        loc="upper left",
        bbox_to_anchor=inset_rect,
        bbox_transform=fig.transFigure,
        axes_class=wcsaxes_mod.WCSAxes,
        axes_kwargs=dict(wcs=av_wcs),
    )

    av_finite = av_data[np.isfinite(av_data)]
    av_vlo = float(np.nanpercentile(av_finite,  1)) if av_finite.size else 0.
    av_vhi = float(np.nanpercentile(av_finite, 99)) if av_finite.size else 1.

    im_ins = axins.imshow(av_data,
                          origin="lower", aspect="equal",
                          interpolation="nearest",
                          vmin=av_vlo, vmax=av_vhi,
                          cmap="magma_r")

    # RA/Dec labels on inset
    ra_a  = axins.coords["ra"]
    dec_a = axins.coords["dec"]
    ra_a.set_axislabel("")
    dec_a.set_axislabel("")
    ra_a.set_major_formatter("hh:mm:ss")
    dec_a.set_major_formatter("dd:mm:ss")
    ra_a.set_ticks(spacing=0.5 * u.arcmin)
    dec_a.set_ticks(spacing=0.5 * u.arcmin)
    ra_a.ticklabels.set_fontsize(SMALL_FONTSIZE)
    dec_a.ticklabels.set_fontsize(SMALL_FONTSIZE)
    ra_a.set_ticklabel_visible(False)
    dec_a.set_ticklabel_visible(False)
    #axins.coords.grid(color="white", alpha=0.15,
    #                  linestyle="--", linewidth=0.3)

    if contour_data is not None and contour_wcs is not None and contour_levels is not None:
        axins.contour(
            contour_data,
            levels=contour_levels,
            colors="cyan",
            linewidths=2.2,
            alpha=0.9,
            transform=axins.get_transform(contour_wcs),
            zorder=8,
        )

    axins.set_title("JWST $A_V$ (zoom-in)", fontsize=BIG_FONTSIZE, pad=3)

    #fig.text(inset_rect[0], inset_rect[1] + inset_rect[3] + 0.003,
    #         "(b) JWST $A_V$ map", fontsize=14, va="bottom", ha="left",
    #         style="italic")

    # ---- Connector lines ---------------------------------------------------
    # Force layout so bbox coords are populated
    fig.canvas.draw()

    # AV colorbar immediately to the right of inset, still inside the figure
    cax_i_rect = [inset_rect[0] + inset_rect[2] - 0.155,
                  inset_rect[1] - 0.015, cb_wid, inset_rect[3] - 0.004]
    cax_i = fig.add_axes(cax_i_rect)
    cb_i  = fig.colorbar(im_ins, cax=cax_i)
    cb_i.set_label("$A_V$ [mag]", fontsize=MIDDLE_FONTSIZE)
    cb_i.ax.tick_params(labelsize=SMALL_FONTSIZE)

    # Connect two diagonal corners of the AV footprint on the main panel
    # to the actual left-side corners of the inset axes:
    #   corner 2 (upper area of footprint) -> top-left of inset axes
    #   corner 0 (lower area of footprint) -> bottom-left of inset axes
    connector_spec = [
        (2, (0, 1)),  # upper-left inset corner
        (0, (0, 0)),  # lower-left inset corner
    ]
    for corner_idx, inset_corner in connector_spec:
        cx = float(av_px_inst[corner_idx, 0])
        cy = float(av_px_inst[corner_idx, 1])
        con = ConnectionPatch(
            xyA=(cx, cy),
            xyB=inset_corner,
            coordsA="data",
            coordsB="axes fraction",
            axesA=ax,
            axesB=axins,
            color="C1",
            linewidth=1.2,
            linestyle="-",
            alpha=0.75,
            zorder=10,
            arrowstyle="-",
        )
        fig.add_artist(con)

    # ---- Save --------------------------------------------------------------
    fig.canvas.draw()
    for ext in ("png", ):# "pdf"):
        path = f"{outbase}.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


# ============================================================================
# Run for all instruments and both BG methods
# ============================================================================
datasets = [
    {
        "simple": CHIMPS_SIMPLE,
        "linear": CHIMPS_LINEAR,
        "instrument": "CHIMPS",
        "line": r"$^{12}$CO(3$-$2)",
        "prefix": "chimps",
    },
    {
        "simple": BEARS_SIMPLE,
        "linear": BEARS_LINEAR,
        "instrument": "Nobeyama BEARS",
        "line": r"$^{12}$CO(1$-$0)",
        "prefix": "nobeyama_bears",
    },
    {
        "simple": FOREST_SIMPLE,
        "linear": FOREST_LINEAR,
        "instrument": "Nobeyama FOREST",
        "line": r"$^{13}$CO(1$-$0)",
        "prefix": "nobeyama_forest",
    },
    {
        "simple": ACES_CS_SIMPLE,
        "linear": ACES_CS_LINEAR,
        "instrument": "ACES",
        "line": r"CS(2$-$1)",
        "prefix": "aces_cs21",
    },
    {
        "simple": ACES_SO32_SIMPLE,
        "linear": ACES_SO32_LINEAR,
        "instrument": "ACES",
        "line": r"SO(3$_2$$-$2$_1$)",
        "prefix": "aces_so32",
    },
    {
        "simple": ACES_HNCO_SIMPLE,
        "linear": ACES_HNCO_LINEAR,
        "instrument": "ACES",
        "line": r"HNCO",
        "prefix": "aces_hnco",
    },
    {
        "simple": ACES_H13COP_SIMPLE,
        "linear": ACES_H13COP_LINEAR,
        "instrument": "ACES",
        "line": r"H$^{13}$CO$^+$",
        "prefix": "aces_h13cop",
    },
]

for ds in datasets:
    for bg_key, bg_label, suffix in [
            ("simple", "simple mean BG subtraction",  "simple"),
            ("linear", "linear-fit BG subtraction",   "linear"),
    ]:
        fits_path = ds[bg_key]
        outbase   = OUTDIR + f"{ds['prefix']}_av_inset_{suffix}"
        print(f"Building {ds['instrument']} {bg_label} figure ...")
        crop, wcs_crop, hdr = load_and_crop(fits_path)
        make_figure(
            main_crop=crop,
            main_wcs=wcs_crop,
            main_header=hdr,
            instrument_label=ds["instrument"],
            line_label=ds["line"],
            bg_method_label=bg_label,
            outbase=outbase,
        )

print("Building Nobeyama BEARS + ACES H13CO+ contour-overlay figure ...")
bears_crop, bears_wcs, bears_hdr = load_and_crop(BEARS_SIMPLE)
h13cop_crop, h13cop_wcs, _ = load_and_crop(ACES_H13COP_SIMPLE, half_size_deg=0.05)
h13cop_finite = h13cop_crop[np.isfinite(h13cop_crop)]
h13cop_pos = h13cop_finite[h13cop_finite > 0]
if h13cop_pos.size > 10:
    h13cop_levels = np.nanpercentile(h13cop_pos, [95, ])
else:
    h13cop_levels = np.nanpercentile(h13cop_finite, [95, ])

make_figure(
    main_crop=bears_crop,
    main_wcs=bears_wcs,
    main_header=bears_hdr,
    instrument_label="Nobeyama BEARS",
    line_label=r"$^{12}$CO(1$-$0)",
    bg_method_label="simple mean BG subtraction",
    outbase=OUTDIR + "nobeyama_bears_av_inset_h13cop_contours",
    contour_data=h13cop_crop,
    contour_wcs=h13cop_wcs,
    contour_levels=h13cop_levels,
)

print("Done.")
