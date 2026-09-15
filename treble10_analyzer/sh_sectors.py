import numpy as np
import spaudiopy as spa

MAX_T_DESIGN_DEGREE = 21  # spa.grids.load_t_design only has designs up to this degree


def build_sector_matrix(sh_order):
    """SH -> spatial-sector matrix, following `sector_edc_loss.sh2sec` in loss.py.

    Uses the smallest t-design valid for `sh_order`, i.e. degree 2*sh_order
    (the minimum required for exact SH quadrature over the sector grid).
    """
    t_design_order = 2 * sh_order
    if t_design_order > MAX_T_DESIGN_DEGREE:
        raise ValueError(
            f"sh_order={sh_order} needs a t-design of degree {t_design_order}, but only "
            f"degrees up to {MAX_T_DESIGN_DEGREE} are available (sh_order <= {MAX_T_DESIGN_DEGREE // 2})."
        )
    sec_azi, sec_zen, _ = spa.utils.cart2sph(*spa.grids.load_t_design(t_design_order).T)
    c_n = spa.sph.maxre_modal_weights(sh_order)
    a_sh2sec, _ = spa.sph.design_sph_filterbank(sh_order, sec_azi, sec_zen, c_n, mode="perfect")
    return a_sh2sec  # (n_sectors, n_sh_channels)


def resolve_sh_order(n_hoa, requested_sh_order):
    """The recording's native ambisonics order, and the order to actually use for sectors.

    Passing a smaller `requested_sh_order` truncates the ACN/SH channels before
    the sector beamformer, trading spatial resolution for a coarser (and,
    for the Bayesian analysis, much cheaper) sector decomposition.
    """
    native_sh_order = round(np.sqrt(n_hoa)) - 1
    if requested_sh_order is None:
        return native_sh_order, native_sh_order
    if requested_sh_order < 1:
        raise ValueError(f"sh_order must be at least 1, got {requested_sh_order}.")
    if requested_sh_order > native_sh_order:
        raise ValueError(
            f"Requested sh_order={requested_sh_order} exceeds the recording's native order "
            f"{native_sh_order} ({n_hoa} HOA channels)."
        )
    return native_sh_order, requested_sh_order
