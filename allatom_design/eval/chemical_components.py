def normalize_ccd_code(code) -> str:
    """Normalize a CCD code / element symbol for membership and key comparison.

    Uppercases and strips whitespace, matching the convention of
    ``allatom_design.data.const.METAL_ELEMENTS`` (which is stored uppercase).
    Do NOT use this for human-readable element symbols (e.g. "Fe"); see
    ``allatom_design.utils.sample_io_utils._normalize_element_symbol`` for that.
    """
    return str(code).strip().upper()
