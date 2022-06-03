__version__ = '0.1.0'


try:
    import lal
    import lalsimulation
except ModuleNotFoundError:
    import os
    os.warn("""
        lalsuite has to be installed from
        https://git.ligo.org/deep.chatterjee/lalsuite/-/tree/taylorf2-ppe.
        Either install it using pip or build it from source in the
        working environment.
    """)