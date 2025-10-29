"""AuroraML Python package initializer.

Loads the compiled C++ extension from the local build directory and exposes
its submodules under the package namespace for convenient imports like:

    import auroraml
    auroraml.linear_model.LinearRegression
"""

import os
import sys
import glob
import importlib.util
import importlib.machinery

__version__ = "0.1.0"

_repo_root = os.path.dirname(os.path.dirname(__file__))
_build_dir = os.path.join(_repo_root, "build")

def _load_extension_from_build():
    if not os.path.isdir(_build_dir):
        return None
    # Find the compiled shared object for this platform
    candidates = []
    # Common filename patterns for CPython extension modules
    patterns = [
        os.path.join(_build_dir, "auroraml.*.so"),
        os.path.join(_build_dir, "auroraml*.pyd"),
        os.path.join(_build_dir, "auroraml*.dll"),
        os.path.join(_build_dir, "auroraml*.dylib"),
    ]
    for pat in patterns:
        candidates.extend(glob.glob(pat))
    if not candidates:
        return None
    # Prefer the first match
    so_path = candidates[0]
    # The extension exports PyInit_auroraml, so the module name must be 'auroraml'
    loader = importlib.machinery.ExtensionFileLoader("auroraml", so_path)
    spec = importlib.util.spec_from_loader("auroraml", loader)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod

# Try to load compiled extension and re-export its symbols
_ext = _load_extension_from_build()
if _ext is not None:
    globals().update({k: v for k, v in vars(_ext).items() if not k.startswith("__")})
    # for introspection
    __all__ = [k for k in vars(_ext).keys() if not k.startswith("__")]
else:
    # Fallback: user must ensure sys.path contains build dir before importing
    pass
