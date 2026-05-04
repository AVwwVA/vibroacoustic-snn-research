from __future__ import annotations
import os
from pathlib import Path

def configure_headless_matplotlib() -> None:
    root = Path.cwd()
    mpl_dir = root / '.mplconfig'
    cache_dir = root / '.cache'
    font_dir = cache_dir / 'fontconfig'
    mpl_dir.mkdir(parents=True, exist_ok=True)
    font_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(mpl_dir))
    os.environ.setdefault('XDG_CACHE_HOME', str(cache_dir))
    import matplotlib
    matplotlib.use('Agg')
