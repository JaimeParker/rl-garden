"""Official FQL/OGBench baseline launcher.

The launcher executes the unmodified upstream JAX ``main.py`` process.  It
does not include cross-implementation parity, checkpoint evaluation, or paper
result aggregation; those research tools are outside this package's scope.
"""

from .presets import FQLPreset, PRESETS

__all__ = ["FQLPreset", "PRESETS"]
