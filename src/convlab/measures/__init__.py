"""The measure catalogue.

Importing this package registers every measure. The submodules are imported
for their decorator side effects, so the order below is the order families
appear in the codebook.
"""

from convlab.measures.base import (
    DYAD_LEVEL,
    PERSON_LEVEL,
    MeasureRegistry,
    MeasureSpec,
    MeasureValue,
    measure,
    registry,
)

# Registration side effects. Kept explicit rather than auto-discovered so
# that a typo in a module name fails loudly at import instead of silently
# producing a smaller catalogue.
from convlab.measures import turntaking as _turntaking  # noqa: F401
from convlab.measures import interruption as _interruption  # noqa: F401
from convlab.measures import backchannel as _backchannel  # noqa: F401
from convlab.measures import lexical as _lexical  # noqa: F401
from convlab.measures import prosodic as _prosodic  # noqa: F401
from convlab.measures import semantic as _semantic  # noqa: F401
from convlab.measures import visual as _visual  # noqa: F401
from convlab.measures import affect as _affect  # noqa: F401
from convlab.measures import laughter as _laughter  # noqa: F401
from convlab.measures import synchrony as _synchrony  # noqa: F401
from convlab.measures import dynamics as _dynamics  # noqa: F401

__all__ = [
    "registry",
    "measure",
    "MeasureRegistry",
    "MeasureSpec",
    "MeasureValue",
    "PERSON_LEVEL",
    "DYAD_LEVEL",
]
