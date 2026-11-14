"""Custom signals for ``qts signal test`` (the extensible part of the registry).

A signal is a Python function with the feature signature ``fn(md: MarketData) ->
Wide``: a date x ticker panel whose row ``t`` uses only rows of ``md`` dated ``<= t``
(trailing windows, ``shift(k >= 0)``; never ``shift(-k)``).  Register it with
:func:`qts.research.signals.register_signal` (as a decorator) here, or in your own
file passed with ``qts signal test NAME --signal-module path/to/file.py``::

    from qts.research.signals import register_signal


    @register_signal("rev_5d", description="minus the 5-day return", sign=1.0)
    def rev_5d(md):
        return -(md.close / md.close.shift(5) - 1.0)

``sign``: the direction you expect to be rewarded (+1: high value -> high return); the
reported IC is of ``sign * signal``.  ``industry_demean=True`` demeans it within
industry before the rank-Gaussian transform, like ``AlphaConfig.industry_demean``.
The test processes it exactly as a built-in feature (winsorise -> [industry demean]
-> rank-Gaussian, eligible names only).  A custom signal is a test of the signal, not a
model input: to trade it, add it to ``qts/alpha/features.py`` (a new feature set).
"""

from __future__ import annotations

from qts.core.types import MarketData, Wide
from qts.research.signals import register_signal


@register_signal(
    "rev_5d",
    description="example custom signal: minus the 5-day return (weekly reversal)",
    sign=1.0,
)
def rev_5d(md: MarketData) -> Wide:
    return -(md.close / md.close.shift(5) - 1.0)
