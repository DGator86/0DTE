"""Autonomous Grok 4.5 paper-trading integration.

The package is deliberately paper-only.  It consumes the framework's current
``TickResult`` and in-memory market snapshot, blinds Legacy/V2/V3 policy
outputs, lets Grok inspect evidence through local tools, and emits at most one
validated ``paper_intent`` for the existing :mod:`paper_broker`.
"""

from .config import GrokConfig
from .integration import GrokCoordinator, register_grok_track

__all__ = ["GrokConfig", "GrokCoordinator", "register_grok_track"]
