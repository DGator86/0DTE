"""
execution package
=================
Empirical execution records and (later) fill models for Prediction Engine V3
Part 3. Existing deterministic priors remain in prediction.models.fill /
execution_cost.py; new empirical provenance lives here.

NOT financial advice.
"""
from execution.fill_records import (
    FILL_RECORD_VERSION,
    FillRecord,
    fill_fraction,
    validate_fill_record,
)

__all__ = [
    "FILL_RECORD_VERSION",
    "FillRecord",
    "fill_fraction",
    "validate_fill_record",
]
