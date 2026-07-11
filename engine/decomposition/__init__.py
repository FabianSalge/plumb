"""Claim decomposition (ADR-0009): rule-based sentence segmentation plus the
token-to-claim reduction off one whole-answer pass. The public surface is this
package — `segmentation` and `reduction` are separately testable halves."""

from engine.decomposition.reduction import ScoredClaim, Span, decompose, reduce_claim
from engine.decomposition.segmentation import Claim, DecompositionError, segment

# The claim unit's identity, bound into every calibration artifact (ADR-0008).
# Bump it whenever segmentation rules or the token-to-claim reduction change —
# the golden segmentation tests changing in the same diff is the reviewer's
# signal — so a calibrator fitted to the old unit refuses to serve.
CLAIM_UNIT = "sentence-maxrisk-v1"

__all__ = [
    "CLAIM_UNIT",
    "Claim",
    "DecompositionError",
    "ScoredClaim",
    "Span",
    "decompose",
    "reduce_claim",
    "segment",
]
