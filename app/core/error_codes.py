"""
Structured error codes for placement and run failures.
Use these keys in return values; map to user-facing messages in the UI.
"""

# Known error keys (returned e.g. from _run_placement)
SAFE_POLY_EMPTY = "safe_poly_empty"
NO_FEASIBLE_CANDIDATE = "no_feasible_candidate"
RUN_FAILED = "run_failed"

# User-facing messages (short, actionable)
USER_MESSAGES: dict[str, str] = {
    SAFE_POLY_EMPTY: "Safe polygon became empty at this padding. Try reducing padding or font size.",
    NO_FEASIBLE_CANDIDATE: "No feasible placement found inside the river. Try reducing padding or font size.",
    RUN_FAILED: "Run failed. Check geometry and inputs.",
}


def user_message(error_key: str | None, fallback: str = "Something went wrong.") -> str:
    """Return a user-facing message for the given error key."""
    if not error_key:
        return fallback
    return USER_MESSAGES.get(error_key, fallback)
