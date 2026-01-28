def math_round(x: float) -> int:
    """
    Round a number to the nearest integer using traditional rounding (0.5 rounds up).

    Args:
        x: The number to round.

    Returns:
        The rounded integer.
    """
    return int(x + 0.5) if x > 0 else int(x - 0.5)
