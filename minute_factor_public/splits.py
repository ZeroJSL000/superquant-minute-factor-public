"""Chronological splits that keep discovery separate from held-out scoring."""

import numpy as np


def chronological_split(days: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use 60/20/20 of labeled signal days in chronological order."""
    labeled_days = days - 1
    if labeled_days < 10:
        raise ValueError("at least eleven days are needed for a full split")
    train_end = int(labeled_days * 0.6)
    validation_end = int(labeled_days * 0.8)
    return (
        np.arange(0, train_end),
        np.arange(train_end, validation_end),
        np.arange(validation_end, labeled_days),
    )


def walk_forward_splits(days: int, year_days: int) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Fit on five sixths of one synthetic year and test on the next year."""
    if year_days < 6:
        raise ValueError("year_days must be at least six")
    complete_years = (days - 1) // year_days
    if complete_years < 2:
        raise ValueError("at least two complete labeled synthetic years are needed")
    output = []
    for year in range(complete_years - 1):
        start = year * year_days
        cutoff = start + year_days * 5 // 6
        next_start = start + year_days
        output.append(
            (
                np.arange(start, cutoff),
                np.arange(cutoff, next_start),
                np.arange(next_start, next_start + year_days),
            )
        )
    return output
