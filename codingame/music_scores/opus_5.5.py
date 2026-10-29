"""Solves CodinGame Music Scores by locating the staff lines, separating the notes as connected ink components once the staff and ledger
lines are erased outside the notes, and reading the pitch from the height of every note head and the duration from its filling."""

import numpy as np
from scipy.ndimage import binary_opening, find_objects, label

PITCHES = "CDEFGABCDEFG"


def main():
    """Decodes the image and takes the most frequent column pattern holding black pixels, that of the columns crossing only the staff,
    as the staff rows, whose five lines give the bottom line center and the spacing between line centers. The staff lines and the ledger
    line of the lower C, lying one spacing below the bottom line, are erased except where the pixels right above and below them are both
    black, so that the 8-connected components of the remaining ink are the notes. Pixels of vertical black runs at least 1.5 spacings
    long belong to stems, and the remaining ink of a note is its head. The middle of the vertical extent of the head gives the pitch,
    and the middle of the horizontal extent of the note across the head rows, where the stem lies flush with the head side, completes
    the head center. White pixels within a fifth of the spacing from the head center, where hollow heads erase the lines too, make
    the note a half note."""
    width, height = map(int, input().split())
    tokens = input().split()
    image = np.repeat(np.array(tokens[::2]) == "B", np.array(tokens[1::2], dtype=int)).reshape(height, width)
    patterns, repeats = np.unique(image.T, axis=0, return_counts=True)
    lines = np.flatnonzero(np.diff(patterns[(repeats * patterns.any(axis=1)).argmax()], prepend=False, append=False)).reshape(-1, 2)
    centers = lines.mean(axis=1) - 0.5
    spacing, ink = (centers[-1] - centers[0]) / 4, image.copy()
    for top, bottom in (*lines, lines[-1] + round(spacing)):
        ink[top:bottom] = image[top - 1] & image[bottom]
    stem = binary_opening(image, np.ones((2 * round(0.75 * spacing) + 1, 1)))
    labels, radius, notes = label(ink, np.ones((3, 3)))[0], spacing / 5, []
    for k, box in enumerate(find_objects(labels), 1):
        part = labels[box] == k
        ys = np.flatnonzero((part & ~stem[box]).any(axis=1))
        xs = np.flatnonzero(part[ys[0]:ys[-1] + 1].any(axis=0))
        y, x = box[0].start + (ys[0] + ys[-1]) / 2, box[1].start + (xs[0] + xs[-1]) / 2
        filled = image[round(y - radius):round(y + radius) + 1, round(x - radius):round(x + radius) + 1].mean() > 0.9
        notes.append((x, PITCHES[round((centers[-1] - y) / spacing * 2) + 2] + ("Q" if filled else "H")))
    print(" ".join(note for _, note in sorted(notes)))


if __name__ == "__main__":
    main()
