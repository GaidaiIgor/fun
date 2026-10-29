"""Solves CodinGame The Resistance by counting the dictionary word sequences whose concatenated Morse encodings form the given sequence."""

from collections import Counter
from string import ascii_uppercase
import sys

MORSE = str.maketrans(dict(zip(ascii_uppercase, ".- -... -.-. -.. . ..-. --. .... .. .--- -.- .-.. -- -. --- .--. --.- .-. ... - ..- ...- .-- -..- -.-- --.."
                               .translate(str.maketrans(".-", "01")).split())))


def main():
    """Reads the sequence and the dictionary and keys every Morse string by the binary number formed by a leading 1 followed by the string
    with dots as 0 and dashes as 1, so that appending a symbol doubles the key and adds the symbol bit. counts maps the encoding keys
    to the numbers of words encoded so, and trie maps the keys of all their prefixes to the same numbers, zero for proper prefixes only.
    Then counts in ways the messages forming every suffix of the sequence, walking the trie from each suffix start."""
    sequence, _, *words = sys.stdin.read().split()
    counts = Counter(int("1" + word.translate(MORSE), 2) for word in words)
    trie = dict.fromkeys((code >> shift for code in counts for shift in range(1, code.bit_length())), 0) | counts
    ways = [0] * len(sequence) + [1]
    for start in reversed(range(len(sequence))):
        node = 1
        for end in range(start, len(sequence)):
            if (count := trie.get(node := 2 * node + (sequence[end] == "-"))) is None:
                break
            ways[start] += count * ways[end + 1]
    print(ways[0])


if __name__ == "__main__":
    main()
