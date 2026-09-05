"""Rewrites the CodinGame arena logs of a directory in place, turning each turn into a block that always starts with the own bot's streams."""
from collections import Counter
from pathlib import Path

BOT_NAME = "IGPro"
ERROR, OUTPUT, SUMMARY = "Standard Error Stream:", "Standard Output Stream:", "Game Summary:"

def main(path: str):
    """Reformats the arena logs of a directory, overwriting each of them with its reformatted content.
    :param path: directory holding the log1.txt to log10.txt logs to reformat"""
    for log_path in (Path(path) / f"log{index}.txt" for index in range(1, 11)):
        log_path.write_text(reformat(log_path.read_text(encoding="utf-8")), encoding="utf-8", newline="\n")

def reformat(text: str) -> str:
    """Converts a raw log into the new format.
    :param text: content of the raw log, where each turn holds player 1's output stream followed by the turn number and the turn total, then player 2's output stream, then an optional game summary, with the own bot's error stream sitting right before the own bot's output stream
    :return: content of the log in the new format, where each turn starts with a "Turn X/Y:" header, lists the own bot's error and output streams, the opponent's output stream and the game summary if any, and ends with an empty line"""
    blocks = []
    for line in text.splitlines():
        if line in (ERROR, OUTPUT, SUMMARY):
            blocks.append((line, []))
        else:
            blocks[-1][1].append(line)

    names = Counter(line.split()[0] for marker, content in blocks if marker == SUMMARY for line in content)
    opponent = next(name for name, _ in names.most_common() if name != BOT_NAME)
    bot_is_first = blocks[0][0] == ERROR
    starts = [index - bot_is_first for index, (marker, content) in enumerate(blocks) if marker == OUTPUT and len(content) > 2]

    result = []
    for start, end in zip(starts, starts[1:] + [len(blocks)]):
        chunk = blocks[start:end]
        outputs = [content for marker, content in chunk if marker == OUTPUT]
        number, total = outputs[0][-2:]
        outputs[0] = outputs[0][:-2]
        own_output, opponent_output = outputs if bot_is_first else outputs[::-1]
        errors = next(content for marker, content in chunk if marker == ERROR)
        summary = next((content for marker, content in chunk if marker == SUMMARY), [])
        result += [f"Turn {int(number)}/{int(total)}:", f"{BOT_NAME} error:", *errors, f"{BOT_NAME} output:", *own_output]
        result += [f"{opponent} output:", *opponent_output, *([SUMMARY, *summary] if summary else []), ""]
    return "\n".join(result)

if __name__ == "__main__":
    path = "."
    main(path)
