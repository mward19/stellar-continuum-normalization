#!/usr/bin/env python3
"""Run the curl lines of a .sh file one at a time with a tqdm progress bar.

usage: python run_downloads.py downloads.sh
Requires: pip install tqdm
"""
import shlex
import subprocess
import sys
import threading
from pathlib import Path

from tqdm import tqdm

# Guards failed_lines.txt, which is shared by concurrent calls to main()
# when scripts are downloaded in parallel across threads.
_fail_log_lock = threading.Lock()


def is_curl(line: str) -> bool:
    return line.lstrip().startswith("curl")


def main(script: str, position: int = 0) -> None:
    # First pass only counts lines (streamed, so the file is never fully in memory);
    # tqdm needs `total` up front to draw a percentage bar.
    with open(script) as f:
        total = sum(1 for line in f if is_curl(line))

    failed = 0
    skipped = 0

    with open(script) as f, open("failed_lines.txt", "a") as fail_log:
        bar = tqdm(total=total, unit="file", leave=False, position=position,
                   desc=Path(script).name)
        for line in f:
            if not is_curl(line):
                continue

            # shlex.split respects the double quotes in your lines, so we can
            # skip shell=True (no quoting/injection surprises).
            args = shlex.split(line)

            # Resumability: if --output target already exists and is non-empty,
            # skip it. curl -f doesn't create the file on HTTP errors, so a
            # non-empty file means a completed download.
            out = args[args.index("--output") + 1] if "--output" in args else None
            if out and Path(out).is_file() and Path(out).stat().st_size > 0:
                skipped += 1
                bar.update(1)
                bar.set_postfix(failed=failed, skipped=skipped)
                continue

            # -sS: silent (so curl's meter doesn't clobber tqdm) but still print
            # errors to stderr, which we capture below.
            args[1:1] = ["-sS"]
            result = subprocess.run(args, capture_output=True, text=True)

            if result.returncode != 0:
                failed += 1
                with _fail_log_lock:
                    fail_log.write(line if line.endswith("\n") else line + "\n")
                    fail_log.flush()
                tqdm.write(f"FAIL ({result.returncode}): {out}  {result.stderr.strip()}")

            bar.update(1)
            bar.set_postfix(failed=failed, skipped=skipped)
        bar.close()

    tqdm.write(f"done: {total} lines, {failed} failed, {skipped} skipped")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])

