"""Batch wrapper for Atmos downloads."""

from __future__ import annotations

import argparse
import subprocess
import uuid
from collections.abc import Iterable, Sequence
from pathlib import Path

LINKS: list[str] = []
DOWNLOADS_HOST = Path(r"E:\Downloads\Wrapper.x86_64.0df45b5\downloads")
STOP_STRINGS = ["connection reset by peer"]
USE_HOST_NETWORK = True
IMAGE = ""

EXIT_STOP_STRING = 2001
EXIT_INTERRUPTED = 130


def ensure_dir(path: Path) -> None:
    """Create directory when missing."""
    path.mkdir(parents=True, exist_ok=True)


def docker_rm_force(container_name: str) -> None:
    """Best-effort cleanup of a container by name."""
    try:
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except Exception:
        pass


def run_one(
    link: str,
    *,
    image: str,
    downloads_host: Path,
    stop_strings: Sequence[str],
    use_host_network: bool,
) -> int:
    """Run one containerized download and stream logs to stdout."""
    container_name = "musicdl_" + uuid.uuid4().hex[:12]

    args = ["docker", "run", "--rm", "--name", container_name]
    if use_host_network:
        args += ["--network", "host"]
    args += ["-v", f"{downloads_host}:/downloads", image, "--atmos", link]

    print("\n==> " + link, flush=True)

    stop_lc = [value.lower() for value in stop_strings if value]
    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            low = line.lower()
            if stop_lc and any(token in low for token in stop_lc):
                print("\n!!! Stop string matched. Aborting...\n", flush=True)
                try:
                    proc.terminate()
                except Exception:
                    pass
                docker_rm_force(container_name)
                return EXIT_STOP_STRING

        return proc.wait()
    except KeyboardInterrupt:
        print("\nStopped (Ctrl+C).", flush=True)
        try:
            proc.terminate()
        except Exception:
            pass
        docker_rm_force(container_name)
        return EXIT_INTERRUPTED
    finally:
        try:
            if proc.poll() is None:
                proc.terminate()
        except Exception:
            pass


def _load_links(path: Path) -> list[str]:
    links: list[str] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            value = line.strip()
            if value and not value.startswith("#"):
                links.append(value)
    return links


def _resolve_links(cli_links: Iterable[str], links_file: str | None) -> list[str]:
    resolved = [value.strip() for value in cli_links if value.strip()]
    if links_file:
        resolved.extend(_load_links(Path(links_file)))
    if not resolved:
        resolved = [value.strip() for value in LINKS if value.strip()]
    return resolved


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for batch Atmos download execution."""
    parser = argparse.ArgumentParser(description="Download Atmos releases.")
    parser.add_argument(
        "links",
        nargs="*",
        help="Music URLs to process. If omitted, falls back to LINKS in this file.",
    )
    parser.add_argument(
        "--links-file",
        default=None,
        help="Optional text file with one URL per line.",
    )
    parser.add_argument(
        "--downloads-host",
        default=str(DOWNLOADS_HOST),
        help=f"Host directory mounted to /downloads inside the container (default: {DOWNLOADS_HOST}).",
    )
    parser.add_argument(
        "--image",
        default=IMAGE,
        help=f"Container image to run (default: {IMAGE}).",
    )
    parser.add_argument(
        "--stop-string",
        action="append",
        default=None,
        help=(
            "Abort when this case-insensitive substring appears in output. "
            "Can be provided multiple times."
        ),
    )
    parser.add_argument(
        "--no-host-network",
        action="store_true",
        help="Disable --network host.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run link batch with fail-fast behavior and optional stop-string matching."""
    args = build_parser().parse_args(argv)
    links = _resolve_links(args.links, args.links_file)
    downloads_host = Path(args.downloads_host)
    ensure_dir(downloads_host)

    stop_strings = (
        [value.strip() for value in args.stop_string if value.strip()]
        if args.stop_string is not None
        else list(STOP_STRINGS)
    )

    index = 0
    total = len(links)
    for link in links:
        code = run_one(
            link,
            image=args.image,
            downloads_host=downloads_host,
            stop_strings=stop_strings,
            use_host_network=USE_HOST_NETWORK and (not args.no_host_network),
        )
        if code == EXIT_INTERRUPTED:
            print(
                "Stopped by user (Ctrl+C). Exiting. "
                f"- stopped at index {index} / {total} - {link}"
            )
            return EXIT_INTERRUPTED
        if code == EXIT_STOP_STRING:
            print(
                "Stopped because output matched a stop string. Exiting. "
                f"- stopped at index {index} / {total} - {link}"
            )
            return 1
        if code != 0:
            print(
                f"Docker exited with code {code}. Exiting. "
                f"- stopped at index {index} / {total} - {link}"
            )
            return code
        index += 1

    print("\nAll done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
