#!/usr/bin/env python3
"""Draws the tier 1 progress tracker in docs/status/ and the Status section of README.md.

Usage, from the repository root after a debug build with tests (Linux, needs Unicorn):

    python3 tools/status/generate_status.py build/debug/bin/linux/bal_status_coverage [QEMU_CSV...]

The coverage tool runs random encodings of every decoder table entry through Ballistic and Unicorn
and reports per entry whether tier 1 translates it and matches. Alias entries that the decoder never
returns (their encodings resolve to the base instruction) are left out.

Unicorn cannot run SVE or SME. Each QEMU_CSV, an output of tests/sve/run_sve_diff.sh (with
BAL_SVE_SPACE=sme for SME), gives the SVE or SME entries it covers their status against
qemu-aarch64 instead; entries both Ballistic and qemu reject count as not started.
"""

import csv
import html
import io
import math
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "docs" / "status"
README = ROOT / "README.md"
MARK_BEGIN = "<!-- BALLISTIC-STATUS:BEGIN -->"
MARK_END = "<!-- BALLISTIC-STATUS:END -->"

VERIFIED, UNVERIFIED, WRONG, TODO = "verified", "unverified", "wrong", "todo"
ORDER = {VERIFIED: 0, UNVERIFIED: 1, WRONG: 2, TODO: 3}
COLORS = {VERIFIED: "#2ea043", UNVERIFIED: "#d29922", WRONG: "#f85149", TODO: "#6e7681"}
BORDER = "#0d1117"

SHORT_NAMES = {
    "Data processing (register)": "DP reg",
    "Data processing (immediate)": "DP imm",
    "Branches and system": "Branches",
}

CLASS_ORDER = [
    "Data processing (register)",
    "Data processing (immediate)",
    "Branches and system",
    "Loads and stores",
    "SIMD and floating point",
    "SVE",
    "SME",
]


def encoding_class(expected):
    """Top-level A64 encoding group from op0, bits 28:25."""
    op0 = (expected >> 25) & 0xF
    if op0 == 0b0000:
        return "SME" if expected >> 31 else "Reserved"
    if op0 == 0b0010:
        return "SVE"
    if op0 in (0b1000, 0b1001):
        return "Data processing (immediate)"
    if op0 in (0b1010, 0b1011):
        return "Branches and system"
    if op0 & 0b0101 == 0b0100:
        return "Loads and stores"
    if op0 & 0b0111 == 0b0101:
        return "Data processing (register)"
    return "SIMD and floating point"


def read_coverage(source):
    if source.endswith(".csv"):
        text = Path(source).read_text()
    else:
        env = dict(os.environ, ASAN_OPTIONS="detect_leaks=0")
        text = subprocess.run([source], capture_output=True, text=True, check=True, env=env).stdout
    entries = []
    for index, name, _mask, expected, status, _translated, _matched, samples in csv.reader(
        io.StringIO(text)
    ):
        if int(samples) == 0:
            continue
        cls = encoding_class(int(expected, 16))
        if cls == "Reserved":
            continue
        entries.append({"index": int(index), "name": name, "class": cls, "status": status})
    return entries


def apply_qemu(entries, paths):
    """Replaces the status of the SVE and SME entries with the qemu comparisons in `paths`."""
    statuses = {}
    for path in paths:
        lines = (line for line in Path(path).read_text().splitlines() if not line.startswith("#"))
        for row in csv.reader(lines):
            statuses[int(row[0])] = {"verified": VERIFIED, "wrong": WRONG}.get(row[4], TODO)
    for entry in entries:
        if entry["class"] in ("SVE", "SME") and entry["index"] in statuses:
            entry["status"] = statuses[entry["index"]]


def squarify(items, x, y, w, h):
    """items: list of (key, weight) sorted descending. Returns {key: rect}."""
    total = sum(v for _, v in items)
    if not items or total <= 0:
        return {}
    scale = w * h / total
    sized = [(k, v * scale) for k, v in items]
    out = {}

    def worst(row, side):
        s = sum(a for _, a in row)
        return max(max(side * side * a / (s * s), (s * s) / (side * side * a)) for _, a in row)

    while sized:
        side = min(w, h)
        row = [sized.pop(0)]
        while sized and worst(row + [sized[0]], side) <= worst(row, side):
            row.append(sized.pop(0))
        s = sum(a for _, a in row)
        if w >= h:
            cw = s / h
            cy = y
            for k, a in row:
                ch = a / cw
                out[k] = (x, cy, cw, ch)
                cy += ch
            x, w = x + cw, w - cw
        else:
            rh = s / w
            cx = x
            for k, a in row:
                cwid = a / rh
                out[k] = (cx, y, cwid, rh)
                cx += cwid
            y, h = y + rh, h - rh
    return out


def cells(paths, statuses, x, y, w, h):
    """Fills a rect with one square per entry, finished entries first."""
    n = len(statuses)
    cols = max(1, round(math.sqrt(n * w / h))) if h > 0 else n
    rows = math.ceil(n / cols)
    cw, ch = w / cols, h / rows
    gap = min(0.5, cw / 5, ch / 5)
    for i, status in enumerate(sorted(statuses, key=ORDER.get)):
        cx, cy = x + (i % cols) * cw, y + (i // cols) * ch
        paths[status].append(
            f"M{cx + gap:.1f} {cy + gap:.1f}h{cw - 2 * gap:.1f}v{ch - 2 * gap:.1f}h{2 * gap - cw:.1f}z"
        )


def label(svg, text, x, y, size=10, weight="normal"):
    svg.append(
        f'<text x="{x + 3:.1f}" y="{y + size + 1:.1f}" font-size="{size}" font-weight="{weight}" '
        f'class="lbl">{html.escape(text)}</text>'
    )


def pct(done, total):
    return f"{100 * done / total:.1f}%" if total else "0%"


def render(entries, commit):
    W, H = 920, 360
    lw, gap, top = 560, 18, 28
    ph = H - top - 24
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        'font-family="-apple-system,Segoe UI,Helvetica,Arial,sans-serif">',
        "<style>.lbl{fill:#fff;paint-order:stroke;stroke:#000;stroke-width:2.5px;"
        "stroke-linejoin:round}.ttl{fill:#8b949e;font-size:13px;font-weight:600}"
        ".txt{fill:#8b949e;font-size:11px}</style>",
    ]

    counts = Counter(e["status"] for e in entries)
    svg.append(
        f'<text x="0" y="17" class="ttl">A64 instructions verified on x86: '
        f"{pct(counts[VERIFIED], len(entries))} ({counts[VERIFIED]}/{len(entries)})</text>"
    )

    by_class = defaultdict(lambda: defaultdict(list))
    for e in entries:
        by_class[e["class"]][e["name"]].append(e["status"])
    class_items = sorted(
        ((c, sum(len(v) for v in m.values())) for c, m in by_class.items()), key=lambda t: -t[1]
    )
    paths = defaultdict(list)
    frames = []
    for cls, (cx, cy, cw, ch) in squarify(class_items, 0, top, lw, ph).items():
        mnemonics = sorted(((k, len(v)) for k, v in by_class[cls].items()), key=lambda t: -t[1])
        header = 14 if ch > 40 else 0
        for mnemonic, (mx, my, mw, mh) in squarify(mnemonics, cx, cy + header, cw, ch - header).items():
            cells(paths, by_class[cls][mnemonic], mx, my, mw, mh)
            frames.append(
                f'<rect x="{mx:.2f}" y="{my:.2f}" width="{mw:.2f}" height="{mh:.2f}" '
                f'fill="none" stroke="{BORDER}" stroke-width="1"/>'
            )
            if mw > 30 and mh > 15:
                label(frames, mnemonic[: max(2, int(mw / 6.5))], mx, my, 9)
        frames.append(
            f'<rect x="{cx:.2f}" y="{cy:.2f}" width="{cw:.2f}" height="{ch:.2f}" '
            f'fill="none" stroke="{BORDER}" stroke-width="2.5"/>'
        )
        if header:
            frames.append(
                f'<rect x="{cx:.2f}" y="{cy:.2f}" width="{cw:.2f}" height="{header}" fill="#161b22"/>'
            )
            label(frames, SHORT_NAMES.get(cls, cls)[: int(cw / 5.6)], cx, cy, 9.5, "600")
    svg.append(f'<rect x="0" y="{top}" width="{lw}" height="{ph}" fill="{BORDER}"/>')
    for status in ORDER:
        if paths[status]:
            svg.append(f'<path fill="{COLORS[status]}" d="{"".join(paths[status])}"/>')
    svg.extend(frames)

    # Right panel: one progress bar per instruction class.
    rx = lw + gap
    rw = W - rx
    svg.append(f'<text x="{rx}" y="17" class="ttl">Progress per instruction class</text>')
    classes = [c for c in CLASS_ORDER if c in by_class]
    row_height = ph / len(classes)
    for i, cls in enumerate(classes):
        statuses = [s for v in by_class[cls].values() for s in v]
        total = len(statuses)
        c = Counter(statuses)
        y = top + i * row_height + 4
        svg.append(f'<text x="{rx}" y="{y + 11:.1f}" class="txt">{html.escape(cls)}</text>')
        svg.append(
            f'<text x="{W}" y="{y + 11:.1f}" class="txt" text-anchor="end">'
            f"{pct(c[VERIFIED], total)} ({c[VERIFIED]}/{total})</text>"
        )
        bar_y, bar_h, bx = y + 16, 10, rx
        svg.append(f'<rect x="{rx}" y="{bar_y:.1f}" width="{rw}" height="{bar_h}" rx="3" fill="#21262d"/>')
        for status in (VERIFIED, UNVERIFIED, WRONG):
            width = rw * c[status] / total
            if width > 0:
                svg.append(
                    f'<rect x="{bx:.2f}" y="{bar_y:.1f}" width="{width:.2f}" height="{bar_h}" '
                    f'fill="{COLORS[status]}"/>'
                )
                bx += width

    ly = H - 8
    lx = 0
    for status, text in (
        (VERIFIED, "matches Unicorn (qemu for SVE, SME)"),
        (UNVERIFIED, "translated, not checkable"),
        (WRONG, "differs from the reference"),
        (TODO, "not started"),
    ):
        svg.append(f'<rect x="{lx}" y="{ly - 9}" width="10" height="10" fill="{COLORS[status]}"/>')
        svg.append(f'<text x="{lx + 14}" y="{ly}" class="txt">{text}</text>')
        lx += 14 + len(text) * 5.6 + 16
    svg.append(
        f'<text x="{W}" y="{ly}" font-size="10" fill="#8b949e" text-anchor="end">@ {commit}</text>'
    )
    svg.append("</svg>")
    return "\n".join(svg)


def badge(left, right, color):
    lw, rw = 7 * len(left) + 12, 7 * len(right) + 12
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{lw + rw}" height="20" '
        f'font-family="Verdana,DejaVu Sans,sans-serif" font-size="11">'
        f'<rect width="{lw}" height="20" fill="#555"/>'
        f'<rect x="{lw}" width="{rw}" height="20" fill="{color}"/>'
        f'<text x="{lw / 2}" y="14" fill="#fff" text-anchor="middle">{left}</text>'
        f'<text x="{lw + rw / 2}" y="14" fill="#fff" text-anchor="middle">{right}</text></svg>'
    )


def badge_color(fraction):
    for limit, color in ((0.9, "#4c1"), (0.6, "#97ca00"), (0.3, "#dfb317"), (0.1, "#fe7d37")):
        if fraction >= limit:
            return color
    return "#e05d44"


def update_readme(summary):
    block = f"""{MARK_BEGIN}
## Status

![integer](docs/status/badge_integer.svg) ![all](docs/status/badge_all.svg)

![Status](docs/status/status.svg)

{summary}

Regenerate after a debug build with `tests/sve/run_sve_diff.sh > sve.csv`,
`BAL_SVE_SPACE=sme BAL_SVE_QEMU=<qemu 10.1 or later> tests/sve/run_sve_diff.sh > sme.csv` and
`python3 tools/status/generate_status.py build/debug/bin/linux/bal_status_coverage sve.csv sme.csv`.
{MARK_END}"""
    text = README.read_text(encoding="utf-8")
    if MARK_BEGIN in text:
        text = re.sub(re.escape(MARK_BEGIN) + ".*?" + re.escape(MARK_END), lambda _: block, text, flags=re.S)
    else:
        text = text.rstrip("\n") + "\n\n" + block + "\n"
    README.write_text(text, encoding="utf-8")


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    entries = read_coverage(sys.argv[1])
    apply_qemu(entries, sys.argv[2:])
    commit = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "--short=8", "HEAD"], capture_output=True, text=True
    ).stdout.strip() or "unknown"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "status.svg").write_text(render(entries, commit), encoding="utf-8")

    integer = [e for e in entries if e["class"].startswith(("Data processing", "Branches"))]
    integer_done = sum(e["status"] == VERIFIED for e in integer) / len(integer)
    all_done = sum(e["status"] == VERIFIED for e in entries) / len(entries)
    (OUT_DIR / "badge_integer.svg").write_text(
        badge("integer + branches", f"{100 * integer_done:.1f}%", badge_color(integer_done)), encoding="utf-8"
    )
    (OUT_DIR / "badge_all.svg").write_text(
        badge("all A64", f"{100 * all_done:.1f}%", badge_color(all_done)), encoding="utf-8"
    )

    counts = Counter(e["status"] for e in entries)
    groups = []
    for name in CLASS_ORDER:
        members = [e for e in entries if e["class"] == name]
        if members:
            done = sum(e["status"] == VERIFIED for e in members) / len(members)
            groups.append(f"{name} {100 * done:.1f}%")
    reference = "Unicorn, and SVE and SME against qemu-aarch64 with 128-bit vectors" if len(sys.argv) > 2 else "Unicorn"
    summary = (
        f"Each of the {len(entries)} reachable A64 encodings in the decoder table is tested with random "
        f"operands against {reference}: {counts[VERIFIED]} verified, {counts[UNVERIFIED]} translated but not "
        f"checkable (Unicorn cannot run the random samples or has a known bug for them), "
        f"{counts[WRONG]} wrong and {counts[TODO]} not started. Verified by group: {', '.join(groups)}."
    )
    update_readme(summary)
    print(summary)


if __name__ == "__main__":
    main()
