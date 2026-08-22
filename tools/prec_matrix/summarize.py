# Attribution summary for the prec_matrix accuracy output: reads
# .tmp/fp4-precision/accuracy.md and reports, per (D, mask, mode), how the
# injected outlier distributions move the error vs the flat baseline, so the
# dominant error source (QK quant vs P/V quant) is directly readable.
# Usage: python tools/prec_matrix/summarize.py [path-to-accuracy.md]
import sys
from collections import defaultdict

PATH = sys.argv[1] if len(sys.argv) > 1 else ".tmp/fp4-precision/accuracy.md"


def parse(path):
  rows = []
  with open(path) as f:
    for line in f:
      if not line.startswith("|") or "cos" in line or set(line) <= set("|- "):
        continue
      cells = [c.strip() for c in line.strip().strip("|").split("|")]
      if len(cells) != 10 or not cells[0].lstrip("-").isdigit():
        continue
      D, N, dtype, mask, dist, seed, mode, cos, mx, mean = cells
      rows.append((
        int(D), int(N), dtype, mask, dist, int(seed), mode, float(cos),
        float(mx), float(mean)
      ))
  return rows


def main():
  rows = parse(PATH)
  # mean over seeds -> (D, mask, dist, mode) metrics
  agg = defaultdict(list)
  for D, N, dtype, mask, dist, seed, mode, cos, mx, mean in rows:
    agg[(D, dtype, mask, dist, mode)].append((cos, mx, mean))
  summary = {
    k: (
      sum(c for c, _, _ in v) / len(v), max(m for _, m, _ in v),
      sum(m for _, _, m in v) / len(v)
    )
    for k, v in agg.items()
  }
  modes = sorted({k[4] for k in summary})
  dists = ["flat", "qk", "v", "all"]
  print(f"parsed {len(rows)} rows from {PATH}\n")
  header = "| D | dtype | mask | mode | flat mean | qk mean | v mean | all mean | qk/flat | v/flat |"
  print(header)
  print("|---|---|---|---|---|---|---|---|---|---|")
  for D, dtype, mask in sorted({k[:3] for k in summary}):
    for mode in modes:
      vals = {}
      for dist in dists:
        entry = summary.get((D, dtype, mask, dist, mode))
        vals[dist] = entry[2] if entry else float("nan")
      base = vals["flat"]
      qk_r = vals["qk"] / base if base > 0 else float("nan")
      v_r = vals["v"] / base if base > 0 else float("nan")
      print(
        f"| {D} | {dtype} | {mask} | {mode} | {vals['flat']:.5f} | "
        f"{vals['qk']:.5f} | {vals['v']:.5f} | {vals['all']:.5f} | "
        f"{qk_r:.2f}x | {v_r:.2f}x |"
      )
  print(
    "\nqk/flat >> 1 => QK-side quant dominates the extra error; "
    "v/flat >> 1 => V/PV-side quant dominates."
  )


if __name__ == "__main__":
  main()
