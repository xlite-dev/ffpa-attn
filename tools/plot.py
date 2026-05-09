import matplotlib.pyplot as plt
import numpy as np

# 300DPI High-definition configuration (Publication-level quality)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

attn_types = [
  'self-attn(F/B)',
  'cross-attn(F/B)',
  'decode-attn(Nq=1,F/B)',
  'gqa(F/B)',
  'causal(F/B)',
  'attn-mask(F/B)',
  'dropout(F/B)',
  'non-aligned(F/B)',
]

# mean fp16, bf16
fwd_speedups = [
  np.mean([2.06, 2.08]),
  np.mean([1.86, 1.87]),
  np.mean([2.86, 2.85]),
  np.mean([2.06, 2.09]),
  np.mean([1.96, 1.99]),
  np.mean([1.70, 1.74]),
  np.mean([1.79, 1.82]),
  np.mean([1.96, 1.98])
]

bwd_speedups = [
  np.mean([2.34, 2.49]),
  np.mean([2.57, 2.51]),
  np.mean([2.54, 2.62]),
  np.mean([2.32, 2.46]),
  np.mean([2.22, 2.56]),
  np.mean([2.06, 2.17]),
  np.mean([2.27, 2.41]),
  np.mean([2.37, 2.67])
]

sdpa_speedups = [1.0] * len(attn_types)

fig, ax = plt.subplots(figsize=(32, 12))

width = 0.20
x = np.arange(len(attn_types))
x_sdpa = x - width
x_fwd = x
x_bwd = x + width

# SDPA Baseline bars (Light gray)
rect_sdpa = ax.bar(x_sdpa, sdpa_speedups, width, label='SDPA Baseline', color='#b0b0b0', edgecolor='white', linewidth=1)

# FFPA Forward bars (Blue)
rect_fwd = ax.bar(
  x_fwd, fwd_speedups, width, label='FFPA Forward (FWD)', color='#2171b5', edgecolor='white', linewidth=1
)

# FFPA Backward bars (Red-Orange)
rect_bwd = ax.bar(
  x_bwd, bwd_speedups, width, label='FFPA Backward (BWD)', color="#fd493c", edgecolor='white', linewidth=1
)

# Dashed baseline line at speedup = 1.0
ax.axhline(y=1, color='#555555', linestyle='--', linewidth=2)


def autolabel(rects):
  """Add speedup value labels on top of each bar"""
  for rect in rects:
    h = rect.get_height()
    offset = 8 if h >= 1 else 20
    va_pos = 'bottom' if h >= 1 else 'top'
    ax.annotate(
      f'{h:.1f}x',
      xy=(rect.get_x() + rect.get_width() / 2, h),
      xytext=(0, offset),
      textcoords='offset points',
      ha='center',
      va=va_pos,
      fontsize=19,
      fontweight='bold',
    )


autolabel(rect_sdpa)
autolabel(rect_fwd)
autolabel(rect_bwd)

ax.set_ylabel('Speedup Ratio (FFPA / SDPA)', fontsize=18)
ax.set_title(
  'FFPA vs SDPA Speedup (FWD & BWD), NVIDIA RTX 5090 Blackwell | B=1, N=8192, H=32, D=512',
  fontsize=22,
  pad=15,
  fontweight='bold'
)

ax.set_xticks(x)
ax.set_xticklabels(attn_types, rotation=0, ha='center', fontsize=22, fontweight='bold')

ax.tick_params(axis='y', labelsize=16)
max_val = max(max(fwd_speedups), max(bwd_speedups))
ax.set_ylim(0, max_val * 1.1)

ax.legend(fontsize=20, loc='upper left')
ax.grid(axis='y', alpha=0.3)

fig.tight_layout()
plt.savefig('ffpa_speedup.png')
