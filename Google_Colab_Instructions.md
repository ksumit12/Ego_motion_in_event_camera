# How to Run Fine-Resolution Analysis on Google Colab (GPU)

## Step-by-Step Instructions (5 minutes setup, 30 minutes run time)

### 1. **Open Google Colab**
- Go to: https://colab.research.google.com/
- Sign in with your Google account

### 2. **Create a New Notebook**
- Click: **File > New notebook**
- Click: **Runtime > Change runtime type** → Select **GPU** (T4 is fine) → Click **Save**

### 3. **Copy-Paste This Code** (5 cells total)

---

#### **CELL 1: Install Dependencies**
```python
# Install required packages
!pip install -q torch tqdm numpy scipy matplotlib
```

---

#### **CELL 2: Upload Files**
```python
# Click the folder icon (📁) on the left sidebar
# Then click the upload button (↑) and upload these files:
# 1. real_events.npy (~660 MB)
# 2. pred_events_dt_00.1ms.npy
# 3. pred_events_dt_00.5ms.npy
# 4. pred_events_dt_01.0ms.npy
# ... (upload as many dt values as you want to analyze)

# Run this cell to verify files are uploaded:
import os
from pathlib import Path

files = list(Path('/content').glob('*.npy'))
print(f"Found {len(files)} .npy files:")
for f in sorted(files):
    size_mb = f.stat().st_size / 1024 / 1024
    print(f"  {f.name}: {size_mb:.1f} MB")
```

---

#### **CELL 3: Define GPU Processing Function**
```python
import numpy as np
import torch
from tqdm.notebook import tqdm
import re
import gc

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Your thesis parameters
DISC_CENTER_X = 665.2710509177888
DISC_CENTER_Y = 337.4668998290816
DISC_RADIUS = 264
EPS_T_MS = 1.0
EPS_XY_PX = 2.0
CHUNK_SIZE = 100_000

def process_dt_gpu(real_path, pred_path, dt_ms):
    """GPU-accelerated cancellation - strictly causal"""
    # Load data
    real_events = np.load(real_path, mmap_mode='r')
    pred_events = np.load(pred_path, mmap_mode='r')
    
    n_real = len(real_events)
    n_pred = len(pred_events)
    
    real_x = real_events[:, 0].astype(np.float32)
    real_y = real_events[:, 1].astype(np.float32)
    real_t = real_events[:, 3].astype(np.float64)
    real_p = real_events[:, 2].astype(np.int8)
    
    pred_x_all = pred_events[:, 0].astype(np.float32)
    pred_y_all = pred_events[:, 1].astype(np.float32)
    pred_t_all = pred_events[:, 3].astype(np.float64)
    pred_p_all = pred_events[:, 2].astype(np.int8)
    
    # Global match tracking (causal state)
    real_matched = np.zeros(n_real, dtype=bool)
    
    eps_t_sec = EPS_T_MS / 1000.0
    eps_xy_sq = EPS_XY_PX * EPS_XY_PX
    disc_r2 = DISC_RADIUS * DISC_RADIUS
    
    n_matches_roi_total = 0
    n_pred_roi_total = 0
    n_chunks = (n_pred + CHUNK_SIZE - 1) // CHUNK_SIZE
    
    pbar = tqdm(range(n_chunks), desc=f"dt={dt_ms:.1f}ms")
    
    for chunk_idx in pbar:
        start = chunk_idx * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, n_pred)
        
        pred_x = pred_x_all[start:end]
        pred_y = pred_y_all[start:end]
        pred_t = pred_t_all[start:end]
        pred_p = pred_p_all[start:end]
        
        # ROI filter
        dx_roi = pred_x - DISC_CENTER_X
        dy_roi = pred_y - DISC_CENTER_Y
        in_roi = (dx_roi*dx_roi + dy_roi*dy_roi) <= disc_r2
        
        if not np.any(in_roi):
            continue
        
        pred_x = pred_x[in_roi]
        pred_y = pred_y[in_roi]
        pred_t = pred_t[in_roi]
        pred_p = pred_p[in_roi]
        n_pred_roi_total += len(pred_t)
        
        # Match each prediction
        for i in range(len(pred_t)):
            t_pred = pred_t[i]
            t_min = t_pred - eps_t_sec
            t_max = t_pred + eps_t_sec
            
            j_start = np.searchsorted(real_t, t_min, side='left')
            j_end = np.searchsorted(real_t, t_max, side='right')
            
            if j_start >= j_end:
                continue
            
            candidates = np.arange(j_start, j_end)
            mask = (~real_matched[candidates]) & (real_p[candidates] != pred_p[i])
            candidates = candidates[mask]
            
            if len(candidates) == 0:
                continue
            
            # GPU distance computation
            if len(candidates) > 10 and torch.cuda.is_available():
                pred_pos = torch.tensor([pred_x[i], pred_y[i]], device=device)
                real_pos = torch.tensor(np.column_stack([real_x[candidates], real_y[candidates]]), device=device)
                dist_sq = torch.sum((real_pos - pred_pos)**2, dim=1)
                dist_sq_np = dist_sq.cpu().numpy()
            else:
                dx = pred_x[i] - real_x[candidates]
                dy = pred_y[i] - real_y[candidates]
                dist_sq_np = dx*dx + dy*dy
            
            valid = dist_sq_np < eps_xy_sq
            if not np.any(valid):
                continue
            
            best_idx = np.argmin(dist_sq_np[valid])
            best_j = candidates[valid][best_idx]
            
            real_matched[best_j] = True
            n_matches_roi_total += 1
        
        if n_pred_roi_total > 0:
            cr = (n_matches_roi_total / n_pred_roi_total) * 100
            pbar.set_postfix({'CR': f'{cr:.1f}%'})
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    cancellation_rate = (n_matches_roi_total / n_pred_roi_total * 100) if n_pred_roi_total > 0 else 0.0
    
    return {
        'dt_ms': dt_ms,
        'cancellation_rate': cancellation_rate,
        'n_matches': n_matches_roi_total,
        'n_predictions': n_pred_roi_total
    }

print("✓ Function defined")
```

---

#### **CELL 4: Process All dt Values**
```python
# Find files
real_path = '/content/real_events.npy'
pred_files = sorted(Path('/content').glob('pred_events_dt_*.npy'))

if not Path(real_path).exists():
    print("❌ real_events.npy not found!")
elif len(pred_files) == 0:
    print("❌ No pred_events_dt_*.npy files found!")
else:
    print(f"Found {len(pred_files)} prediction files\n")
    
    # Extract dt values
    dt_files = []
    for f in pred_files:
        match = re.search(r'dt_(\d+\.\d+)ms', f.name)
        if match:
            dt_ms = float(match.group(1))
            dt_files.append((dt_ms, str(f)))
    
    dt_files.sort()
    
    # Process each
    results = []
    for dt_ms, pred_path in dt_files:
        print(f"\n{'='*50}")
        print(f"Processing dt = {dt_ms:.1f} ms")
        print('='*50)
        
        result = process_dt_gpu(real_path, pred_path, dt_ms)
        results.append(result)
        
        print(f"\n✓ CR = {result['cancellation_rate']:.2f}%, "
              f"matches = {result['n_matches']}/{result['n_predictions']}\n")
    
    # Save
    np.savez('/content/fine_resolution_results.npz',
             dt_values=[r['dt_ms'] for r in results],
             cancellation_rates=[r['cancellation_rate'] for r in results],
             n_matches=[r['n_matches'] for r in results],
             n_predictions=[r['n_predictions'] for r in results])
    
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    for r in results:
        print(f"dt={r['dt_ms']:5.1f}ms: CR={r['cancellation_rate']:5.1f}%")
    
    print("\n✓ Results saved to: fine_resolution_results.npz")
    print("📥 Download from left sidebar!")
```

---

#### **CELL 5: Plot Results**
```python
import matplotlib.pyplot as plt

data = np.load('/content/fine_resolution_results.npz')
dt_values = data['dt_values']
cr_values = data['cancellation_rates']

plt.figure(figsize=(10, 6))
plt.plot(dt_values, cr_values, 'o-', linewidth=2, markersize=8)
plt.xlabel('Prediction Horizon Δt (ms)', fontsize=12)
plt.ylabel('Cancellation Rate (%)', fontsize=12)
plt.title('Fine Resolution: 5-second Window', fontsize=14)
plt.grid(True, alpha=0.3)
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig('/content/cancellation_plot.png', dpi=150)
plt.show()

print("✓ Plot saved!")
```

---

## What You'll Get

**After ~30 minutes, you'll have:**
1. `fine_resolution_results.npz` - Your cancellation rates for all dt values
2. `cancellation_plot.png` - Ready-to-use plot for your thesis

**Download these files** (click them in the left sidebar → 3 dots → Download)

---

## Expected Timeline

- **Upload files**: 5-10 minutes (depending on your internet)
- **Processing**: 20-40 minutes (Colab GPU is much faster than your laptop)
- **Download results**: 1 minute (tiny file)

---

## Notes

- **Free GPU time**: Colab gives you ~12 hours/day of free GPU
- **Files persist**: Keep the tab open; files are deleted when you close it
- **Progress bars**: You'll see live updates as it processes
- **No crashes**: Colab has 16GB RAM + GPU memory, plenty for this task

---

## Troubleshooting

**If GPU is not available:**
- Try: **Runtime > Disconnect and delete runtime**, then reconnect
- Or wait a few hours and try again (free tier has limited availability)

**If upload is slow:**
- Upload just 3-4 dt values first (e.g., 0.5, 1.0, 2.0, 3.0 ms)
- This is enough to show the trend in your thesis

**If a cell fails:**
- Click the cell and press **Shift+Enter** to run it again





