# ============================================================
# PU Bearing Dataset
# CNN vs U-FNO — Cross-Length
# Train on 1s windows ONLY, Test on 1s (same) and 2s (cross)
# ============================================================
#
# EXPECTED FINDING:
#   Both CNN and U-FNO maintain accuracy on 2s windows.
#   Bearing fault signatures are periodic and frequency-based.
#   Fault harmonics appear at the same mel bin regardless of
#   window length. AdaptiveAvgPool/GAP averages over the time
#   axis — more time frames means more repetitions of the same
#   fault pattern → same or better accuracy.
#
# HYPERPARAMS:
#   WIDTH=64, DEPTH=2, MODES=16, UNET_DEPTH=2, DROPOUT=0.30
#   EPOCHS=200, PATIENCE=50
# ============================================================

import os, random, gc, time
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import librosa

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ── Reproducibility ───────────────────────────────────────────
SEED = int(os.environ.get('SEED', 42))
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if DEVICE == 'cuda':
    torch.backends.cudnn.benchmark     = False
    torch.backends.cudnn.deterministic = True
    print('GPU  :', torch.cuda.get_device_name(0))
    print('VRAM :', round(
        torch.cuda.get_device_properties(0).total_memory/1e9, 2), 'GB')

# ── Paths ─────────────────────────────────────────────────────
PU_DIR  = Path(os.environ.get('PU_DATA_DIR', './PU_Data'))
OUT_DIR = Path(os.environ.get('OUTPUTS_DIR', './outputs')) / 'cross_length_1s_to_2s' / f'seed_{SEED}'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── PU Config ─────────────────────────────────────────────────
OPERATING_COND = 'N15_M07_F10'

BEARINGS = {
    'healthy':    ['K001', 'K002', 'K003'],
    'outer_race': ['KA01', 'KA03', 'KA04'],
    'inner_race': ['KI01', 'KI03', 'KI04'],
}
CLASSES     = list(BEARINGS.keys())
NUM_CLASSES = len(CLASSES)

FS         = 64000
TRAIN_REPS = list(range(1, 15))
VAL_REPS   = list(range(15, 18))
TEST_REPS  = list(range(18, 21))

TRAIN_WIN_S = 1
TEST_WIN_S  = [1, 2]
OVERLAP     = 0.50
MAX_SEG     = 80

# ── Spectrogram (no calibration here — F_MAX = FS/2) ─────────
N_MELS     = 128
N_FFT      = 2048
HOP_LENGTH = 512
F_MIN      = 0
F_MAX      = FS // 2

# ── U-FNO config ──────────────────────────────────────────────
WIDTH      = 64
DEPTH      = 2
MODES1     = 16
MODES2     = 16
UNET_DEPTH = 2
DROPOUT    = 0.30

# ── Training ──────────────────────────────────────────────────
BATCH_SIZE      = 32
EPOCHS          = 200
PATIENCE        = 50
LR              = 1e-3
WEIGHT_DECAY    = 1e-4
LABEL_SMOOTHING = 0.05
GRAD_CLIP       = 1.0

print(f'EXPERIMENT   : Train {TRAIN_WIN_S}s -> Test {TEST_WIN_S}')
print(f'Condition    : {OPERATING_COND}')
print(f'U-FNO config : WIDTH={WIDTH}, DEPTH={DEPTH}, '
      f'MODES={MODES1}, UNET_DEPTH={UNET_DEPTH}')

for ws in [TRAIN_WIN_S] + [w for w in TEST_WIN_S if w != TRAIN_WIN_S]:
    mel = librosa.feature.melspectrogram(
        y=np.zeros(ws * FS, dtype=np.float32), sr=FS,
        n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=F_MIN, fmax=F_MAX)
    tag = 'TRAIN+TEST' if ws == TRAIN_WIN_S else 'TEST only (cross)'
    print(f'  {ws}s [{tag}]: (1, {mel.shape[0]}, {mel.shape[1]})')


# ══════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════

def _unwrap(obj):
    while isinstance(obj, np.ndarray) and obj.ndim > 0 and obj.size == 1:
        obj = obj.flat[0]
    return obj

def _iter_channels(arr):
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    for i in range(arr.shape[1]):
        yield arr[0, i]

def _channel_data(ch):
    try:
        d = ch['Data']
        while isinstance(d, np.ndarray) and d.dtype.kind == 'O' and d.size > 0:
            d = d.flat[0]
        while isinstance(d, np.ndarray) and d.ndim > 0 and d.size == 1:
            d = d.flat[0]
        if isinstance(d, np.ndarray) and d.dtype.kind in 'fiu' and d.size >= 1000:
            return d.reshape(-1)
    except Exception:
        pass
    return None

def load_pu_signal(path: Path) -> np.ndarray:
    m         = sio.loadmat(str(path))
    user_keys = [k for k in m if not k.startswith('__')]
    bearing   = _unwrap(m.get('bearing', m[user_keys[0]]))
    candidates = []
    for field in ['Y', 'X']:
        try:
            for ch in _iter_channels(bearing[field]):
                arr = _channel_data(ch)
                if arr is not None and not np.all(np.diff(arr[:200]) > 0):
                    candidates.append(arr)
        except Exception:
            pass
    if not candidates:
        raise RuntimeError(f'No signal: {path.name}')
    return max(candidates, key=len).astype(np.float32)

def make_mel(signal: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=signal, sr=FS,
        n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=F_MIN, fmax=F_MAX)
    mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    mel = (mel - mel.mean()) / (mel.std() + 1e-8)
    return mel

def make_starts(sig_len: int, wl: int, overlap: float, max_keep: int):
    step = int(wl * (1.0 - overlap))
    if step <= 0: step = wl
    if sig_len < wl: return []
    starts = list(range(0, sig_len - wl + 1, step))
    if len(starts) > max_keep:
        idx    = np.linspace(0, len(starts)-1, max_keep, dtype=int)
        starts = [starts[i] for i in idx]
    return starts

def build_rows(reps: list, win_s: int,
               overlap: float, seed: int) -> list:
    rows = []
    wl   = win_s * FS
    for label, bids in BEARINGS.items():
        y = CLASSES.index(label)
        for bid in bids:
            for rep in reps:
                path = PU_DIR / bid / \
                       f'{OPERATING_COND}_{bid}_{rep}.mat'
                if not path.exists():
                    continue
                try:
                    sig = load_pu_signal(path)
                except Exception as e:
                    print(f'  SKIP {path.name}: {e}')
                    continue
                for s in make_starts(len(sig), wl, overlap, MAX_SEG):
                    rows.append({
                        'mel':   make_mel(sig[s:s + wl]),
                        'label': y,
                        'win_s': win_s,
                    })
    rng = np.random.default_rng(seed)
    rng.shuffle(rows)
    return rows


print('\nBuilding data ...')
train_rows = build_rows(TRAIN_REPS, TRAIN_WIN_S, OVERLAP,    SEED)
val_rows   = build_rows(VAL_REPS,   TRAIN_WIN_S, OVERLAP,    SEED + 1)

test_by_len = {}
for ws in TEST_WIN_S:
    test_by_len[ws] = build_rows(
        TEST_REPS, ws, overlap=0.0, seed=SEED + 10 + ws)

print(f'\nData summary:')
print(f'  Train ({TRAIN_WIN_S}s, overlap={OVERLAP}): {len(train_rows)}')
print(f'  Val   ({TRAIN_WIN_S}s, overlap={OVERLAP}): {len(val_rows)}')
for ws in TEST_WIN_S:
    tag = 'same' if ws == TRAIN_WIN_S else 'CROSS'
    print(f'  Test  ({ws}s, no overlap) [{tag}]: {len(test_by_len[ws])}')

for name, rows in [('Train', train_rows), ('Val', val_rows)] + \
                  [(f'Test {ws}s', test_by_len[ws]) for ws in TEST_WIN_S]:
    cnt = np.bincount([r['label'] for r in rows],
                      minlength=NUM_CLASSES)
    print(f'  {name}: ' +
          ', '.join(f'{CLASSES[i]}={cnt[i]}' for i in range(NUM_CLASSES)))


# ══════════════════════════════════════════════════════════════
# DATASET + LOADERS
# ══════════════════════════════════════════════════════════════

class BearingDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        x = torch.from_numpy(r['mel']).float().unsqueeze(0)
        y = torch.tensor(int(r['label']), dtype=torch.long)
        return x, y

def make_loader(rows, shuffle):
    return DataLoader(
        BearingDataset(rows),
        batch_size=BATCH_SIZE, shuffle=shuffle,
        num_workers=0, pin_memory=False, drop_last=shuffle)

train_loader = make_loader(train_rows, shuffle=True)
val_loader   = make_loader(val_rows,   shuffle=False)
test_loaders = {ws: make_loader(test_by_len[ws], shuffle=False)
                for ws in TEST_WIN_S}

labels = [r['label'] for r in train_rows]
cnt    = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float32)
w      = cnt.sum() / np.maximum(cnt, 1.0)
w      = w / w.mean()
CLASS_WEIGHTS = torch.tensor(w, dtype=torch.float32, device=DEVICE)
print('\nClass weights:',
      {c: round(float(x), 3) for c, x in zip(CLASSES, w)})


# ══════════════════════════════════════════════════════════════
# EVALUATE
# ══════════════════════════════════════════════════════════════

def evaluate(model, loader):
    model.eval()
    all_yt, all_yp = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE); yb = yb.to(DEVICE)
            pred = model(xb).argmax(1)
            all_yt.extend(yb.cpu().numpy().tolist())
            all_yp.extend(pred.cpu().numpy().tolist())
    all_yt = np.array(all_yt); all_yp = np.array(all_yp)
    acc    = float((all_yt == all_yp).mean()) if len(all_yt) else 0.0
    cm     = confusion_matrix(all_yt, all_yp,
                              labels=np.arange(NUM_CLASSES))
    return acc, cm


# ══════════════════════════════════════════════════════════════
# CNN MODEL
# ══════════════════════════════════════════════════════════════

class CNNClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        def block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch,  out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch), nn.ReLU(),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch), nn.ReLU(),
                nn.MaxPool2d(2), nn.Dropout2d(0.10),
            )
        self.features = nn.Sequential(
            block(1,   32), block(32,  64), block(64, 128),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(), nn.Dropout(0.40),
            nn.Linear(256, num_classes),
        )
    def forward(self, x):
        return self.classifier(self.features(x))


# ══════════════════════════════════════════════════════════════
# U-FNO MODEL
# ══════════════════════════════════════════════════════════════

class SpectralConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, modes1=12, modes2=12):
        super().__init__()
        self.modes1  = modes1
        self.modes2  = modes2
        scale        = 1.0 / (in_ch * out_ch)
        self.weights = nn.Parameter(
            scale * torch.randn(in_ch, out_ch, modes1, modes2,
                                dtype=torch.cfloat))
    def forward(self, x):
        b, c, h, w = x.shape
        x_ft   = torch.fft.rfft2(x)
        out_ft = torch.zeros(b, self.weights.shape[1], h, w // 2 + 1,
                             device=x.device, dtype=torch.cfloat)
        m1 = min(self.modes1, h)
        m2 = min(self.modes2, w // 2 + 1)
        out_ft[:, :, :m1, :m2] = torch.einsum(
            'bixy,ioxy->boxy',
            x_ft[:, :, :m1, :m2],
            self.weights[:, :, :m1, :m2])
        return torch.fft.irfft2(out_ft, s=(h, w))


class UNetPath(nn.Module):
    def __init__(self, width, unet_depth=1):
        super().__init__()
        self.enc = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(width, width, 3, stride=2, padding=1),
                nn.BatchNorm2d(width), nn.GELU(),
            ) for _ in range(unet_depth)])
        self.dec = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(width * 2, width, 2, stride=2),
                nn.BatchNorm2d(width), nn.GELU(),
            ) for _ in range(unet_depth)])
    def forward(self, x):
        skips = []
        h = x
        for enc in self.enc:
            skips.append(h); h = enc(h)
        for dec, skip in zip(self.dec, reversed(skips)):
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:],
                                  mode='bilinear', align_corners=False)
            h = torch.cat([h, skip], dim=1)
            h = dec(h)
        if h.shape[-2:] != x.shape[-2:]:
            h = F.interpolate(h, size=x.shape[-2:],
                              mode='bilinear', align_corners=False)
        return h


class UFNOBlock(nn.Module):
    def __init__(self, width, modes1=12, modes2=12,
                 unet_depth=1, dropout=0.30):
        super().__init__()
        self.fourier   = SpectralConv2d(width, width, modes1, modes2)
        self.unet      = UNetPath(width, unet_depth)
        self.pointwise = nn.Conv2d(width, width, kernel_size=1)
        self.norm      = nn.BatchNorm2d(width)
        self.drop      = nn.Dropout2d(dropout)
    def forward(self, x):
        y = self.fourier(x) + self.unet(x) + self.pointwise(x)
        y = self.norm(y); y = F.gelu(y); y = self.drop(y)
        return x + y


class UFNOClassifier(nn.Module):
    def __init__(self, num_classes, width=64, depth=2,
                 modes1=12, modes2=12, unet_depth=1, dropout=0.30):
        super().__init__()
        self.in_proj = nn.Conv2d(1, width, kernel_size=1)
        self.blocks  = nn.ModuleList([
            UFNOBlock(width, modes1, modes2, unet_depth, dropout)
            for _ in range(depth)])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(width, 128),
            nn.ReLU(),
            nn.Dropout(0.40),
            nn.Linear(128, num_classes),
        )
    def forward(self, x):
        x = self.in_proj(x)
        for blk in self.blocks:
            x = blk(x)
        return self.head(x)


# ══════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════

def train_model(model, model_name):
    print(f'\n{"="*60}')
    print(f'Training {model_name}  [1s only, n={len(train_rows)}]')
    print(f'  params={sum(p.numel() for p in model.parameters()):,}')
    print(f'{"="*60}')

    criterion = nn.CrossEntropyLoss(
        weight=CLASS_WEIGHTS, label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5,
        patience=8, min_lr=1e-6)

    history      = []
    best_val_acc = -1.0
    best_state   = None
    no_improve   = 0

    for ep in range(1, EPOCHS + 1):
        model.train()
        loss_sum, n = 0.0, 0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE); yb = yb.to(DEVICE)

            b, c, f, t = xb.shape
            fm = random.randint(0, max(1, f // 10))
            fs = random.randint(0, max(0, f - fm))
            xb[:, :, fs:fs+fm, :] = 0.0
            tm = random.randint(0, max(1, t // 10))
            ts = random.randint(0, max(0, t - tm))
            xb[:, :, :, ts:ts+tm] = 0.0

            logits = model(xb)
            loss   = criterion(logits, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), GRAD_CLIP)
            optimizer.step()
            loss_sum += float(loss.item()) * xb.size(0)
            n        += xb.size(0)

        val_acc, _ = evaluate(model, val_loader)
        scheduler.step(val_acc)
        history.append({
            'epoch':      ep,
            'train_loss': loss_sum / max(1, n),
            'val_acc':    val_acc,
            'lr':         float(optimizer.param_groups[0]['lr']),
        })

        if ep % 20 == 0 or ep == 1:
            print(f'  Ep {ep:03d} | '
                  f'loss={loss_sum/max(1,n):.4f} | '
                  f'val_acc={val_acc:.4f}')

        if val_acc > best_val_acc + 1e-6:
            best_val_acc = val_acc
            best_state   = {k: v.detach().cpu().clone()
                            for k, v in model.state_dict().items()}
            no_improve   = 0
        else:
            no_improve  += 1
        if no_improve >= PATIENCE:
            print(f'  Early stop ep={ep} | best={best_val_acc:.4f}')
            break

    model.load_state_dict(best_state)
    return pd.DataFrame(history)


# ══════════════════════════════════════════════════════════════
# RUN BOTH MODELS
# ══════════════════════════════════════════════════════════════

print('\n' + '='*60)
print(f'Train {TRAIN_WIN_S}s -> Test {TEST_WIN_S}')
print('='*60)

cnn_model   = CNNClassifier(num_classes=NUM_CLASSES).to(DEVICE)
cnn_history = train_model(cnn_model, 'CNN')

cnn_results = {}
for ws in TEST_WIN_S:
    acc, cm = evaluate(cnn_model, test_loaders[ws])
    cnn_results[ws] = {'acc': acc, 'cm': cm}
    tag = 'same' if ws == TRAIN_WIN_S else 'CROSS'
    print(f'CNN  {ws}s [{tag}]: {acc:.4f}')
del cnn_model; torch.cuda.empty_cache(); gc.collect()

ufno_model   = UFNOClassifier(
    num_classes=NUM_CLASSES, width=WIDTH, depth=DEPTH,
    modes1=MODES1, modes2=MODES2,
    unet_depth=UNET_DEPTH, dropout=DROPOUT).to(DEVICE)
ufno_history = train_model(ufno_model, 'U-FNO')

ufno_results = {}
for ws in TEST_WIN_S:
    acc, cm = evaluate(ufno_model, test_loaders[ws])
    ufno_results[ws] = {'acc': acc, 'cm': cm}
    tag = 'same' if ws == TRAIN_WIN_S else 'CROSS'
    print(f'UFNO {ws}s [{tag}]: {acc:.4f}')
del ufno_model; torch.cuda.empty_cache(); gc.collect()


# ══════════════════════════════════════════════════════════════
# FINAL COMPARISON
# ══════════════════════════════════════════════════════════════

cnn_drop  = cnn_results[1]['acc']  - cnn_results[2]['acc']
ufno_drop = ufno_results[1]['acc'] - ufno_results[2]['acc']
gap       = cnn_drop - ufno_drop

print('\n' + '='*65)
print(f'FINAL COMPARISON - Train {TRAIN_WIN_S}s -> Test 1s / 2s')
print('='*65)
print(f'{"":28} {"CNN":>10} {"U-FNO":>10}')
print('-'*50)
print(f'{"Test 1s (same)":28} '
      f'{cnn_results[1]["acc"]:>10.4f} {ufno_results[1]["acc"]:>10.4f}')
print(f'{"Test 2s (CROSS)":28} '
      f'{cnn_results[2]["acc"]:>10.4f} {ufno_results[2]["acc"]:>10.4f}')
print(f'{"Drop (1s acc - 2s acc)":28} '
      f'{cnn_drop:>+10.4f} {ufno_drop:>+10.4f}')
print(f'{"Gap (CNN_drop - UFNO_drop)":28} {gap:>+10.4f}')


# ══════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, hist, name in zip(axes, [cnn_history, ufno_history],
                          ['CNN', 'U-FNO']):
    ax.plot(hist['epoch'], hist['val_acc'],
            color='steelblue', lw=2, label=f'val_acc (1s)')
    ax.plot(hist['epoch'], hist['train_loss'],
            color='red', lw=1.5, ls='--', label='train_loss')
    ax.set_title(f'{name} - trained on 1s only')
    ax.set_xlabel('Epoch')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.suptitle('Training Curves (1s training)', fontsize=12)
plt.tight_layout()
plt.savefig(OUT_DIR / 'training_curves.png',
            dpi=150, bbox_inches='tight')
plt.close()

fig2, ax2 = plt.subplots(figsize=(8, 5))
x     = np.arange(2)
width = 0.30
labels_x = ['1s (same length)', '2s (CROSS length)']

b1 = ax2.bar(x - width/2,
             [cnn_results[1]['acc'],  cnn_results[2]['acc']],  width,
             label='CNN',   color='steelblue', alpha=0.85)
b2 = ax2.bar(x + width/2,
             [ufno_results[1]['acc'], ufno_results[2]['acc']], width,
             label='U-FNO', color='coral', alpha=0.85)
for bars in [b1, b2]:
    for bar in bars:
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.01,
                 f'{bar.get_height():.3f}',
                 ha='center', va='bottom', fontsize=9)
ax2.axhline(1/NUM_CLASSES, color='red', ls=':', alpha=0.4, label='chance')
ax2.set_xticks(x); ax2.set_xticklabels(labels_x)
ax2.set_ylabel('Test Accuracy'); ax2.set_ylim(0, 1.10)
ax2.set_title(f'Cross-Length: Train 1s -> Test 1s & 2s\n'
              f'CNN drop={cnn_drop:+.3f}   '
              f'U-FNO drop={ufno_drop:+.3f}')
ax2.legend(fontsize=9); ax2.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / 'accuracy_comparison.png',
            dpi=150, bbox_inches='tight')
plt.close()

fig3, axes3 = plt.subplots(2, 2, figsize=(12, 10))
for ax, res, acc, title in [
    (axes3[0,0], cnn_results[1]['cm'],  cnn_results[1]['acc'],
     'CNN - 1s (same)'),
    (axes3[0,1], cnn_results[2]['cm'],  cnn_results[2]['acc'],
     'CNN - 2s (CROSS)'),
    (axes3[1,0], ufno_results[1]['cm'], ufno_results[1]['acc'],
     'U-FNO - 1s (same)'),
    (axes3[1,1], ufno_results[2]['cm'], ufno_results[2]['acc'],
     'U-FNO - 2s (CROSS)'),
]:
    im = ax.imshow(res, cmap='Blues')
    ax.set_title(f'{title}\nacc={acc:.3f}', fontsize=10)
    ax.set_xticks(np.arange(NUM_CLASSES), CLASSES,
                  rotation=45, ha='right')
    ax.set_yticks(np.arange(NUM_CLASSES), CLASSES)
    mx = int(res.max()) if res.size else 1
    for r in range(NUM_CLASSES):
        for c in range(NUM_CLASSES):
            v = int(res[r, c])
            ax.text(c, r, str(v), ha='center', va='center',
                    fontsize=9,
                    color='white' if v > mx/2 else 'black')
    fig3.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.suptitle('Confusion Matrices - Train 1s -> Test 1s & 2s', fontsize=12)
plt.tight_layout()
plt.savefig(OUT_DIR / 'confusion_matrices.png',
            dpi=150, bbox_inches='tight')
plt.close()


# ══════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════

summary = pd.DataFrame({
    'experiment':    ['cross_length_1s_to_2s'],
    'CNN_same_acc':  [cnn_results[1]['acc']],
    'CNN_cross_acc': [cnn_results[2]['acc']],
    'CNN_drop':      [cnn_drop],
    'UFNO_same_acc': [ufno_results[1]['acc']],
    'UFNO_cross_acc':[ufno_results[2]['acc']],
    'UFNO_drop':     [ufno_drop],
    'gap':           [gap],
})
summary.to_csv(OUT_DIR / 'cross_length_summary.csv', index=False)
cnn_history.to_csv(OUT_DIR  / 'cnn_history.csv',  index=False)
ufno_history.to_csv(OUT_DIR / 'ufno_history.csv', index=False)

for ws in TEST_WIN_S:
    for mname, res in [('cnn', cnn_results), ('ufno', ufno_results)]:
        pd.DataFrame(res[ws]['cm'], index=CLASSES, columns=CLASSES
                     ).to_csv(OUT_DIR / f'cm_{mname}_{ws}s.csv')

print('\nSaved to:', OUT_DIR)
print('\n' + '='*65)
print('CROSS-LENGTH - FULL SUMMARY')
print('='*65)
print(summary.round(4).to_string(index=False))
