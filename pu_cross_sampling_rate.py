# ============================================================
# PU Bearing Dataset
# CNN vs U-FNO — Cross Sampling Rate (Artificial Downsampling)
# Train on 64kHz, Test on 64kHz + 32kHz + 16kHz
# ============================================================
#
# WHY ARTIFICIAL DOWNSAMPLING:
#   PU dataset is always 64kHz — no natural multi-rate variant.
#   We simulate deployment scenarios where a sensor records at
#   lower quality (32kHz or 16kHz) than the training data.
#
# SPECTROGRAM STRATEGY (CALIBRATED):
#   Scale N_FFT and HOP proportionally with FS so that the
#   same physical fault frequencies map to same mel bins.
#   F_MAX capped at 8kHz for all rates (bearing faults < 8kHz).
#
#   64kHz: N_FFT=2048, HOP=512   → (128, ~126) per 1s window
#   32kHz: N_FFT=1024, HOP=256   → (128, ~126) per 1s window
#   16kHz: N_FFT=512,  HOP=128   → (128, ~126) per 1s window
# ============================================================

import os, random, gc
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
OUT_DIR = Path(os.environ.get('OUTPUTS_DIR', './outputs')) / 'cross_sampling_rate' / f'seed_{SEED}'
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

FS_NATIVE  = 64000   # PU native sampling rate
TRAIN_REPS = list(range(1, 15))
VAL_REPS   = list(range(15, 18))
TEST_REPS  = list(range(18, 21))

TRAIN_FS   = 64000
TEST_RATES = [64000, 32000, 16000]

WIN_S          = 1
OVERLAP_RATIO  = 0.0
MAX_SEGMENTS   = 80

# ── CALIBRATED Spectrogram params ────────────────────────────
N_MELS   = 128
F_MIN    = 0
F_MAX    = 8000   # cap at 8kHz for all rates

N_FFT_BASE = 2048
HOP_BASE   = 512

def get_spec_params(fs: int):
    """Scale FFT params proportionally to FS."""
    ratio      = fs // 16000
    base_ratio = TRAIN_FS // 16000
    scale      = ratio / base_ratio
    n_fft      = max(64, int(N_FFT_BASE * scale))
    hop        = max(16, int(HOP_BASE  * scale))
    n_fft = int(2 ** round(np.log2(n_fft)))
    hop   = int(2 ** round(np.log2(hop)))
    return n_fft, hop

print('Spectrogram params and shapes per sampling rate:')
for fs in TEST_RATES:
    n_fft, hop = get_spec_params(fs)
    dummy      = np.zeros(WIN_S * fs, dtype=np.float32)
    mel        = librosa.feature.melspectrogram(
        y=dummy, sr=fs, n_fft=n_fft, hop_length=hop,
        n_mels=N_MELS, fmin=F_MIN, fmax=F_MAX)
    tag = 'TRAIN' if fs == TRAIN_FS else 'CROSS'
    print(f'  {fs}Hz [{tag}]: N_FFT={n_fft}, HOP={hop} '
          f'→ shape (1, {mel.shape[0]}, {mel.shape[1]})')

# ── U-FNO config ──────────────────────────────────────────────
WIDTH      = 64
DEPTH      = 2
MODES1     = 12
MODES2     = 12
UNET_DEPTH = 1
DROPOUT    = 0.30

# ── Training ──────────────────────────────────────────────────
BATCH_SIZE      = 32
EPOCHS          = 200
PATIENCE        = 50
LR              = 1e-3
WEIGHT_DECAY    = 1e-4
LABEL_SMOOTHING = 0.05
GRAD_CLIP       = 1.0


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


def make_mel_for_rate(signal_native: np.ndarray,
                      target_fs: int) -> np.ndarray:
    if target_fs < FS_NATIVE:
        sig = librosa.resample(
            signal_native, orig_sr=FS_NATIVE, target_sr=target_fs)
    else:
        sig = signal_native

    n_fft, hop = get_spec_params(target_fs)
    mel = librosa.feature.melspectrogram(
        y=sig, sr=target_fs,
        n_fft=n_fft, hop_length=hop,
        n_mels=N_MELS, fmin=F_MIN, fmax=F_MAX)
    mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    mel = (mel - mel.mean()) / (mel.std() + 1e-8)
    return mel


def make_starts(sig_len: int, wl: int, max_keep: int):
    step   = wl
    starts = list(range(0, sig_len - wl + 1, step))
    if len(starts) > max_keep:
        idx    = np.linspace(0, len(starts)-1, max_keep, dtype=int)
        starts = [starts[i] for i in idx]
    return starts


def build_rows_for_rates(reps: list, target_rates: list,
                         split_seed: int):
    rows_by_rate = {fs: [] for fs in target_rates}

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

                wl     = WIN_S * FS_NATIVE
                starts = make_starts(len(sig), wl, MAX_SEGMENTS)

                for s in starts:
                    seg_native = sig[s:s + wl]
                    for fs in target_rates:
                        mel = make_mel_for_rate(seg_native, fs)
                        rows_by_rate[fs].append({
                            'mel':   mel,
                            'label': y,
                            'fs':    fs,
                        })

    rng = np.random.default_rng(split_seed)
    for fs in target_rates:
        rng.shuffle(rows_by_rate[fs])

    return rows_by_rate


# ── Build data ────────────────────────────────────────────────
print(f'\nBuilding data for all sampling rates ...')
print(f'  Building train ({TRAIN_FS}Hz) ...')
train_all = build_rows_for_rates(TRAIN_REPS, [TRAIN_FS], SEED)
train_rows = train_all[TRAIN_FS]

print(f'  Building val ({TRAIN_FS}Hz) ...')
val_all  = build_rows_for_rates(VAL_REPS, [TRAIN_FS], SEED + 1)
val_rows = val_all[TRAIN_FS]

print(f'  Building test (all rates: {TEST_RATES}) ...')
test_all = build_rows_for_rates(TEST_REPS, TEST_RATES, SEED + 2)

print(f'\nData summary:')
print(f'  Train ({TRAIN_FS}Hz): {len(train_rows)}')
print(f'  Val   ({TRAIN_FS}Hz): {len(val_rows)}')
for fs in TEST_RATES:
    tag = 'same rate' if fs == TRAIN_FS else f'CROSS ({TRAIN_FS//fs}x down)'
    print(f'  Test  ({fs}Hz): {len(test_all[fs])} [{tag}]')

train_sh = train_rows[0]['mel'].shape if train_rows else None
print('\nShape check:')
for fs in TEST_RATES:
    if test_all[fs]:
        sh  = test_all[fs][0]['mel'].shape
        tag = 'OK' if sh == train_sh else 'MISMATCH - check params'
        print(f'  {fs}Hz: {sh}  [{tag}]')


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

def make_loader(rows, shuffle, batch_size=BATCH_SIZE):
    return DataLoader(
        BearingDataset(rows),
        batch_size=batch_size, shuffle=shuffle,
        num_workers=0, pin_memory=False, drop_last=shuffle)

train_loader = make_loader(train_rows, shuffle=True)
val_loader   = make_loader(val_rows,   shuffle=False)

FIXED_TEST_LOADERS = {
    fs: make_loader(test_all[fs], shuffle=False)
    for fs in TEST_RATES
}

# Class weights
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

def evaluate_all_rates(model):
    model.eval()
    results = {}
    with torch.no_grad():
        for fs, loader in FIXED_TEST_LOADERS.items():
            all_yt, all_yp = [], []
            for xb, yb in loader:
                xb = xb.to(DEVICE); yb = yb.to(DEVICE)
                pred = model(xb).argmax(1)
                all_yt.extend(yb.cpu().numpy().tolist())
                all_yp.extend(pred.cpu().numpy().tolist())
            all_yt = np.array(all_yt)
            all_yp = np.array(all_yp)
            results[fs] = {
                'acc': float((all_yt == all_yp).mean()),
                'cm':  confusion_matrix(all_yt, all_yp,
                                        labels=np.arange(NUM_CLASSES)),
            }
    return results


def evaluate_val(model):
    model.eval()
    all_yt, all_yp = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(DEVICE); yb = yb.to(DEVICE)
            pred = model(xb).argmax(1)
            all_yt.extend(yb.cpu().numpy().tolist())
            all_yp.extend(pred.cpu().numpy().tolist())
    return float((np.array(all_yt) == np.array(all_yp)).mean())


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
                nn.MaxPool2d(2),
                nn.Dropout2d(0.10),
            )
        self.features = nn.Sequential(
            block(1,   32),
            block(32,  64),
            block(64, 128),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.40),
            nn.Linear(256, num_classes),
        )
    def forward(self, x):
        return self.classifier(self.features(x))


# ══════════════════════════════════════════════════════════════
# U-FNO MODEL
# ══════════════════════════════════════════════════════════════

class SpectralConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, modes1=16, modes2=16):
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
    def __init__(self, width, unet_depth=2):
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
            skips.append(h)
            h = enc(h)
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
    def __init__(self, width, modes1=16, modes2=16,
                 unet_depth=2, dropout=0.10):
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
    def __init__(self, num_classes, width=32, depth=4,
                 modes1=16, modes2=16, unet_depth=2, dropout=0.30):
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
    print(f'Training {model_name}  [{TRAIN_FS}Hz only]')
    print(f'  n_train={len(train_rows)}, n_val={len(val_rows)}')
    print(f'  params={sum(p.numel() for p in model.parameters()):,}')
    print(f'{"="*60}')

    criterion = nn.CrossEntropyLoss(
        weight=CLASS_WEIGHTS, label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5,
        patience=5, min_lr=1e-6)

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

        val_acc = evaluate_val(model)
        scheduler.step(val_acc)
        history.append({
            'epoch':      ep,
            'train_loss': loss_sum / max(1, n),
            'val_acc':    val_acc,
        })

        if ep % 10 == 0 or ep == 1:
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

print('\n' + '='*65)
print(f'PU Cross Sampling Rate: Train {TRAIN_FS}Hz → Test {TEST_RATES}')
print('='*65)

cnn_model   = CNNClassifier(num_classes=NUM_CLASSES).to(DEVICE)
cnn_history = train_model(cnn_model, 'CNN')
cnn_results = evaluate_all_rates(cnn_model)

print('\nCNN results:')
base_cnn = cnn_results[TRAIN_FS]['acc']
for fs in TEST_RATES:
    acc  = cnn_results[fs]['acc']
    tag  = 'same' if fs == TRAIN_FS else f'CROSS {TRAIN_FS//fs}x down'
    drop = '' if fs == TRAIN_FS else f' drop={base_cnn-acc:+.4f}'
    print(f'  {fs}Hz [{tag}]: {acc:.4f}{drop}')
del cnn_model; torch.cuda.empty_cache(); gc.collect()

ufno_model   = UFNOClassifier(
    num_classes=NUM_CLASSES, width=WIDTH, depth=DEPTH,
    modes1=MODES1, modes2=MODES2,
    unet_depth=UNET_DEPTH, dropout=DROPOUT).to(DEVICE)
ufno_history = train_model(ufno_model, 'U-FNO')
ufno_results = evaluate_all_rates(ufno_model)

print('\nU-FNO results:')
base_ufno = ufno_results[TRAIN_FS]['acc']
for fs in TEST_RATES:
    acc  = ufno_results[fs]['acc']
    tag  = 'same' if fs == TRAIN_FS else f'CROSS {TRAIN_FS//fs}x down'
    drop = '' if fs == TRAIN_FS else f' drop={base_ufno-acc:+.4f}'
    print(f'  {fs}Hz [{tag}]: {acc:.4f}{drop}')
del ufno_model; torch.cuda.empty_cache(); gc.collect()


# ══════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════

print('\n' + '='*70)
print('FINAL COMPARISON — PU Cross Sampling Rate')
print(f'Trained on {TRAIN_FS}Hz, Tested on {TEST_RATES}')
print('='*70)
print(f'{"rate":>8} {"tag":>12} {"CNN_acc":>10} '
      f'{"UFNO_acc":>10} {"CNN_drop":>10} {"UFNO_drop":>10} {"gap":>8}')
print('-'*72)

rows_out = []
for fs in TEST_RATES:
    ca    = cnn_results[fs]['acc']
    ua    = ufno_results[fs]['acc']
    cdrop = base_cnn  - ca if fs != TRAIN_FS else 0.0
    udrop = base_ufno - ua if fs != TRAIN_FS else 0.0
    gap   = cdrop - udrop
    tag   = 'same' if fs == TRAIN_FS else f'{TRAIN_FS//fs}x down'
    print(f'{fs:>8} {tag:>12} {ca:>10.4f} {ua:>10.4f} '
          f'{cdrop:>+10.4f} {udrop:>+10.4f} {gap:>+8.4f}')
    rows_out.append({'rate_hz': fs, 'tag': tag,
                     'CNN_acc': ca, 'UFNO_acc': ua,
                     'CNN_drop': cdrop, 'UFNO_drop': udrop,
                     'gap': gap})

summary_df = pd.DataFrame(rows_out)
cross_rows = summary_df[summary_df['tag'] != 'same']
if len(cross_rows):
    avg_gap = cross_rows['gap'].mean()
    print(f'\nMean gap: {avg_gap:+.4f}  '
          f'(positive = CNN drops more = U-FNO more robust)')


# ══════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, hist, name in zip(axes, [cnn_history, ufno_history],
                          ['CNN', 'U-FNO']):
    ax.plot(hist['epoch'], hist['val_acc'],
            color='steelblue', lw=2, label='val_acc')
    ax.plot(hist['epoch'], hist['train_loss'],
            color='red', lw=1.5, ls='--', label='train_loss')
    ax.set_title(f'{name} - trained on {TRAIN_FS}Hz')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.suptitle('Training Curves', fontsize=12)
plt.tight_layout()
plt.savefig(OUT_DIR / 'training_curves.png',
            dpi=150, bbox_inches='tight')
plt.close()

fig2, ax2 = plt.subplots(figsize=(9, 5))
x     = np.arange(len(TEST_RATES))
width = 0.30
cnn_accs  = [cnn_results[fs]['acc']  for fs in TEST_RATES]
ufno_accs = [ufno_results[fs]['acc'] for fs in TEST_RATES]
b1 = ax2.bar(x - width/2, cnn_accs,  width,
             label='CNN',   color='steelblue', alpha=0.85)
b2 = ax2.bar(x + width/2, ufno_accs, width,
             label='U-FNO', color='coral', alpha=0.85)
for bars in [b1, b2]:
    for bar in bars:
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.01,
                 f'{bar.get_height():.3f}',
                 ha='center', va='bottom', fontsize=9)
ax2.axhline(1/NUM_CLASSES, color='red', ls=':', alpha=0.4, label='chance')
ax2.set_xticks(x)
ax2.set_xticklabels(
    [f'{fs}Hz\n({"same" if fs==TRAIN_FS else str(TRAIN_FS//fs)+"x down"})'
     for fs in TEST_RATES])
ax2.set_ylabel('Test Accuracy')
ax2.set_ylim(0, 1.10)
ax2.set_title('PU: Robustness to Sensor Quality Degradation\n'
              f'Trained {TRAIN_FS}Hz, tested at lower rates')
ax2.legend(fontsize=9); ax2.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / 'accuracy_vs_rate.png',
            dpi=150, bbox_inches='tight')
plt.close()

fig3, axes3 = plt.subplots(2, len(TEST_RATES),
                           figsize=(6*len(TEST_RATES), 9))
for ri, (model_name, results) in enumerate(
        [('CNN', cnn_results), ('U-FNO', ufno_results)]):
    for ci, fs in enumerate(TEST_RATES):
        ax  = axes3[ri, ci]
        cm  = results[fs]['cm']
        acc = results[fs]['acc']
        im  = ax.imshow(cm, cmap='Blues')
        tag = 'same' if fs == TRAIN_FS else f'CROSS {TRAIN_FS//fs}x'
        ax.set_title(f'{model_name}\n{fs}Hz [{tag}]\nacc={acc:.3f}',
                     fontsize=9)
        ax.set_xticks(np.arange(NUM_CLASSES), CLASSES,
                      rotation=45, ha='right')
        ax.set_yticks(np.arange(NUM_CLASSES), CLASSES)
        mx = int(cm.max()) if cm.size else 1
        for r in range(NUM_CLASSES):
            for c in range(NUM_CLASSES):
                v = int(cm[r, c])
                ax.text(c, r, str(v), ha='center', va='center',
                        fontsize=8,
                        color='white' if v > mx/2 else 'black')
        fig3.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.suptitle('PU Confusion Matrices - Cross Sampling Rate', fontsize=11)
plt.tight_layout()
plt.savefig(OUT_DIR / 'confusion_matrices.png',
            dpi=150, bbox_inches='tight')
plt.close()


# ══════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════

summary_df.to_csv(OUT_DIR / 'cross_rate_summary.csv', index=False)
cnn_history.to_csv(OUT_DIR  / 'cnn_history.csv',  index=False)
ufno_history.to_csv(OUT_DIR / 'ufno_history.csv', index=False)

for fs in TEST_RATES:
    for mname, res in [('cnn', cnn_results), ('ufno', ufno_results)]:
        pd.DataFrame(res[fs]['cm'], index=CLASSES, columns=CLASSES
                     ).to_csv(OUT_DIR / f'cm_{mname}_{fs}hz.csv')

print('\nSaved to:', OUT_DIR)
print('\nFINAL TABLE:')
print(summary_df.round(4).to_string(index=False))
