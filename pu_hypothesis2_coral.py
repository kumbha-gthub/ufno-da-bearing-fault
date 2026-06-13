# ============================================================
# PU Bearing Dataset — Hypothesis 2
# CNN+CORAL  vs  U-FNO+CORAL
# Domain Adaptation: N15_M07_F10 (source) -> 3 limited OPs (targets)
# Target Fraction Sweep [0.1, 0.3, 0.5, 0.7, 0.9]
# ============================================================
#
# CNN model  : CNNEncoder (AdaptiveAvgPool(4,4)) -> feat_dim=2048
#              ~812K parameters
#
# U-FNO model: UFNOEncoder (GAP(1,1)) -> feat_dim=WIDTH=32
#              WIDTH=32, DEPTH=4, MODES=16, UNET_DEPTH=3
#
# Both share identical training loop, CORAL loss, data loading,
# and fixed test set — only the architecture differs.
#
# safe_next() helper prevents StopIteration crash at fraction=0.1
# drop_last=False for CORAL loaders so tiny batches are kept.
# ============================================================

import os, random, gc
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
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

# ════════════════════════════════════════════════════════════════
#                        USER CONFIG
# Set these THREE values before running. Either edit them here, or
# set the environment variables in the cell above the script.
# ════════════════════════════════════════════════════════════════
SEED            = int(os.environ.get('SEED',         0))           # ← 0 then 1 for reseed
PU_DATA_DIR     =     os.environ.get('PU_DATA_DIR',  '/content/drive/MyDrive/PU_Data')
OUTPUTS_DIR     =     os.environ.get('OUTPUTS_DIR',  '/content/drive/MyDrive/PU_Outputs')
# ════════════════════════════════════════════════════════════════

# ── Reproducibility ───────────────────────────────────────────
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
print(f'>>> RUNNING WITH SEED = {SEED} <<<')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if DEVICE == 'cuda':
    torch.backends.cudnn.benchmark     = False
    torch.backends.cudnn.deterministic = True
    print('GPU  :', torch.cuda.get_device_name(0))
    print('VRAM :', round(
        torch.cuda.get_device_properties(0).total_memory/1e9, 2), 'GB')

# ── Paths ─────────────────────────────────────────────────────
PU_DIR  = Path(PU_DATA_DIR)
OUT_DIR = Path(OUTPUTS_DIR) / 'hypothesis2_coral_cnn_vs_ufno' / f'seed_{SEED}'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Sanity check: fail loudly if the data path is wrong
print(f'PU_DIR : {PU_DIR}')
print(f'OUT_DIR: {OUT_DIR}')
if not PU_DIR.exists():
    raise FileNotFoundError(
        f'PU_DIR does not exist: {PU_DIR}\n'
        f'Set PU_DATA_DIR env var or edit USER CONFIG block at top of script.')
sample_bearings = ['K001', 'KA01', 'KI01']
missing = [b for b in sample_bearings if not (PU_DIR / b).exists()]
if missing:
    raise FileNotFoundError(
        f'Missing bearing folders under {PU_DIR}: {missing}\n'
        f'Check that PU_DIR points to the folder containing K001/, KA01/, etc.')
print(f'  Found bearing folders: K001, KA01, KI01 [ok]\n')

# ── Operating conditions ──────────────────────────────────────
ABUNDANT_OP = 'N15_M07_F10'
LIMITED_OPS = ['N15_M01_F10', 'N15_M07_F04', 'N09_M07_F10']
ALL_OPS     = [ABUNDANT_OP] + LIMITED_OPS

# ── Bearings ──────────────────────────────────────────────────
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

# ── Data ──────────────────────────────────────────────────────
WINDOW_OPTIONS_S      = [1, 2]
OVERLAP_RATIO         = 0.0
MAX_SEGMENTS          = 80
MAX_SEGMENTS_ABUNDANT = 9999

# ── Spectrogram ───────────────────────────────────────────────
N_MELS     = 128
N_FFT      = 2048
HOP_LENGTH = 512
F_MIN      = 0
F_MAX      = FS // 2

# ── CNN config ─────────────────────────────────────────────────
CNN_FEAT_DIM = 128 * 4 * 4   # 2048

# ── U-FNO config ──────────────────────────────────────────────
UFNO_WIDTH      = 32
UFNO_DEPTH      = 4
UFNO_MODES1     = 16
UFNO_MODES2     = 16
UFNO_UNET_DEPTH = 3
UFNO_DROPOUT    = 0.40
UFNO_FEAT_DIM   = UFNO_WIDTH

# ── Training (shared) ─────────────────────────────────────────
BATCH_SIZE      = 32
EPOCHS          = 150
PATIENCE        = 60
LR              = 1e-3
WEIGHT_DECAY    = 1e-4
LABEL_SMOOTHING = 0.05
GRAD_CLIP       = 1.0
BETA_MAX        = 1.0

# ── Fraction sweep ────────────────────────────────────────────
TARGET_FRACTIONS = [0.1, 0.3, 0.5, 0.7, 0.9]

print(f'Source       : {ABUNDANT_OP} (100%)')
print(f'Targets      : {LIMITED_OPS}')
print(f'Fractions    : {TARGET_FRACTIONS}')
print(f'Test set     : FIXED reps {TEST_REPS}')
print(f'CORAL beta   : 0 -> {BETA_MAX}')
print(f'CNN          : feat_dim={CNN_FEAT_DIM} (~812K params)')
print(f'U-FNO        : WIDTH={UFNO_WIDTH}, DEPTH={UFNO_DEPTH}, feat_dim={UFNO_FEAT_DIM}')


# ══════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════

def _unwrap(obj):
    while isinstance(obj, np.ndarray) and obj.ndim > 0 and obj.size == 1:
        obj = obj.flat[0]
    return obj

def _iter_channels(arr):
    if arr.ndim == 1: arr = arr.reshape(1, -1)
    for i in range(arr.shape[1]): yield arr[0, i]

def _channel_data(ch):
    try:
        d = ch['Data']
        while isinstance(d, np.ndarray) and d.dtype.kind == 'O' and d.size > 0:
            d = d.flat[0]
        while isinstance(d, np.ndarray) and d.ndim > 0 and d.size == 1:
            d = d.flat[0]
        if isinstance(d, np.ndarray) and d.dtype.kind in 'fiu' and d.size >= 1000:
            return d.reshape(-1)
    except Exception: pass
    return None

def load_pu_signal(path):
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
        except Exception: pass
    if not candidates:
        raise RuntimeError(f'No signal: {path.name}')
    return max(candidates, key=len).astype(np.float32)

def make_mel(signal):
    mel = librosa.feature.melspectrogram(
        y=signal, sr=FS, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=F_MIN, fmax=F_MAX)
    mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    mel = (mel - mel.mean()) / (mel.std() + 1e-8)
    return mel

def make_starts(sig_len, wl, overlap_ratio, max_keep):
    step = int(wl * (1.0 - overlap_ratio)) or wl
    if sig_len < wl: return []
    starts = list(range(0, sig_len - wl + 1, step))
    if len(starts) <= max_keep: return starts
    idx = np.linspace(0, len(starts)-1, max_keep, dtype=int)
    return [starts[i] for i in idx]

def build_rows(op, reps, split_seed):
    rows_by_win = {ws: [] for ws in WINDOW_OPTIONS_S}
    max_seg = MAX_SEGMENTS_ABUNDANT if op == ABUNDANT_OP else MAX_SEGMENTS
    for label, bids in BEARINGS.items():
        y = CLASSES.index(label)
        for bid in bids:
            for rep in reps:
                path = PU_DIR / bid / f'{op}_{bid}_{rep}.mat'
                if not path.exists(): continue
                try: sig = load_pu_signal(path)
                except Exception as e:
                    print(f'  SKIP {path.name}: {e}'); continue
                for ws in WINDOW_OPTIONS_S:
                    wl = ws * FS
                    for s in make_starts(len(sig), wl, OVERLAP_RATIO, max_seg):
                        rows_by_win[ws].append({
                            'mel':    make_mel(sig[s:s+wl]),
                            'label':  y, 'win_s': ws, 'op': op,
                            'domain': 0 if op == ABUNDANT_OP else 1,
                        })
    rng = np.random.default_rng(split_seed)
    for ws in WINDOW_OPTIONS_S: rng.shuffle(rows_by_win[ws])
    return rows_by_win


print('\nBuilding all data in parallel ...')
build_jobs = []
seed_ctr   = 0
for op in ALL_OPS:
    for split, reps in [('train', TRAIN_REPS),
                        ('val',   VAL_REPS),
                        ('test',  TEST_REPS)]:
        build_jobs.append((op, split, reps, SEED + seed_ctr))
        seed_ctr += 1

raw = {}; done = 0
with ThreadPoolExecutor(max_workers=8) as ex:
    fut_map = {
        ex.submit(build_rows, op, reps, seed): (op, split)
        for op, split, reps, seed in build_jobs}
    for fut in as_completed(fut_map):
        op, split = fut_map[fut]
        try:
            raw[(op, split)] = fut.result(); done += 1
            n = sum(len(v) for v in raw[(op, split)].values())
            print(f'  [{done}/{len(build_jobs)}] {op} {split}: {n}')
        except Exception as e:
            print(f'  ERROR {op} {split}: {e}')

all_data = {
    op: {split: raw[(op, split)] for split in ['train','val','test']}
    for op in ALL_OPS}


# ══════════════════════════════════════════════════════════════
# DATASET + LOADERS
# ══════════════════════════════════════════════════════════════

class BearingDataset(Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        x = torch.from_numpy(r['mel']).float().unsqueeze(0)
        y = torch.tensor(int(r['label']), dtype=torch.long)
        d = torch.tensor(int(r['domain']), dtype=torch.long)
        return x, y, d

def make_loaders_by_win(rows_by_win, shuffle, batch_size=BATCH_SIZE,
                        drop_last=None):
    result = {}
    for ws in WINDOW_OPTIONS_S:
        rows = rows_by_win.get(ws)
        if not rows: continue
        n  = len(rows)
        bs = min(batch_size, n)
        dl = drop_last if drop_last is not None else shuffle
        if dl and n < bs: dl = False
        result[ws] = DataLoader(
            BearingDataset(rows), batch_size=bs, shuffle=shuffle,
            num_workers=0, pin_memory=False, drop_last=dl)
    return result

def combine_rows_by_win(list_of_rows_by_win):
    combined = {ws: [] for ws in WINDOW_OPTIONS_S}
    for rbw in list_of_rows_by_win:
        for ws in WINDOW_OPTIONS_S:
            combined[ws].extend(rbw.get(ws, []))
    return combined

FIXED_TEST_LOADERS = {
    op: make_loaders_by_win(all_data[op]['test'], shuffle=False)
    for op in ALL_OPS}

print('\nFixed test set:')
for op in ALL_OPS:
    tag = '(source)' if op == ABUNDANT_OP else '(target)'
    for ws in WINDOW_OPTIONS_S:
        if ws in FIXED_TEST_LOADERS[op]:
            n = len(FIXED_TEST_LOADERS[op][ws].dataset)
            print(f'  {op} {tag} {ws}s: {n}')


# ══════════════════════════════════════════════════════════════
# CORAL LOSS + BETA SCHEDULE
# ══════════════════════════════════════════════════════════════

def coral_loss(source_feat, target_feat):
    d = source_feat.size(1)
    if source_feat.size(0) < 2 or target_feat.size(0) < 2:
        return torch.tensor(0.0, device=source_feat.device)
    src_c = source_feat - source_feat.mean(0, keepdim=True)
    tgt_c = target_feat - target_feat.mean(0, keepdim=True)
    cov_s = (src_c.t() @ src_c) / (source_feat.size(0) - 1)
    cov_t = (tgt_c.t() @ tgt_c) / (target_feat.size(0) - 1)
    return (cov_s - cov_t).pow(2).sum() / (4.0 * d * d)

def get_beta(epoch, total_epochs, beta_max=1.0):
    p = epoch / max(1, total_epochs)
    return beta_max * (2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)


# ══════════════════════════════════════════════════════════════
# CNN MODEL
# ══════════════════════════════════════════════════════════════

class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        def block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch,  out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch), nn.ReLU(),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch), nn.ReLU(),
                nn.MaxPool2d(2), nn.Dropout2d(0.10))
        self.net = nn.Sequential(
            block(1,   32), block(32,  64), block(64, 128),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten())

    def forward(self, x):
        return self.net(x)


class CNNClassifierHead(nn.Module):
    def __init__(self, feat_dim, num_classes, dropout=0.40):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(256, num_classes))

    def forward(self, x):
        return self.net(x)


class CNN_CORAL(nn.Module):
    def __init__(self, num_classes, feat_dim):
        super().__init__()
        self.encoder  = CNNEncoder()
        self.cls_head = CNNClassifierHead(feat_dim, num_classes)

    def forward(self, x):
        feat   = self.encoder(x)
        logits = self.cls_head(feat)
        return logits, feat


# ══════════════════════════════════════════════════════════════
# U-FNO MODEL
# ══════════════════════════════════════════════════════════════

class SpectralConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, modes1=16, modes2=16):
        super().__init__()
        self.modes1  = modes1; self.modes2 = modes2
        scale        = 1.0 / (in_ch * out_ch)
        self.weights = nn.Parameter(
            scale * torch.randn(in_ch, out_ch, modes1, modes2,
                                dtype=torch.cfloat))

    def forward(self, x):
        b, c, h, w = x.shape
        x_ft   = torch.fft.rfft2(x)
        out_ft = torch.zeros(b, self.weights.shape[1], h, w//2+1,
                             device=x.device, dtype=torch.cfloat)
        m1 = min(self.modes1, h); m2 = min(self.modes2, w//2+1)
        out_ft[:,:,:m1,:m2] = torch.einsum(
            'bixy,ioxy->boxy',
            x_ft[:,:,:m1,:m2], self.weights[:,:,:m1,:m2])
        return torch.fft.irfft2(out_ft, s=(h, w))


class UNetPath(nn.Module):
    def __init__(self, width, unet_depth=2):
        super().__init__()
        self.enc = nn.ModuleList([
            nn.Sequential(nn.Conv2d(width, width, 3, stride=2, padding=1),
                          nn.BatchNorm2d(width), nn.GELU())
            for _ in range(unet_depth)])
        self.dec = nn.ModuleList([
            nn.Sequential(nn.ConvTranspose2d(width*2, width, 2, stride=2),
                          nn.BatchNorm2d(width), nn.GELU())
            for _ in range(unet_depth)])

    def forward(self, x):
        skips = []; h = x
        for enc in self.enc: skips.append(h); h = enc(h)
        for dec, skip in zip(self.dec, reversed(skips)):
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:],
                                  mode='bilinear', align_corners=False)
            h = torch.cat([h, skip], dim=1); h = dec(h)
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


class UFNOEncoder(nn.Module):
    def __init__(self, width=32, depth=4, modes1=16, modes2=16,
                 unet_depth=2, dropout=0.40):
        super().__init__()
        self.in_proj = nn.Conv2d(1, width, kernel_size=1)
        self.blocks  = nn.ModuleList([
            UFNOBlock(width, modes1, modes2, unet_depth, dropout)
            for _ in range(depth)])
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.in_proj(x)
        for blk in self.blocks: x = blk(x)
        return self.gap(x).flatten(1)


class UFNOClassifierHead(nn.Module):
    def __init__(self, feat_dim, num_classes, dropout=0.40):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(128, num_classes))

    def forward(self, x):
        return self.net(x)


class UFNO_CORAL(nn.Module):
    def __init__(self, num_classes, feat_dim,
                 width=32, depth=4, modes1=16, modes2=16,
                 unet_depth=2, dropout=0.40):
        super().__init__()
        self.encoder  = UFNOEncoder(width, depth, modes1, modes2,
                                    unet_depth, dropout)
        self.cls_head = UFNOClassifierHead(feat_dim, num_classes, dropout)

    def forward(self, x):
        feat   = self.encoder(x)
        logits = self.cls_head(feat)
        return logits, feat


# ══════════════════════════════════════════════════════════════
# EVALUATE
# ══════════════════════════════════════════════════════════════

def evaluate_loaders(model, loaders_by_win):
    model.eval()
    crit = nn.CrossEntropyLoss()
    all_yt, all_yp = [], []
    per_win = {}
    with torch.no_grad():
        for ws, loader in loaders_by_win.items():
            yt_w, yp_w = [], []
            ls, n = 0.0, 0
            for xb, yb, _ in loader:
                xb = xb.to(DEVICE); yb = yb.to(DEVICE)
                logits, _ = model(xb)
                pred = logits.argmax(1)
                ls  += float(crit(logits, yb).item()) * xb.size(0)
                n   += xb.size(0)
                yt_w.extend(yb.cpu().numpy().tolist())
                yp_w.extend(pred.cpu().numpy().tolist())
            yt_w = np.array(yt_w); yp_w = np.array(yp_w)
            acc  = float((yt_w == yp_w).mean()) if len(yt_w) else 0.0
            per_win[ws] = {'acc': acc, 'loss': ls/max(1,n), 'n': len(yt_w)}
            all_yt.append(yt_w); all_yp.append(yp_w)
    all_yt = np.concatenate(all_yt); all_yp = np.concatenate(all_yp)
    return float((all_yt == all_yp).mean()), all_yt, all_yp, per_win


# ══════════════════════════════════════════════════════════════
# SINGLE EXPERIMENT (one model, one fraction)
# ══════════════════════════════════════════════════════════════

def run_one(model, model_name, fraction, run_seed):
    rng = np.random.default_rng(run_seed)

    train_abundant = all_data[ABUNDANT_OP]['train']
    train_limited_by_op = {}
    for op in LIMITED_OPS:
        lim = {}
        for ws in WINDOW_OPTIONS_S:
            full   = all_data[op]['train'][ws]
            n_keep = max(NUM_CLASSES, int(len(full) * fraction))
            idx    = rng.choice(len(full), size=n_keep, replace=False)
            lim[ws] = [full[i] for i in idx]
        train_limited_by_op[op] = lim

    train_combined = combine_rows_by_win(
        [train_abundant] + list(train_limited_by_op.values()))

    ab_total  = sum(len(train_abundant[ws]) for ws in WINDOW_OPTIONS_S)
    lim_total = sum(len(train_limited_by_op[op][ws])
                    for op in LIMITED_OPS for ws in WINDOW_OPTIONS_S)
    grand_total = sum(len(v) for v in train_combined.values())

    print(f'\n  {"-"*58}')
    print(f'  [{model_name}] TRAINING DATA  (fraction={fraction})')
    print(f'  source={ab_total}  target={lim_total}  total={grand_total}')
    print(f'  {"-"*58}')

    all_labels = [r['label'] for ws in WINDOW_OPTIONS_S
                  for r in train_combined[ws]]
    cnt = np.bincount(all_labels, minlength=NUM_CLASSES).astype(np.float32)
    w   = cnt.sum() / np.maximum(cnt, 1.0); w = w / w.mean()
    cw  = torch.tensor(w, dtype=torch.float32, device=DEVICE)
    ce_crit = nn.CrossEntropyLoss(weight=cw, label_smoothing=LABEL_SMOOTHING)

    train_loaders = make_loaders_by_win(train_combined, shuffle=True)
    val_loaders   = make_loaders_by_win(
        all_data[ABUNDANT_OP]['val'], shuffle=False)

    src_loaders = make_loaders_by_win(
        all_data[ABUNDANT_OP]['train'], shuffle=True, drop_last=False)
    tgt_combined = combine_rows_by_win(list(train_limited_by_op.values()))
    tgt_loaders  = make_loaders_by_win(
        tgt_combined, shuffle=True, drop_last=False,
        batch_size=max(2, BATCH_SIZE // 2))

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6)

    history      = []
    best_val_acc = -1.0
    best_state   = None
    no_improve   = 0

    src_iters = {ws: iter(src_loaders[ws]) for ws in src_loaders}
    tgt_iters = {ws: iter(tgt_loaders[ws]) for ws in tgt_loaders}

    def safe_next(iters_dict, loaders_dict, ws):
        if ws not in loaders_dict: return None
        if len(loaders_dict[ws].dataset) == 0: return None
        try:
            return next(iters_dict[ws])
        except StopIteration:
            iters_dict[ws] = iter(loaders_dict[ws])
            try:    return next(iters_dict[ws])
            except StopIteration: return None

    print(f'  [{model_name}] Training ...')
    for ep in range(1, EPOCHS + 1):
        model.train()
        beta = get_beta(ep, EPOCHS, BETA_MAX)
        ce_s, coral_s, n = 0.0, 0.0, 0

        for ws in WINDOW_OPTIONS_S:
            if ws not in train_loaders: continue
            for xb, yb, _ in train_loaders[ws]:
                xb = xb.to(DEVICE); yb = yb.to(DEVICE)

                b, c, f, t = xb.shape
                fm = random.randint(0, max(1, f//10))
                fs = random.randint(0, max(0, f-fm))
                xb[:,:,fs:fs+fm,:] = 0.0
                tm = random.randint(0, max(1, t//10))
                ts = random.randint(0, max(0, t-tm))
                xb[:,:,:,ts:ts+tm] = 0.0

                logits, _ = model(xb)
                loss_ce   = ce_crit(logits, yb)

                loss_coral = torch.tensor(0.0, device=DEVICE)
                sb = safe_next(src_iters, src_loaders, ws)
                tb = safe_next(tgt_iters, tgt_loaders, ws)
                if sb is not None and tb is not None:
                    xs = sb[0].to(DEVICE); xt = tb[0].to(DEVICE)
                    _, sf = model(xs); _, tf = model(xt)
                    loss_coral = coral_loss(sf, tf)

                loss = loss_ce + beta * loss_coral
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), GRAD_CLIP)
                optimizer.step()

                ce_s    += float(loss_ce.item())    * xb.size(0)
                coral_s += float(loss_coral.item()) * xb.size(0)
                n       += xb.size(0)

        val_acc, _, _, val_pw = evaluate_loaders(model, val_loaders)
        scheduler.step(val_acc)

        row = {'epoch': ep, 'beta': beta,
               'ce_loss': ce_s/max(1,n),
               'coral_loss': coral_s/max(1,n),
               'val_acc': val_acc,
               'lr': float(optimizer.param_groups[0]['lr'])}
        for ws in WINDOW_OPTIONS_S:
            if ws in val_pw: row[f'val_acc_{ws}s'] = val_pw[ws]['acc']
        history.append(row)

        if ep % 15 == 0 or ep == 1:
            ws_str = '  '.join(f'{ws}s={val_pw[ws]["acc"]:.3f}'
                               for ws in WINDOW_OPTIONS_S if ws in val_pw)
            print(f'  [{model_name}] frac={fraction} Ep{ep:03d} '
                  f'b={beta:.2f} ce={ce_s/max(1,n):.4f} '
                  f'coral={coral_s/max(1,n):.4f} '
                  f'val={val_acc:.4f} [{ws_str}]')

        if val_acc > best_val_acc + 1e-6:
            best_val_acc = val_acc
            best_state   = {k: v.detach().cpu().clone()
                            for k, v in model.state_dict().items()}
            no_improve   = 0
        else:
            no_improve += 1
        if no_improve >= PATIENCE:
            print(f'  [{model_name}] early stop ep={ep} '
                  f'best={best_val_acc:.4f}')
            break

    model.load_state_dict(best_state)

    test_results = {}
    for op in ALL_OPS:
        acc, yt, yp, pw = evaluate_loaders(model, FIXED_TEST_LOADERS[op])
        cm  = confusion_matrix(yt, yp, labels=np.arange(NUM_CLASSES))
        test_results[op] = {'acc': acc, 'per_win': pw, 'cm': cm,
                            'yt': yt.tolist(), 'yp': yp.tolist()}
        tag = '(src)' if op == ABUNDANT_OP else '(tgt)'
        print(f'  [{model_name}] frac={fraction} {op}{tag}: {acc:.4f}  ' +
              '  '.join(f'{ws}s={pw[ws]["acc"]:.4f}'
                        for ws in WINDOW_OPTIONS_S if ws in pw))

    return {
        'fraction': fraction, 'history': pd.DataFrame(history),
        'best_val': best_val_acc, 'test_results': test_results,
        'n_abundant': ab_total, 'n_limited': lim_total,
    }


# ══════════════════════════════════════════════════════════════
# MAIN SWEEP
# ══════════════════════════════════════════════════════════════

print('\n' + '='*70)
print('Hypothesis 2: CNN+CORAL vs U-FNO+CORAL - PU Dataset')
print('='*70)

cnn_results  = {}
ufno_results = {}
all_rows     = []

for fi, frac in enumerate(TARGET_FRACTIONS):
    print(f'\n{"="*70}')
    print(f'[{fi+1}/{len(TARGET_FRACTIONS)}] FRACTION = {frac}')
    print('='*70)

    print(f'\n  --- CNN+CORAL ---')
    cnn_model = CNN_CORAL(num_classes=NUM_CLASSES,
                          feat_dim=CNN_FEAT_DIM).to(DEVICE)
    print(f'  CNN params: {sum(p.numel() for p in cnn_model.parameters()):,}')
    cnn_res = run_one(cnn_model, 'CNN', frac, SEED + fi)
    cnn_results[frac] = cnn_res
    del cnn_model; torch.cuda.empty_cache(); gc.collect()

    print(f'\n  --- U-FNO+CORAL ---')
    ufno_model = UFNO_CORAL(
        num_classes=NUM_CLASSES, feat_dim=UFNO_FEAT_DIM,
        width=UFNO_WIDTH, depth=UFNO_DEPTH,
        modes1=UFNO_MODES1, modes2=UFNO_MODES2,
        unet_depth=UFNO_UNET_DEPTH, dropout=UFNO_DROPOUT).to(DEVICE)
    print(f'  UFNO params: {sum(p.numel() for p in ufno_model.parameters()):,}')
    ufno_res = run_one(ufno_model, 'UFNO', frac, SEED + fi)
    ufno_results[frac] = ufno_res
    del ufno_model; torch.cuda.empty_cache(); gc.collect()

    for mname, res in [('CNN', cnn_res), ('UFNO', ufno_res)]:
        for op in ALL_OPS:
            tr = res['test_results'][op]
            all_rows.append({
                'model':       mname,
                'fraction':    frac,
                'op':          op,
                'is_source':   op == ABUNDANT_OP,
                'overall_acc': tr['acc'],
                'acc_1s':      tr['per_win'].get(1, {}).get('acc', None),
                'acc_2s':      tr['per_win'].get(2, {}).get('acc', None),
            })


# ══════════════════════════════════════════════════════════════
# SUMMARY TABLES
# ══════════════════════════════════════════════════════════════

sweep_df   = pd.DataFrame(all_rows)
limited_df = sweep_df[~sweep_df['is_source']]

avg_limited = (limited_df
               .groupby(['model','fraction'])['overall_acc']
               .mean().unstack('model').round(3))

print('\n' + '='*70)
print('HYPOTHESIS 2 RESULTS - Average over 3 target OPs')
print('='*70)
print(avg_limited.to_string())

for mname, label in [('CNN', 'CNN+CORAL'), ('UFNO', 'U-FNO+CORAL')]:
    print(f'\n--- {label} (per operating condition) ---')
    piv = (sweep_df[sweep_df['model']==mname]
           .pivot_table(index='fraction', columns='op',
                        values='overall_acc').round(3))
    print(piv.to_string())


# ══════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for mname, color, ls in [('CNN', 'steelblue', '--'),
                         ('UFNO', 'coral',    '-')]:
    mdf  = limited_df[limited_df['model'] == mname]
    avg  = mdf.groupby('fraction')['overall_acc'].mean()
    avg1 = mdf.groupby('fraction')['acc_1s'].mean()
    avg2 = mdf.groupby('fraction')['acc_2s'].mean()
    lbl  = f'{"CNN" if mname=="CNN" else "U-FNO"}+CORAL'
    axes[0].plot(avg.index,  avg.values,  marker='o', lw=2,
                 color=color, ls=ls, label=lbl)
    axes[1].plot(avg1.index, avg1.values, marker='o', lw=2,
                 color=color, ls=ls, label=lbl)
    axes[2].plot(avg2.index, avg2.values, marker='s', lw=2,
                 color=color, ls=ls, label=lbl)

for ax, title in zip(axes, ['Overall', '1s Window', '2s Window']):
    ax.axhline(1/NUM_CLASSES, color='red', ls=':', alpha=0.4,
               label='chance')
    ax.set_xlabel('Fraction of Target Data')
    ax.set_ylabel('Avg Accuracy (3 target OPs)')
    ax.set_title(f'PU: CNN+CORAL vs U-FNO+CORAL\n{title}')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.set_xticks(TARGET_FRACTIONS); ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig(OUT_DIR / 'cnn_vs_ufno_avg_targets.png',
            dpi=150, bbox_inches='tight')
plt.close()

fig2, axes2 = plt.subplots(1, len(LIMITED_OPS),
                           figsize=(6*len(LIMITED_OPS), 5))
for i, op in enumerate(LIMITED_OPS):
    ax = axes2[i]
    for mname, color, ls in [('CNN', 'steelblue', '--'),
                             ('UFNO', 'coral', '-')]:
        mdf = sweep_df[(sweep_df['model']==mname) & (sweep_df['op']==op)]
        lbl = f'{"CNN" if mname=="CNN" else "U-FNO"}+CORAL'
        ax.plot(mdf['fraction'], mdf['overall_acc'], marker='o',
                lw=2, color=color, ls=ls, label=lbl)
    ax.axhline(1/NUM_CLASSES, color='red', ls=':', alpha=0.4)
    ax.set_title(f'Target: {op}', fontsize=10)
    ax.set_xlabel('Target Data Fraction')
    ax.set_ylabel('Test Accuracy')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.set_xticks(TARGET_FRACTIONS); ax.set_ylim(0, 1.05)
plt.suptitle('PU: CNN+CORAL vs U-FNO+CORAL per Target OP', fontsize=11)
plt.tight_layout()
plt.savefig(OUT_DIR / 'per_op_comparison.png',
            dpi=150, bbox_inches='tight')
plt.close()

for mname, results_dict, color in [
    ('CNN',   cnn_results,  'steelblue'),
    ('U-FNO', ufno_results, 'coral'),
]:
    hist = results_dict[0.1]['history']
    fig3, ax3 = plt.subplots(1, 2, figsize=(10, 4))
    ax3[0].plot(hist['epoch'], hist['ce_loss'],
                color=color, lw=2, label='CE loss')
    ax3[0].plot(hist['epoch'], hist['val_acc'],
                color='green', ls='--', lw=2, label='val_acc')
    ax3[0].set_title(f'{mname}+CORAL - CE & val_acc (frac=0.1)')
    ax3[0].legend(); ax3[0].grid(alpha=0.3)
    ax3[1].plot(hist['epoch'], hist['coral_loss'],
                color=color, lw=2, label='CORAL loss')
    ax3b = ax3[1].twinx()
    ax3b.plot(hist['epoch'], hist['beta'],
              color='purple', lw=1, alpha=0.5, label='beta')
    ax3b.set_ylabel('beta', color='purple')
    ax3[1].set_title(f'{mname}+CORAL - CORAL loss (frac=0.1)')
    ax3[1].legend(); ax3[1].grid(alpha=0.3)
    plt.suptitle(f'{mname}+CORAL Training Dynamics', fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f'loss_{mname.lower().replace("-","")}_frac01.png',
                dpi=150, bbox_inches='tight')
    plt.close()


# ══════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════

sweep_df['seed'] = SEED
# Collapse flag: src accuracy < 0.5 = training stuck at chance
sweep_df['collapsed'] = sweep_df.apply(
    lambda r: bool(sweep_df[(sweep_df['model']==r['model']) &
                            (sweep_df['fraction']==r['fraction']) &
                            (sweep_df['is_source'])]['overall_acc'].iloc[0] < 0.5), axis=1)
sweep_df.to_csv(OUT_DIR / 'sweep_results.csv', index=False)
avg_limited.to_csv(OUT_DIR / 'avg_limited_ops.csv')

# Per-sample predictions for McNemar tests later
import json as _json
preds_dump = {}
for frac in TARGET_FRACTIONS:
    for mname, res in [('cnn', cnn_results[frac]),
                       ('ufno', ufno_results[frac])]:
        for op in ALL_OPS:
            tr = res['test_results'][op]
            key = f'{mname}__frac{frac}__{op}'
            preds_dump[key] = {'yt': tr['yt'], 'yp': tr['yp']}
with open(OUT_DIR / 'predictions.json', 'w') as f:
    _json.dump({'seed': SEED, 'predictions': preds_dump}, f)
print(f'  Saved per-sample predictions to {OUT_DIR / "predictions.json"}')

for frac in TARGET_FRACTIONS:
    tag = str(frac).replace('.', 'p')
    cnn_results[frac]['history'].to_csv(
        OUT_DIR / f'cnn_history_frac{tag}.csv', index=False)
    ufno_results[frac]['history'].to_csv(
        OUT_DIR / f'ufno_history_frac{tag}.csv', index=False)
    for op in ALL_OPS:
        for mname, res in [('cnn', cnn_results[frac]),
                           ('ufno', ufno_results[frac])]:
            cm = res['test_results'][op]['cm']
            pd.DataFrame(cm, index=CLASSES, columns=CLASSES
                         ).to_csv(OUT_DIR / f'cm_{mname}_{op}_frac{tag}.csv')

print('\nSaved to:', OUT_DIR)
print('\n' + '='*70)
print('FINAL SUMMARY')
print('='*70)
print('\nAverage accuracy on target OPs (CNN vs U-FNO):')
print(avg_limited.round(3).to_string())
