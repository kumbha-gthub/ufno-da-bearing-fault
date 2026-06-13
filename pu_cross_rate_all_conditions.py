# ============================================================
# PU Bearing Dataset — H1 Cross-Rate: All Operating Conditions
# + Plain FNO Ablation (no UNet path)
#
# Runs cross-rate experiment (Train 64kHz -> Test 64/32/16kHz)
# for ALL 4 operating conditions:
#   N15_M07_F10, N15_M01_F10, N15_M07_F04, N09_M07_F10
#
# Also runs PLAIN FNO (no UNet path) on N15_M07_F10 to
# answer the parameter-count ablation question:
#   CNN (812K) vs Plain FNO (~1.2M) vs U-FNO (1.34M)
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
OUT_DIR = Path(OUTPUTS_DIR) / 'cross_rate_all_conditions' / f'seed_{SEED}'
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

# ── PU Config ─────────────────────────────────────────────────
ALL_CONDITIONS = [
    'N15_M07_F10',
    'N15_M01_F10',
    'N15_M07_F04',
    'N09_M07_F10',
]

BEARINGS = {
    'healthy':    ['K001', 'K002', 'K003'],
    'outer_race': ['KA01', 'KA03', 'KA04'],
    'inner_race': ['KI01', 'KI03', 'KI04'],
}
CLASSES     = list(BEARINGS.keys())
NUM_CLASSES = len(CLASSES)

FS_NATIVE  = 64000
TRAIN_REPS = list(range(1, 15))
VAL_REPS   = list(range(15, 18))
TEST_REPS  = list(range(18, 21))

TRAIN_FS   = 64000
TEST_RATES = [64000, 32000, 16000]

WIN_S        = 1
OVERLAP      = 0.0
MAX_SEGMENTS = 80

# ── Calibrated spectrogram ────────────────────────────────────
N_MELS     = 128
F_MIN      = 0
F_MAX      = 8000

N_FFT_BASE = 2048
HOP_BASE   = 512

def get_spec_params(fs):
    ratio  = fs // 16000
    base_r = FS_NATIVE // 16000
    scale  = ratio / base_r
    n_fft  = int(2 ** round(np.log2(max(64,  int(N_FFT_BASE * scale)))))
    hop    = int(2 ** round(np.log2(max(16,  int(HOP_BASE   * scale)))))
    return n_fft, hop

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

print('Operating conditions:', ALL_CONDITIONS)
print('Test rates:', TEST_RATES)
print(f'\nU-FNO hyperparams: WIDTH={WIDTH}, DEPTH={DEPTH}, '
      f'MODES={MODES1}x{MODES2}, UNET_DEPTH={UNET_DEPTH}')
print(f'Expected params  : CNN=812,643  U-FNO=1,337,091\n')

print('Spectrogram shapes:')
for fs in TEST_RATES:
    nf, h = get_spec_params(fs)
    dummy = np.zeros(WIN_S * fs, dtype=np.float32)
    mel   = librosa.feature.melspectrogram(
        y=dummy, sr=fs, n_fft=nf, hop_length=h,
        n_mels=N_MELS, fmin=F_MIN, fmax=F_MAX)
    print(f'  {fs}Hz: N_FFT={nf}, HOP={h} -> (1,{mel.shape[0]},{mel.shape[1]})')


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

def make_mel_at(signal_native, target_fs):
    sig = librosa.resample(signal_native, orig_sr=FS_NATIVE,
                           target_sr=target_fs) \
          if target_fs < FS_NATIVE else signal_native
    nf, h = get_spec_params(target_fs)
    mel   = librosa.feature.melspectrogram(
        y=sig, sr=target_fs, n_fft=nf, hop_length=h,
        n_mels=N_MELS, fmin=F_MIN, fmax=F_MAX)
    mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    mel = (mel - mel.mean()) / (mel.std() + 1e-8)
    return mel

def make_starts(sig_len, wl, max_keep):
    starts = list(range(0, sig_len - wl + 1, wl))
    if len(starts) > max_keep:
        idx = np.linspace(0, len(starts)-1, max_keep, dtype=int)
        starts = [starts[i] for i in idx]
    return starts

def build_rows(op_cond, reps, target_rates, seed):
    rows = {fs: [] for fs in target_rates}
    for label, bids in BEARINGS.items():
        y = CLASSES.index(label)
        for bid in bids:
            for rep in reps:
                path = PU_DIR / bid / f'{op_cond}_{bid}_{rep}.mat'
                if not path.exists(): continue
                try: sig = load_pu_signal(path)
                except Exception as e:
                    print(f'  SKIP {path.name}: {e}'); continue
                wl = WIN_S * FS_NATIVE
                for s in make_starts(len(sig), wl, MAX_SEGMENTS):
                    seg = sig[s:s+wl]
                    for fs in target_rates:
                        rows[fs].append({
                            'mel':   make_mel_at(seg, fs),
                            'label': y, 'fs': fs})
    rng = np.random.default_rng(seed)
    for fs in target_rates: rng.shuffle(rows[fs])
    return rows


# ══════════════════════════════════════════════════════════════
# DATASET + LOADERS
# ══════════════════════════════════════════════════════════════

class BD(Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        return (torch.from_numpy(r['mel']).float().unsqueeze(0),
                torch.tensor(int(r['label']), dtype=torch.long))

def mk(rows, shuf):
    return DataLoader(BD(rows), batch_size=BATCH_SIZE, shuffle=shuf,
                      num_workers=0, pin_memory=False, drop_last=shuf)

def eval_all(model, test_loaders):
    model.eval(); res = {}
    with torch.no_grad():
        for fs, loader in test_loaders.items():
            yt, yp = [], []
            for xb, yb in loader:
                xb=xb.to(DEVICE); yb=yb.to(DEVICE)
                pred = model(xb).argmax(1)
                yt.extend(yb.cpu().numpy().tolist())
                yp.extend(pred.cpu().numpy().tolist())
            yt = np.array(yt); yp = np.array(yp)
            res[fs] = {'acc': float((yt==yp).mean()),
                       'cm': confusion_matrix(yt, yp,
                               labels=np.arange(NUM_CLASSES)),
                       'yt': yt.tolist(), 'yp': yp.tolist()}
    return res

def eval_val(model, val_loader):
    model.eval(); yt, yp = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb=xb.to(DEVICE); yb=yb.to(DEVICE)
            pred = model(xb).argmax(1)
            yt.extend(yb.cpu().numpy().tolist())
            yp.extend(pred.cpu().numpy().tolist())
    return float((np.array(yt)==np.array(yp)).mean())


# ══════════════════════════════════════════════════════════════
# MODELS
# ══════════════════════════════════════════════════════════════

class CNNClassifier(nn.Module):
    def __init__(self, nc):
        super().__init__()
        def blk(a,b): return nn.Sequential(
            nn.Conv2d(a,b,3,padding=1),nn.BatchNorm2d(b),nn.ReLU(),
            nn.Conv2d(b,b,3,padding=1),nn.BatchNorm2d(b),nn.ReLU(),
            nn.MaxPool2d(2),nn.Dropout2d(0.10))
        self.feat = nn.Sequential(blk(1,32),blk(32,64),blk(64,128),
                                  nn.AdaptiveAvgPool2d((4,4)),nn.Flatten())
        self.clf  = nn.Sequential(nn.Linear(128*16,256),nn.ReLU(),
                                  nn.Dropout(0.40),nn.Linear(256,nc))
    def forward(self,x): return self.clf(self.feat(x))


class SC2d(nn.Module):
    def __init__(self,ic,oc,m1,m2):
        super().__init__(); self.m1=m1; self.m2=m2
        self.w=nn.Parameter(1./(ic*oc)*
               torch.randn(ic,oc,m1,m2,dtype=torch.cfloat))
    def forward(self,x):
        b,c,h,w=x.shape; xf=torch.fft.rfft2(x)
        of=torch.zeros(b,self.w.shape[1],h,w//2+1,
                       device=x.device,dtype=torch.cfloat)
        m1=min(self.m1,h); m2=min(self.m2,w//2+1)
        of[:,:,:m1,:m2]=torch.einsum('bixy,ioxy->boxy',
            xf[:,:,:m1,:m2],self.w[:,:,:m1,:m2])
        return torch.fft.irfft2(of,s=(h,w))

class UNetPath(nn.Module):
    def __init__(self,w,d=1):
        super().__init__()
        self.enc=nn.ModuleList([nn.Sequential(
            nn.Conv2d(w,w,3,stride=2,padding=1),nn.BatchNorm2d(w),nn.GELU())
            for _ in range(d)])
        self.dec=nn.ModuleList([nn.Sequential(
            nn.ConvTranspose2d(w*2,w,2,stride=2),nn.BatchNorm2d(w),nn.GELU())
            for _ in range(d)])
    def forward(self,x):
        sk=[]; h=x
        for e in self.enc: sk.append(h); h=e(h)
        for d,s in zip(self.dec,reversed(sk)):
            if h.shape[-2:]!=s.shape[-2:]:
                h=F.interpolate(h,size=s.shape[-2:],
                                mode='bilinear',align_corners=False)
            h=torch.cat([h,s],1); h=d(h)
        if h.shape[-2:]!=x.shape[-2:]:
            h=F.interpolate(h,size=x.shape[-2:],
                            mode='bilinear',align_corners=False)
        return h

class UFNOBlock(nn.Module):
    def __init__(self,w,m1,m2,ud,do):
        super().__init__()
        self.f=SC2d(w,w,m1,m2); self.u=UNetPath(w,ud)
        self.p=nn.Conv2d(w,w,1); self.n=nn.BatchNorm2d(w)
        self.d=nn.Dropout2d(do)
    def forward(self,x):
        y=self.f(x)+self.u(x)+self.p(x)
        y=self.n(y); y=F.gelu(y); y=self.d(y); return x+y

class UFNOClassifier(nn.Module):
    def __init__(self,nc,w=64,d=2,m1=12,m2=12,ud=1,do=0.30):
        super().__init__()
        self.ip=nn.Conv2d(1,w,1)
        self.blks=nn.ModuleList([UFNOBlock(w,m1,m2,ud,do) for _ in range(d)])
        self.head=nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),nn.Flatten(),
                                nn.Linear(w,128),nn.ReLU(),
                                nn.Dropout(0.40),nn.Linear(128,nc))
    def forward(self,x):
        x=self.ip(x)
        for b in self.blks: x=b(x)
        return self.head(x)

class PlainFNOBlock(nn.Module):
    def __init__(self,w,m1,m2,do):
        super().__init__()
        self.f=SC2d(w,w,m1,m2); self.p=nn.Conv2d(w,w,1)
        self.n=nn.BatchNorm2d(w); self.d=nn.Dropout2d(do)
    def forward(self,x):
        y=self.f(x)+self.p(x)
        y=self.n(y); y=F.gelu(y); y=self.d(y); return x+y

class PlainFNOClassifier(nn.Module):
    def __init__(self,nc,w=64,d=2,m1=12,m2=12,do=0.30):
        super().__init__()
        self.ip=nn.Conv2d(1,w,1)
        self.blks=nn.ModuleList([PlainFNOBlock(w,m1,m2,do) for _ in range(d)])
        self.head=nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),nn.Flatten(),
                                nn.Linear(w,128),nn.ReLU(),
                                nn.Dropout(0.40),nn.Linear(128,nc))
    def forward(self,x):
        x=self.ip(x)
        for b in self.blks: x=b(x)
        return self.head(x)


# ══════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════

def train_model(model, train_loader, val_loader, name):
    crit  = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR,
                              weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='max', factor=0.5, patience=8, min_lr=1e-6)

    best=-1.; bst=None; ni=0
    for ep in range(1, EPOCHS+1):
        model.train(); ls=0.; n=0
        for xb,yb in train_loader:
            xb=xb.to(DEVICE); yb=yb.to(DEVICE)
            b,c,f,t=xb.shape
            fm=random.randint(0,max(1,f//10))
            fs_=random.randint(0,max(0,f-fm))
            xb[:,:,fs_:fs_+fm,:]=0.
            tm=random.randint(0,max(1,t//10))
            ts=random.randint(0,max(0,t-tm))
            xb[:,:,:,ts:ts+tm]=0.
            logits=model(xb); loss=crit(logits,yb)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),GRAD_CLIP)
            opt.step(); ls+=float(loss.item())*xb.size(0); n+=xb.size(0)
        va = eval_val(model, val_loader); sched.step(va)
        if ep%20==0 or ep==1:
            print(f'    [{name}] Ep{ep:03d} loss={ls/max(1,n):.4f} '
                  f'val={va:.4f}')
        if va>best+1e-6:
            best=va; bst={k:v.detach().cpu().clone()
                          for k,v in model.state_dict().items()}; ni=0
        else: ni+=1
        if ni>=PATIENCE:
            print(f'    [{name}] early stop ep={ep} best={best:.4f}')
            break
    model.load_state_dict(bst)
    return best


# ══════════════════════════════════════════════════════════════
# MAIN LOOP — ALL CONDITIONS
# ══════════════════════════════════════════════════════════════

all_rows = []
preds_dump_h1 = {}

for op_cond in ALL_CONDITIONS:
    print(f'\n{"="*65}')
    print(f'CONDITION: {op_cond}')
    print(f'{"="*65}')

    print('  Building data ...')
    train_d = build_rows(op_cond, TRAIN_REPS, [TRAIN_FS], SEED)
    val_d   = build_rows(op_cond, VAL_REPS,   [TRAIN_FS], SEED+1)
    test_d  = build_rows(op_cond, TEST_REPS,  TEST_RATES, SEED+2)

    train_loader = mk(train_d[TRAIN_FS], True)
    val_loader   = mk(val_d[TRAIN_FS],   False)
    test_loaders = {fs: mk(test_d[fs], False) for fs in TEST_RATES}

    n_train = len(train_d[TRAIN_FS])
    print(f'  n_train={n_train}, n_val={len(val_d[TRAIN_FS])}, '
          f'n_test={len(test_d[TRAIN_FS])} per rate')

    results_this = {}

    for mname, ModelClass, mkw in [
        ('CNN',      CNNClassifier,     {}),
        ('PlainFNO', PlainFNOClassifier,
             {'w':WIDTH,'d':DEPTH,'m1':MODES1,'m2':MODES2,'do':DROPOUT}),
        ('U-FNO',    UFNOClassifier,
             {'w':WIDTH,'d':DEPTH,'m1':MODES1,'m2':MODES2,
              'ud':UNET_DEPTH,'do':DROPOUT}),
    ]:
        # Ablation only on primary condition
        if mname == 'PlainFNO' and op_cond != 'N15_M07_F10':
            continue

        model = ModelClass(NUM_CLASSES, **mkw).to(DEVICE)
        nparams = sum(p.numel() for p in model.parameters())
        print(f'\n  Training {mname} ({nparams:,} params) ...')

        EXPECTED = {'CNN': 812_643, 'U-FNO': 1_337_091}
        if mname in EXPECTED and nparams != EXPECTED[mname]:
            raise RuntimeError(
                f'WRONG HYPERPARAMS: {mname} has {nparams:,} params '
                f'but expected {EXPECTED[mname]:,}. '
                f'Check MODES1/MODES2/UNET_DEPTH match the paper config.'
            )
        train_model(model, train_loader, val_loader, mname)
        res = eval_all(model, test_loaders)
        results_this[mname] = res

        base = res[TRAIN_FS]['acc']
        print(f'  {mname} results:')
        for fs in TEST_RATES:
            acc  = res[fs]['acc']
            tag  = 'same' if fs==TRAIN_FS else f'{TRAIN_FS//fs}x down'
            drop = base - acc if fs != TRAIN_FS else 0.0
            print(f'    {fs}Hz [{tag}]: {acc:.4f}  drop={drop:+.4f}')

        del model; torch.cuda.empty_cache(); gc.collect()

    for mname, res in results_this.items():
        base = res[TRAIN_FS]['acc']
        for fs in TEST_RATES:
            acc  = res[fs]['acc']
            drop = base - acc if fs != TRAIN_FS else 0.0
            all_rows.append({
                'condition': op_cond,
                'model':     mname,
                'rate_hz':   fs,
                'tag':       'same' if fs==TRAIN_FS else f'{TRAIN_FS//fs}x down',
                'acc':       acc,
                'drop':      drop,
            })
            preds_dump_h1[f'{mname}__{op_cond}__{fs}Hz'] = {
                'yt': res[fs]['yt'], 'yp': res[fs]['yp']}


# ══════════════════════════════════════════════════════════════
# SUMMARY TABLES
# ══════════════════════════════════════════════════════════════

import json as _json
with open(OUT_DIR / 'predictions.json', 'w') as f:
    _json.dump({'seed': SEED, 'predictions': preds_dump_h1}, f)
print(f'\nSaved per-sample predictions to {OUT_DIR / "predictions.json"}')

df = pd.DataFrame(all_rows)
df['seed'] = SEED

print('\n' + '='*70)
print('CROSS-RATE SUMMARY: CNN vs U-FNO across all conditions')
print('='*70)

gap_rows = []
for cond in ALL_CONDITIONS:
    sub = df[(df['condition']==cond) & (df['model'].isin(['CNN','U-FNO']))]
    for fs in [32000, 16000]:
        tag   = f'{TRAIN_FS//fs}x down'
        c_row = sub[(sub['model']=='CNN')   & (sub['rate_hz']==fs)]
        u_row = sub[(sub['model']=='U-FNO') & (sub['rate_hz']==fs)]
        if len(c_row) and len(u_row):
            cd  = float(c_row['drop'].iloc[0])
            ud  = float(u_row['drop'].iloc[0])
            gap = cd - ud
            gap_rows.append({
                'condition': cond, 'rate': tag,
                'CNN_acc':  float(c_row['acc'].iloc[0]),
                'UFNO_acc': float(u_row['acc'].iloc[0]),
                'CNN_drop': cd, 'UFNO_drop': ud, 'gap': gap,
            })
            print(f'  {cond} {tag}: CNN drop={cd:+.4f} '
                  f'UFNO drop={ud:+.4f} gap={gap:+.4f}')

gap_df = pd.DataFrame(gap_rows)
if not gap_df.empty:
    mean_gap_2x = gap_df[gap_df['rate']=='2x down']['gap'].mean()
    mean_gap_4x = gap_df[gap_df['rate']=='4x down']['gap'].mean()
    print(f'\nMean gap @ 2x down: {mean_gap_2x:+.4f}')
    print(f'Mean gap @ 4x down: {mean_gap_4x:+.4f}')

print('\n' + '='*70)
print('ABLATION: CNN vs Plain FNO vs U-FNO (N15_M07_F10)')
print('='*70)
abl = df[df['condition']=='N15_M07_F10']
for mname in ['CNN', 'PlainFNO', 'U-FNO']:
    sub = abl[abl['model']==mname]
    if len(sub) == 0: continue
    base = float(sub[sub['rate_hz']==TRAIN_FS]['acc'].iloc[0])
    print(f'\n  {mname}:')
    for fs in TEST_RATES:
        r   = sub[sub['rate_hz']==fs]
        acc = float(r['acc'].iloc[0]) if len(r) else float('nan')
        drop= base - acc if fs != TRAIN_FS else 0.0
        print(f'    {fs}Hz: {acc:.4f}  drop={drop:+.4f}')


# ── Plot: gap per condition ───────────────────────────────────
if not gap_df.empty:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, rate_tag, title in [
        (axes[0], '2x down', 'Accuracy Drop at 32kHz (2x down)'),
        (axes[1], '4x down', 'Accuracy Drop at 16kHz (4x down)'),
    ]:
        sub = gap_df[gap_df['rate']==rate_tag]
        x   = np.arange(len(sub))
        w   = 0.25
        ax.bar(x - w, sub['CNN_drop'],  w, label='CNN drop',  color='steelblue', alpha=0.85)
        ax.bar(x,     sub['UFNO_drop'], w, label='U-FNO drop',color='coral',     alpha=0.85)
        ax.bar(x + w, sub['gap'],       w, label='Gap',       color='green',     alpha=0.70)
        ax.set_xticks(x)
        ax.set_xticklabels(sub['condition'], rotation=20, ha='right', fontsize=9)
        ax.set_ylabel('Accuracy Drop')
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)
        ax.axhline(0, color='black', lw=0.8)

    plt.suptitle('PU Cross-Rate: CNN vs U-FNO drop per operating condition',
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'gap_per_condition.png', dpi=150, bbox_inches='tight')
    plt.close()

# ── Save ──────────────────────────────────────────────────────
df.to_csv(OUT_DIR / 'all_conditions_results.csv', index=False)
gap_df.to_csv(OUT_DIR / 'gap_per_condition.csv', index=False)

print('\nSaved to:', OUT_DIR)
if not gap_df.empty:
    print('\nFINAL GAP TABLE:')
    print(gap_df.round(4).to_string(index=False))
