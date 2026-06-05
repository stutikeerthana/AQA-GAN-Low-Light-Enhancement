"""
AQA-UNet  --  Adaptive Quality-Aware UNet GAN
==============================================
Combines LOL-v2 Real + LSRW for ~1 000+ training pairs.
Restores quality conditioning via FiLM (Feature-wise Linear Modulation),
which is cleaner than the original additive-bias approach.

Datasets
  LOL-v2 Real : D:\\Aloo\\LOL-v2\\Real_captured\\  (Train/Low + Train/Normal)
  LSRW        : D:\\Aloo\\our485\\                  (low/ + high/)
  Combined    : LOL train + LSRW  ->  split 80/20 for train/val
  Test        : LOL-v2 Real Test set only (clean benchmark)

Quality levels  (NIQE proxy)
  0 = severe    score > 6.0   (very dark, heavy noise)
  1 = moderate  score > 4.0
  2 = mild      score <= 4.0  (slightly underexposed)

Architecture
  Generator  : UNet  (base_ch=32, ~3.6M params) + FiLM at bottleneck
  Discriminator : 70x70 PatchGAN (base_ch=64) + quality bias on output
  FiLM init  : gamma=1, beta=0  ->  identity at epoch 0, diverges as training progresses

Usage
  python lol_aqa_unet.py                              # train + report + plots
  python lol_aqa_unet.py --mode train
  python lol_aqa_unet.py --mode train --resume checkpoints/ckpt_epoch30.pth
  python lol_aqa_unet.py --mode train --no-adv        # pixel-only baseline
  python lol_aqa_unet.py --mode report
  python lol_aqa_unet.py --mode plots
  python lol_aqa_unet.py --mode test --tta
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os, re, csv, json, time, sys, logging, argparse, shutil
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms, models
from torchvision.utils import save_image
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
LOL_PATH  = r"D:\Aloo\LOL-v2\Real_captured"   # contains Train/ and Test/
LSRW_PATH = r"D:\Aloo\our485"                  # contains low/ and high/
OUTPUT_DIR = r"D:\Aloo\Output_aqa_unet"

IMG_SIZE   = 256
BATCH_SIZE = 4
LR         = 2e-4
EPOCHS     = 150
SEED       = 42

# Loss weights
LAMBDA_ADV     = 0.5
LAMBDA_CHAR    = 100.0
LAMBDA_SSIM    = 5.0
LAMBDA_PERCEPT = 0.05

# Augmentation
MIXUP_ALPHA = 0.4       # set 0.0 to disable

# Quality thresholds (NIQE proxy)
NIQE_HIGH  = 6.0        # above -> level 0 (severe)
NIQE_LOW   = 4.0        # above -> level 1 (moderate), below -> level 2 (mild)
N_LEVELS   = 3

# Discriminator label smoothing
D_REAL = 0.85
D_FAKE = 0.15

# Runtime
USE_AMP     = True
RESUME_CKPT = None

CKPT_EVERY = 10
PLOT_EVERY = 20

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DIRS = {k: os.path.join(OUTPUT_DIR, k)
        for k in ("samples", "checkpoints", "plots", "logs")}
for d in DIRS.values():
    os.makedirs(d, exist_ok=True)

HISTORY_PATH = os.path.join(DIRS["logs"], "history.json")


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
def setup_logger():
    log = logging.getLogger("aqa_unet")
    log.setLevel(logging.INFO); log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    ch  = logging.StreamHandler(sys.stdout)
    if hasattr(ch.stream, "reconfigure"):
        ch.stream.reconfigure(encoding="utf-8")
    ch.setFormatter(fmt); log.addHandler(ch)
    fh = logging.FileHandler(os.path.join(DIRS["logs"], "train.log"),
                             mode="a", encoding="utf-8")
    fh.setFormatter(fmt); log.addHandler(fh)
    return log


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def compute_psnr(p, t):
    mse = F.mse_loss(p, t)
    return 100.0 if mse < 1e-12 else (10 * torch.log10(1.0 / mse)).item()


def _gauss(w=11, s=1.5, c=3, dev="cpu"):
    x = torch.arange(w, dtype=torch.float32, device=dev) - w // 2
    g = torch.exp(-x**2 / (2*s**2)); g /= g.sum()
    return (g.unsqueeze(0) * g.unsqueeze(1)).expand(c, 1, w, w)


@torch.no_grad()
def compute_ssim(a, b, w=11):
    C1, C2, c = 0.01**2, 0.03**2, a.shape[1]
    k   = _gauss(w, dev=a.device); pad = w // 2
    mu1 = F.conv2d(a,   k, padding=pad, groups=c)
    mu2 = F.conv2d(b,   k, padding=pad, groups=c)
    s1  = F.conv2d(a*a, k, padding=pad, groups=c) - mu1**2
    s2  = F.conv2d(b*b, k, padding=pad, groups=c) - mu2**2
    s12 = F.conv2d(a*b, k, padding=pad, groups=c) - mu1*mu2
    return ((2*mu1*mu2+C1)*(2*s12+C2) / ((mu1**2+mu2**2+C1)*(s1+s2+C2))).mean().item()


def niqe_proxy(t):
    """Fast no-reference proxy on a (C,H,W) tensor in [-1,1]."""
    img  = torch.clamp((t + 1) / 2, 0, 1)
    gray = 0.299*img[0] + 0.587*img[1] + 0.114*img[2]
    return ((1 - gray.mean()) * 5 + (1 - gray.std(unbiased=False)) * 3).item()


def assign_quality(t):
    """Returns 0 (severe), 1 (moderate), 2 (mild) based on NIQE proxy."""
    s = niqe_proxy(t)
    return 0 if s > NIQE_HIGH else (1 if s > NIQE_LOW else 2)


# ─────────────────────────────────────────────────────────────────────────────
# DATASET  --  LOL-v2 + LSRW combined
# ─────────────────────────────────────────────────────────────────────────────
_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _num(fname):
    m = re.search(r"(\d+)", fname)
    return m.group(1) if m else None


def build_lol_pairs(split="Train"):
    """LOL-v2: matches Low/<file> to Normal/<file> by embedded number."""
    ld = os.path.join(LOL_PATH, split, "Low")
    nd = os.path.join(LOL_PATH, split, "Normal")
    if not os.path.isdir(ld) or not os.path.isdir(nd):
        return []
    nmap = {_num(f): f for f in os.listdir(nd) if _num(f)}
    return [
        (os.path.join(ld, f), os.path.join(nd, nmap[_num(f)]))
        for f in sorted(os.listdir(ld))
        if _num(f) and _num(f) in nmap
    ]


def build_lsrw_pairs():
    """LSRW: matches low/<stem>.* to high/<stem>.* by filename stem."""
    ld = os.path.join(LSRW_PATH, "low")
    hd = os.path.join(LSRW_PATH, "high")
    if not os.path.isdir(ld) or not os.path.isdir(hd):
        print(f"[WARN] LSRW not found at {LSRW_PATH} -- using LOL only")
        return []
    # Build stem -> filename map for high/
    hmap = {os.path.splitext(f)[0]: f
            for f in os.listdir(hd)
            if os.path.splitext(f)[1].lower() in _EXTS}
    pairs = []
    for f in sorted(os.listdir(ld)):
        if os.path.splitext(f)[1].lower() not in _EXTS:
            continue
        stem = os.path.splitext(f)[0]
        if stem in hmap:
            pairs.append((os.path.join(ld, f), os.path.join(hd, hmap[stem])))
    return pairs


class PairedDataset(Dataset):
    """Generic paired low/high dataset with quality label."""
    def __init__(self, pairs, tf=None):
        self.pairs = pairs
        self.tf    = tf

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        lp, hp = self.pairs[idx]
        low  = Image.open(lp).convert("RGB")
        high = Image.open(hp).convert("RGB")
        if self.tf:
            seed = torch.randint(0, 2**31, (1,)).item()
            torch.manual_seed(seed); low  = self.tf(low)
            torch.manual_seed(seed); high = self.tf(high)
        ql = assign_quality(low)
        return low, high, torch.tensor(ql, dtype=torch.long)


def get_loaders():
    to_norm = transforms.Normalize([0.5]*3, [0.5]*3)

    train_tf = transforms.Compose([
        transforms.Resize(320),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(15, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.08),
        transforms.ToTensor(), to_norm,
    ])
    test_tf = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(), to_norm,
    ])

    # Build pair lists
    lol_train = build_lol_pairs("Train")
    lsrw      = build_lsrw_pairs()
    lol_test  = build_lol_pairs("Test")

    all_train = lol_train + lsrw        # combine both sources
    if not all_train:
        raise RuntimeError("No training pairs found -- check LOL_PATH and LSRW_PATH")

    # 80/20 split on combined training data
    val_n = int(0.2 * len(all_train)); trn_n = len(all_train) - val_n
    g     = torch.Generator().manual_seed(SEED)
    # Use index split (not random_split on Dataset, so we can share pairs list)
    import random; rng = random.Random(SEED)
    idx   = list(range(len(all_train))); rng.shuffle(idx)
    trn_pairs = [all_train[i] for i in idx[:trn_n]]
    val_pairs = [all_train[i] for i in idx[trn_n:]]

    trn_ds  = PairedDataset(trn_pairs, tf=train_tf)
    val_ds  = PairedDataset(val_pairs, tf=test_tf)
    test_ds = PairedDataset(lol_test,  tf=test_tf)

    nw = min(4, os.cpu_count() or 2); pw = nw > 0
    kw = dict(num_workers=nw, pin_memory=True, persistent_workers=pw)
    return (
        DataLoader(trn_ds,  BATCH_SIZE, shuffle=True,  **kw),
        DataLoader(val_ds,  BATCH_SIZE, shuffle=False, **kw),
        DataLoader(test_ds, 1,          shuffle=False, num_workers=nw),
        len(trn_ds), len(val_ds), len(test_ds),
        len(lol_train), len(lsrw),
    )


def mixup(l1, h1, l2, h2):
    lam = float(np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA))
    return lam*l1 + (1-lam)*l2, lam*h1 + (1-lam)*h2


# ─────────────────────────────────────────────────────────────────────────────
# GENERATOR  --  Quality-Conditioned UNet  (AQA-UNet)
# ─────────────────────────────────────────────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ic, oc, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(oc, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(oc, oc, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(oc, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
    def forward(self, x): return self.net(x)


class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation: scale + shift feature maps by quality level.
    Initialized to identity (gamma=1, beta=0) so epoch-0 behavior is unchanged.
    As training progresses the embeddings diverge and conditioning kicks in.
    """
    def __init__(self, channels, n_levels=N_LEVELS):
        super().__init__()
        self.gamma = nn.Embedding(n_levels, channels)
        self.beta  = nn.Embedding(n_levels, channels)
        nn.init.ones_(self.gamma.weight)    # identity init: scale = 1
        nn.init.zeros_(self.beta.weight)    # identity init: shift = 0

    def forward(self, x, quality):
        # quality: (B,) LongTensor with values 0/1/2
        g = self.gamma(quality).unsqueeze(-1).unsqueeze(-1)   # (B, C, 1, 1)
        b = self.beta(quality).unsqueeze(-1).unsqueeze(-1)
        return x * g + b


class AQAUNet(nn.Module):
    """
    UNet generator conditioned on quality level via FiLM at the bottleneck.

    base_ch=32  ->  ~3.6M params   (default -- suitable for ~1000 pairs)
    base_ch=48  ->  ~8.1M params   (try if PSNR plateaus after 100 epochs)

    Residual output: tanh(inp + 2*out)
    Gives dark inputs (≈-0.9) headroom to reach bright output values.
    """
    def __init__(self, base_ch=32, n_levels=N_LEVELS):
        super().__init__()
        c = base_ch

        # Encoder: 256 -> 128 -> 64 -> 32 -> 16
        self.e1 = ConvBlock(3,   c)
        self.e2 = ConvBlock(c,   c*2)
        self.e3 = ConvBlock(c*2, c*4)
        self.e4 = ConvBlock(c*4, c*8)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck + FiLM quality conditioning
        self.bot  = nn.Sequential(ConvBlock(c*8, c*8), ConvBlock(c*8, c*8))
        self.film = FiLM(c*8, n_levels)

        # Decoder: 16 -> 32 -> 64 -> 128 -> 256
        self.d4 = ConvBlock(c*8 + c*8, c*4)
        self.d3 = ConvBlock(c*4 + c*4, c*2)
        self.d2 = ConvBlock(c*2 + c*2, c)
        self.d1 = ConvBlock(c   + c,   c)

        self.head = nn.Conv2d(c, 3, 1)

    def _up(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([x, skip], dim=1)

    def forward(self, x, quality):
        inp = x
        e1  = self.e1(x)
        e2  = self.e2(self.pool(e1))
        e3  = self.e3(self.pool(e2))
        e4  = self.e4(self.pool(e3))

        b   = self.film(self.bot(self.pool(e4)), quality)

        d4  = self.d4(self._up(b,  e4))
        d3  = self.d3(self._up(d4, e3))
        d2  = self.d2(self._up(d3, e2))
        d1  = self.d1(self._up(d2, e1))

        return torch.tanh(inp + self.head(d1) * 2.0)


# ─────────────────────────────────────────────────────────────────────────────
# DISCRIMINATOR  --  Quality-conditioned 70x70 PatchGAN
# ─────────────────────────────────────────────────────────────────────────────
class PatchGANAQA(nn.Module):
    """
    Standard 70x70 PatchGAN with a learned quality bias on the output map.
    The quality embedding adds a (B,1,1,1) offset to patch predictions,
    making D's realness judgement sensitive to how degraded the input was.
    """
    def __init__(self, base_ch=64, n_levels=N_LEVELS):
        super().__init__()
        c = base_ch

        def blk(ic, oc, norm=True, stride=2):
            L = [nn.Conv2d(ic, oc, 4, stride, 1, bias=not norm)]
            if norm: L.append(nn.InstanceNorm2d(oc, affine=True))
            L.append(nn.LeakyReLU(0.2, inplace=True))
            return L

        self.net = nn.Sequential(
            *blk(3,   c,   norm=False),   # 128
            *blk(c,   c*2),               # 64
            *blk(c*2, c*4),               # 32
            *blk(c*4, c*8, stride=1),     # 31
            nn.Conv2d(c*8, 1, 4, 1, 1),  # 30x30 patch logits
        )
        # Quality bias: small embedding projected to a scalar offset
        self.q_emb  = nn.Embedding(n_levels, 128)
        self.q_proj = nn.Linear(128, 1)
        nn.init.zeros_(self.q_proj.weight)  # start with no quality bias
        nn.init.zeros_(self.q_proj.bias)

    def forward(self, x, quality):
        feat  = self.net(x)
        q_bias = self.q_proj(self.q_emb(quality)).view(-1, 1, 1, 1)
        return feat + q_bias

    def d_loss(self, real, fake, quality):
        rp = self(real.detach(), quality)
        fp = self(fake.detach(), quality)
        loss = 0.5 * (
            F.binary_cross_entropy_with_logits(rp, torch.ones_like(rp)  * D_REAL) +
            F.binary_cross_entropy_with_logits(fp, torch.zeros_like(fp) + D_FAKE)
        )
        return loss, torch.sigmoid(rp).mean().item(), torch.sigmoid(fp).mean().item()

    def g_loss(self, fake, quality):
        fp = self(fake, quality)
        return F.binary_cross_entropy_with_logits(fp, torch.ones_like(fp))


# ─────────────────────────────────────────────────────────────────────────────
# LOSSES
# ─────────────────────────────────────────────────────────────────────────────
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__(); self.e2 = eps*eps
    def forward(self, p, t):
        return torch.sqrt((p-t)**2 + self.e2).mean()


class SSIMLoss(nn.Module):
    def __init__(self, w=11, s=1.5):
        super().__init__(); self.w, self.s = w, s
    def forward(self, p, t):
        a, b   = (p+1)/2, (t+1)/2
        C1, C2, c = 0.01**2, 0.03**2, a.shape[1]
        k = _gauss(self.w, self.s, c, a.device); pad = self.w // 2
        mu1 = F.conv2d(a,   k, padding=pad, groups=c)
        mu2 = F.conv2d(b,   k, padding=pad, groups=c)
        s1  = F.conv2d(a*a, k, padding=pad, groups=c) - mu1**2
        s2  = F.conv2d(b*b, k, padding=pad, groups=c) - mu2**2
        s12 = F.conv2d(a*b, k, padding=pad, groups=c) - mu1*mu2
        return 1.0 - ((2*mu1*mu2+C1)*(2*s12+C2) / ((mu1**2+mu2**2+C1)*(s1+s2+C2))).mean()


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.feats = nn.Sequential(*list(vgg.features)[:9]).eval()
        for p in self.feats.parameters(): p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485,0.456,0.406]).view(1,3,1,1))
        self.register_buffer("std",  torch.tensor([0.229,0.224,0.225]).view(1,3,1,1))
    def forward(self, enh, ref):
        e = ((enh+1)/2 - self.mean)/self.std
        r = ((ref+1)/2 - self.mean)/self.std
        return F.mse_loss(self.feats(e.float()), self.feats(r.float()))


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def gnorm(m):
    return sum(p.grad.data.norm(2)**2
               for p in m.parameters() if p.grad is not None).item()**0.5

def gpu_peak():
    if not torch.cuda.is_available(): return 0.0
    return torch.cuda.max_memory_allocated() / 1024**3


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN ONE EPOCH
# ─────────────────────────────────────────────────────────────────────────────
def train_epoch(G, D, g_opt, d_opt, loader,
                char_fn, ssim_fn, percept_fn,
                use_adv, scaler, epoch):
    G.train(); D.train()
    S  = {k: 0.0 for k in ["g","d","char","ssim","percept","adv",
                            "d_real","d_fake","gg","dg","skip"]}
    n  = 0
    ql_counts = [0, 0, 0]   # track quality-level distribution per epoch

    for low, nrm, quality in tqdm(loader, desc=f"Ep {epoch:03d}", leave=False):
        low     = low.to(DEVICE, non_blocking=True)
        nrm     = nrm.to(DEVICE, non_blocking=True)
        quality = quality.to(DEVICE, non_blocking=True)
        bs      = low.size(0)

        for ql in quality.cpu().tolist():
            ql_counts[ql] += 1

        # MixUp within batch
        if MIXUP_ALPHA > 0 and bs > 1:
            idx = torch.randperm(bs, device=DEVICE)
            low, nrm = mixup(low, nrm, low[idx], nrm[idx])

        # ── Discriminator ────────────────────────────────────────────────────
        d_opt.zero_grad(set_to_none=True)
        with autocast(enabled=USE_AMP):
            with torch.no_grad():
                fake_d = G(low, quality)
            d_loss, r_mean, f_mean = D.d_loss(nrm, fake_d, quality)

        d_skip = int(r_mean > 0.80 and f_mean < 0.20)
        if not d_skip:
            if USE_AMP:
                scaler.scale(d_loss).backward()
                scaler.unscale_(d_opt)
                torch.nn.utils.clip_grad_norm_(D.parameters(), 5.0)
                dg_ = gnorm(D); scaler.step(d_opt)
            else:
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(D.parameters(), 5.0)
                dg_ = gnorm(D); d_opt.step()
        else:
            dg_ = 0.0

        # ── Generator ────────────────────────────────────────────────────────
        g_opt.zero_grad(set_to_none=True)
        with autocast(enabled=USE_AMP):
            fake      = G(low, quality)
            l_char    = char_fn(fake, nrm)
            l_ssim    = ssim_fn(fake, nrm)
            l_percept = percept_fn(fake, nrm)
            l_adv     = (D.g_loss(fake, quality) if use_adv
                         else torch.tensor(0., device=DEVICE))
            g_loss    = (LAMBDA_CHAR    * l_char    +
                         LAMBDA_SSIM    * l_ssim    +
                         LAMBDA_PERCEPT * l_percept +
                         LAMBDA_ADV     * l_adv)

        if USE_AMP:
            scaler.scale(g_loss).backward()
            scaler.unscale_(g_opt)
            torch.nn.utils.clip_grad_norm_(G.parameters(), 5.0)
            gg_ = gnorm(G); scaler.step(g_opt); scaler.update()
        else:
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), 5.0)
            gg_ = gnorm(G); g_opt.step()

        S["g"]       += g_loss.item()    * bs
        S["d"]       += d_loss.item()    * bs
        S["char"]    += l_char.item()    * bs
        S["ssim"]    += l_ssim.item()    * bs
        S["percept"] += l_percept.item() * bs
        S["adv"]     += l_adv.item()     * bs
        S["d_real"]  += r_mean           * bs
        S["d_fake"]  += f_mean           * bs
        S["gg"]      += gg_              * bs
        S["dg"]      += dg_              * bs
        S["skip"]    += d_skip           * bs
        n += bs

    metrics = {k: v/n for k, v in S.items()}
    metrics["ql_counts"] = ql_counts
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATE
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def validate(G, loader, epoch, save_sample=True):
    G.eval()
    P, S, NI, NO, by_ql = [], [], [], [], {0:[], 1:[], 2:[]}
    saved = False

    for low, nrm, quality in loader:
        low     = low.to(DEVICE)
        nrm     = nrm.to(DEVICE)
        quality = quality.to(DEVICE)
        enh     = G(low, quality)
        e01     = (enh+1)/2; n01 = (nrm+1)/2
        p = compute_psnr(e01, n01); s = compute_ssim(e01, n01)
        P.append(p); S.append(s)
        for i in range(low.size(0)):
            NI.append(niqe_proxy(low[i])); NO.append(niqe_proxy(enh[i]))
            by_ql[quality[i].item()].append(p)   # PSNR per quality level

        if save_sample and not saved:
            grid = torch.cat([low[:4], enh[:4], nrm[:4]], 0)
            save_image((grid+1)/2,
                       os.path.join(DIRS["samples"], f"ep{epoch:03d}.png"), nrow=4)
            saved = True

    return {
        "psnr":     float(np.mean(P)),
        "ssim":     float(np.mean(S)),
        "niqe_in":  float(np.mean(NI)),
        "niqe_out": float(np.mean(NO)),
        # Per quality-level PSNR  (tells you if conditioning is working)
        "psnr_ql0": float(np.mean(by_ql[0])) if by_ql[0] else 0.0,
        "psnr_ql1": float(np.mean(by_ql[1])) if by_ql[1] else 0.0,
        "psnr_ql2": float(np.mean(by_ql[2])) if by_ql[2] else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# HISTORY
# ─────────────────────────────────────────────────────────────────────────────
HKEYS = ["epoch","g","d","char","ssim_l","percept","adv",
         "d_real","d_fake","skip","gg","dg",
         "val_psnr","val_ssim","val_niqe_in","val_niqe_out",
         "val_psnr_ql0","val_psnr_ql1","val_psnr_ql2",
         "lr_g","lr_d","time_s","tput","gpu_peak"]


def save_hist(h):
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(h, f, indent=2)
    n = len(h["epoch"])
    with open(os.path.join(DIRS["logs"], "history.csv"),
              "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(HKEYS)
        for i in range(n):
            w.writerow([h.get(k, [""]*n)[i] if i < len(h.get(k,[])) else ""
                        for k in HKEYS])


def load_hist():
    if not os.path.exists(HISTORY_PATH):
        raise FileNotFoundError("No history.json -- run --mode train first.")
    with open(HISTORY_PATH, encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────
def save_plot(h, epoch):
    e = h["epoch"]
    if len(e) < 2: return
    fig, ax = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle(f"AQA-UNet  epoch {epoch}", fontsize=13)

    ax[0,0].plot(e, h["g"], label="G", color="#4fc3f7")
    ax[0,0].plot(e, h["d"], label="D", color="#ff7043")
    ax[0,0].set_title("Total losses"); ax[0,0].legend(); ax[0,0].grid(alpha=0.3)

    ax[0,1].semilogy(e, h["char"],    label="Charbonnier", color="#fff176")
    ax[0,1].semilogy(e, h["ssim_l"],  label="SSIM",        color="#80cbc4")
    ax[0,1].semilogy(e, h["percept"], label="Perceptual",  color="#ce93d8")
    ax[0,1].semilogy(e, h["adv"],     label="Adversarial", color="#a5d6a7")
    ax[0,1].set_title("G components (log)"); ax[0,1].legend(); ax[0,1].grid(alpha=0.3)

    ax[0,2].plot(e, h["d_real"], label="D(real)", color="#66bb6a")
    ax[0,2].plot(e, h["d_fake"], label="D(fake)", color="#ef5350")
    ax[0,2].axhline(0.5, color="gray", lw=0.8, ls="--")
    ax[0,2].set_ylim(0,1); ax[0,2].set_title("D equilibrium")
    ax[0,2].legend(); ax[0,2].grid(alpha=0.3)

    ax[0,3].plot(e, h["gg"], label="G grad", color="#4fc3f7")
    ax[0,3].plot(e, h["dg"], label="D grad", color="#ff7043")
    ax[0,3].set_yscale("log"); ax[0,3].set_title("Grad norms")
    ax[0,3].legend(); ax[0,3].grid(alpha=0.3)

    ax[1,0].plot(e, h["val_psnr"], color="#69f0ae", marker="o", ms=3)
    ax[1,0].axhline(25, color="red",    lw=0.8, ls="--", alpha=0.7, label="25dB")
    ax[1,0].axhline(20, color="orange", lw=0.8, ls=":",  alpha=0.7, label="20dB")
    ax[1,0].set_title("Val PSNR (dB)"); ax[1,0].legend(); ax[1,0].grid(alpha=0.3)

    ax[1,1].plot(e, h["val_ssim"], color="#40c4ff", marker="o", ms=3)
    ax[1,1].axhline(0.80, color="red", lw=0.8, ls="--", alpha=0.7, label="0.80")
    ax[1,1].set_title("Val SSIM"); ax[1,1].legend(); ax[1,1].grid(alpha=0.3)

    ax[1,2].plot(e, h["val_niqe_in"],  label="NIQE in",  color="gray", ls="--")
    ax[1,2].plot(e, h["val_niqe_out"], label="NIQE out", color="#ff9800")
    ax[1,2].set_title("NIQE proxy"); ax[1,2].legend(); ax[1,2].grid(alpha=0.3)

    # Per-quality PSNR -- key AQA diagnostic
    # If conditioning works, severe (ql0) should improve faster than mild (ql2)
    ax[1,3].plot(e, h.get("val_psnr_ql0", [0]*len(e)),
                 label="Severe (ql=0)",   color="#ef5350", marker=".", ms=3)
    ax[1,3].plot(e, h.get("val_psnr_ql1", [0]*len(e)),
                 label="Moderate (ql=1)", color="#fff176", marker=".", ms=3)
    ax[1,3].plot(e, h.get("val_psnr_ql2", [0]*len(e)),
                 label="Mild (ql=2)",     color="#69f0ae", marker=".", ms=3)
    ax[1,3].set_title("PSNR per quality level (AQA check)")
    ax[1,3].legend(); ax[1,3].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(DIRS["plots"], f"diag_{epoch:03d}.png")
    plt.savefig(path, dpi=100); plt.close()
    return path


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────
def train(resume=None, use_adv=True):
    set_seed(SEED)
    log = setup_logger()
    log.info("=" * 65)
    log.info(f"AQA-UNet training  [{datetime.now():%Y-%m-%d %H:%M}]")
    log.info(f"AMP={USE_AMP}  MixUp={MIXUP_ALPHA}  adv={use_adv}")
    log.info("=" * 65)

    G = AQAUNet(base_ch=32, n_levels=N_LEVELS).to(DEVICE)
    D = PatchGANAQA(base_ch=64, n_levels=N_LEVELS).to(DEVICE)

    ng = sum(p.numel() for p in G.parameters())
    nd = sum(p.numel() for p in D.parameters())
    log.info(f"G params: {ng:,}  (UNet + FiLM)   D params: {nd:,}  (PatchGAN + q-bias)")

    g_opt = optim.Adam(G.parameters(), lr=LR,       betas=(0.9, 0.999))
    d_opt = optim.Adam(D.parameters(), lr=LR * 0.1, betas=(0.9, 0.999))
    g_sch = optim.lr_scheduler.CosineAnnealingLR(g_opt, T_max=EPOCHS, eta_min=1e-6)
    d_sch = optim.lr_scheduler.CosineAnnealingLR(d_opt, T_max=EPOCHS, eta_min=1e-7)

    char_fn    = CharbonnierLoss().to(DEVICE)
    ssim_fn    = SSIMLoss().to(DEVICE)
    percept_fn = PerceptualLoss().to(DEVICE)
    scaler     = GradScaler(enabled=USE_AMP and torch.cuda.is_available())

    tr_ld, va_ld, te_ld, ntr, nva, nte, n_lol, n_lsrw = get_loaders()
    log.info(f"LOL-v2: {n_lol}  LSRW: {n_lsrw}  "
             f"Total train: {ntr}  Val: {nva}  Test: {nte}")
    if torch.cuda.is_available():
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")

    h     = {k: [] for k in HKEYS}
    best  = {"psnr": -1.0, "ssim": -1.0}
    start = 1

    if resume and os.path.isfile(resume):
        ck = torch.load(resume, map_location=DEVICE)
        G.load_state_dict(ck["G"]); D.load_state_dict(ck["D"])
        g_opt.load_state_dict(ck["g_opt"]); d_opt.load_state_dict(ck["d_opt"])
        g_sch.load_state_dict(ck["g_sch"]); d_sch.load_state_dict(ck["d_sch"])
        h = ck.get("h", h); best = ck.get("best", best)
        start = ck["epoch"] + 1
        log.info(f"Resumed epoch {start-1}  PSNR={best['psnr']:.2f}  SSIM={best['ssim']:.4f}")
    elif resume:
        log.warning(f"Checkpoint not found: {resume}  -- starting fresh")

    for ep in range(start, EPOCHS + 1):
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
        t0 = time.time()

        tm = train_epoch(G, D, g_opt, d_opt, tr_ld,
                         char_fn, ssim_fn, percept_fn, use_adv, scaler, ep)
        vm = validate(G, va_ld, ep)
        dt = time.time() - t0

        h["epoch"].append(ep)
        for k, src in [("g","g"),("d","d"),("char","char"),("ssim_l","ssim"),
                       ("percept","percept"),("adv","adv"),
                       ("d_real","d_real"),("d_fake","d_fake"),
                       ("skip","skip"),("gg","gg"),("dg","dg")]:
            h[k].append(tm[src])
        for k in ["psnr","ssim","niqe_in","niqe_out",
                  "psnr_ql0","psnr_ql1","psnr_ql2"]:
            h[f"val_{k}"].append(vm[k])
        h["lr_g"].append(g_opt.param_groups[0]["lr"])
        h["lr_d"].append(d_opt.param_groups[0]["lr"])
        h["time_s"].append(round(dt, 1))
        h["tput"].append(round(ntr / dt, 1))
        h["gpu_peak"].append(round(gpu_peak(), 3))
        save_hist(h)

        ql = tm["ql_counts"]
        log.info(
            f"[{ep:03d}/{EPOCHS}] "
            f"G={tm['g']:.3f}  D={tm['d']:.3f}  "
            f"char={tm['char']:.4f}  ssim={tm['ssim']:.4f}  "
            f"perc={tm['percept']:.4f}  adv={tm['adv']:.4f}"
        )
        log.info(
            f"          "
            f"D(r)={tm['d_real']:.3f}  D(f)={tm['d_fake']:.3f}  "
            f"skip={tm['skip']*100:.1f}%  "
            f"QL[sev/mod/mild]={ql[0]}/{ql[1]}/{ql[2]}"
        )
        log.info(
            f"          "
            f"PSNR={vm['psnr']:.2f}dB  SSIM={vm['ssim']:.4f}  "
            f"PSNR/ql=[{vm['psnr_ql0']:.1f}/{vm['psnr_ql1']:.1f}/{vm['psnr_ql2']:.1f}]  "
            f"{dt:.0f}s  GPU={gpu_peak():.2f}GB"
        )

        g_sch.step(); d_sch.step()

        if vm["psnr"] > best["psnr"]:
            best["psnr"] = vm["psnr"]
            torch.save({"epoch": ep, "G": G.state_dict(), "D": D.state_dict()},
                       os.path.join(DIRS["checkpoints"], "best_psnr.pth"))
            log.info(f"          * Best PSNR: {vm['psnr']:.3f} dB")
            # Save the visualization and plot for best PSNR
            shutil.copy(os.path.join(DIRS["samples"], f"ep{ep:03d}.png"),
                        os.path.join(DIRS["samples"], "best_psnr_sample.png"))
            save_plot(h, ep)

        if vm["ssim"] > best["ssim"]:
            best["ssim"] = vm["ssim"]
            torch.save({"epoch": ep, "G": G.state_dict(), "D": D.state_dict()},
                       os.path.join(DIRS["checkpoints"], "best_ssim.pth"))
            log.info(f"          * Best SSIM: {vm['ssim']:.4f}")
            # Save the visualization for best SSIM
            shutil.copy(os.path.join(DIRS["samples"], f"ep{ep:03d}.png"),
                        os.path.join(DIRS["samples"], "best_ssim_sample.png"))

        if ep % CKPT_EVERY == 0 or ep == EPOCHS:
            p = os.path.join(DIRS["checkpoints"], f"ckpt_epoch{ep}.pth")
            torch.save({"epoch": ep,
                        "G": G.state_dict(), "D": D.state_dict(),
                        "g_opt": g_opt.state_dict(), "d_opt": d_opt.state_dict(),
                        "g_sch": g_sch.state_dict(), "d_sch": d_sch.state_dict(),
                        "h": h, "best": best}, p)
            log.info(f"          Checkpoint -> {p}")

        if ep % PLOT_EVERY == 0 or ep == EPOCHS:
            pp = save_plot(h, ep)
            log.info(f"          Plot -> {pp}")

    log.info(f"\nDone -- Best PSNR:{best['psnr']:.3f}dB  SSIM:{best['ssim']:.4f}")
    test_eval(G, te_ld, log)
    return G


# ─────────────────────────────────────────────────────────────────────────────
# TEST EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def test_eval(G, loader, log=None, tta=False):
    G.eval()
    P, S, NI, NO = [], [], [], []

    for low, nrm, quality in tqdm(loader, desc="Test"):
        low     = low.to(DEVICE)
        nrm     = nrm.to(DEVICE)
        quality = quality.to(DEVICE)

        if tta:
            preds = []
            for flip in (False, True):
                x = low.flip(-1) if flip else low
                for k in range(4):
                    xr = torch.rot90(x, k, [-2,-1])
                    pr = torch.rot90(G(xr, quality), -k, [-2,-1])
                    preds.append(pr.flip(-1) if flip else pr)
            enh = torch.stack(preds).mean(0)
        else:
            enh = G(low, quality)

        P.append(compute_psnr((enh+1)/2, (nrm+1)/2))
        S.append(compute_ssim((enh+1)/2, (nrm+1)/2))
        NI.append(niqe_proxy(low[0])); NO.append(niqe_proxy(enh[0]))

    res = {"psnr":             float(np.mean(P)),
           "ssim":             float(np.mean(S)),
           "niqe_improve_pct": (np.mean(NI)-np.mean(NO))/np.mean(NI)*100,
           "tta": tta}
    msg = (f"Test{'[TTA]' if tta else ''} "
           f"PSNR:{res['psnr']:.3f}dB  SSIM:{res['ssim']:.4f}  "
           f"NIQE improve:{res['niqe_improve_pct']:.1f}%")
    (log.info if log else print)(msg)
    with open(os.path.join(DIRS["logs"], "test_results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    return res


# ─────────────────────────────────────────────────────────────────────────────
# METRICS REPORT
# ─────────────────────────────────────────────────────────────────────────────
G_ = "\033[92m"; Y_ = "\033[93m"; R_ = "\033[91m"
C_ = "\033[96m"; B_ = "\033[1m";  E_ = "\033[0m"

def _c(v, lo, hi, f=".3f"):
    return (G_ if v>=hi else Y_ if v>=lo else R_) + f"{v:{f}}" + E_
def _cl(v, lo, hi, f=".4f"):
    return (G_ if v<=lo else Y_ if v<=hi else R_) + f"{v:{f}}" + E_


def run_report():
    h = load_hist(); n = len(h["epoch"])
    print("=" * 130)
    print(B_ + C_ + "  AQA-UNet  METRICS REPORT  (LOL-v2 + LSRW)".center(130) + E_)
    print("=" * 130)

    hdr = [("Ep",4),("G",8),("D",8),("Char",8),("SSIM",7),("Perc",7),("Adv",7),
           ("D(r)",6),("D(f)",6),("Skip%",6),
           ("PSNR",7),("SSIM",7),("QL0",6),("QL1",6),("QL2",6),
           ("Time",7),("img/s",6)]
    print(B_ + "".join(f"{nm:>{w}}" for nm, w in hdr) + E_)
    print("-" * 130)

    for i in range(n):
        e = h["epoch"][i]
        print(
            f"{e:>4}"
            + _cl(h["g"][i],       1.2, 2.0).rjust(8)
            + _cl(h["d"][i],       0.4, 0.7).rjust(8)
            + _cl(h["char"][i],    0.01, 0.05).rjust(8)
            + f"{h['ssim_l'][i]:>7.4f}"
            + f"{h['percept'][i]:>7.4f}"
            + f"{h['adv'][i]:>7.4f}"
            + _c(h["d_real"][i],       0.4, 0.6, ".3f").rjust(6)
            + _c(1-h["d_fake"][i],     0.4, 0.6, ".3f").rjust(6)
            + f"{h['skip'][i]*100:>6.1f}"
            + _c(h["val_psnr"][i],     20., 25., ".2f").rjust(7)
            + _c(h["val_ssim"][i],     0.70, 0.80, ".4f").rjust(7)
            + f"{h.get('val_psnr_ql0',[0]*n)[i]:>6.1f}"
            + f"{h.get('val_psnr_ql1',[0]*n)[i]:>6.1f}"
            + f"{h.get('val_psnr_ql2',[0]*n)[i]:>6.1f}"
            + f"{h['time_s'][i]:>7.1f}"
            + f"{h['tput'][i]:>6.1f}"
        )
    print("=" * 130)

    bp = max(h["val_psnr"]); bs = max(h["val_ssim"])
    bp_ep = h["epoch"][h["val_psnr"].index(bp)]
    bs_ep = h["epoch"][h["val_ssim"].index(bs)]
    tt    = sum(h["time_s"])
    print(f"\n  Best PSNR  : {B_}{bp:.3f} dB{E_} (epoch {bp_ep})  "
          f"{''+G_+'OK'+E_ if bp>=25 else Y_+f'gap {25-bp:.2f}dB'+E_}")
    print(f"  Best SSIM  : {B_}{bs:.4f}{E_}   (epoch {bs_ep})  "
          f"{''+G_+'OK'+E_ if bs>=0.80 else Y_+f'gap {0.80-bs:.4f}'+E_}")
    print(f"  Total time : {str(timedelta(seconds=int(tt)))}  ({tt/n:.1f}s/epoch)\n")

    # AQA conditioning check
    last = slice(max(0,n-10), n)
    ql0 = np.mean(h.get("val_psnr_ql0",[0]*n)[last])
    ql2 = np.mean(h.get("val_psnr_ql2",[0]*n)[last])
    print(f"  AQA check  : PSNR severe={ql0:.2f}dB  mild={ql2:.2f}dB  "
          f"gap={ql2-ql0:.2f}dB  (larger gap = FiLM conditioning active)\n")

    recs = []
    if bp < 20:
        recs.append((R_+"[CRITICAL]"+E_,
                     "PSNR<20dB -- verify pair filenames match, check LAMBDA_CHAR=100"))
    elif bp < 25:
        recs.append((Y_+"[HINT]"+E_,
                     f"PSNR={bp:.2f}dB -- try base_ch=48 or EPOCHS=200"))
    if abs(ql0 - ql2) < 0.3:
        recs.append((Y_+"[AQA]"+E_,
                     "FiLM gap<0.3dB -- conditioning not yet active; "
                     "try pip install pyiqa for accurate quality labels"))
    if bs < 0.80:
        recs.append((Y_+"[HINT]"+E_, "SSIM<0.80 -- raise LAMBDA_SSIM to 8.0"))
    recs.append((C_+"[TIP]"+E_, "pip install pyiqa  -- real NIQE for better quality labels"))
    recs.append((C_+"[TIP]"+E_, "Free +0.4dB: python lol_aqa_unet.py --mode test --tta"))
    for tag, msg in recs:
        print(f"  {tag}  {msg}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="AQA-UNet  LOL-v2 + LSRW")
    p.add_argument("--mode",   choices=["train","report","plots","all","test"],
                   default="all")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--no-adv", action="store_true",
                   help="Pixel losses only (fast baseline, no GAN)")
    p.add_argument("--tta",    action="store_true",
                   help="8-fold test-time augmentation at eval")
    a = p.parse_args()

    if a.mode in ("train", "all"):
        print("\n" + "="*65)
        print("  TRAINING  [AQA-UNet  LOL-v2 + LSRW]")
        print("="*65)
        train(resume=a.resume or RESUME_CKPT, use_adv=not a.no_adv)

    if a.mode in ("report", "all"):
        print("\n" + "="*65)
        print("  METRICS REPORT")
        print("="*65)
        run_report()

    if a.mode in ("plots", "all"):
        print("\n" + "="*65)
        print("  PLOTS")
        print("="*65)
        h = load_hist(); pp = save_plot(h, h["epoch"][-1])
        print(f"  Saved -> {pp}")

    if a.mode == "test":
        ckpt = os.path.join(DIRS["checkpoints"], "best_psnr.pth")
        _, _, te_ld, *_ = get_loaders()
        G = AQAUNet(base_ch=32).to(DEVICE)
        G.load_state_dict(torch.load(ckpt, map_location=DEVICE)["G"])
        test_eval(G, te_ld, tta=a.tta)


if __name__ == "__main__":
    main()