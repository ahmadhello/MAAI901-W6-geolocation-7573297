"""Models, data loading, scene labels, training and evaluation for the study."""
import io, os, json, math, time, random, zipfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageEnhance

MODELS = {
    "clip":   dict(ckpt="openai/clip-vit-base-patch16",
                   mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711), feat=768),
    "vit":    dict(ckpt="google/vit-base-patch16-224-in21k", mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), feat=768),
    "resnet": dict(ckpt="microsoft/resnet-50", mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), feat=2048),
}
NAMES = {"clip": "CLIP ViT-B/16", "vit": "ViT-B/16 (IN-21k)", "resnet": "ResNet-50"}
N_CLASSES = 25


# ------------------------------------------------------------------ data
class ShardStore:
    """Reads cached JPEGs from shard zips; each worker opens its own handles."""
    def __init__(self, shard_paths):
        self.paths = list(shard_paths)
        self.where = {}
        for p in self.paths:
            with zipfile.ZipFile(p) as z:
                for n in z.namelist():
                    self.where[n[:-4]] = p
        self._open, self._pid = {}, os.getpid()

    def get(self, i):
        if os.getpid() != self._pid:          # new DataLoader worker: never share file handles across processes
            self._open, self._pid = {}, os.getpid()
        p = self.where[i]
        z = self._open.get(p)
        if z is None:
            z = self._open[p] = zipfile.ZipFile(p)
        return Image.open(io.BytesIO(z.read(f"{i}.jpg"))).convert("RGB")

    def __getstate__(self):
        d = dict(self.__dict__); d["_open"] = {}; return d


def random_resized_crop(im, size=224, scale=(0.8, 1.0), ratio=(3 / 4, 4 / 3), rng=random):
    W, H = im.size
    area = W * H
    for _ in range(10):
        a = area * rng.uniform(*scale)
        r = math.exp(rng.uniform(math.log(ratio[0]), math.log(ratio[1])))
        w, h = int(round(math.sqrt(a * r))), int(round(math.sqrt(a / r)))
        if 0 < w <= W and 0 < h <= H:
            x, y = rng.randint(0, W - w), rng.randint(0, H - h)
            return im.crop((x, y, x + w, y + h)).resize((size, size), Image.BICUBIC)
    return im.resize((size, size), Image.BICUBIC)


def to_tensor(im, mean, std):
    a = np.asarray(im, dtype=np.float32) / 255.0
    a = (a - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    return torch.from_numpy(a.transpose(2, 0, 1).copy())


class ImageSet(torch.utils.data.Dataset):
    def __init__(self, store, ids, labels, mean, std, train=False, center_crop=False):
        self.store, self.ids, self.labels = store, list(ids), list(labels)
        self.mean, self.std, self.train, self.center_crop = mean, std, train, center_crop

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, k):
        im = self.store.get(self.ids[k])
        if self.train:
            im = random_resized_crop(im)
            im = ImageEnhance.Brightness(im).enhance(random.uniform(0.9, 1.1))
            im = ImageEnhance.Contrast(im).enhance(random.uniform(0.9, 1.1))
        elif self.center_crop:           # used only for the Places365 scene classifier
            im = im.crop((16, 16, 240, 240))
        else:
            im = im.resize((224, 224), Image.BICUBIC)
        return to_tensor(im, self.mean, self.std), self.labels[k]


def _worker_init(wid):
    s = torch.initial_seed() % 2**31
    random.seed(s + wid); np.random.seed((s + wid) % 2**31)


def loader(ds, batch, shuffle, seed=0, workers=2, drop_last=False):
    g = torch.Generator(); g.manual_seed(seed)
    return torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=shuffle, generator=g, num_workers=workers,
                                       pin_memory=torch.cuda.is_available(), drop_last=drop_last,
                                       worker_init_fn=_worker_init, persistent_workers=False)


# ------------------------------------------------------------------ models
class Net(nn.Module):
    def __init__(self, kind):
        super().__init__()
        from transformers import CLIPVisionModel, ViTModel, ResNetModel
        self.kind = kind
        ck = MODELS[kind]["ckpt"]
        if kind == "clip":
            self.backbone = CLIPVisionModel.from_pretrained(ck)
        elif kind == "vit":
            self.backbone = ViTModel.from_pretrained(ck, add_pooling_layer=False)
        else:
            self.backbone = ResNetModel.from_pretrained(ck)
        self.head = nn.Linear(MODELS[kind]["feat"], N_CLASSES)

    def features(self, x):
        if self.kind == "clip":
            return self.backbone(pixel_values=x).pooler_output        # CLS after final layer norm, before projection
        if self.kind == "vit":
            return self.backbone(pixel_values=x).last_hidden_state[:, 0]   # class token
        return self.backbone(pixel_values=x).pooler_output.flatten(1)       # 2,048-d pooled features

    def forward(self, x, frozen=False):
        if frozen:
            with torch.no_grad():
                f = self.features(x)
            return self.head(f.float())
        return self.head(self.features(x).float())


def param_groups(model, wd):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 or "norm" in n.lower() or n.endswith(".bias") else decay).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


def warmup_cosine(opt, total, warm_frac=0.05):
    warm = max(1, int(total * warm_frac))
    def f(step):
        if step < warm:
            return (step + 1) / warm
        p = (step - warm) / max(1, total - warm)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


@torch.no_grad()
def predict(model, dl, device, frozen=False):
    model.eval()
    out, ys = [], []
    for x, y in dl:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            out.append(model(x).float().cpu())
        ys.append(y)
    return torch.cat(out).numpy(), torch.cat(ys).numpy()


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def run_stage(model, stage, train_dl_fn, val_dl, epochs, lr, device, ckpt_path=None, log=print,
              wd=0.05, clip=1.0, start_state=None):
    """One training stage. stage 1 = head only on a frozen backbone, stage 2 = all weights.
    Saves a checkpoint after every epoch (replacing the previous one).
    Returns per-epoch validation accuracy and the state dict of the best epoch."""
    frozen = stage == 1
    model.backbone.requires_grad_(not frozen)
    opt = torch.optim.AdamW(param_groups(model, wd), lr=lr, betas=(0.9, 0.999))
    steps_per_epoch = len(train_dl_fn(0))
    sched = warmup_cosine(opt, epochs * steps_per_epoch)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    hist, best, best_state, start_ep = [], -1.0, None, 0
    if ckpt_path and os.path.exists(ckpt_path):
        st = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        scaler.load_state_dict(st["scaler"]); hist, best, start_ep = st["hist"], st["best"], st["epoch"] + 1
        best_state = st.get("best_state")
        log(f"  resumed at epoch {start_ep + 1}/{epochs}")
    for ep in range(start_ep, epochs):
        t0 = time.time()
        if frozen:
            model.eval(); model.head.train()
        else:
            model.train()
        tot, n = 0.0, 0
        for x, y in train_dl_fn(ep):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                loss = F.cross_entropy(model(x, frozen=frozen), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], clip)
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * len(y); n += len(y)
        logits, ys = predict(model, val_dl, device)
        acc = float((logits.argmax(1) == ys).mean())
        hist.append({"epoch": ep + 1, "train_loss": tot / n, "val_acc": acc, "minutes": (time.time() - t0) / 60})
        log(f"  stage {stage} epoch {ep + 1}/{epochs}: loss {tot / n:.3f}  val acc {acc:.4f}  ({(time.time() - t0) / 60:.1f} min)")
        if acc > best:
            best = acc
            best_state = {k: v.detach().half().cpu() if v.is_floating_point() else v.cpu() for k, v in model.state_dict().items()} \
                if stage == 2 else {k: v.detach().cpu() for k, v in model.head.state_dict().items()}
        if ckpt_path:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "scaler": scaler.state_dict(), "hist": hist, "best": best, "epoch": ep,
                        "best_state": best_state}, ckpt_path + ".tmp")
            os.replace(ckpt_path + ".tmp", ckpt_path)
    return hist, best_state


# ------------------------------------------------------------------ Places365 scene classifier
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inp, planes, stride=1, down=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inp, planes, 1, bias=False); self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False); self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, 1, bias=False); self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True); self.downsample = down
    def forward(self, x):
        idt = x if self.downsample is None else self.downsample(x)
        x = self.relu(self.bn1(self.conv1(x))); x = self.relu(self.bn2(self.conv2(x))); x = self.bn3(self.conv3(x))
        return self.relu(x + idt)


class ResNet50(nn.Module):
    """Same layout and parameter names as torchvision's resnet50, so the Places365 weights load directly."""
    def __init__(self, n=365):
        super().__init__()
        self.inp = 64
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False); self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True); self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make(64, 3); self.layer2 = self._make(128, 4, 2)
        self.layer3 = self._make(256, 6, 2); self.layer4 = self._make(512, 3, 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1); self.fc = nn.Linear(2048, n)
    def _make(self, planes, blocks, stride=1):
        down = None
        if stride != 1 or self.inp != planes * 4:
            down = nn.Sequential(nn.Conv2d(self.inp, planes * 4, 1, stride, bias=False), nn.BatchNorm2d(planes * 4))
        layers = [Bottleneck(self.inp, planes, stride, down)]
        self.inp = planes * 4
        layers += [Bottleneck(self.inp, planes) for _ in range(1, blocks)]
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(torch.flatten(self.avgpool(x), 1))


def load_places365(path):
    m = ResNet50()
    st = torch.load(path, map_location="cpu", weights_only=False)
    st = st.get("state_dict", st)
    m.load_state_dict({k.replace("module.", ""): v for k, v in st.items()})
    return m.eval()


def scene_groups(probs, W):
    """probs: (N, 365) softmax; W: (365, 3) membership (indoor, natural, man-made).
    A category in several groups is split equally between them, so the three totals sum to one."""
    Wn = W / W.sum(1, keepdims=True)
    g = probs @ Wn
    lab = np.array(["indoor", "natural", "urban"])[g.argmax(1)]
    return g, lab
