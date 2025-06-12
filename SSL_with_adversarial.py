# fine-tuning + lots of augmentations + adversarial + subject classifier
import os
import glob
import copy
import numpy as np
import mne
import torch
import torch.nn.functional as F
from torch import nn,optim
from torch.utils.data import Dataset,TensorDataset,DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 0) Hyperparameters for contrastive pre-training
BATCH_SIZE   = 256
WINDOW_SIZE  = 320         # samples per segment (≈2 s @160 Hz)
LR           = 1e-5        # scheduled
EPOCHS       = 50
MOMENTUM     = 0.99
TAU          = 0.15      # contrastive loss temperature

# NN classification

epochs_2class = 100
batch_size_2class = 64
LR_2class = 1e-3          # scheduled

# 1) DATA AUGMENTATIONS
class TemporalCutout:
    def __init__(self, max_width): self.max_width = max_width
    def __call__(self, x):
        w = np.random.randint(1, self.max_width + 1)
        start = np.random.randint(0, x.shape[1] - w + 1)
        x[:, start:start + w] = 0
        return x

class AddNoise:
    def __init__(self, scale): self.scale = scale
    def __call__(self, x):
        return x + np.random.randn(*x.shape).astype(np.float32) * self.scale

class Compose:
    def __init__(self, transforms): self.transforms = transforms
    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

class Scaling:
    """Randomly scale the signal amplitude by a factor in [low, high]."""
    def __init__(self, low=0.9, high=1.1):
        self.low, self.high = low, high
    def __call__(self, x):
        factor = np.random.uniform(self.low, self.high)
        return (x * factor).astype(np.float32)

class TemporalShift:
    """Circularly shift the signal by up to max_shift samples."""
    def __init__(self, max_shift):
        self.max_shift = max_shift
    def __call__(self, x):
        shift = np.random.randint(-self.max_shift, self.max_shift + 1)
        return np.roll(x, shift, axis=1).astype(np.float32)
class ChannelDropout:
    """Randomly zero‐out a fraction of channels each call."""
    def __init__(self, drop_prob):
        self.drop_prob = drop_prob
    def __call__(self, x):
        mask = np.random.rand(x.shape[0]) > self.drop_prob
        return (x * mask[:, None]).astype(np.float32)

class FrequencyMask:
    """Randomly mask out a band of frequencies in the FFT domain."""
    def __init__(self, max_width):
        self.max_width = max_width
    def __call__(self, x):
        # x: (C, L)
        # FFT along time
        Xf = np.fft.rfft(x, axis=1)
        freq_bins = Xf.shape[1]
        f0 = np.random.randint(0, freq_bins - self.max_width)
        w  = np.random.randint(1, self.max_width + 1)
        Xf[:, f0:f0 + w] = 0
        x2 = np.fft.irfft(Xf, n=x.shape[1], axis=1)
        return x2.astype(np.float32)
        
transform_q = Compose([ Scaling(0.8, 1.2), TemporalCutout(80), TemporalShift(20),FrequencyMask(30), AddNoise(0.2), ChannelDropout(0.1)])
transform_k = Compose([ Scaling(0.9, 1.1), TemporalCutout(60),TemporalShift(10), FrequencyMask(20), AddNoise(0.25), ChannelDropout(0.05)])

# 2) MoCo‐STYLE DATASET (returns (C, L) pairs), C--eeg channel count , L---window_size , MoCo --Momentum Contrast
class MoCoDataset(Dataset):
    def __init__(self, edf_paths, window_size, t_q, t_k):
        self.segments = []
        self.subj_labels = []
        # build subject→int mapping
        subjects = sorted({os.path.dirname(p).split(os.sep)[-1]
                           for p in edf_paths})
        self.subj2idx = {s:i for i,s in enumerate(subjects)}
        for p in edf_paths:
            subj = os.path.dirname(p).split(os.sep)[-1]
            lab  = self.subj2idx[subj]
            raw  = mne.io.read_raw_edf(p, preload=True, verbose=False)
            data = raw.get_data()*10e6  #  shape = (n_channels, n_times)  n_times = total number of samples per channel = (sampling_rate × recording_duration)= 160.0 × 120 s = 19200 samples 
             
            n_window  = data.shape[1] // window_size
            
            for i in range(n_window):
                seg = data[:, i*window_size:(i+1)*window_size].astype(np.float32)  # seg.shape == (n_channels, window_size)
                self.segments.append(seg)
                self.subj_labels.append(lab)

        self.t_q = t_q  
        self.t_k = t_k  

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        seg = self.segments[idx]    # numpy array (C, L)
        label = self.subj_labels[idx]
        v_q = self.t_q(seg.copy())  # apply  q  transforms
        v_k = self.t_k(seg.copy())  # apply  k transforms
        # ensure float32 before creating tensor
        return torch.from_numpy(v_q).float(), torch.from_numpy(v_k).float(), torch.tensor(label, dtype=torch.long)   # shape = (C, L)  C--eeg channel count , L---window_size

# 3) MODEL DEFINITIONS
# 1D Residual Block
class BasicBlock(nn.Module):
    def __init__(self, in_c, out_c, k, s=1, down=None):    #in_c = input channel number, out_c = output channel number, k = kernel size, s = stride
        super().__init__()
        self.bn0   = nn.BatchNorm1d(in_c)
        self.elu   = nn.ELU(inplace=True)
        self.conv1 = nn.Conv1d(in_c, out_c, k, s, k//2, bias=False)
        self.bn1   = nn.BatchNorm1d(out_c)
        self.conv2 = nn.Conv1d(out_c, out_c, k, 1, k//2, bias=False)
        self.down  = down

    def forward(self, x):
        out = self.elu(self.bn0(x))
        idt = out
        out = self.elu(self.bn1(self.conv1(out)))
        out = self.conv2(out)
        if self.down:
            idt = self.down(x)
        return out + idt   # shape (B, out_c, L_out), B--Batch

class Resnet8(nn.Module):
    def __init__(self, n_ch):
        super().__init__()
        self.inp   = 32
        self.conv1 = nn.Conv1d(n_ch, 32, 13, 2, 3, bias=False)
        self.l1    = self._make_layer(32, 11, blocks=1, stride=1)
        self.l2    = self._make_layer(128, 9,  blocks=1, stride=1)
        self.l3    = self._make_layer(256, 7,  blocks=1, stride=2)
        self.elu   = nn.ELU(inplace=True)
        self.ap    = nn.AdaptiveAvgPool1d(1)

    def _make_layer(self, planes, k, blocks, stride):
        down = None
        if stride != 1 or self.inp != planes:
            down = nn.Sequential(
                nn.Conv1d(self.inp, planes, 1, stride, bias=False),
                nn.BatchNorm1d(planes)
            )
        layers = [BasicBlock(self.inp, planes, k, stride, down)]
        self.inp = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inp, planes, k))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, C, L)
        x = self.conv1(x)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.elu(x)
        x = self.ap(x)
        return torch.flatten(x, 1)  # (B, 256)

class ProjectionHead(nn.Module): #F
    def __init__(self, i=256, h=128, o=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(i, h),
            nn.ReLU(True),
            nn.Linear(h, o),
        )
    def forward(self, x):
        return self.net(x)
        
class SubjectClassifier(nn.Module):
    def __init__(self, feat_dim, n_subj):
        super().__init__()
        self.fc = nn.Linear(feat_dim, n_subj)
    def forward(self, h):
        return self.fc(h)   # logits over N_subj

# 4) INFO‐NCE LOSS & MOMENTUM UPDATE
def info_nce_loss(z1, z2, T):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    logits = torch.mm(z1, z2.T) / T
    B = z1.size(0)
    labels = torch.arange(B, device=z1.device)
    return F.cross_entropy(logits,     labels)

@torch.no_grad()
def momentum_update(q_enc, k_enc, q_proj, k_proj, m):
    for p, pk in zip(q_enc.parameters(), k_enc.parameters()):
        pk.data.mul_(m).add_(p.data, alpha=1-m)    # pk = m * pk + (1-m) * p
    for p, pk in zip(q_proj.parameters(), k_proj.parameters()):
        pk.data.mul_(m).add_(p.data, alpha=1-m)

def main(): 
    # ── 1) GATHER ALL RUNS 03,04,07,08,11,12 ───────────────────
    root      = "eeg_dataset_5sub/eeg_dataset_5sub"
    runs      = ["03","04","07","08","11","12"]
    
    # — pick multiple subject here —    
    edf_paths = []
    for subj in sorted(os.listdir(root)):
        for r in runs:
            f = f"{subj}R{r}.edf"
            p = os.path.join(root,subj,f)
            if os.path.exists(p): edf_paths.append(p)
            
        
    # ── 2) CONTRASTIVE PRE-TRAINING ─────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    raw0   = mne.io.read_raw_edf(edf_paths[0], preload=False, verbose=False)
    n_ch   = len(raw0.ch_names)
    
    qe     = Resnet8(n_ch).to(device)   # q encoder
    ke     = copy.deepcopy(qe).to(device);  [p.requires_grad_(False) for p in ke.parameters()]  # k encoder
    qp     = ProjectionHead().to(device)    # q projectionhead
    kp     = copy.deepcopy(qp).to(device); [p.requires_grad_(False) for p in kp.parameters()]  # k projectionhead
    
    ds_ssl = MoCoDataset(edf_paths, WINDOW_SIZE, transform_q, transform_k)
    ld_ssl = DataLoader(ds_ssl, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    N_subj = len(ds_ssl.subj2idx)
    C_sub  = SubjectClassifier(feat_dim=256, n_subj=N_subj).to(device)
    
    opt_q   = torch.optim.Adam( list(qe.parameters()) + list(qp.parameters()), lr=LR)
    opt_sub = torch.optim.Adam(C_sub.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(opt_q, step_size=30, gamma=0.1)
    
    # ── 3) MoCo loop with adversarial steps ─────────
    λ = 1.0
    for ep in range(1, EPOCHS+1):
        tot = 0.0
        for vq, vk, subj_lab in ld_ssl:
            vq, vk, subj_lab = vq.to(device), vk.to(device), subj_lab.to(device)
            B = vq.size(0)
    
            # — Step A: train C_sub to minimize Cross Entropy (subject, logits) —
            with torch.no_grad():
                h_detach = qe(vq)          # freeze G here
            logits_sub = C_sub(h_detach)   # (B, N_subj)   # logits- model's pre-softmax score
            loss_sub   = F.cross_entropy(logits_sub, subj_lab)
    
            opt_sub.zero_grad()
            loss_sub.backward()
            opt_sub.step()
    
            # — Step B: train (G,F) to minimize InfoNCE  – λ·r_sub
            #    (subject head frozen)
            zq       = qp(qe(vq))   # G → F on v_q, produces embeddings hq
            with torch.no_grad():
                zk = kp(ke(vk))      # Gk → Fk on v_k, produces embeddings hk
            loss_con = info_nce_loss(zq, zk, TAU)
    
          
            h2        = qe(vq)           # now gradients flow into qe ( q encoder)
            logits2   = C_sub(h2)        # C_sub frozen
            probs   = F.softmax(logits2, dim=1)  # (B, N_subj)

            
            p_true = probs[torch.arange(B), subj_lab]  # (B,)
    
            #  regularization term,  r_sub = mean_i [ -log(1 - p_true[i]) ]
            eps    = 1e-8
            r_sub  = -torch.log(1.0 - p_true + eps).mean()
    
             
            loss_total = loss_con + λ * r_sub
    
            # — disable subject‐head grads before adversarial backward prop —
            for p in C_sub.parameters():
                p.requires_grad_(False)
            
            # backprop only into (qe, qp)
            opt_q.zero_grad()
            loss_total.backward()
            opt_q.step()
            
            # — re‐enable subject‐head grads —
            for p in C_sub.parameters():
                p.requires_grad_(True)

    
            # — Momentum‐update key network —
            momentum_update(qe, ke, qp, kp, MOMENTUM)
    
            tot += loss_con.item() * B
    
        scheduler.step()
        print(f"Epoch {ep:02d} — InfoNCE: {tot/len(ds_ssl):.4f}")

    
    # ── 4) EPOCHING T1/T2 FOR CLASSIFICATION ────────────────────
    event_id = {'T1': 0, 'T2': 1}   # desired label mapping
    
    feats, labs = [], []
    for p in edf_paths:
        raw = mne.io.read_raw_edf(p, preload=True, verbose=False)
    
        events, annot_map = mne.events_from_annotations(raw, verbose=False)
    
        id_T1 = annot_map['T1']
        id_T2 = annot_map['T2']
    
        epochs = mne.Epochs(
            raw,
            events,
            event_id={'T1': id_T1, 'T2': id_T2},
            tmin=-0.2,
            tmax=2.0,
            baseline=(None, 0),
            preload=True,
            verbose=False
        )

        X = epochs.get_data()             # shape (n_epochs, C, T)
        ys_raw = epochs.events[:, 2]      # label ids
        ys = (ys_raw == id_T2).astype(int) # convert to 0/1
    
        feats.append(torch.from_numpy(X).float())
        labs.append(torch.from_numpy(ys).long())
    
    X = torch.cat(feats, dim=0)  # EEG segments (N, C, T)
    y = torch.cat(labs, dim=0)   # labels (N,)

    # Split
    Xtr, Xte, ytr, yte = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    
    #  Build TensorDatasets
    train_ds = TensorDataset(Xtr, ytr)
    test_ds  = TensorDataset(Xte, yte)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size_2class, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size_2class)

    # ── 5) Fine-Tuning Encoder + Classifier ─────────────────────────────────────
    
    # Build a small classifier (same as your previous one)
    class FFNClassifier(nn.Module):
        def __init__(self, in_dim=256, h_dim=128, out_dim=2):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(h_dim, out_dim),
            )
        def forward(self, x):
            return self.net(x)

    # Build the full Fine-Tune model
    class FineTuneModel(nn.Module):
        def __init__(self, encoder, classifier):
            super().__init__()
            self.encoder = encoder
            self.classifier = classifier
    
        def forward(self, x):
            x = self.encoder(x)
            x = self.classifier(x)
            return x

    #  Instantiate
    clf_net = FFNClassifier().to(device)
    finetune_model = FineTuneModel(qe, clf_net).to(device)
    
    # Define loss, optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(finetune_model.parameters(), lr=LR_2class)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    #  Fine-tuning loop
    for epoch in range(1, epochs_2class+1):
        finetune_model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = finetune_model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        
        scheduler.step()
        
        print(f"[Fine-tune Train] Epoch {epoch:02d}, Loss: {total_loss/len(train_ds):.4f}")
    
    #  Evaluation
    finetune_model.eval()
    correct = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = finetune_model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
    
    acc = correct / len(test_ds)
    print(f"[Fine-tune Test] Segment-level accuracy: {acc:.4f}")
if __name__ == "__main__":
    main()

