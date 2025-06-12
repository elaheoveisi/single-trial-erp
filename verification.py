import torch
import torch.nn.functional as F


#%%

''' InfoNCE Loss'''

''' Manual '''

def info_nce_eqn(q, k, tau = 0.2):
    """
    q, k: float-tensors of shape (B, D)
    tau: temperature scalar
    returns: scalar loss = mean_i ℓ_i
    where ℓ_i = -log( exp(q_i·k_i/τ) / sum_j exp(q_i·k_j/τ) )
    """
    #  normalize
    q_norm = torch.nn.functional.normalize(q, dim=1)  # shape (B, D)
    k_norm = torch.nn.functional.normalize(k, dim=1)

    #  similarity matrix (B×B)
    sim = q_norm @ k_norm.t()           # (B, B)  each entry = q_i·k_j
    sim = sim / tau                     # divide by temperature

    #  exponentiate
    exp_sim = torch.exp(sim)            # (B, B)

    #  numerator = exp(q_i·k_i/τ)  → the diagonal
    numer = exp_sim.diag()              # shape (B,)

    #  denominator = ∑_j exp(q_i·k_j/τ)  → row-sums
    denom = exp_sim.sum(dim=1)          # shape (B,)

    # per-sample loss ℓ_i
    loss_per = -torch.log(numer / denom)  # shape (B,)

    #  mean over batch
    return loss_per.mean()


# toy batch
q = torch.tensor([[5.,4.],[2.,1.]])
k = torch.tensor([[1.,2.],[8.,1.]])

print(f"Manual InfoNCE Loss = {info_nce_eqn(q, k):.4f}")

''' Pytorch InfoNCE'''

def info_nce(z1, z2, T= 0.2):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    logits = z1 @ z2.t() / T
    B = logits.size(0)
    labels = torch.arange(B)
    return F.cross_entropy(logits,     labels)   # one_way
    # return 0.5 * (F.cross_entropy(logits,     labels)   # symmetric
    #             + F.cross_entropy(logits.t(), labels))


z1 = torch.tensor([[5.,4.],[2.,1.]])
z2 = torch.tensor([[1.,2.],[8.,1.]])


print(f"Pytorch InfoNCE Loss = {info_nce(z1, z2, T= 0.2):.4f}")






#%% 

''' Cross Entropy Loss'''


import torch
import torch.nn.functional as F
from torch import nn, optim


''' Manual'''
#  the classifier weights and bias
W = torch.tensor([[1., 0.],
                  [0., 1.],
                  [1., 1.]])
b = torch.zeros(3)

# a single latent vector h_i
h_i = torch.tensor([0.5, 1.0])

#  logits: z = W h_i + b
logits = W @ h_i + b  # shape: (3,)
print("Logits:", logits.numpy())

#  predicted probabilities via softmax
probs = F.softmax(logits, dim=0)
print("Probabilities:", probs.numpy())

#  a true subject label (e.g., s_i = 2)
true_label = torch.tensor(2)

# Manual cross-entropy for label 2: -log(p_true)
loss_manual = -torch.log(probs[true_label])
print(f"Manual CE loss (label={true_label.item()}): {loss_manual.item():.4f}")


''' Subject Classifier  implementation'''

class SubjectClassifier(nn.Module):
    def __init__(self, feat_dim, n_subj):
        super().__init__()
        self.fc = nn.Linear(feat_dim, n_subj)
    def forward(self, h):
        return self.fc(h)   # logits over N_subj

 

#  feat_dim=2, n_subj=3
C_sub = SubjectClassifier(feat_dim = 2, n_subj = 3)

#  set its weights and zero biases for reproducibility:
#    W = [[1,0],[0,1],[1,1]], b = [0,0,0]
with torch.no_grad():
    C_sub.fc.weight.copy_(torch.tensor([[1., 0.],
                                        [0., 1.],
                                        [1., 1.]]))
    C_sub.fc.bias.zero_()

#  A single latent vector h_i and its true label (e.g. subject #2)
h_i      = torch.tensor([[0.5, 1.0]])   # shape (1,2)
subj_lab = torch.tensor([2])            # shape (1,)

logits_sub = C_sub(h_i)                 # shape (1,3)
print("Classifier Logits:", logits_sub)            


probs_sub = F.softmax(logits_sub, dim=1)
print("Classifier probabilities:", probs_sub)
 
#  cross-entropy via PyTorch
loss_pt = F.cross_entropy(logits_sub, subj_lab)
print("PyTorch CE:", loss_pt.item())     # ~0.6800

 
#%%
''' Regularization'''

''' Manual'''


import torch
import torch.nn.functional as F
from torch import nn, optim


''' Manual'''
#  the classifier weights and bias
W = torch.tensor([[1., 0.],
                  [0., 1.],
                  [1., 1.]])
b = torch.zeros(3)

# a single latent vector h_i
h_i = torch.tensor([0.5, 1.0])

#  logits: z = W h_i + b
logits = W @ h_i + b  # shape: (3,)
 
#  predicted probabilities via softmax
probs = F.softmax(logits, dim=0)
 
#  a true subject label (e.g., s_i = 2)
true_label = torch.tensor(2)

# Manual regularization for label 2: -log(1-p_true)
loss_manual = -torch.log(1- probs[true_label])
print(f"Manual regularization term (label={true_label.item()}): {loss_manual.item():.4f}")


''' Regularization implementation'''

class SubjectClassifier(nn.Module):
    def __init__(self, feat_dim, n_subj):
        super().__init__()
        self.fc = nn.Linear(feat_dim, n_subj)
    def forward(self, h):
        return self.fc(h)   # logits over N_subj

 

#  feat_dim=2, n_subj=3
C_sub = SubjectClassifier(feat_dim = 2, n_subj = 3)

#  set its weights and zero biases for reproducibility:
#    W = [[1,0],[0,1],[1,1]], b = [0,0,0]
with torch.no_grad():
    C_sub.fc.weight.copy_(torch.tensor([[1., 0.],
                                        [0., 1.],
                                        [1., 1.]]))
    C_sub.fc.bias.zero_()

#  A single latent vector h_i and its true label (e.g. subject #2)
h_i      = torch.tensor([[0.5, 1.0]])       # shape (1,2)
subj_lab = torch.tensor([2])                # shape (1,)

logits_sub = C_sub(h_i)                     # shape (1,3)
probs_sub = F.softmax(logits_sub, dim=1)
p_true = probs_sub[torch.arange(h_i.size(0)), subj_lab]  # shape (1,)

# 3) compute r_sub = -log(1 - p_true)
r_sub = -torch.log(1.0 - p_true)
 
print("Classifier Regularization term:", r_sub)      


