 

import torch
import torch.nn.functional as F
from torch import nn, optim


''' InfoNCE Loss'''

 

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


print(f"InfoNCE_Loss = {info_nce(z1, z2, T= 0.2):.4f}")


''' Output

InfoNCE_Loss = 0.4810

'''


''' Cross Entropy Loss'''


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
     


probs_sub = F.softmax(logits_sub, dim=1)
 
 
#  cross-entropy via PyTorch
loss_pt = F.cross_entropy(logits_sub, subj_lab)
print("Cross Entropy Loss:", loss_pt.item())     

''' Output

 Cross Entropy Loss: 0.68026

'''

''' Regularization'''
 

class SubjectClassifier(nn.Module):
    def __init__(self, feat_dim, n_subj):
        super().__init__()
        self.fc = nn.Linear(feat_dim, n_subj)
    def forward(self, h):
        return self.fc(h)   # logits over N_subj

 

#  feat_dim=2, n_subj=3
C_sub = SubjectClassifier(feat_dim = 2, n_subj = 3)

#    weights and zero biases for reproducibility:
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


''' Output

Classifier Regularization term: tensor([0.7062]

'''
