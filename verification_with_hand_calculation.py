

import torch
import torch.nn.functional as F
from torch import nn, optim

''' InfoNCE Loss'''



def info_nce_loss(q, k, tau=0.2):
   
    
    q_norm = F.normalize(q, p=2, dim=1)  # shape [N, D] , N = No of samples in the batch, D = dimension of vector
    k_norm = F.normalize(k, p=2, dim=1)  # p = 2 for L2 Norm, dim = 1 for  row-wise normalization 

    N = q.size(0)    # No of rows
    loss = 0.0

    #  similarity matrix: S_ij = q_i · k_j
    sim_matrix = torch.mm(q_norm, k_norm.T)  # shape [N, N]
    #print(sim_matrix)

    for i in range(N):
        # numerator = exp(q_i ⋅ k_i / tau)
        numerator = torch.exp(sim_matrix[i, i] / tau)
 
        # denominator = sum_j exp(q_i ⋅ k_j / tau)
        denominator = torch.sum(torch.exp(sim_matrix[i] / tau))
        
        l_i = -torch.log(numerator / denominator)
        loss += l_i

    return loss / N  # mean loss 


sample_1 = torch.tensor([[5.,4.],[2.,1.]])
sample_2 = torch.tensor([[1.,2.],[8.,1.]])


L_infoNCE = info_nce_loss(sample_1, sample_2)
 

print(f"InfoNCE_Loss = {L_infoNCE:.4f}")



''' Subject Classifier  implementation'''

class SubjectClassifier(nn.Module):
    def __init__(self, feat_dim = 2, n_subj = 3):
        super().__init__()
        self.fc = nn.Linear(feat_dim, n_subj, bias=True)

    def forward(self, h):
        return self.fc(h)  # returns logits
 

#  feat_dim=2, n_subj=3
C_sub = SubjectClassifier(feat_dim = 2, n_subj = 3)

#   weights and zero biases:
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

predicted_class = torch.argmax(logits_sub, dim=1)
print("Predicted Subject Class:", predicted_class.item())


'''cross-entropy via PyTorch'''

loss_CE = F.cross_entropy(logits_sub, subj_lab)
print("Cross Entropy Loss:", loss_CE.item())     

''' Regularization'''
probs_sub = F.softmax(logits_sub, dim=1)
p_true = probs_sub[torch.arange(h_i.size(0)), subj_lab]  # shape (1,)

#  r_sub = -log(1 - p_true)
r_sub = -torch.log(1.0 - p_true)
 
print("Regularization term:", r_sub)     


''' Final Loss''' 

lambda_reg = 1.0

Final_loss = L_infoNCE + lambda_reg * r_sub

 
Sub_loss = loss_CE