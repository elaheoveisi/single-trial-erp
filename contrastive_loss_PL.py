import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class SubjectClassifier(nn.Module):
    def __init__(self, feat_dim=2, n_subj=3):
        super().__init__()
        self.fc = nn.Linear(feat_dim, n_subj, bias=True)
        with torch.no_grad():
            self.fc.weight.copy_(torch.tensor([[1., 0.],
                                               [0., 1.],
                                               [1., 1.]]))
            self.fc.bias.zero_()

    def forward(self, h):
        return self.fc(h)

class Contrastive_Module(pl.LightningModule):
    def __init__(self, feat_dim=2, n_subj=3, tau=0.2, lambda_reg=1.0, lr=1e-3):
        super().__init__()
        self.tau = tau
        self.lambda_reg = lambda_reg
        self.lr = lr
        self.subject_classifier = SubjectClassifier(feat_dim, n_subj)

    def info_nce_loss(self, q, k):
        q_norm = F.normalize(q, p=2, dim=1)
        k_norm = F.normalize(k, p=2, dim=1)
        N = q.size(0)
        sim_matrix = torch.mm(q_norm, k_norm.T)
        loss = 0.0
        for i in range(N):
            numerator = torch.exp(sim_matrix[i, i] / self.tau)
            denominator = torch.sum(torch.exp(sim_matrix[i] / self.tau))
            l_i = -torch.log(numerator / denominator)
            loss += l_i
        return loss / N

    def training_step(self, batch, batch_idx):
        q, k, h_i, subj_lab = batch

        L_infoNCE = self.info_nce_loss(q, k)

        logits_sub = self.subject_classifier(h_i)
        loss_CE = F.cross_entropy(logits_sub, subj_lab)

        probs_sub = F.softmax(logits_sub, dim=1)
        p_true = probs_sub[torch.arange(h_i.size(0)), subj_lab]
        r_sub = -torch.log(1.0 - p_true)
        reg_term = torch.mean(r_sub)
         
        final_loss = L_infoNCE + self.lambda_reg * reg_term

        self.log("info_nce_loss", L_infoNCE)
        self.log("cross_entropy_loss", loss_CE)
        self.log("regularization", reg_term)
        self.log("final_loss", final_loss)

        return final_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

# Example usage
if __name__ == "__main__":
    model = Contrastive_Module()
    sample_1 = torch.tensor([[5.,4.],[2.,1.]])
    sample_2 = torch.tensor([[1.,2.],[8.,1.]])
    h_i = torch.tensor([[0.5, 1.0]])
    subj_lab = torch.tensor([2])
    batch = (sample_1, sample_2, h_i, subj_lab)
    output = model.training_step(batch, 0)
    print("Output Loss:", output.item())
