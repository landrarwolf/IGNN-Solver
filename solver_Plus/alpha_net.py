import torch
import torch.nn as nn
import torch.nn.functional as F

from new.GCN_layers import alpha_gcn


class AlphaNet(nn.Module):
    def __init__(self, nhid, learn_beta, ninner, alpha_model="alpha_tcn"):  
        super().__init__()
        self.alpha_model = alpha_model
        self.nhid = nhid
        self.ninner = ninner
        self.learn_beta = learn_beta
        self.seq_fc = None  # Should be set by subclass

        #
        if self.alpha_model == "alpha_rnn":  # multi-layer gated recurrent unit (GRU) RNN 
            self.alpha_rnn = nn.GRU(ninner, ninner, batch_first=True, bidirectional=True)  
            hidden_factor = 2
        elif self.alpha_model == "alpha_tcn":
            self.alpha_tcn = nn.Sequential(nn.Conv1d(ninner, ninner, kernel_size=5, padding=2),
                                           nn.GroupNorm(5, ninner),
                                           nn.ReLU(),
                                           nn.Conv1d(ninner, ninner, kernel_size=5, padding=2),
                                           )  #

            hidden_factor = 1

        elif self.alpha_model == "alpha_gcn":
            self.alpha_gcn = alpha_gcn(ninner)  
            hidden_factor = 1

        elif self.alpha_model == "alpha_nn":
            self.alpha_nn = nn.Sequential(
                nn.Linear(ninner, ninner),
                nn.ReLU(),
                nn.Linear(ninner, ninner),
            )  
            hidden_factor = 1

        else:
            raise ValueError("alpha_xxn should be True!")


        self.alpha_predictor = nn.Sequential(nn.Linear(ninner * hidden_factor, ninner),
                                             nn.ReLU(),
                                             nn.Linear(ninner, 1))
        if learn_beta:
            # print("Learning beta by beta=beta(G)!")
            self.beta_predictor = nn.Sequential(nn.Linear(ninner * hidden_factor, ninner),
                                                nn.ReLU(),
                                                nn.Linear(ninner, 1)
                                                )

    def forward_RR(self, G, **kwargs): 
        raise NotImplemented("This function is not implemented in the abstract class AlphaNet")

    def forward(self, RR, n, k, m, up, A):
        bsz = RR.shape[0]
        if not self.training:
            # Re-organize the order because the indexing wraps around in eval mode
            if k >= m:
                RR = torch.cat([RR[:, up:], RR[:, :up]], dim=1)  

        if "alpha_tcn" in self.__dict__['_modules']:
            RR = RR.clone().transpose(1, 2)  
            raw_ab = self.alpha_tcn(RR) + RR  
            beta = (self.beta_predictor(F.avg_pool1d(raw_ab, kernel_size=raw_ab.shape[-1]).squeeze(-1))
                    .view(bsz, 1)) if self.learn_beta else None 
            raw_ab = raw_ab.transpose(1, 2)

        elif "alpha_rnn" in self.__dict__['_modules']:
            raw_ab, _ = self.alpha_rnn(RR.clone()) 
            raw_ab_bidir = raw_ab.view(bsz, -1, 2, self.ninner)  
            beta = (self.beta_predictor(torch.cat([raw_ab_bidir[:, -1, 0], raw_ab_bidir[:, 0, 1]], dim=-1))
                    .view(bsz, 1)) if self.learn_beta else None  

        elif "alpha_gcn" in self.__dict__['_modules']:
            RR = RR.clone().transpose(1, 2)  
            raw_ab = self.alpha_gcn(RR, adj=A) + RR  
            beta = (self.beta_predictor(F.avg_pool1d(raw_ab, kernel_size=raw_ab.shape[-1]).squeeze(-1))
                    .view(bsz, 1)) if self.learn_beta else None 
            raw_ab = raw_ab.transpose(1, 2) 

        elif "alpha_nn" in self.__dict__['_modules']:
            RR_list = torch.unbind(RR.clone(), dim=1)
            raw_ab = torch.zeros((bsz, len(RR_list), self.ninner)).to(RR.device)
            for i in range(len(RR_list)):
                raw_ab[:, i, :] = self.alpha_nn(RR_list[i]) 

            raw_ab = raw_ab.transpose(1, 2)
            beta = (self.beta_predictor(F.avg_pool1d(raw_ab, kernel_size=raw_ab.shape[-1]).squeeze(-1))
                    .view(bsz, 1)) if self.learn_beta else None
            raw_ab = raw_ab.transpose(1, 2)

        else:
            raise ValueError("Either alpha_rnn or alpha_tcn should be True!")


        raw_alpha = self.alpha_predictor(raw_ab).view(bsz, n)
        alpha = raw_alpha + (1 - raw_alpha.sum(1, keepdim=True)) / n  # Make sure alpha sums to 1. Shape: (bsz x n)

        if not self.training:
            if k >= m:
                alpha = torch.cat([alpha[:, (n - up):], alpha[:, :(n - up)]], dim=1)
        return alpha, beta 


class SequenceAlphaNet(AlphaNet):
    def __init__(self, nhid, learn_beta, ninner, alpha_model="alpha_tcn"): 
        super(SequenceAlphaNet, self).__init__(nhid=nhid, ninner=ninner, learn_beta=learn_beta)
        self.seq_fc = nn.Sequential(nn.Linear(nhid, ninner), nn.ReLU(), nn.Linear(ninner, ninner))

    def forward_RR(self, G, **kwargs):
        out = self.seq_fc(G) 

        return out
