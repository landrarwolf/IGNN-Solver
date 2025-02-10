import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse
from torch import autograd
from torch.nn import Module
from torch.nn import Parameter

from solver_Plus.Hyper_anderson import anderson, learnable_aa
from solver_Plus.initializer import Embedding4Initializer
from utils import get_spectral_rad, projection_norm_inf


class ImplicitGraph(Module):
    def __init__(self, in_features, out_features, num_node, kappa=0.99):
        super(ImplicitGraph, self).__init__()
        self.in_features = in_features
        self.m = out_features
        self.n = num_node
        self.k = kappa  # if set kappa=0, projection will be disabled at forward feeding.

        self.W = Parameter(torch.FloatTensor(self.m, self.m))  
        self.B = Parameter(torch.FloatTensor(self.in_features, self.m))  

        self.init() 

    def init(self):
        stdv = 1. / math.sqrt(self.W.size(1)) 
        self.W.data.uniform_(-stdv, stdv) 
        self.B.data.uniform_(-stdv, stdv)

    def forward(self, X_pre, A, U, phi=F.relu):
        A_U_B = torch.spmm(torch.spmm(A, U), self.B)

        if self.k != 0:  
            A_rho = get_spectral_rad(A)  
            self.W = projection_norm_inf(self.W, kappa=self.k / A_rho)  

        X_W = torch.spmm(X_pre, self.W) 
        A_X_W = torch.spmm(A, X_W) 
        # X_ = self.W @ X_pre

        X_new = phi(A_X_W + A_U_B)  

        return X_new


class IGNN_solver(nn.Module):
    def __init__(self, nfeat, nhid, nclass, num_node, kappa, dropout,
                 model_mode="implicit", 
                 grad_mode="solver_grad",  
                 fw_solver=anderson, bw_solver=anderson, 
                 **kwargs):

        super(IGNN_solver, self).__init__()
        self.hook = None
        self.kwargs = kwargs
        self.dropout = dropout
        self.f = ImplicitGraph(nfeat, nhid, num_node, kappa) 
        self.Z_0 = Parameter(torch.zeros(num_node, nhid), requires_grad=False) 
        self.V = nn.Linear(nhid, nclass, bias=False)  
        self.model_mode = model_mode  
        if model_mode == "implicit":
            self.grad_mode = grad_mode 
            if grad_mode == "solver_grad":
                self.fw_solver = fw_solver
                self.bw_solver = bw_solver

    def forward(self, U, A, **kwargs):
        # the initial estimate of the fixed point.(0 or random sample from N(0,1))
        z0 = torch.zeros_like(self.Z_0)  
        z_star = z0
        threshold = kwargs.get('threshold', 20)

        if self.model_mode == "explicit":
            n_layer = kwargs.get('n_layer', 3)
            for i in range(n_layer):
                z_star = self.f(z_star, A, U)
            new_z_star = z_star

        elif self.model_mode == "implicit":  
            # Compute the fixed point(using e.g. Anderson Acceleration)
            with torch.no_grad(): 
                # x is the input injection;
                result_fw = self.fw_solver(lambda z: self.f(z, A, U), x0=z0, threshold=threshold)
                new_z_star = result_fw['result']
                # nstep_fw = result_fw['nstep']
                # print(nstep_fw)

            # While training:
            # (Prepare for) Backward pass
            if self.training: 
                new_z_star = self.f(z_star.requires_grad_(), A, U)

                if self.grad_mode == "explicit_grad": 
                    return

                elif self.grad_mode == "solver_grad": 
                    new_z_star = self.f(z_star.requires_grad_(), A, U)  # Re-engage the autodiff tape

                    def backward_hook(grad):
                        if self.hook is not None:
                            self.hook.remove()
                            torch.cuda.synchronize()  # To avoid infinite recursion 
                        f_g = lambda y: autograd.grad(new_z_star, z_star, y, retain_graph=True)[0] + grad
                        result_bw = self.bw_solver(f_g, x0=torch.zeros_like(grad), **self.kwargs)
                        new_grad = result_bw['result']
                        # nstep_bw = result_bw['nstep']
                        # print(nstep_bw)
                        return new_grad

                    self.hook = new_z_star.register_hook(backward_hook) 

        output = F.dropout(new_z_star, self.dropout, training=self.training) 
        output = self.V(output) 

        out = {"label_pred": output,
               "emb": new_z_star,
               }

        return out


class IGNN_solver_plus(nn.Module):
    def __init__(self, nfeat, nhid, nclass, num_node, kappa, dropout, **kwargs):
        """

        """
        super(IGNN_solver_plus, self).__init__()
        self.num_node = num_node
        self.nhid = nhid
        self.hook = None
        self.kwargs = kwargs
        self.dropout = dropout
        self.f = ImplicitGraph(nfeat, nhid, num_node, kappa)  
        if self.f is not None:
            print("Number of model.f parameters: ",
                  sum(p.nelement() for p in self.f.parameters() if p.requires_grad))
        self.V = nn.Linear(nhid, nclass, bias=False)  

        self.crit = nn.BCEWithLogitsLoss() 
        self.hypsolver = learnable_aa(learn_alpha=True, learn_beta=True, alpha_nhid=nhid)  #
        print("Number of hypsolver parameters: ",
              sum(p.nelement() for p in self.hypsolver.parameters() if p.requires_grad))

        self.hypsolver.initializer = Embedding4Initializer(nfeat, nhid)


        if self.hypsolver.initializer is not None:
            print("Number of initializer parameters: ",
                  sum(p.nelement() for p in self.hypsolver.initializer.parameters() if p.requires_grad))
        else:
            print("No initializer is used.")

        # self.solver = anderson
        self.solver = learnable_aa(learn_alpha=False, learn_beta=False) 

    def forward(self, U, A, **kwargs):

        simple = kwargs.get('simple', False)
        train_step = kwargs.get('ep', 1)
        threshold = kwargs.get('threshold', 30)
        A_s = kwargs.get('A_s', A)

        if self.hypsolver.initializer is not None:
            z0 = self.hypsolver.initializer(U)
        else:
            z0 = torch.zeros(self.num_node, self.nhid).cuda()

        if not simple:
            # freeze self.hypsolver
            for param in self.hypsolver.parameters():
                param.requires_grad = False

            # release self.f
            for param in self.f.parameters():
                param.requires_grad = True if self.training else False

            with torch.no_grad(): 
                result_fw = self.hypsolver(lambda z: self.f(z, A, U), x0=z0, graph_A=A, threshold=threshold,
                                           print_intermediate=False, print_galphas=False)
                new_z_star = result_fw['result']

            if self.training:  
                z_star = z0.clone()
                new_z_star = self.f(z_star.requires_grad_(), A, U)  # Re-engage the autodiff tape

                def backward_hook(grad): 
                    if self.hook is not None:
                        self.hook.remove()
                        torch.cuda.synchronize()  # To avoid infinite recursion
                    f_g = lambda y: autograd.grad(new_z_star, z_star, y, retain_graph=True)[0] + grad
                    result_bw = self.hypsolver(f_g, x0=torch.zeros_like(grad), **self.kwargs, graph_A=A)
                    new_grad = result_bw['result']
                    return new_grad

                self.hook = new_z_star.register_hook(backward_hook)

            output = F.dropout(new_z_star, self.dropout, training=self.training)  # Dropout
            output = self.V(output)  # 
            out = {"label_pred": output,
                   "emb": new_z_star,  #
                   }
            return out

        if simple:
            # release self.hypsolver
            for param in self.hypsolver.parameters():
                param.requires_grad = True if self.training else False

            # freeze self.f
            for param in self.f.parameters():
                param.requires_grad = False

            hyp_result = self.hypsolver(lambda z: self.f(z, A, U), x0=z0, threshold=threshold,
                                        print_intermediate=False, print_galphas=True, graph_A=A_s)

            with torch.no_grad():  #
                anderson_result = self.solver(lambda z: self.f(z, A, U), x0=torch.zeros_like(z0), threshold=threshold)
                z_targ = anderson_result['result']
                z_ref = z_targ.clone()
                torch.cuda.empty_cache()

                ce_loss = 0
                hyp_ce_loss = 0  
                ref_ce_loss = 0  
                ini_ce_loss = 0  


            def deq_hypersolver_loss(model, z_targ, z_ref, rel_trace, hyp_X, Galphas, z_init, train_step):
                """
                all_losses = deq_hypersolver_loss(self,
                                                  z_targ,          
                                                  z1s_ref,                 
                                                  hyp_result['rel_trace'], 
                                                  hyp_result['X'],   
                                                  hyp_result['Galphas'],    
                                                  z_init,                
                                                  train_step=train_step   
                                                  )
                """

                lam1 = 0.1  
                lam2 = 5  
                lam3 = 1e-4  
                loss_len = rel_trace.shape[1]  
                ratio = max(1 - train_step / 1000, 0.005) 


                diff = hyp_X[:, -loss_len:] - z_targ[:, None]
                reco_losses = diff.norm(dim=2).mean(0) 

                if model.training:
                    loss_weights = torch.arange(0, 1, 0.02).cuda() ** 1 
                    loss_weights = loss_weights[:loss_len] / loss_weights[:loss_len].sum() 
                    reco_losses = reco_losses * loss_weights * lam1
                    loss = reco_losses.sum()  
                else:
                    loss = torch.tensor(0.0).to(z_targ)

                #
                if not (z_init == 0).all():
                    init_est_loss = F.mse_loss(z_init, z_targ) * lam2
                else:
                    init_est_loss = torch.tensor(0.0).cuda()
                loss = loss + init_est_loss

                #
                # lam3 alpha loss（Galphas）
                if model.hypsolver.learn_alpha:
                    alpha_aux_loss = Galphas.mean() * lam3  
                    loss = loss + alpha_aux_loss * ratio / 20000

                return (loss.view(-1, 1), 
                        )

            all_losses = deq_hypersolver_loss(self, z_targ, z_ref, hyp_result['rel_trace'],
                                              hyp_result['X'], hyp_result['Galphas'], z0,
                                              train_step=train_step)

            z_star_hyp = hyp_result['result']
            output = F.dropout(z_star_hyp, self.dropout, training=self.training) 
            output = self.V(output)  
            loss = all_losses[0]
            out = {
                "loss": loss,
                "emb": output,
                "hyp_result": hyp_result,
                "anderson_result": anderson_result,
            }
            return out
