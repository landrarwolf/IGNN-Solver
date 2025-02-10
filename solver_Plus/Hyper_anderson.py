import torch
import torch.nn as nn


def anderson(f, x0, m=6, lam=1e-4, threshold=20, eps=1e-4, stop_mode='rel', beta=1.0):
    """
        Anderson acceleration for fixed point iteration.
    """
    
    bsz, d = x0.shape  
    alternative_mode = 'rel' if stop_mode == 'abs' else 'abs'
    X = torch.zeros(bsz, m, d, dtype=x0.dtype, device=x0.device) 
    F = torch.zeros(bsz, m, d, dtype=x0.dtype, device=x0.device) 
    X[:, 0], F[:, 0] = x0.reshape(bsz, -1), f(x0).reshape(bsz, -1)
    X[:, 1], F[:, 1] = F[:, 0], f(F[:, 0].reshape_as(x0)).reshape(bsz, -1)
    
    H = torch.zeros(bsz, m + 1, m + 1, dtype=x0.dtype, device=x0.device)
    H[:, 0, 1:] = H[:, 1:, 0] = 1
    y = torch.zeros(bsz, m + 1, 1, dtype=x0.dtype, device=x0.device)
    y[:, 0] = 1
    
    trace_dict = {'abs': [],
                  'rel': []}
    lowest_dict = {'abs': 1e8,
                   'rel': 1e8}
    lowest_step_dict = {'abs': 0,
                        'rel': 0}
    
    
    for k in range(2, threshold):
        n = min(k, m)
        G = F[:, :n] - X[:, :n]
        H[:, 1:n + 1, 1:n + 1] = torch.bmm(G, G.transpose(1, 2)) + lam * torch.eye(n, dtype=x0.dtype, device=x0.device)[
            None]
        
        alpha = torch.linalg.solve(H[:, :n + 1, :n + 1], y[:, :n + 1])[:, 1:n + 1, 0] 
        
        X[:, k % m] = beta * (alpha[:, None] @ F[:, :n])[:, 0] + (1 - beta) * (alpha[:, None] @ X[:, :n])[:, 0]
        

        F[:, k % m] = f(X[:, k % m].reshape_as(x0)).reshape(bsz, -1)
        gx = (F[:, k % m] - X[:, k % m]).view_as(x0)
        abs_diff = gx.norm().item()  
        rel_diff = abs_diff / (1e-5 + F[:, k % m].norm().item()) 
        diff_dict = {'abs': abs_diff,
                     'rel': rel_diff}
        trace_dict['abs'].append(abs_diff)
        trace_dict['rel'].append(rel_diff)
        
        for mode in ['rel', 'abs']:
            if diff_dict[mode] < lowest_dict[mode]:
                if mode == stop_mode:
                    lowest_xest, lowest_gx = X[:, k % m].view_as(x0).clone().detach(), gx.clone().detach()
                lowest_dict[mode] = diff_dict[mode]
                lowest_step_dict[mode] = k
        
        # if trace_dict[stop_mode][-1] < eps:
        #     for _ in range(threshold - 1 - k):
        #         trace_dict[stop_mode].append(lowest_dict[stop_mode])
        #         trace_dict[alternative_mode].append(lowest_dict[alternative_mode])
        #     break
    
    out = {"result": lowest_xest,
           "lowest": lowest_dict[stop_mode],
           "nstep": lowest_step_dict[stop_mode],
           "prot_break": False,
           "abs_trace": trace_dict['abs'],
           "rel_trace": trace_dict['rel'],
           "eps": eps,
           "threshold": threshold}
    X = F = None
    return out

########################################################################################################################
# Path: solver_Plus/Hyper_anderson.py


class learnable_aa(nn.Module):
    def __init__(self, alpha_net_dict=None, m=6, stop_mode='rel', learn_alpha=False, alpha_nhid=200, learn_beta=False,
                 **kwargs):
        """
        Args:
            alpha_net_dict (dict): The type of alpha prediction network to use. This will differ based on data type 
                                   (e.g., sequence, image, feature tensor, etc.), and has the following format:
                                   {
                                       'name': [CLS_NAME],
                                       'kwargs': {...}
                                   }

                                   This can be None if learn_alpha=False.
            m (int, optional): [Number of Anderson's past-step slots]. Defaults to 6.
            stop_mode (str, optional): [Whether residuals measured in absolute ("abs") or relative ("rel") mode]. 
                                       Defaults to 'rel'.
            learn_alpha (bool, optional): [If True, alpha of Anderson will be learned]. Defaults to True.
            alpha_nhid (int, optional): [Input dimension used to predict alpha]. Defaults to 200.
            learn_beta (bool, optional): [If True, beta of Anderson will be learned]. Defaults to False.
            hyperload (str, optional): [Path to load a pretrained hypersolver state dict]. Defaults to "".
        """
        super().__init__()
        self.stop_mode = 'rel'
        self.m = m  # 5
        assert m > 2, "You should have m > 2 to satisfy AA prototype"
        self.alternative_mode = 'rel' if stop_mode == 'abs' else 'abs'
        self.learn_alpha = learn_alpha
        self.learn_beta = learn_beta
        if learn_alpha:  
            alpha_net_dict = {'name': 'SequenceAlphaNet',
                              'kwargs': {'ninner': 20, 'alpha_model': "alpha_tcn"
                                         }
                              }  # 'ninner': 100
            
            assert alpha_net_dict is not None and 'name' in alpha_net_dict, "alpha_net_dict CANNOT be None if learn_alpha"
            self.alpha_nhid = alpha_nhid 
            self.alpha_net = eval(alpha_net_dict['name'])(alpha_nhid, learn_beta,
                                                          **alpha_net_dict['kwargs'])  
            if not learn_beta:  
                # print("Not learning beta!")
                self.beta = [1.0] * 100
        
        if not learn_alpha:
            if learn_beta: 
                self.beta = nn.Parameter(torch.zeros(100, ) + 1.0) 
            else: 
                # print("Not learning beta!")
                self.beta = [1.0] * 100 
    
    def forward(self, f, x0, lam=1e-4, tol=1e-3, threshold=30, print_intermediate=False, print_galphas=False, **kwargs):
        """
        Args:
            f (function): [layer's function form]
            x0 (torch.Tensor): [Initial estimate of the fixed point]
            lam (float, optional): [Anderson's lambda]. Defaults to 1e-4.
            tol (float, optional): [Anderson's tolerance level; works with the stop_mode]. Defaults to 1e-3.
            threshold (int, optional): [Max number of forward iterations]. Defaults to 30.
            print_intermediate (bool, optional): [If True]. Defaults to False.
            print_galphas (bool, optional): [If True, returns the ||G*alpha|| values]. Defaults to True.
            kwargs: [Cutoffs, etc. extra information that is to be passed into `forward_RR`]

        Returns:
            [dict]: [The result of fixed point solving by (learnable) Anderson]
        """

        bsz, d = x0.shape
        m = self.m  #
        X = torch.zeros(bsz, threshold if self.training else m, d, dtype=x0.dtype,
                        device=x0.device)  
        F = torch.zeros(bsz, threshold if self.training else m, d, dtype=x0.dtype,
                        device=x0.device)  
        X[:, 0], F[:, 0] = x0, f(x0) 
        if (x0 == 0).all():
            # Started with all zeros (i.e., no initializer)
            X[:, 1], F[:, 1] = F[:, 0], f(F[:, 0])
            k_start = 2
        else:
            k_start = 1
        
        if self.learn_alpha:
            dim = self.alpha_net.ninner 
            R = torch.zeros(bsz, threshold if self.training else m, dim, dtype=x0.dtype,
                            device=x0.device) 
            for i in range(k_start):
                R[:, i] = self.alpha_net.forward_RR((F[:, i] - X[:, i]).view_as(x0), **kwargs).reshape(bsz, -1)

        else:
            H = torch.zeros(bsz, m + 1, m + 1, dtype=x0.dtype, device=x0.device)
            H[:, 0, 1:] = H[:, 1:, 0] = 1
            y = torch.zeros(bsz, m + 1, 1, dtype=x0.dtype, device=x0.device)
            y[:, 0] = 1

        trace_dict = {'abs': [], 'rel': []}
        lowest_dict = {'abs': 1e8, 'rel': 1e8}
        lowest_step_dict = {'abs': 0, 'rel': 0}

        F = F.detach() 
        X = X.detach()
        Inits = Galphas = None
        Alphas = []
        if print_intermediate:
            Inits = [X[:, 1].clone().detach()[None]]
        if print_galphas:
            Galphas = []
        
        for k in range(k_start, threshold):
            n = min(k, m)
            if self.training:
                FF, XX, up = F[:, max(k - m, 0):k], X[:, max(k - m, 0):k], k
                if self.learn_alpha:
                    RR = R[:, max(k - m, 0):k]
                else:
                    FF, XX = torch.cat([FF[:, n - (k % m):], FF[:, :n - (k % m)]], dim=1), torch.cat(
                        [XX[:, n - (k % m):], XX[:, :n - (k % m)]], dim=1)
            else:
                FF, XX, up = F[:, :n], X[:, :n], k % m
                if self.learn_alpha:
                    RR = R[:, :n]        
            G = FF - XX  # chronological order if learn_alpha; wrap-around if not.
            
            if self.learn_alpha:
                A = kwargs.get('graph_A', None)
                alpha, beta = self.alpha_net(RR, n, k, m, up, A)

                if not self.learn_beta:
                    beta = self.beta[k]
                if print_galphas:
                    Galphas.append(torch.einsum('bni,bn->bi', G, alpha).norm(1).mean(0))
            else:
                H[:, 1:n + 1, 1:n + 1] = torch.bmm(G, G.transpose(1, 2)) + lam * \
                                         torch.eye(n, dtype=x0.dtype, device=x0.device)[None]
                alpha = torch.linalg.solve(H[:, :n + 1, :n + 1].clone(), y[:, :n + 1].clone())[:, 1:n + 1, 0]
                beta = self.beta[k]

            
            alpha_clone = alpha[:, None].clone()
            FF_clone = FF.clone()
            XX_clone = XX.clone()
            X[:, up] = beta * (alpha_clone @ FF_clone)[:, 0] + (1 - beta) * (alpha_clone @ XX_clone)[:, 0]
            F[:, up] = f(X[:, up])
            temp = F[:, up].clone() 
            gx = F[:, up] - X[:, up]
            if self.learn_alpha:
                R[:, up] = self.alpha_net.forward_RR(gx, **kwargs).reshape(bsz, -1) 

            if print_intermediate:
                Inits.append(X[:, up].clone().detach()[None])
                Alphas.append(alpha.clone().detach()[None])

            abs_diff = gx.norm()
            rel_diff = abs_diff / (1e-5 + temp.norm())
            diff_dict = {'abs': abs_diff, 'rel': rel_diff}
            trace_dict['abs'].append(abs_diff)
            trace_dict['rel'].append(rel_diff)
        
        return {'result': X[:, up].view_as(x0),
                # 'X': X.view(bsz, -1, *x0.shape[1:]),
                'X': X,
                'Inits': torch.cat(Inits, dim=0) if Inits else None,
                'rel_trace': torch.stack(trace_dict['rel']).view(1, -1),
                'abs_trace': torch.stack(trace_dict['abs']).view(1, -1), 
                'Galphas': torch.stack(Galphas).view(1, -1) if Galphas else None,
                'beta': beta,
                # 'time': time.time() - t,
                }
