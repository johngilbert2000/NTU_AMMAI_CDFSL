import torch
from collections import defaultdict

# Lookahead paper by Zhang et al.:   https://arxiv.org/abs/1907.08610

# Inspired by Lookahead implementations by:
#   Zhang et al.:   https://github.com/michaelrzhang/lookahead
#   Liam (alphadl): https://github.com/alphadl/lookahead.pytorch


class NestedLookahead(torch.optim.Optimizer):
    """
    A Nested Lookahead Optimizer, in which the Lookahead loop uses Lookahead
    
    opt: inner optimizer for fast weights
    lr: learning rate of inner optimizer
    k: number of fast steps before updating inner slow step
    a (α): inner slow weight step size, a ∈ [0,1]
    s: number of inner (nested) slow steps before updating outer slow step
    h: outer slow weight step size, h ∈ [0,1]
    pullback: pullback momentum; can apply to only one scope (i.e., inner slow weights or outer slow weights) if specified;
        (pullback options: None, "reset", "pullback", "reset_inner", "reset_outer", "pullback_inner", "pullback_outer")
    weight_decay: weight decay of inner optimizer
    
    Note: Becomes original lookahead optimizer if s == 0
    """
    def __init__(self, opt, lr=0.001, k=5, a=0.5, s=0, h=0.5, pullback=None, weight_decay=0):
        if isinstance(pullback, str):
            pullback = pullback.lower()
        
        if (k < 1):
            raise ValueError("Fast k steps must be a natural number")
        elif (s < 0):
            raise ValueError("Inner slow s steps must be a natural number")
        elif (pullback not in [None, "reset", "pullback", "reset_inner", "reset_outer", "pullback_inner", "pullback_outer"]):
            raise ValueError("Pullback momentum options: None, 'reset', 'pullback', \
            'reset_inner', 'reset_outer', 'pullback_inner', 'pullback_outer'")
        elif (a < 0) or (a > 1):
            raise ValueError("Requirement: a ∈ [0,1]; (recommended: 0.5)")
        elif (h < 0) or (h > 1):
            raise ValueError("Requirement: h ∈ [0,1]; (recommended: 0.5)")
        elif (weight_decay < 0) or (weight_decay > 1):
            raise ValueError("Requirement: weight_decay ∈ [0,1]")
        elif (lr < 0) or (lr > 1):
            raise ValueError("Requirement: lr ∈ [0,1]")
        
        opt.param_groups[0]["lr"] = lr
        opt.param_groups[0]["weight_decay"] = weight_decay
        self.opt = opt
        self.k = int(k)
        self.a = a
        self.s = int(s)
        self.h = h
        self.pullback = pullback
        self._k_ct = 0 # counter for k
        self._s_ct = 0 # counter for s
        
        self.param_groups = self.opt.param_groups
        self.state = defaultdict(dict)
        self.defaults = {"lr":0.001, "k":5, "a":0.5, "s":0, "h":0.5, "pullback":None}

    def decay_lr_alpha(self, decay_rate):
        """Multiplies lr and α by (1-decay_rate); decay_rate ∈ [0,1]"""
        assert (decay_rate >= 0) and (decay_rate <= 1)
        self.opt.param_groups[0]["lr"] *= (1-decay_rate)
        self.a *= (1-decay_rate)
    
    def __getstate__(self):
        return {
            'state': self.state,
            'opt': self.opt,
            'a': self.a,
            'b': self.b,
            'pullback': self.pullback,
            'k': self.k,
            's': self.s,
            '_k_ct': self._k_ct,
            '_s_ct': self._s_ct,
        }

    def state_dict(self):
        return self.opt.state_dict()

    def load_state_dict(self, state_dict):
        self.opt.load_state_dict(state_dict)
        
    def add_param_group(self, param_group):
        self.opt.add_param_group(param_group)
        
    def zero_grad(self):
        self.opt.zero_grad()
        
    def update_momentum(self, p, scope="inner"):
        """Update momentum buffer
        
        p: parameter in group['params']
        scope: 'inner' or 'outer'
        """
        assert (scope == "inner") or (scope == "outer")
        if "momentum_buffer" in self.opt.state[p].keys():            
            if (self.pullback == "pullback") or (self.pullback == f"pullback_{scope}"):
                param_state = self.state[p]
                buffer = self.opt.state[p]["momentum_buffer"]
                if "cached_momentum" in param_state.keys():
                    cached = param_state["cached_momentum"]
                    buffer = self.a*buffer + (1-self.a)*cached
                    self.opt.state[p]["momentum_buffer"] = buffer
                param_state["cached_momentum"] = buffer
                
            elif (self.pullback == "reset") or (self.pullback == f"reset_{scope}"):
                self.opt.state[p]["momentum_buffer"] = torch.zeros_like(p.data)

    def update_inner(self, group):
        "Update inner slow weights with fast weights using lookahead"
        for fast in group["params"]:
            param_state = self.state[fast]
            if "slow_param" not in param_state.keys():
                param_state["slow_param"] = torch.zeros_like(fast.data)
                param_state["slow_param"].copy_(fast.data)
            slow = param_state["slow_param"]
            slow += self.a * (fast.data - slow)
            fast.data.copy_(slow)
            self.update_momentum(fast, scope="inner")

    def update_outer(self, group):
        "Update outer slow weights with inner slow weights using lookahead"
        for fast in group["params"]:
            param_state = self.state[fast]
            if "slow_param" not in param_state.keys():
                param_state["slow_param"] = torch.zeros_like(fast.data)
                param_state["slow_param"].copy_(fast.data)
            
            slow = param_state["slow_param"]
            if "outer_param" not in param_state.keys():
                param_state["outer_param"] = torch.zeros_like(slow)
                param_state["outer_param"].copy_(slow)
            outer = param_state["outer_param"]
            outer += self.h * (slow - outer)
            fast.data.copy_(outer)
            self.update_momentum(fast, scope="outer")
    
    def update(self):
        "Update inner slow weights and outer slow weights"
        # check counters
        update_inner_check = self._k_ct == 0
        if update_inner_check: 
            self._s_ct += 1

        update_outer_check = (self._s_ct == 0) and (self.s > 0)            
        
        # update loop
        for group in self.param_groups:
            if update_inner_check:
                self.update_inner(group)
            if update_outer_check:
                self.update_outer(group)
        self._k_ct += 1

        # reset counter at end of loop
        if self._s_ct >= self.s:
            self._s_ct = 0
        if self._k_ct >= self.k:
            self._k_ct = 0

    def step(self, closure=None):
        loss = self.opt.step(closure)
        self.update()
        return loss