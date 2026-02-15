import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import pyro
import pyro.distributions as dist
import pyro.distributions.transforms as T

from pyro.nn import DenseNN
from pyro.infer.reparam.transform import TransformReparam
from pyro.distributions.conditional import ConditionalTransformedDistribution

from .pgm_layer import (
    MLP, CNN,  # fmt: skip
    ConditionalGumbelMax,
    ConditionalAffineTransform,
    ConditionalTransformedDistributionGumbelMax,
)


class BasePGM(nn.Module):
    def __init__(self):
        super().__init__()

    def scm(self, *args, **kwargs):
        def config(msg):
            if isinstance(msg["fn"], dist.TransformedDistribution):
                return TransformReparam()
            else:
                return None

        return pyro.poutine.reparam(self.model, config=config)(*args, **kwargs)

    def sample_scm(self, n_samples=1, t=None):
        with pyro.plate("obs", n_samples):
            samples = self.scm(t)
        return samples

    def sample(self, n_samples=1, t=None):
        with pyro.plate("obs", n_samples):
            samples = self.model(t)  # model defined in parent class
        return samples

    def infer_exogeneous(self, obs):
        batch_size = list(obs.values())[0].shape[0]
        # assuming that we use transformed distributions for everything:
        cond_model = pyro.condition(self.sample, data=obs)
        cond_trace = pyro.poutine.trace(cond_model).get_trace(batch_size)

        output = {}
        for name, node in cond_trace.nodes.items():
            if "z" in name or "fn" not in node.keys():
                continue
            fn = node["fn"]
            if isinstance(fn, dist.Independent):
                fn = fn.base_dist
            if isinstance(fn, dist.TransformedDistribution):
                # compute exogenous base dist (created with TransformReparam) at all sites
                output[name + "_base"] = T.ComposeTransform(fn.transforms).inv(
                    node["value"]
                )
        return output

    def counterfactual(self, obs, intervention, num_particles=1, detach=True, t=None):
        dag_variables = self.variables.keys()
        # assert set(obs.keys()) == set(dag_variables)
        avg_cfs = {k: torch.zeros_like(obs[k]) for k in obs.keys()}
        batch_size = list(obs.values())[0].shape[0]

        for _ in range(num_particles):
            # Abduction
            exo_noise = self.infer_exogeneous(obs)
            exo_noise = {k: v.detach() if detach else v for k, v in exo_noise.items()}
            # condition on root node variables (no exogeneous noise available)
            for k in dag_variables:
                if k not in intervention.keys():
                    if k not in [i.split("_base")[0] for i in exo_noise.keys()]:
                        exo_noise[k] = obs[k]
            # Abducted SCM
            abducted_scm = pyro.poutine.condition(self.sample_scm, data=exo_noise)
            # Action
            counterfactual_scm = pyro.poutine.do(abducted_scm, data=intervention)
            # Prediction
            counterfactuals = counterfactual_scm(batch_size, t)

            if hasattr(self, "discrete_variables"):  # hack for MIMIC
                # Check if we should change "finding", i.e. if its parents and/or
                # itself are not intervened, then we use its observed value.
                # This is needed due to stochastic abduction of discrete variables.
                if (
                    "age" not in intervention.keys()
                    and "finding" not in intervention.keys()
                ):
                    counterfactuals["finding"] = obs["finding"]

            for k, v in counterfactuals.items():
                avg_cfs[k] += v / num_particles
        return avg_cfs

class FlowPGM(BasePGM):
    def __init__(self, args):
        super().__init__()
        self.variables = {
            "age": "countinuous",
            # "svol": "countinuous",
            "diagnosis": "binary",
            "parietal":"countinuous",
            "frontal": "countinuous",
            "insula": "countinuous",
            "cingulate":"countinuous",
            "occipital": "countinuous",
            "temporal": "countinuous",
            "whitem": "countinuous",

        }
        # define base distributions
        # for k in ["a", "s", "p", "f", "i"]:
        for k in ["a", "p", "f", "i", "c", "o", "w","t"]:
            self.register_buffer(f"{k}_base_loc", torch.zeros(1))
            self.register_buffer(f"{k}_base_scale", torch.ones(1))

        # age spline flow
        self.age_flow_components = T.ComposeTransformModule([T.Spline(1)])
        self.age_flow = T.ComposeTransform(
            [
                self.age_flow_components,
                # self.age_constraints,
            ]
        )

        # # svol spline flow 
        # self.svol_flow_components = T.ComposeTransformModule([T.Spline(1)])
        # self.svol_flow = T.ComposeTransform(
        #     [
        #         self.svol_flow_components,
        #         # self.age_constraints,
        #     ]
        # )

        # parietal (conditional) flow, (age, svol, diagnosis) -> (parietal)
        par_net = DenseNN(2, [8,16], param_dims=[1,1], nonlinearity=nn.LeakyReLU(0.1))
        self.par_flow = ConditionalAffineTransform(context_nn=par_net, event_dim=0)

        # frontal (conditiona  ) flow, (age, svol, diagnosis) -> (frontal)
        front_net = DenseNN(2, [8,16], param_dims=[1,1], nonlinearity=nn.LeakyReLU(0.1))
        self.front_flow = ConditionalAffineTransform(context_nn=front_net, event_dim=0)

        # insula (conditional) flow, (age, svol, diagnosis) -> (insula)
        insu_net = DenseNN(2, [8,16], param_dims=[1,1], nonlinearity=nn.LeakyReLU(0.1))
        self.insu_flow = ConditionalAffineTransform(context_nn=insu_net, event_dim=0)

        # cin (conditional) flow, (age, svol, diagnosis) -> (parietal)
        cin_net = DenseNN(2, [8,16], param_dims=[1,1], nonlinearity=nn.LeakyReLU(0.1))
        self.cin_flow = ConditionalAffineTransform(context_nn=cin_net, event_dim=0)

        # occ (conditiona  ) flow, (age, svol, diagnosis) -> (frontal)
        occ_net = DenseNN(2, [8,16], param_dims=[1,1], nonlinearity=nn.LeakyReLU(0.1))
        self.occ_flow = ConditionalAffineTransform(context_nn=occ_net, event_dim=0)

        # tmporal (conditional) flow, (age, svol, diagnosis) -> (insula)
        tep_net = DenseNN(2, [8,16], param_dims=[1,1], nonlinearity=nn.LeakyReLU(0.1))
        self.tep_flow = ConditionalAffineTransform(context_nn=tep_net, event_dim=0)
       
       # white (conditional) flow, (age, svol, diagnosis) -> (insula)
        wht_net = DenseNN(2, [8,16], param_dims=[1,1], nonlinearity=nn.LeakyReLU(0.1))
        self.wht_flow = ConditionalAffineTransform(context_nn=wht_net, event_dim=0)

        # log space for diagnosis
        self.pd_logit = nn.Parameter(np.log(1 / 2) * torch.ones(1))

        


    def model(self, t=None):
        pyro.module("BrainPGM", self)
        # # p(s), svol flow
        # ps_base = dist.Normal(self.s_base_loc, self.s_base_scale).to_event(1)
        # ps = dist.TransformedDistribution(ps_base, self.svol_flow)
        # svol = pyro.sample("svol", ps)
        # _ = self.svol_flow_components  # register with pyro


        # p(a), age flow
        pa_base = dist.Normal(self.a_base_loc, self.a_base_scale).to_event(1)
        pa = dist.TransformedDistribution(pa_base, self.age_flow)
        age = pyro.sample("age", pa)
        _ = self.age_flow_components  # register with pyro

        # p(d), diagnosis dist
        pd= dist.Bernoulli(logits=self.pd_logit).to_event(1)
        diagnosis = pyro.sample("diagnosis", pd)

        # p(p), parietal flow conditioned on svol, age, diagnosis
        # print(f'{svol.shape}, {age.shape}, {diagnosis.shape}')
        pp_sad_base = dist.Normal(self.p_base_loc, self.p_base_scale).to_event(1)
        pp_sad = ConditionalTransformedDistribution(
            pp_sad_base, [self.par_flow] ## par_net
        ).condition(torch.cat([age, diagnosis], dim=1))
        # ).condition(torch.cat([svol, age, diagnosis], dim=1))
        parietal = pyro.sample("parietal", pp_sad)        
    
        # p(f), frontal flow conditioned on svol, age, diagnosis
        pf_sad_base = dist.Normal(self.f_base_loc, self.f_base_scale).to_event(1)
        pf_sad = ConditionalTransformedDistribution(
            pf_sad_base, [self.front_flow]
        ).condition(torch.cat([age, diagnosis], dim=1))
        frontal = pyro.sample("frontal", pf_sad)

        # p(i), insula flow conditioned on svol, age, diagnosis
        pi_sad_base = dist.Normal(self.i_base_loc, self.i_base_scale).to_event(1)
        pi_sad = ConditionalTransformedDistribution(
            pi_sad_base, [self.insu_flow]
        ).condition(torch.cat([age, diagnosis], dim=1))
        insula = pyro.sample("insula", pi_sad)



        # p(p), temporal flow conditioned on svol, age, diagnosis
        pt_sad_base = dist.Normal(self.t_base_loc, self.t_base_scale).to_event(1)
        pt_sad = ConditionalTransformedDistribution(
            pt_sad_base, [self.tep_flow] ## par_net
        ).condition(torch.cat([age, diagnosis], dim=1))
        # ).condition(torch.cat([svol, age, diagnosis], dim=1))
        temporal = pyro.sample("temporal", pt_sad)        
    
        # p(f), cingulate flow conditioned on svol, age, diagnosis
        pc_sad_base = dist.Normal(self.c_base_loc, self.c_base_scale).to_event(1)
        pc_sad = ConditionalTransformedDistribution(
            pc_sad_base, [self.cin_flow]
        ).condition(torch.cat([age, diagnosis], dim=1))
        cingulate = pyro.sample("cingulate", pc_sad)

        # p(i), insula flow conditioned on svol, age, diagnosis
        po_sad_base = dist.Normal(self.o_base_loc, self.o_base_scale).to_event(1)
        po_sad = ConditionalTransformedDistribution(
            po_sad_base, [self.occ_flow]
        ).condition(torch.cat([age, diagnosis], dim=1))
        occipital = pyro.sample("occipital", po_sad)
        # p(i), insula flow conditioned on svol, age, diagnosis
        pw_sad_base = dist.Normal(self.w_base_loc, self.w_base_scale).to_event(1)
        pw_sad = ConditionalTransformedDistribution(
            pw_sad_base, [self.wht_flow]
        ).condition(torch.cat([age, diagnosis], dim=1))
        whitem = pyro.sample("whitem", pw_sad)

        return {
            # "svol":svol,
            "age": age,
            "diagnosis": diagnosis,
            "parietal":parietal,
            "frontal":frontal,
            "insula":insula,
            "cingulate":cingulate,
            "occipital": occipital,
            "temporal": temporal,
            "whitem": whitem,
        }
        
    def svi_model(self, **obs):
        with pyro.plate("observations", obs["age"].shape[0]):
            pyro.condition(self.model, data=obs)()

    def guide_pass(self, **obs):
        pass


