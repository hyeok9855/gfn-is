import torch
import numpy as np
from collections import defaultdict
from itertools import chain
from line_profiler import profile

from . import utils


"""
Helper functions
"""


def collate_states_scores(states, probs):
    """Collates states, scores (summing) to be unique via state hash.
    Retains input order of states.
    Adds predicted probs for duplicate states.
    Removes invalid states (None).
    Differentiable wrt probs.

    Assumes that states and probs are aligned.
    Importantly, states must always be in the same order.
    States are ordered by the actions used to generate them,
    which is expected to be in a consistent,
    deterministic order as a function of state (in get_fwd/back_actions).

    Input
    -----
    states: List of [State], length n
    scores: Torch tensor, shape (n)

    Returns
    -------
    states: List of [State], length m < n (unique), in same order.
    scores: Torch tensor, shape (m).
    """
    if len(states) == 0 or len(probs) == 0:
        # raise Exception(f'Problematic collate input. {states=} {probs=}')
        raise Exception(f"Problematic collate input. states={states} probs={probs}")
    if len(states) != len(probs):
        # msg = f'Problematic collate input; lengths differ. {states=} {probs=}'
        msg = f"Problematic collate input; lengths differ. states={states} probs={probs}"
        raise Exception(msg)

    d = defaultdict(lambda: 0)
    for state, prob in zip(states, probs):
        if state is not None:
            d[state] += prob

    collated_states = [state for state in d]
    collated_probs = [d[state] for state in d]
    collated_probs = torch.stack(collated_probs)
    collated_probs = collated_probs / torch.sum(collated_probs)
    return collated_states, collated_probs


def logp_to_p(logps):
    """Convert logps to ps. Batched."""
    if type(logps) == torch.Tensor:
        scores = torch.exp(logps - torch.logsumexp(logps, -1, keepdim=True))
        return scores / torch.sum(scores)
    elif type(logps) == list:
        # list of tensors of variable length
        result = []
        for lp in logps:
            lp_norm = torch.exp(lp - torch.logsumexp(lp, -1, keepdim=True))
            result.append(lp_norm / torch.sum(lp_norm))
        return result
    # raise Exception(f'{type(logps)=}')
    raise Exception(f"type(logps)={type(logps)}")


def unique_keep_order_remove_nones(items):
    """Remove duplicates, keeping order. Uses hashing."""
    return [x for x in dict.fromkeys(items) if x is not None]


"""
BasePolicies
"""


class BasePolicySA:
    """Base policy class - inherited and specified in MDPs.

    A policy is a deep neural net that samples actions from states.
    The network architecture depends heavily on the specific MDP.

    Policy outputs scores for possible actions given an input state.
    MDP logic translates actions into states, using transition & get_action
    functions. Importantly, the order of actions must always be the same
    for the same input State object.

    *** self.net and self.state_map_f outputs must be aligned:
    the i-th self.net(state) output must be predicted score for
    the i-th state in state_map_f(state).

      (This is slightly trickier for graph neural nets: self.net must
      flatten graph output into a vector first.)

    BaseTBGFlowNet objects contain two Policy objects - forward and
    backward - and own the optimizers and training logic for the
    Policies.
    """

    def __init__(self, args, mdp, actor, net, state_map_f):
        """Initialize policy, SA

        Inputs
        ------
        args:         AttrDict; user arguments
        mdp:          MDP object
        actor:        Actor object
        net:          torch.nn.module, mapping List of States -> torch.tensor
        state_map_f:  Function mapping State -> List of [State].
                      e.g., get_children or get_parents.
        """
        self.args = args
        self.mdp = mdp
        self.actor = actor
        self.net = net
        self.state_map_f = state_map_f

    def parameters(self):
        """Retrieve trainable parameters, send to optimizer."""
        return self.net.parameters()

    def eval(self):
        self.net.eval()

    def train(self):
        self.net.train()

    def to(self, device):
        self.net.to(device)

    def state_dict(self):
        return self.net.state_dict()

    def load_state_dict(self, state_dict):
        self.net.load_state_dict(state_dict)

    """
    Logps unique
    """

    def logps_unique(self, batch):
        """Differentiable; output logps of unique children/parents.

        Typical logic flow (example for getting children)
        1. Run network on state - returns high-dim actions
        2. Translate actions into list of states - not unique
        3. Filter invalid child states, including stop action/terminal state
            if the state does not have stop.
        4. Reduce states to unique, using hash property of states.
           Need to add predicted probabilities.
        5. Normalize probs to sum to 1

        Input
        -----
        batch: List of [State], n items, or State
        f:     function, get_children or get_parents

        Returns
        -------
        state_to_logp: List of dicts mapping state to torch.tensor
        """
        batched = bool(type(batch) is list)
        if not batched:
            batch = [batch]

        # Run network on states -> (b, o)
        logps_bo = self.net(batch)
        ps_bo = logp_to_p(logps_bo)

        # Iterate over batch ...; Reduce size (o) to (# unique)
        batch_dicts = []
        for state, ps_o in zip(batch, ps_bo):
            children = self.state_map_f(state)
            childs_uniq, ps_uniq = collate_states_scores(children, ps_o)

            state_to_logp = {child: torch.log(p) for child, p in zip(childs_uniq, ps_uniq)}
            batch_dicts.append(state_to_logp)

        return batch_dicts if batched else batch_dicts[0]

    """
    Sample
    """

    def sample(self, batch, epsilon):
        """Non-differentiable; sample a child or parent.

        Epsilon chance of sampling a unique child
        uniformly.

        Input: batch: List of [State], or State
               f:     function, get_children or get_parents
        Output: List of [State], or State
        """
        batched = bool(type(batch) is list)
        if not batched:
            batch = [batch]

        # Run network on state. -> (b, actions)
        logps_bo = self.net(batch)
        ps_bo = logp_to_p(logps_bo)

        batch_samples = []
        for state, ps_o in zip(batch, ps_bo):
            children = self.state_map_f(state)
            childs_uniq, ps_uniq = collate_states_scores(children, ps_o)

            if np.random.random() < epsilon:
                sample = np.random.choice(childs_uniq)
            else:
                ps = utils.tensor_to_np(ps_uniq, reduce_singleton=False)
                sample = np.random.choice(childs_uniq, p=ps)
            batch_samples.append(sample)

        return batch_samples if batched else batch_samples[0]


class BasePolicySSR:
    """Base policy class - inherited and specified in MDPs.

    SSR: State x state -> R (log energy)

    A policy is a deep neural net that samples actions from states.
    The network architecture depends on the specific MDP.

    BaseTBGFlowNet objects contain two Policy objects - forward and
    backward - and own the optimizers and training logic for the
    Policies.
    """

    def __init__(self, args, mdp, actor, encoder, scorer, state_map_f):
        """Initialize policy, SSR

        Inputs
        ------
        args:         AttrDict; user arguments
        mdp:          MDP object
        actor:        Actor object
        encoder:      torch.nn.module, mapping List of States -> torch.tensor
        scorer:       torch.nn.module, mapping [z1, z2] tensor -> scalar.
        state_map_f:  Function mapping State -> List of [State].
                      e.g., get_children or get_parents.
        """
        self.args = args
        self.mdp = mdp
        self.actor = actor
        self.encoder = encoder
        self.scorer = scorer
        self.nets = (self.encoder, self.scorer)
        self.state_map_f = state_map_f

    def parameters(self):
        """Retrieve trainable parameters, send to optimizer."""
        return chain(self.encoder.parameters(), self.scorer.parameters())

    def eval(self):
        for net in self.nets:
            net.eval()

    def train(self):
        for net in self.nets:
            net.train()

    def to(self, device):
        for net in self.nets:
            net.to(device)

    def state_dict(self):
        return (self.encoder.state_dict(), self.scorer.state_dict())

    def load_state_dict(self, state_dicts):
        encoder_sd, scorer_sd = state_dicts
        self.encoder.load_state_dict(encoder_sd)
        self.scorer.load_state_dict(scorer_sd)

    """
    logps unique
    """

    @profile
    def logps_unique(self, batch):
        """Differentiable; output logps of unique children/parents.

        Typical logic flow (example for getting children)
        For each state in batch ...
        1. Get children, reduce to valid and unique
        2. Set up input: (state, child) for child in children
        3. Run SSR network on batch
        4. Normalize probs to sum to 1

        Input
        -----
        batch: List of [State], n items, or State
        f:     function, get_children or get_parents

        Returns
        -------
        state_to_logp: List of dicts mapping state to torch.tensor
        """
        batched = bool(type(batch) is list)
        if not batched:
            batch = [batch]

        batch_dicts = []
        for state in batch:
            children = self.state_map_f(state)
            childs_uniq = unique_keep_order_remove_nones(children)

            childs_uniq, logps_uniq = self.__forward(state, childs_uniq)

            state_to_logp = {child: logp for child, logp in zip(childs_uniq, logps_uniq)}
            batch_dicts.append(state_to_logp)

        return batch_dicts if batched else batch_dicts[0]

    """
    values
    """

    def values_unique(self, batch):
        batched = bool(type(batch) is list)
        if not batched:
            batch = [batch]

        batches = []
        batch_dicts = []
        for state in batch:
            children = self.state_map_f(state)
            childs_uniq = unique_keep_order_remove_nones(children)

            childs_uniq, logps_uniq, values = self.__forward_v(state, childs_uniq)
            state_to_logp = {child: logp for child, logp in zip(childs_uniq, logps_uniq)}

            batches.append(values)
            batch_dicts.append(state_to_logp)
        return batches if batched else batches[0]

    """
    sample
    """

    @profile
    def sample(self, batch, epsilon=0.0):
        batched = bool(type(batch) is list)
        if not batched:
            batch = [batch]

        batch_samples = []
        for state in batch:
            children = self.state_map_f(state)
            # Reduce children to valid, unique
            childs_uniq = unique_keep_order_remove_nones(children)
            childs_uniq, logps_uniq = self.__forward(state, childs_uniq)

            # Sample
            ps_uniq = torch.exp(logps_uniq)
            if np.random.random() < epsilon:
                sample = np.random.choice(childs_uniq)
            else:
                ps = utils.tensor_to_np(ps_uniq, reduce_singleton=False)
                sample = np.random.choice(childs_uniq, p=ps)
            batch_samples.append(sample)

        return batch_samples if batched else batch_samples[0]

    """
    Forward
    """

    @profile
    def __forward(self, state, childs_uniq):
        """Single state -> (unique child states, logps) efficiently.

        With encoder, scorer framework, we have:
          encoder: state -> z
          scorer:  [z1, z2] -> R
        Call encoder on [state, c1, c2, ...] for c in children
          -> [z_state, z1, z2, ...]
        Call scorer on [z_state, z_i] for each i.

        Naive approach calls encode(state) C times, this does it once.
        """
        # Encode [state, c1, c2, ...] for c in children
        encoder_inp = [state] + childs_uniq
        # (C+1, states) -> (C+1, e)
        embeds = self.encoder(encoder_inp)
        embed_inp_state = embeds[0]
        embed_children = embeds[1:]

        # Score [z_state, z_i] for each i
        scorer_inp = torch.cat(
            [embed_inp_state.unsqueeze(0).repeat(len(childs_uniq), 1), embed_children], dim=1
        )

        # (C, 2e) -> (C, 1) -> (C)
        scores_uniq = torch.clip(
            torch.squeeze(self.scorer(scorer_inp), -1),
            min=self.args.clip_policy_logit_min,
            max=self.args.clip_policy_logit_max,
        )
        scores_uniq = torch.nan_to_num(scores_uniq, neginf=self.args.clip_policy_logit_min)

        logps_uniq = scores_uniq - torch.logsumexp(scores_uniq, -1)
        return childs_uniq, logps_uniq

    def __forward_v(self, state, childs_uniq):
        """Single state -> (unique child states, logps) efficiently.

        With encoder, scorer framework, we have:
          encoder: state -> z
          scorer:  [z1, z2] -> R
        Call encoder on [state, c1, c2, ...] for c in children
          -> [z_state, z1, z2, ...]
        Call scorer on [z_state, z_i] for each i.

        Naive approach calls encode(state) C times, this does it once.
        """
        # Encode [state, c1, c2, ...] for c in children
        encoder_inp = [state] + childs_uniq
        # (C+1, states) -> (C+1, e)
        embeds = self.encoder(encoder_inp)
        embed_inp_state = embeds[0]
        embed_children = embeds[1:]

        # Score [z_state, z_i] for each i
        scorer_inp = torch.cat(
            [embed_inp_state.unsqueeze(0).repeat(len(childs_uniq), 1), embed_children], dim=1
        )
        # (C, 2e) -> (C, 1) -> (C)
        scores_uniq = torch.clip(
            torch.squeeze(self.scorer(scorer_inp), -1),
            min=self.args.clip_policy_logit_min,
            max=self.args.clip_policy_logit_max,
        )
        scores_uniq = torch.nan_to_num(scores_uniq, neginf=self.args.clip_policy_logit_min)
        return childs_uniq, scores_uniq, torch.logsumexp(scores_uniq, -1)
