# Some parts adapted from https://github.com/alexis-jacq/LOLA_DiCE/blob/master/ipd_DiCE.py
# Some parts adapted from Chris Lu's MOFOS repo

# import jnp
import math
# import jnp.nn as nn
# from jnp.distributions import Categorical
import numpy as np
import argparse
import os
import datetime

import jax
import jax.numpy as jnp
from jax import jit, vmap, pmap
import functools
import optax
from functools import partial

import flax
from flax import struct
from flax import linen as nn
import jax.numpy as jnp
from typing import NamedTuple, Callable, Any
from flax.training.train_state import TrainState

from flax.training import checkpoints

# tensorflow probability === tf/jax/cuda version hell
#from tensorflow_probability.substrates import jax as tfp
#tfd = tfp.distributions

from coin_game_jax import CoinGame
from ipd_jax import IPD

deepmap = jax.tree_util.tree_map

@struct.dataclass
class Agent:
    pol: "TrainState"
    val: "TrainState"

    def extract_params(self):
        return dict(pol=self.pol.params, val=self.val.params)
    def replace_params(self, params):
        return self.replace(pol=self.pol.replace(params=params["pol"]),
                            val=self.val.replace(params=params["val"]) if use_baseline else self.val)
    def apply_gradients(self, grad):
        return self.replace(pol=self.pol.apply_gradients(grads=grad["pol"]),
                            val=self.val.apply_gradients(grads=grad["val"]) if use_baseline else self.val)

    def init_state(self, batch_size):
        # obtain the module through the apply_fn -__-
        return dict(pol=self.pol.apply_fn.__self__.init_state(batch_size),
                    val=self.val.apply_fn.__self__.init_state(batch_size) if use_baseline else None)

    @classmethod
    def make(cls, key, action_size, input_size):
        if args.architecture == "rnn":
          pol = RNN(num_outputs=action_size, num_hidden_units=args.hidden_size,
                    layers_before_gru=args.layers_before_gru)
          val = RNN(num_outputs=1, num_hidden_units=args.hidden_size,
                    layers_before_gru=args.layers_before_gru)
        elif args.architecture == "conv":
          pol = Conv(num_outputs=action_size, num_hidden_units=args.hidden_size, window=3)
          val = Conv(num_outputs=1, num_hidden_units=args.hidden_size, window=3)
        else: raise KeyError(args.architecture)

        key_pol, key_val = jax.random.split(key)
        pol_params = pol.init_params(key_pol, jnp.ones([args.batch_size, input_size]))
        val_params = val.init_params(key_val, jnp.ones([args.batch_size, input_size]))

        if args.optim.lower() == 'adam':
            pol_optimizer = optax.adam(learning_rate=args.lr_out)
            val_optimizer = optax.adam(learning_rate=args.lr_v)
        elif args.optim.lower() == 'sgd':
            pol_optimizer = optax.sgd(learning_rate=args.lr_out)
            val_optimizer = optax.sgd(learning_rate=args.lr_v)
        else:
            raise Exception("Unknown or Not Implemented Optimizer")

        pol_ts = TrainState.create(apply_fn=pol.apply, params=pol_params, tx=pol_optimizer)
        val_ts = TrainState.create(apply_fn=val.apply, params=val_params, tx=val_optimizer)
        return Agent(pol=pol_ts, val=val_ts)

class RNN(nn.Module):
    num_outputs: int
    num_hidden_units: int
    layers_before_gru: int

    def setup(self):
        if self.layers_before_gru >= 1:
            self.linear1 = nn.Dense(features=self.num_hidden_units)
        if self.layers_before_gru >= 2:
            self.linear2 = nn.Dense(features=self.num_hidden_units)
        self.GRUCell = nn.GRUCell()
        self.linear_end = nn.Dense(features=self.num_outputs)

    def __call__(self, x, carry):
        if self.layers_before_gru >= 1:
            x = self.linear1(x)
            x = nn.relu(x)
        if self.layers_before_gru >= 2:
            x = self.linear2(x)
        carry, x = self.GRUCell(carry, x)
        outputs = self.linear_end(x)
        return carry, outputs

    @nn.nowrap
    def init_state(self, batch_size):
        return jnp.zeros([batch_size, self.num_hidden_units])

    @nn.nowrap
    def init_params(self, key, x):
        return self.init(key, x, self.init_state(x.shape[0]))

class Conv(nn.Module):
    num_outputs: int
    num_hidden_units: int
    window: int = 3

    def setup(self):
        self.linear1 = nn.Dense(features=self.num_hidden_units)
        self.filter = nn.Dense(features=self.num_hidden_units)
        self.linear_end = nn.Dense(features=self.num_outputs)

    def __call__(self, x, queue):
        x = nn.relu(self.linear1(x))
        assert len(queue) == self.window
        queue = [*queue[:-1], x]
        x = nn.relu(self.filter(jnp.concatenate(queue, axis=-1)))
        y = self.linear_end(x)
        return queue, y

    @nn.nowrap
    def init_state(self, batch_size):
        return [jnp.zeros([batch_size, self.num_hidden_units])
                for _ in range(self.window)]

    @nn.nowrap
    def init_params(self, key, x):
        return self.init(key, x, self.init_state(x.shape[0]))

def reverse_cumsum(x, axis):
    return x + jnp.sum(x, axis=axis, keepdims=True) - jnp.cumsum(x, axis=axis)

#@jit
def magic_box(x):
    return jnp.exp(x - jax.lax.stop_gradient(x))

#@jit
def update_gae_with_delta_backwards(gae, delta):
    gae = gae * args.gamma * args.gae_lambda + delta
    return gae, gae

#@jit
def get_gae_advantages(rewards, values, next_val_history):
    deltas = rewards + args.gamma * jax.lax.stop_gradient(next_val_history) - jax.lax.stop_gradient(values)
    gae = jnp.zeros_like(deltas[0, :])
    deltas = jnp.flip(deltas, axis=0)
    gae, flipped_advantages = jax.lax.scan(update_gae_with_delta_backwards, gae, deltas, deltas.shape[0])
    advantages = jnp.flip(flipped_advantages, axis=0)
    return advantages

#@jit
def dice_objective(self_logprobs, other_logprobs, rewards, values, end_state_v):
    cum_discount = jnp.cumprod(args.gamma * jnp.ones(rewards.shape), axis=0) / args.gamma
    discounted_rewards = rewards * cum_discount
    stochastic_nodes = self_logprobs + other_logprobs

    if use_baseline:
        assert values.shape[0] == args.rollout_len  # if so, use concatenate to construct this
        next_val_history = jnp.zeros((args.rollout_len, args.batch_size))
        next_val_history = next_val_history.at[:args.rollout_len-1].set(values[1:args.rollout_len])
        next_val_history = next_val_history.at[-1].set(end_state_v)

        if args.zero_vals:
            next_val_history = jnp.zeros_like(next_val_history)
            values = jnp.zeros_like(values)

        advantages = get_gae_advantages(rewards, values, next_val_history)
        discounted_advantages = advantages * cum_discount
        deps_up_to_t = jnp.cumsum(stochastic_nodes, axis=0)  # == `dependencies`?
        deps_less_than_t = deps_up_to_t - stochastic_nodes  # take out the dependency in the given time step

        # Look at Loaded DiCE and GAE papers to see where this formulation comes from
        dice_obj = ((magic_box(deps_up_to_t) - magic_box(deps_less_than_t))
                    * discounted_advantages).sum(axis=0).mean()
    else:
        # dice objective:
        # REMEMBER that in this jax code the axis 0 is the rollout_len (number of time steps in the environment)
        # and axis 1 is the batch.
        dependencies = jnp.cumsum(stochastic_nodes, axis=0)
        dice_obj = (magic_box(dependencies) * discounted_rewards).sum(axis=0).mean()
    return -dice_obj  # want to minimize -objective

#@jit
def dice_objective_plus_value_loss(self_logprobs, other_logprobs, rewards, values, end_state_v):
    # Essentially a wrapper function for the objective to put all the control flow in one spot
    # The reasoning behind this function here is that the reward_loss has a stop_gradient
    # on all of the nodes related to the value function
    # and the value function has no nodes related to the policy
    # Then we can actually take the respective grads like the way I have things set up now
    # And I should be able to update both policy and value functions
    reward_loss = dice_objective(self_logprobs, other_logprobs, rewards, values, end_state_v)
    if use_baseline:
        val_loss = value_loss(rewards, values, end_state_v)
        return reward_loss + val_loss
    else:
        return reward_loss

#@jit
def value_loss(rewards, values, final_state_vals):
    final_state_vals = jax.lax.stop_gradient(final_state_vals)
    discounts = jnp.cumprod(args.gamma * jnp.ones(rewards.shape), axis=0) / args.gamma
    gamma_t_r_ts = rewards * discounts
    G_ts = reverse_cumsum(gamma_t_r_ts, axis=0)
    R_ts = G_ts / discounts
    final_val_discounted_to_curr = (args.gamma * jnp.flip(discounts, axis=0)) * final_state_vals
    # You DO need a detach on these. Because it's the target - it should be detached. It's a target value.
    # Essentially a Monte Carlo style type return for R_t, except for the final state we also use the estimated final state value.
    # This becomes our target for the value function loss. So it's kind of a mix of Monte Carlo and bootstrap, but anyway you need the final value
    # because otherwise your value calculations will be inconsistent
    # TODO(cooijmat) figure out if we do or do not need a detach here then
    values_loss = (R_ts + final_val_discounted_to_curr - values) ** 2
    values_loss = values_loss.sum(axis=0).mean()
    return values_loss

#@jit
def act(key, agent, astate, obs):
    new_astate = dict()
    new_astate["pol"], logits = agent.pol.apply_fn(agent.pol.params, obs, astate["pol"])
    if use_baseline:
        new_astate["val"], values = agent.val.apply_fn(agent.val.params, obs, astate["val"])
        values = values.squeeze(-1)
    else:
        new_astate["val"], values = None, None
    key, subkey = jax.random.split(key)
    logits = nn.log_softmax(logits)
    actions = jax.random.categorical(subkey, logits)
    logps = jax.vmap(lambda z, a: z[a])(logits, actions)
    return dict(a=actions, l=logits, v=values, astate=new_astate,
                logp=logps, p=jnp.exp(logits))

#@jit
def _scan_act(key, agent, astate, obsseq):
    key, subkey = jax.random.split(key)
    @jax.named_call
    def scan_act_body(carry, obs):
        key, astate = carry
        key, subkey = jax.random.split(key)
        aux = act(subkey, agent, astate, obs)
        return (key, aux["astate"]), (aux["p"], aux["v"])
    obsseq = jnp.stack(obsseq[:args.rollout_len], axis=0) # TODO(cooijmat) stack this in caller
    carry, aux = jax.lax.scan(scan_act_body, (subkey, astate), obsseq, args.rollout_len)
    return aux

#@jit
def scan_act(key, agent, obsseq):
    astate = agent.init_state(args.batch_size)
    return _scan_act(key, agent, astate, obsseq)

#@jit
def scan_act_onebatch(key, agent, obsseq):
    astate = agent.init_state(1)
    return _scan_act(key, agent, astate, obsseq)

def get_policies_for_states(key, agent, obsseq):
    return scan_act(key, agent, obsseq)[0]
def get_policies_for_states_onebatch(key, agent, obsseq):
    return scan_act_onebatch(key, agent, obsseq)[0]

def dicttranspose(dikts):
    assert all(set(dikt) == set(dikts[0]) for dikt in dikts[1:])
    return {key: [dikt[key] for dikt in dikts] for key in dikts[0]}

def env_step(key, env_state, obss, agents, astates):
    key, sk1, sk2, skenv = jax.random.split(key, 4)
    subkeys = [sk1, sk2]
    aux = dicttranspose([act(*args) for args in zip(subkeys, agents, astates, obss)])
    skenv = jax.random.split(skenv, args.batch_size)
    env_state, new_obs, rewards, aux_info = vec_env_step(env_state, *aux["a"], skenv)
    new_obss = [new_obs, new_obs]
    stuff = (key, env_state, new_obss, aux["astate"])
    return stuff, (dict(obss=new_obss, r=rewards, p=aux["p"], logp=aux["logp"], v=aux["v"], a=aux["a"]), aux_info)

def do_env_rollout(key, agents, agent_for_state_history):
    keys = jax.random.split(key, args.batch_size + 1)
    key, env_subkeys = keys[0], keys[1:]
    env_state, obsv = vec_env_reset(env_subkeys)
    obss = [obsv, obsv]
    astates = [agent.init_state(args.batch_size) for agent in agents]

    @jax.named_call
    def env_step_body(carry, _):
        (key, env_state, obss, astates) = carry
        return env_step(key, env_state, obss, agents, astates)
    carry = (key, env_state, obss, astates)
    carry, (aux, aux_info) = jax.lax.scan(env_step_body, carry, None, args.rollout_len)

    if agent_for_state_history == 1:
        obsseq = [obss[0], *aux["obss"][0]]
    elif agent_for_state_history == 2:
        obsseq = [obss[1], *aux["obss"][1]]
    else: raise ValueError(agent_for_state_history)

    return carry, aux, obsseq

def kl_div_jax(curr, target):
    # TODO(cooijmat) wrong way around??
    return (curr * (jnp.log(curr) - jnp.log(target))).sum(axis=-1).mean()


def eval_vs_alld_selfagent1(stuff, unused):
    key, agent, env_state, obsv, astate = stuff

    key, subkey = jax.random.split(key)
    aux = act(subkey, agent, astate, obsv)
    a, astate = aux["a"], aux["astate"]

    keys = jax.random.split(key, args.batch_size + 1)
    key, env_subkeys = keys[0], keys[1:]

    i_am_red_agent = True
    opp_is_red_agent = False

    if args.env == "ipd":
        a_opp = jnp.zeros_like(a) # Always defect
    elif args.env == "coin":
        a_opp = env.get_moves_shortest_path_to_coin(env_state, opp_is_red_agent)

    a1 = a
    a2 = a_opp
    env_state, new_obs, (r1, r2), aux_info = vec_env_step(env_state, a1, a2, env_subkeys)
    obsv = new_obs

    score1 = r1.mean()
    score2 = r2.mean()

    stuff = (key, agent, env_state, obsv, astate)
    aux = (score1, score2)
    return stuff, aux

def eval_vs_alld_selfagent2(stuff, unused):
    key, agent, env_state, obsv, astate = stuff

    key, subkey = jax.random.split(key)
    aux = act(subkey, agent, astate, obsv)
    a, astate = aux["a"], aux["astate"]

    keys = jax.random.split(key, args.batch_size + 1)
    key, env_subkeys = keys[0], keys[1:]

    i_am_red_agent = False
    opp_is_red_agent = True

    if args.env == "ipd":
        a_opp = jnp.zeros_like(a) # Always defect
    elif args.env == "coin":
        a_opp = env.get_moves_shortest_path_to_coin(env_state, opp_is_red_agent)

    a2 = a
    a1 = a_opp

    env_state, new_obs, (r1, r2), aux_info = vec_env_step(env_state, a1, a2, env_subkeys)
    obsv = new_obs

    score1 = r1.mean()
    score2 = r2.mean()

    stuff = (key, agent, env_state, obsv, astate)
    aux = (score1, score2)
    return stuff, aux

def eval_vs_allc_selfagent1(stuff, unused):
    key, agent, env_state, obsv, astate = stuff

    key, subkey = jax.random.split(key)
    aux = act(subkey, agent, astate, obsv)
    a, astate = aux["a"], aux["astate"]

    keys = jax.random.split(key, args.batch_size + 1)
    key, env_subkeys = keys[0], keys[1:]

    i_am_red_agent = True
    opp_is_red_agent = False

    if args.env == "ipd":
        a_opp = jnp.ones_like(a) # Always cooperate
    elif args.env == "coin":
        a_opp = env.get_coop_action(env_state, opp_is_red_agent)

    a1 = a
    a2 = a_opp

    env_state, new_obs, (r1, r2), aux_info = vec_env_step(env_state, a1, a2, env_subkeys)
    obsv = new_obs

    score1 = r1.mean()
    score2 = r2.mean()

    stuff = (key, agent, env_state, obsv, astate)
    aux = (score1, score2)
    return stuff, aux

def eval_vs_allc_selfagent2(stuff, unused):
    key, agent, env_state, obsv, astate = stuff

    key, subkey = jax.random.split(key)
    aux = act(subkey, agent, astate, obsv)
    a, astate = aux["a"], aux["astate"]

    keys = jax.random.split(key, args.batch_size + 1)
    key, env_subkeys = keys[0], keys[1:]

    i_am_red_agent = False
    opp_is_red_agent = True

    if args.env == "ipd":
        a_opp = jnp.ones_like(a) # Always cooperate
    elif args.env == "coin":
        a_opp = env.get_coop_action(env_state, opp_is_red_agent)

    a2 = a
    a1 = a_opp

    env_state, new_obs, (r1, r2), aux_info = vec_env_step(env_state, a1, a2, env_subkeys)
    obsv = new_obs

    score1 = r1.mean()
    score2 = r2.mean()

    stuff = (key, agent, env_state, obsv, astate)
    aux = (score1, score2)
    return stuff, aux

def eval_vs_tft_selfagent1(stuff, unused):
    key, agent, env_state, obsv, astate, prev_a, prev_agent_coin_collected_same_col, r1, r2 = stuff

    key, subkey = jax.random.split(key)
    aux = act(subkey, agent, astate, obsv)
    a, astate = aux["a"], aux["astate"]

    keys = jax.random.split(key, args.batch_size + 1)
    key, env_subkeys = keys[0], keys[1:]

    if args.env == "ipd":
        # Copy last move of agent; assumes prev_a = all coop
        a_opp = prev_a
        prev_agent_coin_collected_same_col = None
    elif args.env == "coin":
        r_opp = r2
        # Agent here means me, the agent we are testing
        prev_agent_coin_collected_same_col = jnp.where(r_opp < 0, 0, prev_agent_coin_collected_same_col)
        prev_agent_coin_collected_same_col = jnp.where(r_opp > 0, 1, prev_agent_coin_collected_same_col)

        a_opp_defect = env.get_moves_shortest_path_to_coin(env_state, False)
        a_opp_coop = env.get_coop_action(env_state, False)

        a_opp = jax.lax.stop_gradient(a_opp_coop)
        a_opp = jnp.where(prev_agent_coin_collected_same_col == 0, a_opp_defect, a_opp)

    a1 = a
    a2 = a_opp

    env_state, new_obs, (r1, r2), aux_info = vec_env_step(env_state, a1, a2, env_subkeys)
    obsv = new_obs

    score1 = r1.mean()
    score2 = r2.mean()

    stuff = (key, agent, env_state, obsv, astate, a, prev_agent_coin_collected_same_col, r1, r2)
    aux = (score1, score2)
    return stuff, aux

def eval_vs_tft_selfagent2(stuff, unused):
    key, agent, env_state, obsv, astate, prev_a, prev_agent_coin_collected_same_col, r1, r2 = stuff

    key, subkey = jax.random.split(key)
    aux = act(subkey, agent, astate, obsv)
    a, astate = aux["a"], aux["astate"]

    keys = jax.random.split(key, args.batch_size + 1)
    key, env_subkeys = keys[0], keys[1:]

    if args.env == "ipd":
        # Copy last move of agent; assumes prev_a = all coop
        a_opp = prev_a
        prev_agent_coin_collected_same_col = None
    elif args.env == "coin":
        r_opp = r1
        # Agent here means me, the agent we are testing
        prev_agent_coin_collected_same_col = jnp.where(r_opp < 0, 0, prev_agent_coin_collected_same_col)
        prev_agent_coin_collected_same_col = jnp.where(r_opp > 0, 1, prev_agent_coin_collected_same_col)

        a_opp_defect = env.get_moves_shortest_path_to_coin(env_state, True)
        a_opp_coop = env.get_coop_action(env_state, True)

        a_opp = jax.lax.stop_gradient(a_opp_coop)
        a_opp = jnp.where(prev_agent_coin_collected_same_col == 0, a_opp_defect, a_opp)

    a1 = a_opp
    a2 = a

    env_state, new_obs, (r1, r2), aux_info = vec_env_step(env_state, a1, a2, env_subkeys)
    obsv = new_obs

    score1 = r1.mean()
    score2 = r2.mean()

    stuff = (key, agent, env_state, obsv, astate, a, prev_agent_coin_collected_same_col, r1, r2)
    aux = (score1, score2)
    return stuff, aux

def eval_vs_fixed_strategy(key, agent, strat="alld", self_agent=1):
    keys = jax.random.split(key, args.batch_size + 1)
    key, env_subkeys = keys[0], keys[1:]

    env_state, obsv = vec_env_reset(env_subkeys) # note this works only with the same obs, otherwise you would have to switch things up a bit here

    astate = agent.init_state(args.batch_size)

    if strat == "alld":
        stuff = key, agent, env_state, obsv, astate
        if self_agent == 1:
            stuff, aux = jax.lax.scan(eval_vs_alld_selfagent1, stuff, None, args.rollout_len)
        else:
            stuff, aux = jax.lax.scan(eval_vs_alld_selfagent2, stuff, None, args.rollout_len)
    elif strat == "allc":
        stuff = key, agent, env_state, obsv, astate
        if self_agent == 1:
            stuff, aux = jax.lax.scan(eval_vs_allc_selfagent1, stuff, None, args.rollout_len)
        else:
            stuff, aux = jax.lax.scan(eval_vs_allc_selfagent2, stuff, None, args.rollout_len)
    elif strat == "tft":
        if args.env == "ipd":
            prev_a = jnp.ones(
                args.batch_size, dtype=int)  # assume agent (self) cooperated for the init time step when the opponent is using TFT
            r1 = jnp.zeros(args.batch_size)  # these don't matter for IPD,
            r2 = jnp.zeros(args.batch_size)
            prev_agent_coin_collected_same_col = None
        elif args.env == "coin":
            if self_agent == 1:
                prev_a = env.get_coop_action(env_state, red_agent_perspective=False)  # doesn't matter for coin
            else:
                prev_a = env.get_coop_action(env_state, red_agent_perspective=True)  # doesn't matter for coin
            prev_agent_coin_collected_same_col = jnp.ones(
                args.batch_size, dtype=int)  # 0 = defect, collect other agent coin. Init with 1 (coop)
            r1 = jnp.zeros(args.batch_size)
            r2 = jnp.zeros(args.batch_size)
        else:
            raise NotImplementedError
        stuff = (key, agent, env_state, obsv, astate, prev_a,
                 prev_agent_coin_collected_same_col, r1, r2)
        if self_agent == 1:
            stuff, aux = jax.lax.scan(eval_vs_tft_selfagent1, stuff, None, args.rollout_len)
        else:
            stuff, aux = jax.lax.scan(eval_vs_tft_selfagent2, stuff, None, args.rollout_len)

    score1, score2 = aux
    score1 = score1.mean()
    score2 = score2.mean()
    return (score1, score2), None

def inspect_ipd(agents):
    assert args.env == 'ipd'
    unused_keys = jax.random.split(jax.random.PRNGKey(0), args.batch_size)
    state, obsv = vec_env_reset(unused_keys)
    init_state = env.init_state
    for i in range(2):
        for j in range(2):
            state1 = env.states[i, j]
            for ii in range(2):
                for jj in range(2):
                    state2 = env.states[ii, jj]
                    state_history = [init_state, state1, state2]
                    print(state_history)
                    pol_probs1 = get_policies_for_states_onebatch(jax.random.PRNGKey(0), agents[0], state_history)
                    pol_probs2 = get_policies_for_states_onebatch(jax.random.PRNGKey(0), agents[1], state_history)
                    print(pol_probs1)
                    print(pol_probs2)
    # Build state history artificially for all combs, and pass those into the pol_probs.

@jit
def eval_progress(subkey, agents):
    keys = jax.random.split(subkey, args.batch_size + 1)
    key, env_subkeys = keys[0], keys[1:]
    env_state, obsv = vec_env_reset(env_subkeys)
    obss = [obsv, obsv]
    astates = [agent.init_state(args.batch_size) for agent in agents]
    key, subkey = jax.random.split(key)

    def body_fn(carry, _):
        (key, env_state, obss, astates) = carry
        return env_step(key, env_state, obss, agents, astates)
    carry = (subkey, env_state, obss, astates)
    carry, (aux, aux_info) = jax.lax.scan(body_fn, carry, None, args.rollout_len)

    [r1,r2] = aux["r"]

    score1rec = []
    score2rec = []

    print("Eval vs Fixed Strategies:")
    for strat in ["alld", "allc", "tft"]:
        # print(f"Playing against strategy: {strat.upper()}")
        key, subkey = jax.random.split(key)
        score1, _ = eval_vs_fixed_strategy(subkey, agents[0], strat, self_agent=1)
        score1rec.append(score1[0])
        # print(f"Agent 1 score: {score1[0]}")
        key, subkey = jax.random.split(key)
        score2, _ = eval_vs_fixed_strategy(subkey, agents[1], strat, self_agent=2)
        score2rec.append(score2[1])
        # print(f"Agent 2 score: {score2[1]}")

    score1rec = jnp.stack(score1rec)
    score2rec = jnp.stack(score2rec)

    avg_rew1 = r1.mean()
    avg_rew2 = r2.mean()

    if args.env == 'coin':
        rr, rb, br, bb = aux_info
        rr = rr.sum(axis=0).mean()
        rb = rb.sum(axis=0).mean()
        br = br.sum(axis=0).mean()
        bb = bb.sum(axis=0).mean()
        return avg_rew1, avg_rew2, rr, rb, br, bb, score1rec, score2rec
    else:
        return avg_rew1, avg_rew2, None, None, None, None, score1rec, score2rec

# outer step functions are called in a plain python loop so should be jitted,
# and input structure should match output structure
@jit
def one_outer_step_update_selfagent1(key, agent1, ref_agents):
    ref_agent1, ref_agent2 = ref_agents
    outer_agent1 = agent1
    def fn(params, key):
        agent1 = outer_agent1.replace_params(params)
        key, subkey = jax.random.split(key)
        agent2_ahead = inner_update_agent2(subkey, agent1, ref_agent2)
        objective = out_lookahead(key, [agent1, agent2_ahead], ref_agent1, self_agent=1)
        return objective
    key, subkey = jax.random.split(key)
    grad = jax.grad(fn)(outer_agent1.extract_params(), subkey)
    agent1 = agent1.apply_gradients(grad)
    return key, agent1, ref_agents

@jit
def one_outer_step_update_selfagent2(key, agent2, ref_agents):
    ref_agent1, ref_agent2 = ref_agents
    outer_agent2 = agent2
    def fn(params, key):
        agent2 = outer_agent2.replace_params(params)
        key, subkey = jax.random.split(key)
        agent1_ahead = inner_update_agent1(subkey, ref_agent1, agent2)
        objective = out_lookahead(key, [agent1_ahead, agent2], ref_agent2, self_agent=2)
        return objective
    key, subkey = jax.random.split(key)
    grad = jax.grad(fn)(outer_agent2.extract_params(), subkey)
    agent2 = agent2.apply_gradients(grad)
    return key, agent2, ref_agents

def inner_update_agent1(key, agent1, agent2):
    ref_agent1 = agent1
    agent1 = agent1.replace(pol=TrainState.create(apply_fn=agent1.pol.apply_fn,
                                                  params=agent1.pol.params,
                                                  tx=optax.sgd(learning_rate=args.lr_in)),
                            val=TrainState.create(apply_fn=agent1.val.apply_fn,
                                                  params=agent1.val.params,
                                                  tx=optax.sgd(learning_rate=args.lr_v)))

    def body_fn(carry, _):
        key, agent1 = carry
        key, subkey = jax.random.split(key)
        def fn(params):
          agent1_ = agent1.replace_params(params)
          return in_lookahead(subkey, [agent1_, agent2], ref_agent1, other_agent=1)
        grad = jax.grad(fn)(agent1.extract_params())
        agent1 = agent1.apply_gradients(grad)
        return (key, agent1), None

    key, reused_subkey = jax.random.split(key)
    key, subkey = jax.random.split(key)
    assert not args.old_kl_div

    # do one step with reused_subkey as in the original code
    carry = (reused_subkey, agent1)
    carry, _ = body_fn(carry, None)
    agent1 = carry[1]

    key, subkey = jax.random.split(key)
    if args.inner_steps > 1:
        carry = (subkey, agent1)
        carry, aux = jax.lax.scan(body_fn, carry, None, args.inner_steps - 1)
        agent1 = carry[1]

    return agent1

def inner_update_agent2(key, agent1, agent2):
    ref_agent2 = agent2
    agent2 = agent2.replace(pol=TrainState.create(apply_fn=agent2.pol.apply_fn,
                                                  params=agent2.pol.params,
                                                  tx=optax.sgd(learning_rate=args.lr_in)),
                            val=TrainState.create(apply_fn=agent2.val.apply_fn,
                                                  params=agent2.val.params,
                                                  tx=optax.sgd(learning_rate=args.lr_v)))

    def body_fn(carry, _):
        key, agent2, params_for_bug1 = carry
        key, subkey = jax.random.split(key)
        def fn(params):
            agent2_ = agent2.replace_params(params)
            return in_lookahead(subkey, [agent1, agent2_], ref_agent2, other_agent=2)
        grad = jax.grad(fn)(agent2.extract_params())
        agent2 = agent2.apply_gradients(grad)
        return (key, agent2, agent2.extract_params()), None

    key, reused_subkey = jax.random.split(key)
    key, subkey = jax.random.split(key)
    assert not args.old_kl_div

    # do one step with reused_subkey as in the original code
    carry = (reused_subkey, agent2, agent2.extract_params())
    carry, _ = body_fn(carry, None)
    agent2 = carry[1]

    key, subkey = jax.random.split(key)
    if args.inner_steps > 1:
        carry = (subkey, agent2, agent2.extract_params())
        carry, aux = jax.lax.scan(body_fn, carry, None, args.inner_steps - 1)
        agent2 = carry[1]

    return agent2

def in_lookahead(key, agents, ref_agent, other_agent=2):
    carry, auxseq, obsseq = do_env_rollout(key, agents, agent_for_state_history=other_agent)
    (key, env_state, obss, astates) = carry

    # we are in the inner loop, so `other_agent` is us
    us = other_agent-1
    them = 1-us

    # act just to get the final state values
    key, *subkeys = jax.random.split(key, 3)
    auxend = act(subkeys[us], agents[us], astates[us], obss[us])
    objective = dice_objective_plus_value_loss(self_logprobs=auxseq["logp"][us],
                                               other_logprobs=auxseq["logp"][them],
                                               rewards=auxseq["r"][us],
                                               values=auxseq["v"][us],
                                               end_state_v=auxend["v"])
    # print(f"Inner Agent (Agent {other_agent}) episode return avg {auxseq['r'][us].sum(axis=0).mean()}")

    assert not args.old_kl_div
    key, sk1, sk2 = jax.random.split(key, 3)
    probseq = get_policies_for_states(sk1, agents[us], obsseq)
    probseq_ref = get_policies_for_states(sk2, ref_agent, obsseq)
    kl_div = kl_div_jax(probseq, probseq_ref)

    return objective + args.inner_beta * kl_div  # we want to min kl div

def out_lookahead(key, agents, ref_agent, self_agent=1):
    carry, auxseq, obsseq = do_env_rollout(key, agents, agent_for_state_history=self_agent)
    (key, env_state, obss, astates) = carry

    # we are in the outer loop, so `self_agent` is us
    us = self_agent-1
    them = 1-us

    # act just to get the final state values
    key, subkey = jax.random.split(key)
    auxend = act(subkey, agents[us], astates[us], obss[us])
    objective = dice_objective_plus_value_loss(self_logprobs=auxseq["logp"][us],
                                               other_logprobs=auxseq["logp"][them],
                                               rewards=auxseq["r"][us],
                                               values=auxseq["v"][us],
                                               end_state_v=auxend["v"])
    # print(f"Agent {self_agent} episode return avg {auxseq['r'][us].sum(axis=0).mean()}")

    assert not args.old_kl_div
    key, sk1, sk2 = jax.random.split(key, 3)
    probseq = get_policies_for_states(sk1, agents[us], obsseq)
    probseq_ref = get_policies_for_states(sk2, ref_agent, obsseq)
    kl_div = kl_div_jax(probseq, probseq_ref)

    return objective + args.outer_beta * kl_div

stop = jax.lax.stop_gradient

def update_agents(key, agents):
    assert not args.old_kl_div  # too gnarly
    agent1, agent2 = list(agents)

    # --- AGENT 1 UPDATE ---
    key, subkey = jax.random.split(key)
    carry = (subkey, agent1, agents)
    for _ in range(args.outer_steps):
        carry = one_outer_step_update_selfagent1(*carry)
    agent1 = carry[1]

    # --- AGENT 2 UPDATE ---
    key, subkey = jax.random.split(key)
    carry = (subkey, agent2, agents)
    for _ in range(args.outer_steps):
        carry = one_outer_step_update_selfagent2(*carry)
    agent2 = carry[1]

    return key, [agent1, agent2]

# redefine play, simplified so we can actually see what's happening (cooijmat)
def play(key, agents):
    record = dict(score1=[], score2=[], rr=[], rb=[], br=[], bb=[], score1rec=[], score2rec=[])

    print("start iterations with", args.inner_steps, "inner steps and", args.outer_steps, "outer steps:")

    key, subkey = jax.random.split(key)
    score1, score2, rr, rb, br, bb, score1rec, score2rec = eval_progress(key, agents)
    for k,v in dict(score1=score1,score2=score2,score1rec=score1rec,score2rec=score2rec,rr=rr,rb=rb,br=br,bb=bb).items():
        record[k].append(v)

    for update in range(args.n_update):
        key, agents = update_agents(key, agents)

        # evaluate progress:
        key, subkey = jax.random.split(key)
        score1, score2, rr, rb, br, bb, score1rec, score2rec = eval_progress(key, agents)
        for k,v in dict(score1=score1,score2=score2,score1rec=score1rec,score2rec=score2rec,rr=rr,rb=rb,br=br,bb=bb).items():
            record[k].append(v)

        # print
        if (update + 1) % args.print_every == 0:
            print("*" * 10)
            print("Epoch: {}".format(update + 1), flush=True)
            print(f"scores {score1} {score2}")
            if args.env == 'coin':
                print(f"same {rr+bb} diff {rb+br}")
                print(f"rr {rr} rb {rb} br {br} bb {bb}")
            print("Scores vs fixed strats ALLD, ALLC, TFT:")
            print(score1rec)
            print(score2rec)
            if args.env == 'ipd':
                if args.inspect_ipd:
                    inspect_ipd(agents)
        if (update + 1) % args.checkpoint_every == 0:
            now = datetime.datetime.now()
            checkpoints.save_checkpoint(ckpt_dir=args.save_dir,
                                        target=agents,
                                        step=update + 1, prefix=f"checkpoint_{now.strftime('%Y-%m-%d_%H-%M')}_seed{args.seed}_epoch")
            np.savez_compressed("record.incoming.npz", **record)
            os.rename("record.incoming.npz", "record.npz")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("POLA")
    parser.add_argument("--inner_steps", type=int, default=1, help="inner loop steps for DiCE")
    parser.add_argument("--outer_steps", type=int, default=1, help="outer loop steps for POLA")
    parser.add_argument("--lr_out", type=float, default=0.005,
                        help="outer loop learning rate: same learning rate across all policies for now")
    parser.add_argument("--lr_in", type=float, default=0.03,
                        help="inner loop learning rate (eta): this has no use in the naive learning case. Used for the gradient step done for the lookahead for other agents during LOLA (therefore, often scaled to be higher than the outer learning rate in non-proximal LOLA). Note that this has a different meaning for the Taylor approx vs. actual update versions. A value of eta=1 is perfectly reasonable for the Taylor approx version as this balances the scale of the gradient with the naive learning term (and will be multiplied by the outer learning rate after), whereas for the actual update version with neural net, 1 is way too big an inner learning rate. For prox, this is the learning rate on the inner prox loop so is not that important - you want big enough to be fast-ish, but small enough to converge.")
    parser.add_argument("--lr_v", type=float, default=0.001,
                        help="same learning rate across all policies for now. Should be around maybe 0.001 or less for neural nets to avoid instability")
    parser.add_argument("--gamma", type=float, default=0.96, help="discount rate")
    parser.add_argument("--n_update", type=int, default=5000, help="number of epochs to run")
    parser.add_argument("--rollout_len", type=int, default=50, help="How long we want the time horizon of the game to be (number of steps before termination/number of iterations of the IPD)")
    parser.add_argument("--batch_size", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=1, help="for seed")
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--print_every", type=int, default=1, help="Print every x number of epochs")
    parser.add_argument("--outer_beta", type=float, default=0.0, help="for outer kl penalty with POLA")
    parser.add_argument("--inner_beta", type=float, default=0.0, help="for inner kl penalty with POLA")
    parser.add_argument("--save_dir", type=str, default='.', help="Where to save checkpoints")
    parser.add_argument("--checkpoint_every", type=int, default=50, help="Epochs between checkpoint save")
    parser.add_argument("--load_dir", type=str, default=None, help="Directory for loading checkpoint")
    parser.add_argument("--load_prefix", type=str, default=None, help="Prefix for loading checkpoint")
    parser.add_argument("--diff_coin_reward", type=float, default=1.0, help="changes problem setting (the reward for picking up coin of different colour)")
    parser.add_argument("--diff_coin_cost", type=float, default=-2.0, help="changes problem setting (the cost to the opponent when you pick up a coin of their colour)")
    parser.add_argument("--same_coin_reward", type=float, default=1.0, help="changes problem setting (the reward for picking up coin of same colour)")
    parser.add_argument("--grid_size", type=int, default=3, help="Grid size for Coin Game")
    parser.add_argument("--optim", type=str, default="adam", help="Used only for the outer agent (in the out_lookahead)")
    parser.add_argument("--no_baseline", action="store_true", help="Use NO Baseline (critic) for variance reduction. Default is baseline using Loaded DiCE with GAE")
    parser.add_argument("--env", type=str, default="coin",
                        choices=["ipd", "coin"])
    parser.add_argument("--hist_one", action="store_true", help="Use one step history (no gru or rnn, just one step history)")
    parser.add_argument("--print_info_each_outer_step", action="store_true", help="For debugging/curiosity sake")
    parser.add_argument("--init_state_coop", action="store_true", help="For IPD only: have the first state be CC instead of a separate start state")
    parser.add_argument("--split_coins", action="store_true", help="If true, then when both agents step on same coin, each gets 50% of the reward as if they were the only agent collecting that coin. Only tested with OGCoin so far")
    parser.add_argument("--zero_vals", action="store_true", help="For testing/debug. Can also serve as another way to do no_baseline. Set all values to be 0 in Loaded Dice Calculation")
    parser.add_argument("--gae_lambda", type=float, default=1,
                        help="lambda for GAE (1 = monte carlo style, 0 = TD style)")
    parser.add_argument("--val_update_after_loop", action="store_true", help="Update values only after outer POLA loop finishes, not during the POLA loop")
    parser.add_argument("--std", type=float, default=0.1, help="standard deviation for initialization of policy/value parameters")
    parser.add_argument("--old_kl_div", action="store_true", help="Use the old version of KL div relative to just one batch of states at the beginning")
    parser.add_argument("--inspect_ipd", action="store_true", help="Detailed (2 steps + start state) policy information in the IPD with full history")
    parser.add_argument("--layers_before_gru", type=int, default=2, choices=[0, 1, 2], help="Number of linear layers (with ReLU activation) before GRU, supported up to 2 for now")
    parser.add_argument("--contrib_factor", type=float, default=1.33, help="contribution factor to vary difficulty of IPD")
    parser.add_argument("--architecture", type=str, default="rnn", choices="rnn conv".split())
    args = parser.parse_args()
    assert not args.old_kl_div

    np.random.seed(args.seed)
    if args.env == 'coin':
        #assert args.grid_size == 3  # rest not implemented yet
        input_size = args.grid_size ** 2 * 4
        action_size = 4
        env = CoinGame(grid_size=args.grid_size)
    elif args.env == 'ipd':
        input_size = 6 # 3 * n_agents
        action_size = 2
        env = IPD(init_state_coop=args.init_state_coop, contrib_factor=args.contrib_factor)
    else:
        raise NotImplementedError("unknown env")
    vec_env_reset = jax.vmap(env.reset)
    vec_env_step = jax.vmap(env.step)

    key = jax.random.PRNGKey(args.seed)
    key, key1, key2 = jax.random.split(key, 3)
    agents = [Agent.make(key1, action_size, input_size),
              Agent.make(key2, action_size, input_size)]

    if args.load_dir is not None:
        epoch_num = int(args.load_prefix.split("epoch")[-1])
        if epoch_num % 10 == 0:
            epoch_num += 1  # Kind of an ugly temporary fix to allow for the updated checkpointing system which now has
            # record of rewards/eval vs fixed strat before the first training - important for IPD plots. Should really be applied to
            # all checkpoints with the new updated code I have, but the coin checkpoints above are from old code

        assert args.load_prefix is not None
        agents = checkpoints.restore_checkpoint(ckpt_dir=args.load_dir,
                                                target=agents,
                                                prefix=args.load_prefix)

    use_baseline = True
    if args.no_baseline:
        use_baseline = False

    assert args.inner_steps >= 1
    # Use 0 lr if you want no inner steps... TODO allow for this functionality (naive learning)?
    assert args.outer_steps >= 1

    play(key, agents)
