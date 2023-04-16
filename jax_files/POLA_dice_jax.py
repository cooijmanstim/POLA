# Some parts adapted from https://github.com/alexis-jacq/LOLA_DiCE/blob/master/ipd_DiCE.py
# Some parts adapted from Chris Lu's MOFOS repo

import math, argparse, os, pickle, datetime, functools
from functools import partial
import numpy as np

import jax, jax.numpy as jnp
from jax import jit, vmap, pmap
import optax

import flax, flax.struct as struct, flax.linen as nn
from flax.training.train_state import TrainState
from flax.training import checkpoints

from coin_game_jax import CoinGame
from ipd_jax import IPD

deepmap = jax.tree_util.tree_map
stop = jax.lax.stop_gradient

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

def magic_box(x):
    return jnp.exp(x - stop(x))

def get_gae_advantages(rewards, curr_values, next_values):
    deltas = rewards + args.gamma * next_values - curr_values
    gae = jnp.zeros_like(deltas[0, :])
    deltas = jnp.flip(deltas, axis=0)
    def fn(gae, delta):
        gae = gae * args.gamma * args.gae_lambda + delta
        return gae, gae
    gae, flipped_advantages = jax.lax.scan(fn, gae, deltas, deltas.shape[0])
    advantages = jnp.flip(flipped_advantages, axis=0)
    return advantages

def dice_objective(self_logprobs, other_logprobs, rewards, values):
    cum_discount = args.gamma ** jnp.arange(len(rewards))[:,None]
    discounted_rewards = rewards * cum_discount
    stochastic_nodes = self_logprobs + other_logprobs

    if use_baseline:
        if args.zero_vals:
            values = jnp.zeros_like(values)
        curr_values, next_values = values[:-1], values[1:]

        advantages = get_gae_advantages(rewards, curr_values, next_values)
        discounted_advantages = advantages * cum_discount
        deps_up_to_t = jnp.cumsum(stochastic_nodes, axis=0)  # == `dependencies`?
        deps_less_than_t = deps_up_to_t - stochastic_nodes  # take out the dependency in the given time step

        # Look at Loaded DiCE and GAE papers to see where this formulation comes from
        dice_obj = ((magic_box(deps_up_to_t) - magic_box(deps_less_than_t))
                    * stop(discounted_advantages)).sum(axis=0).mean()
    else:
        # dice objective:
        dependencies = jnp.cumsum(stochastic_nodes, axis=0)
        dice_obj = (magic_box(dependencies) * discounted_rewards).sum(axis=0).mean()
    return -dice_obj  # want to minimize -objective

def compute_objectives(self_logprobs, other_logprobs, rewards, values):
    reward_loss = dice_objective(self_logprobs, other_logprobs, rewards, values)
    val_loss = value_loss(rewards, values) if use_baseline else 0
    return reward_loss, val_loss

def value_loss(rewards, values):
    discounts = args.gamma ** jnp.arange(len(rewards))[:,None]
    if args.use_a2c:  # a2c
        deltas = rewards + args.gamma*values[1:] - curr_values[:-1]
        return huber(deltas).sum(axis=0).mean()
    else:
        # the original value loss is a mix of various amounts of bootstrapping -- the loss
        # term for time step t is a T-t step TD error
        discounted_rewards = rewards * discounts
        R_ts = reverse_cumsum(discounted_rewards, axis=0) / discounts
        final_state_vals = values[-1]
        curr_values = values[:-1]
        final_val_discounted_to_curr = (args.gamma * jnp.flip(discounts, axis=0)) * final_state_vals
        #loss = (R_ts + stop(final_val_discounted_to_curr) - values) ** 2
        loss = huber(R_ts + stop(final_val_discounted_to_curr) - curr_values)
        return loss.sum(axis=0).mean()

def huber(x):
    return jnp.where(jnp.abs(x)<1, x**2, jnp.abs(x))

def act(key, agent, astate, obs):
    new_astate = dict(astate)
    new_astate["pol"], logits = agent.pol.apply_fn(agent.pol.params, obs, astate["pol"])
    key, subkey = jax.random.split(key)
    logits = nn.log_softmax(logits)
    actions = jax.random.categorical(subkey, logits)
    logps = jax.vmap(lambda z, a: z[a])(logits, actions)
    return dict(astate=new_astate, a=actions, l=logits, logp=logps, p=jnp.exp(logits))

def compute_values(agent, obsseq):
    T,B,*_ = obsseq.shape
    valstate = agent.init_state(B)["val"]
    def body_fn(valstate, obs):
        valstate, value = agent.val.apply_fn(agent.val.params, obs, valstate)
        return valstate, value.squeeze(-1)
    _, values = jax.lax.scan(body_fn, valstate, obsseq)
    return values

def compute_probs(key, agent, astate, obsseq):
    T,B,*_ = obsseq.shape
    astate = agent.init_state(B)
    key, subkey = jax.random.split(key)
    @jax.named_call
    def body_fn(carry, obs):
        key, astate = carry
        key, subkey = jax.random.split(key)
        astate, aux = act(subkey, agent, astate, obs)
        return (key, astate), aux["p"]
    obsseq = jnp.stack(obsseq[:args.rollout_len], axis=0) # TODO(cooijmat) stack this in caller
    carry, probs = jax.lax.scan(body_fn, (subkey, astate), obsseq, args.rollout_len)
    return probs

def dicttranspose(dikts):
    assert all(set(dikt) == set(dikts[0]) for dikt in dikts[1:])
    return {key: [dikt[key] for dikt in dikts] for key in dikts[0]}

def env_step(key, env_state, obss, agents, astates):
    key, sk1, sk2, skenv = jax.random.split(key, 4)
    subkeys = [sk1, sk2]
    aux = dicttranspose([act(*args) for args in zip(subkeys, agents, astates, obss)])
    skenv = jax.random.split(skenv, args.batch_size)
    env_state, new_obss, rewards, aux_info = vec_env_step(env_state, *aux["a"], skenv)
    stuff = (key, env_state, new_obss, aux["astate"])
    return stuff, (dict(obss=new_obss, r=rewards, p=aux["p"], logp=aux["logp"], a=aux["a"]), aux_info)

def do_env_rollout(key, agents):
    keys = jax.random.split(key, args.batch_size + 1)
    key, env_subkeys = keys[0], keys[1:]
    env_state, obss = vec_env_reset(env_subkeys)
    astates = [agent.init_state(args.batch_size) for agent in agents]
    @jax.named_call
    def env_step_body(carry, _):
        (key, env_state, obss, astates) = carry
        return env_step(key, env_state, obss, agents, astates)
    carry = (key, env_state, obss, astates)
    carry, (auxseq, aux_info) = jax.lax.scan(env_step_body, carry, None, args.rollout_len)
    obsseq = [jnp.concatenate([obss[i][None], auxseq["obss"][i]], axis=0)
              for i in range(len(agents))]
    return carry, auxseq, obsseq

def kl_div_jax(curr, target):
    # NOTE reverse kl, https://github.com/Silent-Zebra/POLA/issues/5
    return (curr * (jnp.log(curr) - jnp.log(target))).sum(axis=-1).mean()

def eval_vs_alld_selfagent1(stuff, unused):
    key, agent, env_state, obss, astate = stuff

    key, subkey = jax.random.split(key)
    aux = act(subkey, agent, astate, obss[0])
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
    env_state, obss, (r1, r2), aux_info = vec_env_step(env_state, a1, a2, env_subkeys)

    score1 = r1.mean()
    score2 = r2.mean()

    stuff = (key, agent, env_state, obss, astate)
    aux = (score1, score2)
    return stuff, aux

def eval_vs_alld_selfagent2(stuff, unused):
    key, agent, env_state, obss, astate = stuff

    key, subkey = jax.random.split(key)
    aux = act(subkey, agent, astate, obss[1])
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

    env_state, obss, (r1, r2), aux_info = vec_env_step(env_state, a1, a2, env_subkeys)

    score1 = r1.mean()
    score2 = r2.mean()

    stuff = (key, agent, env_state, obss, astate)
    aux = (score1, score2)
    return stuff, aux

def eval_vs_allc_selfagent1(stuff, unused):
    key, agent, env_state, obss, astate = stuff

    key, subkey = jax.random.split(key)
    aux = act(subkey, agent, astate, obss[0])
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

    env_state, obss, (r1, r2), aux_info = vec_env_step(env_state, a1, a2, env_subkeys)

    score1 = r1.mean()
    score2 = r2.mean()

    stuff = (key, agent, env_state, obss, astate)
    aux = (score1, score2)
    return stuff, aux

def eval_vs_allc_selfagent2(stuff, unused):
    key, agent, env_state, obss, astate = stuff

    key, subkey = jax.random.split(key)
    aux = act(subkey, agent, astate, obss[1])
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

    env_state, obss, (r1, r2), aux_info = vec_env_step(env_state, a1, a2, env_subkeys)

    score1 = r1.mean()
    score2 = r2.mean()

    stuff = (key, agent, env_state, obss, astate)
    aux = (score1, score2)
    return stuff, aux

def eval_vs_tft_selfagent1(stuff, unused):
    key, agent, env_state, obss, astate, prev_a, prev_agent_coin_collected_same_col, r1, r2 = stuff

    key, subkey = jax.random.split(key)
    aux = act(subkey, agent, astate, obss[0])
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

    env_state, obss, (r1, r2), aux_info = vec_env_step(env_state, a1, a2, env_subkeys)

    score1 = r1.mean()
    score2 = r2.mean()

    stuff = (key, agent, env_state, obss, astate, a, prev_agent_coin_collected_same_col, r1, r2)
    aux = (score1, score2)
    return stuff, aux

def eval_vs_tft_selfagent2(stuff, unused):
    key, agent, env_state, obss, astate, prev_a, prev_agent_coin_collected_same_col, r1, r2 = stuff

    key, subkey = jax.random.split(key)
    aux = act(subkey, agent, astate, obss[1])
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

    env_state, obss, (r1, r2), aux_info = vec_env_step(env_state, a1, a2, env_subkeys)

    score1 = r1.mean()
    score2 = r2.mean()

    stuff = (key, agent, env_state, obss, astate, a, prev_agent_coin_collected_same_col, r1, r2)
    aux = (score1, score2)
    return stuff, aux

def eval_vs_fixed_strategy(key, agent, strat="alld", self_agent=1):
    keys = jax.random.split(key, args.batch_size + 1)
    key, env_subkeys = keys[0], keys[1:]
    env_state, obss = vec_env_reset(env_subkeys)
    astate = agent.init_state(args.batch_size)

    if strat == "alld":
        stuff = key, agent, env_state, obss, astate
        if self_agent == 1:
            stuff, aux = jax.lax.scan(eval_vs_alld_selfagent1, stuff, None, args.rollout_len)
        else:
            stuff, aux = jax.lax.scan(eval_vs_alld_selfagent2, stuff, None, args.rollout_len)
    elif strat == "allc":
        stuff = key, agent, env_state, obss, astate
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
        stuff = (key, agent, env_state, obss, astate, prev_a,
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
    state, obss = vec_env_reset(unused_keys)
    init_state = env.init_state
    for i in range(2):
        for j in range(2):
            state1 = env.states[i, j]
            for ii in range(2):
                for jj in range(2):
                    state2 = env.states[ii, jj]
                    state_history = [init_state, state1, state2]
                    assert False # FIXME construct flipped state_history for player 2
                    print(state_history)
                    pol_probs1 = compute_probs(jax.random.PRNGKey(0), agents[0], state_history)
                    pol_probs2 = compute_probs(jax.random.PRNGKey(0), agents[1], state_history)
                    print(pol_probs1)
                    print(pol_probs2)

@jit
def train_exploiters_step(key, agents, exploiters):
    exploiters_ = exploiters
    def fn(paramss):
        exploiters = [e.replace_params(params) for e,params in zip(exploiters_,paramss)]
        carry0,auxseq0,obsseq0 = do_env_rollout(key, [exploiters[0],agents[1]])
        carry1,auxseq1,obsseq1 = do_env_rollout(key, [agents[0],exploiters[1]])
        values0 = compute_values(exploiters[0], obsseq0[0])
        values1 = compute_values(exploiters[1], obsseq1[1])
        r0,l0 = compute_objectives(self_logprobs=auxseq0["logp"][0],
                                   other_logprobs=auxseq0["logp"][1],
                                   rewards=auxseq0["r"][0], values=values0)
        r1,l1 = compute_objectives(self_logprobs=auxseq1["logp"][1],
                                   other_logprobs=auxseq1["logp"][0],
                                   rewards=auxseq1["r"][1], values=values1)
        return r0+l0+r1+l1
    grads = jax.grad(fn)([e.extract_params() for e in exploiters])
    exploiters = [exploiter.apply_gradients(grad) for exploiter,grad in zip(exploiters,grads)]
    return exploiters
def train_exploiters(key, agents):
    key, key1,key2 = jax.random.split(key,3)
    exploiters = [Agent.make(key1, action_size, input_size),
                  Agent.make(key2, action_size, input_size)]
    for _ in range(300):  # enough?
        key, subkey = jax.random.split(key)
        exploiters = train_exploiters_step(subkey, agents, exploiters)
    [rx2,ra2],_ = eval_matchup(key, [exploiters[0],agents[1]])
    [ra1,rx1],_ = eval_matchup(key, [agents[0],exploiters[1]])
    return dict(rx1=rx1.mean(), ra1=ra1.mean(), ra2=ra2.mean(), rx2=rx2.mean())

@jit
def eval_matchup(key, agents):
    keys = jax.random.split(key, args.batch_size + 1)
    key, env_subkeys = keys[0], keys[1:]
    env_state, obss = vec_env_reset(env_subkeys)
    astates = [agent.init_state(args.batch_size) for agent in agents]
    key, subkey = jax.random.split(key)
    def body_fn(carry, _):
        (key, env_state, obss, astates) = carry
        return env_step(key, env_state, obss, agents, astates)
    carry = (subkey, env_state, obss, astates)
    carry, (aux, aux_info) = jax.lax.scan(body_fn, carry, None, args.rollout_len)
    return aux["r"], aux_info

@jit
def eval_progress_1(key, agents):
    key, subkey = jax.random.split(key)

    [r1,r2],aux_info = eval_matchup(subkey, agents)

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

    if args.env == 'coin':
        rr, rb, br, bb = aux_info
        rr = rr.sum(axis=0).mean()
        rb = rb.sum(axis=0).mean()
        br = br.sum(axis=0).mean()
        bb = bb.sum(axis=0).mean()
    else:
        rr,rb,br,bb = 4*[None]
    return dict(score1=r1.mean(),score2=r2.mean(),score1rec=score1rec,score2rec=score2rec,
                rr=rr,rb=rb,br=br,bb=bb)

def eval_progress(key, agents):
    key, subkey1, subkey2 = jax.random.split(key,3)
    aux = eval_progress_1(subkey1, agents)
    ex = train_exploiters(subkey2, agents)
    return dict(score1x=ex["rx1"],score1a=ex["ra1"],
                score2x=ex["rx2"],score2a=ex["ra2"],
                **aux)

# outer step functions are called in a plain python loop so should be jitted,
# and input structure should match output structure
@jit
def one_outer_step_update_selfagent1(key, agent1, ref_agents):
    ref_agent1, ref_agent2 = ref_agents
    outer_agent1 = agent1
    def fn(params, key):
        agent1 = outer_agent1.replace_params(params)
        key, subkey = jax.random.split(key)
        agent2_ahead, iaux = inner_update_agent2(subkey, agent1, ref_agent2)
        objective, aux = out_lookahead(key, [agent1, agent2_ahead], ref_agent1, self_agent=1)
        return objective, [aux["loss"], iaux["gradnorms"]]
    key, subkey = jax.random.split(key)
    (_, [loss,inner_gradnorms]), grad = jax.value_and_grad(fn, has_aux=True)(outer_agent1.extract_params(), subkey)
    agent1 = agent1.apply_gradients(grad)
    outer_gradnorms = deepmap(lambda x: (x**2).mean(), grad)
    return (key, agent1, ref_agents), dict(agent1_loss=loss,
                                           outer_agent1_gradnorms=outer_gradnorms,
                                           inner_agent2_gradnorms=inner_gradnorms)

@jit
def one_outer_step_update_selfagent2(key, agent2, ref_agents):
    ref_agent1, ref_agent2 = ref_agents
    outer_agent2 = agent2
    def fn(params, key):
        agent2 = outer_agent2.replace_params(params)
        key, subkey = jax.random.split(key)
        agent1_ahead, iaux = inner_update_agent1(subkey, ref_agent1, agent2)
        objective, aux = out_lookahead(key, [agent1_ahead, agent2], ref_agent2, self_agent=2)
        return objective, [aux["loss"], iaux["gradnorms"]]
    key, subkey = jax.random.split(key)
    (objective, [loss,inner_gradnorms]), grad = jax.value_and_grad(fn, has_aux=True)(outer_agent2.extract_params(), subkey)
    agent2 = agent2.apply_gradients(grad)
    outer_gradnorms = deepmap(lambda x: (x**2).mean(), grad)
    return (key, agent2, ref_agents), dict(agent2_loss=loss,
                                           inner_agent1_gradnorms=inner_gradnorms,
                                           outer_agent2_gradnorms=outer_gradnorms)

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
        (_, aux), grad = jax.value_and_grad(fn,has_aux=True)(agent1.extract_params())
        agent1 = agent1.apply_gradients(grad)
        gradnorms = deepmap(lambda x: (x**2).mean(), grad)
        return (key, agent1), dict(loss=aux["loss"], gradnorms=gradnorms)

    key, subkey = jax.random.split(key)
    carry = (subkey, agent1)
    carry, aux = jax.lax.scan(body_fn, carry, None, args.inner_steps)
    agent1 = carry[1]

    aux = deepmap(lambda x: x.mean(axis=0), aux)  # avg across time
    return agent1, aux

def inner_update_agent2(key, agent1, agent2):
    ref_agent2 = agent2
    agent2 = agent2.replace(pol=TrainState.create(apply_fn=agent2.pol.apply_fn,
                                                  params=agent2.pol.params,
                                                  tx=optax.sgd(learning_rate=args.lr_in)),
                            val=TrainState.create(apply_fn=agent2.val.apply_fn,
                                                  params=agent2.val.params,
                                                  tx=optax.sgd(learning_rate=args.lr_v)))

    def body_fn(carry, _):
        key, agent2 = carry
        key, subkey = jax.random.split(key)
        def fn(params):
            agent2_ = agent2.replace_params(params)
            return in_lookahead(subkey, [agent1, agent2_], ref_agent2, other_agent=2)
        (_,aux), grad = jax.value_and_grad(fn,has_aux=True)(agent2.extract_params())
        agent2 = agent2.apply_gradients(grad)
        gradnorms = deepmap(lambda x: (x**2).mean(), grad)
        return (key, agent2), dict(loss=aux["loss"], gradnorms=gradnorms)

    key, subkey = jax.random.split(key)
    carry = (subkey, agent2)
    carry, aux = jax.lax.scan(body_fn, carry, None, args.inner_steps)
    agent2 = carry[1]

    aux = deepmap(lambda x: x.mean(axis=0), aux)  # avg across time
    return agent2, aux

def in_lookahead(key, agents, ref_agent, other_agent=2):
    carry, auxseq, obsseq = do_env_rollout(key, agents)
    (key, env_state, obss, astates) = carry

    # we are in the inner loop, so `other_agent` is us
    us = other_agent-1
    them = 1-us

    values = compute_values(agents[us], obsseq[us])
    objective,loss = compute_objectives(self_logprobs=auxseq["logp"][us],
                                        other_logprobs=auxseq["logp"][them],
                                        rewards=auxseq["r"][us], values=values)
    # print(f"Inner Agent (Agent {other_agent}) episode return avg {auxseq['r'][us].sum(axis=0).mean()}")

    key, sk1, sk2 = jax.random.split(key, 3)
    probseq = compute_probs(sk1, agents[us], obsseq[us])
    probseq_ref = compute_probs(sk2, ref_agent, obsseq[us])
    kl_div = kl_div_jax(probseq, probseq_ref)

    return (objective + loss + args.inner_beta * kl_div,  # we want to min kl div
            dict(loss=loss))

def out_lookahead(key, agents, ref_agent, self_agent=1):
    carry, auxseq, obsseq = do_env_rollout(key, agents)
    (key, env_state, obss, astates) = carry

    # we are in the outer loop, so `self_agent` is us
    us = self_agent-1
    them = 1-us

    values = compute_values(agents[us], obsseq[us])
    objective,loss = compute_objectives(self_logprobs=auxseq["logp"][us],
                                        other_logprobs=auxseq["logp"][them],
                                        rewards=auxseq["r"][us], values=values)
    # print(f"Agent {self_agent} episode return avg {auxseq['r'][us].sum(axis=0).mean()}")

    key, sk1, sk2 = jax.random.split(key, 3)
    probseq = compute_probs(sk1, agents[us], obsseq[us])
    probseq_ref = compute_probs(sk2, ref_agent, obsseq[us])
    kl_div = kl_div_jax(probseq, probseq_ref)

    return (objective + loss + args.outer_beta * kl_div,
            dict(loss=loss))

def update_agents(key, agents):
    agent1, agent2 = list(agents)

    key, subkey = jax.random.split(key)
    carry = (subkey, agent1, agents)
    for _ in range(args.outer_steps):
        carry, aux1 = one_outer_step_update_selfagent1(*carry)
    agent1 = carry[1]

    key, subkey = jax.random.split(key)
    carry = (subkey, agent2, agents)
    for _ in range(args.outer_steps):
        carry, aux2 = one_outer_step_update_selfagent2(*carry)
    agent2 = carry[1]

    return key, [agent1, agent2], {**aux1, **aux2}

def play(key, update, agents):
    #record = dict()

    print("start iterations with", args.inner_steps, "inner steps and", args.outer_steps, "outer steps:")

    #key, subkey = jax.random.split(key)
    #logframe = eval_progress(subkey,agents)
    #for k,v in logframe.items():
    #    record.setdefault(k,[]).append(v)
    #if args.wandb:
    #    wandb.log(logframe)

    while True:  # update < args.n_update
        key, agents, optstats = update_agents(key, agents)
        optstats = {key: {"_".join(map(str, path)): np.array(value)
                          for path, value in iterate_nested_dict(thing)}
                    for key, thing in optstats.items()}
        update += 1

        # evaluate progress:
        key, subkey = jax.random.split(key)
        logframe = eval_progress(subkey,agents)
        #for k,v in logframe.items():
        #    record[k].append(v)
        logframe["opt"] = optstats
        if args.wandb:
            wandb.log(logframe,step=update)

        # print
        if update % args.print_every == 0:
            def printframe(update,score1,score2,rr,rb,br,bb,score1rec,score2rec,**kwargs):
                print("*" * 10)
                print("Epoch: {}".format(update), flush=True)
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
            printframe(update,**logframe)

        if update % args.checkpoint_every == 0:
            checkpoints.save_checkpoint(ckpt_dir=".",
                                        target=(update,agents),
                                        step=update,
                                        prefix="ckpt_epoch")
            #np.savez_compressed("record.incoming.npz", **record)
            #os.rename("record.incoming.npz", "record.npz")

        if update % 100 == 0:
            paramss = jax.tree_util.tree_map(np.array, [agent.extract_params()["pol"] for agent in agents])
            with open(f"agents_epoch{update}.pkl.incoming", "wb") as file:
                pickle.dump(paramss, file)
            os.rename(f"agents_epoch{update}.pkl.incoming", f"agents_epoch{update}.pkl")

from collections import abc
def iterate_nested_dict(node):
    if not isinstance(node, abc.Mapping):
        yield (), node
    else:
        for key, child in node.items():
            for path, leaf in iterate_nested_dict(child):
                yield (key, *path), leaf

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
    parser.add_argument("--checkpoint_every", type=int, default=50, help="Epochs between checkpoint save")
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
    parser.add_argument("--inspect_ipd", action="store_true", help="Detailed (2 steps + start state) policy information in the IPD with full history")
    parser.add_argument("--layers_before_gru", type=int, default=2, choices=[0, 1, 2], help="Number of linear layers (with ReLU activation) before GRU, supported up to 2 for now")
    parser.add_argument("--contrib_factor", type=float, default=1.33, help="contribution factor to vary difficulty of IPD")
    parser.add_argument("--architecture", type=str, default="rnn", choices="rnn conv".split())
    parser.add_argument("--use_a2c", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    if args.wandb:
        import wandb
        wandb.init(project="pola", resume=True)
        wandb.config.update(args)

    np.random.seed(args.seed)
    if args.env == 'coin':
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
    update,agents = checkpoints.restore_checkpoint(ckpt_dir=".",
                                                   target=(0,agents),
                                                   prefix="ckpt_epoch")
    if update > 0:
      print("resuming after", update, "updates")

    use_baseline = True
    if args.no_baseline:
        use_baseline = False

    assert args.inner_steps >= 1
    # Use 0 lr if you want no inner steps... TODO allow for this functionality (naive learning)?
    assert args.outer_steps >= 1

    play(key, update, agents)
