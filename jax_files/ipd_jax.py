import jax.numpy as jnp

class IPD:
    """
    A two-agent vectorized environment.
    Possible actions for each agent are (C)ooperate and (D)efect.
    """
    def __init__(self, init_state_coop=False, contrib_factor=1.33):
        cc = contrib_factor - 1.
        dd = 0.
        dc = contrib_factor / 2. # I defect when opp coop
        cd = contrib_factor / 2. - 1 # I coop when opp defect
        self.payout_mat = jnp.array([[dd, dc],[cd, cc]])
        # One hot state representation because this would scale to n agents
        self.states = jnp.array([[[1, 0, 0, 1, 0, 0], #DD (WE ARE BACK TO THE REPR OF FIRST AGENT, SECOND AGENT)
                                  [1, 0, 0, 0, 1, 0]], #DC
                                 [[0, 1, 0, 1, 0, 0], #CD
                                  [0, 1, 0, 0, 1, 0]]]) #CC
        if init_state_coop:
            self.init_state = jnp.array([0, 1, 0, 0, 1, 0])
        else:
            self.init_state = jnp.array([0, 0, 1, 0, 0, 1])

    def reset(self, unused_key):
        obss = [self.init_state, self.flip_obs(self.init_state)]
        return self.init_state, obss

    def step(self, unused_state, ac0, ac1, unused_key):
        r0 = self.payout_mat[ac0, ac1]
        r1 = self.payout_mat[ac1, ac0]
        state = self.states[ac0, ac1]
        observation = state
        reward = (r0, r1)
        obss = [state, self.flip_obs(state)]
        return state, obss, reward, None

    @classmethod
    def flip_obs(cls, obs):
        return obs.reshape([2,3])[::-1].reshape([6])

class MiladIPD:
    def __init__(self, contrib_factor=1.33):
        # NOTE in milad's version, cooperate is 0 and defect is 1
        cc = contrib_factor - 1.
        dd = 0.
        dc = contrib_factor / 2. # I defect when opp coop
        cd = contrib_factor / 2. - 1 # I coop when opp defect
        self.payout_mat = jnp.array([[cc, cd],[dc, dd]])
        self.states = jnp.array([[[0, 1, 0, 1],   #DD
                                  [0, 1, 1, 0]],  #DC
                                 [[1, 0, 0, 1],   #CD
                                  [1, 0, 1, 0]]]) #CC
        self.init_state = jnp.array([0, 0, 0, 0])

    def reset(self, unused_key):
        obss = [self.init_state, self.flip_obs(self.init_state)]
        return self.init_state, obss

    def step(self, unused_state, ac0, ac1, unused_key):
        r0 = self.payout_mat[ac0, ac1]
        r1 = self.payout_mat[ac1, ac0]
        state = self.states[ac0, ac1]
        observation = state
        reward = (r0, r1)
        obss = [state, self.flip_obs(state)]
        return state, obss, reward, None

    @classmethod
    def flip_obs(cls, obs):
        return obs.reshape([2,2])[::-1].reshape([4])

