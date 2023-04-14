import jax
import jax.numpy as jnp
from typing import NamedTuple
from typing import Tuple


class CoinGameState(NamedTuple):
    red_pos: jnp.ndarray
    blue_pos: jnp.ndarray
    coin_pos: jnp.ndarray
    is_red_coin: jnp.ndarray
    step_count: jnp.ndarray


# class CoinGameJAX:
MOVES = jax.device_put(
    jnp.array(
        [
            [0, 1], # right
            [0, -1], # left
            [1, 0], # down
            [-1, 0], # up
        ]
    )
)


class CoinGame:
    def __init__(self, grid_size=3):
        self.grid_size = grid_size

    def generate_coins(self, subkey, red_pos_flat, blue_pos_flat):
        subkey, sk1, sk2 = jax.random.split(subkey, 3)
        coin_pos_max_val = self.grid_size ** 2 - 2
        coin_pos_max_val += (red_pos_flat == blue_pos_flat)
        stacked_pos = jnp.stack((red_pos_flat, blue_pos_flat))
        min_pos = jnp.min(stacked_pos)
        max_pos = jnp.max(stacked_pos)
        coin_pos_flat = jax.random.randint(sk2, shape=[], minval=0,
                                           maxval=coin_pos_max_val)
        coin_pos_flat += (coin_pos_flat >= min_pos)
        coin_pos_flat += jnp.logical_and((coin_pos_flat >= max_pos), (red_pos_flat != blue_pos_flat))
        coin_pos = jnp.stack((coin_pos_flat // self.grid_size, coin_pos_flat % self.grid_size))
        return coin_pos

    def reset(self, subkey) -> Tuple[jnp.ndarray, CoinGameState]:
        subkey, sk1, sk2, sk3 = jax.random.split(subkey, 4)
        red_pos_flat = jax.random.randint(sk1, shape=[], minval=0, maxval=self.grid_size**2)
        red_pos = jnp.stack((red_pos_flat // self.grid_size, red_pos_flat % self.grid_size))
        blue_pos_flat = jax.random.randint(sk2, shape=[], minval=0, maxval=self.grid_size**2)
        blue_pos = jnp.stack((blue_pos_flat // self.grid_size, blue_pos_flat % self.grid_size))
        coin_pos = self.generate_coins(sk3, red_pos_flat, blue_pos_flat)
        step_count = jnp.zeros([], dtype="int32")
        is_red_coin = jax.random.randint(sk3, shape=[], minval=0, maxval=2).astype("bool")
        state = CoinGameState(red_pos, blue_pos, coin_pos, is_red_coin, step_count)
        obss = self.state_to_obss(state)
        return state, obss

    def state_to_obss(self, state: CoinGameState) -> jnp.ndarray:
        obs = jnp.zeros((4, self.grid_size, self.grid_size))
        obs = obs.at[0, state.red_pos[0], state.red_pos[1]].set(1.0)
        obs = obs.at[1, state.blue_pos[0], state.blue_pos[1]].set(1.0)
        # red coin pos
        obs = obs.at[2, state.coin_pos[0], state.coin_pos[1]].set(state.is_red_coin)
        # blue coin pos
        obs = obs.at[3, state.coin_pos[0], state.coin_pos[1]].set(~state.is_red_coin)
        obss = [obs, obs[((1,0,3,2),)]] # flip perspective for blue player
        return [obs.reshape(4*self.grid_size**2) for obs in obss] # flatten

    def step(self, state: CoinGameState, action_0: int, action_1: int, subkey: jnp.ndarray) -> Tuple[jnp.ndarray, list]:
        new_red_pos = (state.red_pos + MOVES[action_0]) % self.grid_size
        new_blue_pos = (state.blue_pos + MOVES[action_1]) % self.grid_size
        rr = jnp.all(new_red_pos  == state.coin_pos, axis=-1) & state.is_red_coin
        rb = jnp.all(new_red_pos  == state.coin_pos, axis=-1) &~state.is_red_coin
        br = jnp.all(new_blue_pos == state.coin_pos, axis=-1) & state.is_red_coin
        bb = jnp.all(new_blue_pos == state.coin_pos, axis=-1) &~state.is_red_coin
        red_reward = (rr+rb-2*br).astype("float32")
        blue_reward = (bb+br-2*rb).astype("float32")

        need_new_coins = rr | rb | br | bb
        new_red_pos_flat = new_red_pos[0] * self.grid_size + new_red_pos[1]
        new_blue_pos_flat = new_blue_pos[0] * self.grid_size + new_blue_pos[1]
        generated_coins = self.generate_coins(subkey, new_red_pos_flat, new_blue_pos_flat)
        new_coin_pos = jnp.where(need_new_coins, generated_coins, state.coin_pos)
        new_is_red_coin = jnp.where(need_new_coins, ~state.is_red_coin, state.is_red_coin)

        new_state = CoinGameState(new_red_pos, new_blue_pos, new_coin_pos, new_is_red_coin,
                                  state.step_count+1)
        obss = self.state_to_obss(new_state)
        return new_state, obss, (red_reward, blue_reward), (rr, rb, br, bb)

    def get_moves_shortest_path_to_coin(self, state, red_agent_perspective=True):
        # Ties broken arbitrarily, in this case, since I check the vertical distance later
        # priority is given to closing vertical distance (making up or down moves)
        # before horizontal moves
        agent_pos = state.red_pos if red_agent_perspective else state.blue_pos

        horiz_dist_right = (state.coin_pos[:,1] - agent_pos[:,1]) % self.grid_size
        horiz_dist_left = (agent_pos[:,1] - state.coin_pos[:,1]) % self.grid_size
        vert_dist_down = (state.coin_pos[:,0] - agent_pos[:,0]) % self.grid_size
        vert_dist_up = (agent_pos[:,0] - state.coin_pos[:,0]) % self.grid_size

        actions = jax.random.randint(jax.random.PRNGKey(0), shape=(1,), minval=0, maxval=0)
        actions = jnp.where(horiz_dist_right < horiz_dist_left, 0, actions)
        actions = jnp.where(horiz_dist_left < horiz_dist_right, 1, actions)
        actions = jnp.where(vert_dist_down < vert_dist_up, 2, actions)
        actions = jnp.where(vert_dist_up < vert_dist_down, 3, actions)

        return actions

    def get_moves_away_from_coin(self, moves_towards_coin):
        return jnp.array([1,0,3,2])[moves_towards_coin]

    def get_coop_action(self, state, red_agent_perspective=True):
        # move toward coin if same colour, away if opposite colour
        # An agent that always does this is considered to 'always cooperate'
        moves_towards_coin = self.get_moves_shortest_path_to_coin(state, red_agent_perspective=red_agent_perspective)
        moves_away_from_coin = self.get_moves_away_from_coin(moves_towards_coin)
        is_my_coin = state.is_red_coin if red_agent_perspective else ~state.is_red_coin
        return jnp.where(is_my_coin, moves_towards_coin, moves_away_from_coin)

