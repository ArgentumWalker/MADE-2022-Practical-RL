from world.realm import Realm
from world.envs import OnePlayerEnv, VersusBotEnv, TwoPlayerEnv
from world.utils import RenderedEnvWrapper
from world.map_loaders.base import MixedMapLoader
from world.map_loaders.single_team import SingleTeamLabyrinthMapLoader, SingleTeamRocksMapLoader
from world.scripted_agents import ClosestTargetAgent

if __name__ == "__main__":
    # Task 1
    env = OnePlayerEnv(Realm(
        MixedMapLoader((SingleTeamLabyrinthMapLoader(), SingleTeamRocksMapLoader())),
        1
    ))
    env = RenderedEnvWrapper(env)
    agent = ClosestTargetAgent()
    for i in range(4):
        state, info = env.reset()
        agent.reset(state, 0)
        done = False
        while not done:
            state, done, info = env.step(agent.get_actions(state, 0))
        env.render(f"render/one_player_env/{i}")