import numpy as np

from stable_baselines3 import DQN


class CoverageDQN(DQN):

    def __init__(
        self,
        *args,
        coverage_exploration=True,
        **kwargs,
    ):

        self.coverage_exploration = (
            coverage_exploration
        )

        super().__init__(
            *args,
            **kwargs
        )

    def predict(
        self,
        observation,
        state=None,
        episode_start=None,
        deterministic=False,
    ):

        # --------------------------------------------------
        # Coverage-driven exploration
        # --------------------------------------------------

        if (
            self.coverage_exploration
            and not deterministic
        ):

            env = self.get_env()

            if env is not None:

                forced_action = env.env_method(
                    "get_coverage_action"
                )[0]

                return (
                    np.array(
                        [forced_action],
                        dtype=np.int64
                    ),
                    state,
                )

        # --------------------------------------------------
        # Normal DQN prediction
        # --------------------------------------------------

        return super().predict(
            observation,
            state=state,
            episode_start=episode_start,
            deterministic=deterministic,
        )