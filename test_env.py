from pytorch_lightning import seed_everything
from mode.evaluation.utils import get_default_mode_and_env
from hydra import compose, initialize
from mode.evaluation.multistep_sequences import get_sequences
from mode.evaluation.utils import get_env_state_for_initial_condition

display = print

with initialize(config_path="conf"):
    cfg = compose(config_name="mode_evaluate")

seed_everything(0, workers=True)  # type:ignore
lang_embeddings = None
env = None
results = {}
plans = {}

print(cfg.device)
model, env, _, lang_embeddings = get_default_mode_and_env(
    cfg.train_folder,
    cfg.dataset_path,
    cfg.checkpoint,
    env=env,
    lang_embeddings=lang_embeddings,
    eval_cfg_overwrite=cfg.eval_cfg_overwrite,
    device_id=cfg.device,
)

del model

import pyhash; from optree import tree_map; hasher = pyhash.fnv1_64(); teha = lambda y: tree_map(lambda x: hasher(str(x.tolist() if hasattr(x, "tolist") else x)), y)

initial_state, eval_sequence = get_sequences(1)[0]

robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

obs = env.get_obs()
obs_raw = env.last_raw_obs

start_info = env.get_info()

display(hasher(str(start_info)))
# display(teha(obs_raw))
display(hasher(str(teha(obs_raw))))

goal = {}
goal['lang_text'] = "langinfo"

display(hasher(str(goal)))
display(teha(obs))