import os
import subprocess

root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
python_exec = os.path.join(root, ".venv", "Scripts", "python.exe")
log_folder = os.path.join(root, "logs", "eval")

discrete_envs = ["Acrobot-v1", "CartPole-v0", "CartPole-v1", "MountainCar-v0"]
continuous_envs = ["MountainCarContinuous-v0", "Pendulum-v1"]
all_envs = discrete_envs + continuous_envs

algo_envs = {
    "reinforce": all_envs,
    "minibatch_reinforce": all_envs,
    "a2c": all_envs,
    "ppo": all_envs,
    "trpo": all_envs,
    "ars": continuous_envs,
    "ddpg": continuous_envs,
    "sac": continuous_envs,
    "td3": continuous_envs,
    "tqc": continuous_envs,
}

seeds = [1, 2]

if not os.path.isdir(log_folder):
    os.makedirs(log_folder, exist_ok=True)

for algo, envs in algo_envs.items():
    for env_id in envs:
        for seed in seeds:
            cmd = [
                python_exec,
                os.path.join(root, "train.py"),
                "--algo",
                algo,
                "--env",
                env_id,
                "--log-folder",
                log_folder,
                "--n-timesteps",
                "100000",
                "--seed",
                str(seed),
                "--eval-freq",
                "25000",
                "--eval-episodes",
                "5",
                "--verbose",
                "0",
            ]
            print("Running:", " ".join(cmd))
            try:
                subprocess.run(cmd, cwd=root, check=True)
            except subprocess.CalledProcessError as e:
                print(f"FAILED: {algo} {env_id} seed={seed}")
                print(e)
                raise

print("Batch evaluation finished")
