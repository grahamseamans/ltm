import subprocess

tasks = [
    "Ant-v4",
    # "HalfCheetah-v4",
    # "Hopper-v4",
    # "HumanoidStandup-v4",
]
files = [
    ("mem_new", "ltm/test_tianshou_ppo.py"),
    # ("regular", "/home/cibo/code/ltm/tianshou/examples/mujoco/mujoco_ppo.py"),
]
epochs = 25


runs = []
for task in tasks:
    for runner in files:
        # x = f"python {runner[1]} --task={task} --logger=wandb --wandb-project=ltm --wandb-run={runner[0]}-{task} --epoch={epochs}"
        x = f"python {runner[1]} --task={task} --logger=wandb --wandb-project=ltm --wandb-run=vanilla --epoch={epochs}"
        subprocess.run(x, shell=True)
