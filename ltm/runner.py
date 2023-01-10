import subprocess

"""
code todo:
    make it so that everythign is actually changeable by args

    figure out how to have strings change the net class

    So what is up with it doing dumb stuff if it has memory?
        maybe it needs to use the stupider decide?
        Maybe the memory needs to only return actions?
            Backprop is messy from obs and ret which aren't actions?
        returning only actions fixed this, but not it's very cautios
    
    So we can make it bored and we can make it explore, but how
    do we make it go back to doing something good from before?
    How to we make it want to try out other good strategies?

    give it a bonus to reward if it's in a cluster of old memories!
    - this way it will want to revisit old areas...
    - if the 10 nearest memories are older then that's better...
    - could also scale it by the ret's from those 10 closest ones...
        - dont want to revisit bad memories, want to revisit good ones?
    - average the times of the two memories when you're removing one...
        - can randomly choose which mem to get rid of too
        - both do the same thing...
    - some function where the diff between makes it larger. (subtraction? div?)
    - it would be cool to try averaging the mems we're getting rid of too...
    - I want to have it somehow really dislike strategies that sometimes dont work
    - like I want it to like a consistent strategy...
    - That is to say I'm not saying I want it to consistently pick one strategy
    - I want it to pick a consistentry strategy. A dependable strategy...
    - is there a way to make a penalty or loss for this?
    - when we're finding memories shouldnt we not keep the old ones based on returns?
    - Like if something has gone badly in the past dont we want to get rid of that?
    - Things are different now. I think we just want to do newest obs action pairs?

    I think it's a bit of a problem

testing plans:
    check half cheeta with other way to do the decide (attn_decide)

"""

tasks = [
    # "Ant-v4",
    # "HalfCheetah-v4",
    "Walker2d-v4",
    # "Hopper-v4",
    # "Humanoid-v4",
]
epochs = 40

runs = []
for task in tasks:
    x = f"python ltm/test_tianshou_ppo.py --task={task} --logger=wandb --wandb-project=ltm --wandb-run=walker_Hybrid_boredom_nostalgiaV2_dream_!ret --epoch={epochs}"
    subprocess.run(x, shell=True)
