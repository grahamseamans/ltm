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

    I think it's a bit of a problem of how the memories have returns rather than just rewards
    like we dont know what's a good 
        - one option is just for old memories to slowly get their returns increased.
            - The older the memory the more enticing it is to relive.
            - Then you relive it and realize that maybe it isnt what you thought it was
            - might want to do slow replacement or something so you actually explore these ideas...
            - Make it so that terminal states don't get interesting over time, but allow the things 
            - leading up to them to be interesting.
            - We want to have a library of terminal observations, and then make it so that those
            - arent getting better... (mabye a little just to retest it I guess...)
            OOh There's the close to death rush! - adrenaline junkie
            - This is a reward for being close to a terminal state but not having a termination!
        - just remember saves
            - Save a bunch of terminal states
            - If a trajectory that get's near one of these terminal states but then continues 
                - to a good state? or just continues?
            - Save that trajectory? Save just the area around where it avoided termination?
            - We have no need to save the reward for saves...
            - We could also just save the trajectories leading up to a terminal state in memory
            - we could have a memory just for these sorts of things - terminal states?
            - could have terminal fails and terminal saves?
            - Then we could have everything else - and these would get fonder over time?
            - Then we do 2 memory queries and then its a 3 piece attention at the end?
            - We can just do exploration / novelty as a loss term then too?
        
        - Danger Bit
            - find the end obs from episodes
            - check to see if it's from running out of time
                - if so ignore it
            - stick the obs into the danger table
            - On forward pass:
                - find closest memory in danger table
                - take cosine similarty
                - that's the danger bit
                - append it to the obs or somewhere in the network.

            
        We have memory banks for:
            - terminal states
            - Trajectories that get near terminal states, but then live on
            - Everything else and it's returns (This has nostalgia?)

        - Want to somehow isolate
            - The actually bad thing - falling on your face
            - the thing that led up to it - walking...
            - You want to think, hey maybe walking is a good idea, 
            - can I do it without falling on my face?
        - We want it to avoid catastrophic,
        - But maybe not avoid things that have come before catastrophe?
        - What we want to avoid is everything similar to a loss.
        - But what really is a loss?
        - I honestly think we might just need some sort of planning thing
        - This way we can reward it for getting to new states, but more so finding
        - new connections between known states.
        - So if A -> B, and we know this
        - Then if we find A -> C this is really exiciting...


        - We want to somehow be aware of the causal nature of these things...

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
    x = f"python ltm/test_tianshou_ppo.py --task={task} --logger=wandb --wandb-project=ltm --wandb-run=walker_danger_bit --epoch={epochs}"
    subprocess.run(x, shell=True)
