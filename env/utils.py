'''
Author: Chris Yoon
Github: https://github.com/cyoon1729/Policy-Gradient-Methods.git
'''

import numpy as np
import math
import gym
#import torch
import pdb
import sys

from collections.abc import Mapping

#print('wArgsTools v0.0.4 is still under development!!!!')

class struct(Mapping):
    def __init__(somthing_completely_different_from_self,
                 **kwargs):
        somthing_completely_different_from_self.__dict__.update(kwargs)

    def __iadd__(somthing_completely_different_from_self, moreargs):
        somthing_completely_different_from_self.__dict__.update(moreargs.__dict__)
        return somthing_completely_different_from_self
        
    def __len__(somthing_completely_different_from_self):
        return len(vars(somthing_completely_different_from_self).keys())

    def __getitem__(somthing_completely_different_from_self, name):
        return vars(somthing_completely_different_from_self)[name]

    def keys(somthing_completely_different_from_self):
        return vars(somthing_completely_different_from_self).keys()
        
    def __iter__(somthing_completely_different_from_self):
        return iter(somthing_completely_different_from_self.keys())

    def __repr__(somthing_completely_different_from_self):
        return str(somthing_completely_different_from_self.__dict__)



def mini_batch_train(env, agent, max_episodes, max_steps, batch_size, visualize=False):
    episode_rewards = []

    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            #print(step)
            #pdb.set_trace()
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)

            if visualize:
                env.render()
            
            agent.push_into_replay_buffer(state, action, reward, next_state, done)
            episode_reward += reward

            if len(agent.replay_buffer) > batch_size:
                agent.update(batch_size)   

            if done or step == max_steps-1:
                episode_rewards.append(episode_reward)
                print("Episode {}:\t{: d}\t {: 4.3f}".format(episode, step+1, episode_reward))
                break

            state = next_state

    return episode_rewards

def mini_batch_train_frames(env, agent, max_frames, batch_size):
    episode_rewards = []
    state = env.reset()
    episode_reward = 0

    for frame in range(max_frames):
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.replay_buffer.push(state, action, reward, next_state, done)
        episode_reward += reward

        if len(agent.replay_buffer) > batch_size:
            agent.update(batch_size)   

        if done:
            episode_rewards.append(episode_reward)
            print("Frame " + str(frame) + ": " + str(episode_reward))
            state = env.reset()
            episode_reward = 0
        
        state = next_state
            
    return episode_rewards

# process episode rewards for multiple trials
def process_episode_rewards(many_episode_rewards):
    minimum = [np.min(episode_reward) for episode_reward in episode_rewards]
    maximum = [np.max(episode_reward) for episode_reward in episode_rewards]
    mean = [np.mean(episode_reward) for episode_reward in episode_rewards]

    return minimum, maximum, mean


class stdlog:
    lines = []
    text = ''
    stdout = sys.stdout
    stderr = sys.stderr
    tee = None
    tee_options = {'stdout': sys.stdout,
                   'stderr': sys.stderr}
    
    def __init__(self, enabled=False, tee=None):
        self.__dict__.update(locals())
        if enabled:
            self.enable()

    def disable(self):
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        return self
        
    def enable(self):
        sys.stdout = self
        sys.stderr = self
        return self
    
    def write(self, v):
        self.lines.append(str(v))
        if self.tee is not None:
            print(v, file=self.tee_options[self.tee], end='')

    def flush(self):
        self.text = ''.join(self.lines)
        return self.text

    def __repr__(self):
        return self.flush()

    def __str__(self):
        return self.__repr__()


    
def _genkeys(d, key=''): 
    packedkeys='' 
    for k, v in d.items(): 
        if isinstance(v, dict): 
            k=_genkeys(v, key+k+'/') 
            packedkeys+=k+',' 
        else: 
            packedkeys+=key+k+','

    # print(packedkeys)
      
    return packedkeys.replace(',,', ',') 

