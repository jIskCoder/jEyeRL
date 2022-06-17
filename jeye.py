
##Written by Mo Hossny

import env.seed
#from env.seed import __set_seed__
import pdb

import sys
import gym
import copy
import torch
import numpy as np
#import pandas as pd

from env import WrappedEyeHeadF
from networks import Actor
from ddpg import DDPGAgent

env.seed.__set_seed__(0)

def train(env, agent,max_episodes=100,max_steps=10,batch_size=64):
    episode_rewards = []
    dy=0
    dz=0
    max_reward=-1000000
    for episode in range(max_episodes):
        if episode%2==0:
            dy=np.random.uniform(-0.1,0.1)
            dz = 0
        else :
            dy=0
        dz=np.random.uniform(-0.1,0.1)
        x = np.random.uniform(0.99, 1.01)
        y = np.random.uniform(dy-0.01, 0.01+dy)
        z = np.random.uniform(dz-0.01, 0.01+dz)
        state = env.reset(x,y,z)
        episode_reward = 0
        
        for step in range(max_steps):
            action = agent.get_action(torch.FloatTensor(state))
            action=action.flatten().tolist()
            #print(action.flatten().tolist(), state)
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.push(np.squeeze(state), action, reward, np.squeeze(next_state), done)
            episode_reward += reward

            if len(agent.replay_buffer) > batch_size:
                agent.update(batch_size)   

            if done or step == max_steps-1:
                if episode_reward>max_reward:
                    max_reward=episode_reward

                    path = "./trainedmodels/actor_{}.ddpg".format(np.round(episode_reward,decimals=2))
                    torch.save(agent.actor.state_dict(),path)
                episode_rewards.append(episode_reward)
                print("Episode " + str(episode) + ": " + str(episode_reward))
                break

            state = next_state

    return episode_rewards
    
def play(env, ctrl,dy,dz, n_episodes=50, n_episode_len=100, visualize=True):
    log = []
    for epsd in range(n_episodes):
        x = np.random.uniform(0.99, 1.01)
        y = np.random.uniform(dy-0.01, 0.01+dy)
        z = np.random.uniform(dz-0.01, 0.01+dz)
        state = env.reset(x,y,z)
        epsd_rwrd = 0
        for stp in range(n_episode_len):
            action = ctrl(torch.FloatTensor(state))
            new_state, reward, done, info = env.step(action.flatten().tolist())
            #print(np.concatenate(([epsd], [stp],[dy],[dz], np.squeeze(new_state).tolist(), [reward])))
            log.append(np.concatenate(([epsd], [stp],[dy],[dz], np.squeeze(new_state).tolist(), [reward],[env.L_dist,env.R_dist])))
            if visualize:
                env.render()

            epsd_rwrd += reward
            state = new_state
    with open('./data_'+str(np.round(dy,decimal=1))+"_"+str(np.round(dz,decimal=1))+".csv", 'w') as f:
        f.write("Episode, Step, dy,dz,ballx,bally,ballz,rpogx,rpogy,rpogz,lpogx,lpogy,lpogz,rcoordx,rcoody,rcoordz,lcoordx,lcoordy,lcoordz,rlr,rmr,rsr,rir,rso,rio,llr,lmr,lsr,lir,lso,lio,reward,Ldist,Rdist\n")
        np.savetxt(f, log, delimiter=',',fmt='%5s')

       
def main(filename=''):
    env = WrappedEyeHeadF(episode_len=100, by=(-.16, .16), bz=(-.32, .32),
                          visualize=True, scaled_state=False)


    agent=DDPGAgent(env,0.99,1e-2,1000000,1e-3,1e-3)
    #Remove next commentted code to start training
    #Don't forget to create directory (trainedmodels) to save models
    #train(env,agent,1000,100)
    ctrl = Actor()
    state_dict = torch.load('jeyeF3216__99___-8.ddpg')
    ctrl.load_state_dict(state_dict['actor'])

    print(ctrl)
    
    #pdb.set_trace()
    play(env, ctrl,0,0)
    play(env, ctrl,0,0.1)
    play(env, ctrl,0,-0.1)
    play(env, ctrl,0.1,0)
    play(env, ctrl,-0.1,0)
    play(env, ctrl,0.1,0.1)
    play(env, ctrl,0.1,-0.1)
    play(env, ctrl,-0.1,0.1)
    play(env, ctrl,-0.1,-0.1)
    pass


if __name__ == '__main__':
    #main(sys.argv[1])
    main()










    
