import os
from osim.env import OsimEnv
import numpy as np
from gym import spaces, logger
from .geohelpers import *
#import sklearn
#import sklearn.preprocessing
import numpy as np
from opensim import Vec3

import sys
import pdb


class EyeHead(OsimEnv):
    print("Eye Version 4.1", file=sys.stderr)
    max_step=0
    model_path = os.path.join(os.path.dirname(__file__), './models/EyeSkullModel_1.osim')
    LR_dist=0
    L_dist=0
    R_dist=0
    ball=0
    lCoord=[]
    rCoord=[]
    Effort=0
    scaler=None
    LACT=[]; PrevLACT=[]
    RACT = []
    PrevRACT = []
    def __init__(self, devang_flag=False, lrdist_flag=False, rlpog_flag=False, torsion_flag=False, scaled_state=True,
                 visualize=True, integrator_accuracy=5e-5, report=None,maxstep=100):
        
        self.devang_flag=devang_flag
        self.lrdist_flag = lrdist_flag
        self.rlpog_flag = rlpog_flag
        self.torsion_flag=torsion_flag
        self.scaled_state=scaled_state
        self.Effort=0
        super(EyeHead, self).__init__(visualize=visualize, integrator_accuracy=integrator_accuracy)

        if visualize:
            self.viz2 = self.osim_model.model.getVisualizer().getSimbodyVisualizer()
            self.viz2.setBackgroundColor(Vec3(0, 0, 0))
            self.viz2.setBackgroundType(0)

        self.max_step=maxstep
        self.osim_model.stepsize=0.01

        #self.reset()
    #function to normalize states
    def scale_state(self,state):                 #requires input shape=(2,)
        if self.scaler!=None:
            scaled = self.scaler.transform([state])
            return np.squeeze(scaled)

    def reset(self, bx=1,by=0,bz=0,project=True, seed=None, init_pose=None, obs_as_dict=False):
        self.LACT = []
        self.PrevLACT = []
        self.RACT = []
        self.PrevRACT = []
        self.lEffort=0
        self.rEffort = 0
        self.istep=0
        self.t = 0
         # initialize state
        self.osim_model.state = self.osim_model.model.initializeState()
        self.osim_model.state.setTime(0)
        self.osim_model.istep = 0
        self.osim_model.model.getCoordinateSet().get("ball_tx").set_locked(False)
        self.osim_model.model.initStateFromProperties(self.osim_model.state)
        self.osim_model.model.getCoordinateSet().get("ball_tx").setDefaultValue(bx)
        self.osim_model.model.initStateFromProperties(self.osim_model.state)
        self.osim_model.model.getCoordinateSet().get("ball_tx").set_locked(True)
        self.osim_model.model.initStateFromProperties(self.osim_model.state)

        self.osim_model.model.getCoordinateSet().get("ball_ty").set_locked(False)
        self.osim_model.model.initStateFromProperties(self.osim_model.state)
        self.osim_model.model.getCoordinateSet().get("ball_ty").setDefaultValue(by)
        self.osim_model.model.initStateFromProperties(self.osim_model.state)
        self.osim_model.model.getCoordinateSet().get("ball_ty").set_locked(True)
        self.osim_model.model.initStateFromProperties(self.osim_model.state)

        self.osim_model.model.getCoordinateSet().get("ball_tz").set_locked(False)
        self.osim_model.model.initStateFromProperties(self.osim_model.state)
        self.osim_model.model.getCoordinateSet().get("ball_tz").setDefaultValue(bz)
        self.osim_model.model.initStateFromProperties(self.osim_model.state)
        self.osim_model.model.getCoordinateSet().get("ball_tz").set_locked(True)
        self.osim_model.model.initStateFromProperties(self.osim_model.state)
        self.osim_model.reset_manager()
        if self.scaled_state:
            return self.scale_state(self.get_observation())
        else:
            return self.get_observation()
    def moveROI(self, bx,by,bz):
        self.osim_model.model.getCoordinateSet().get("ball_tx").set_locked(False)
        self.osim_model.model.initStateFromProperties(self.osim_model.state)
        self.osim_model.model.getCoordinateSet().get("ball_tx").setDefaultValue(bx)
        self.osim_model.model.initStateFromProperties(self.osim_model.state)
        self.osim_model.model.getCoordinateSet().get("ball_tx").set_locked(True)
        self.osim_model.model.initStateFromProperties(self.osim_model.state)

        self.osim_model.model.getCoordinateSet().get("ball_ty").set_locked(False)
        self.osim_model.model.initStateFromProperties(self.osim_model.state)
        self.osim_model.model.getCoordinateSet().get("ball_ty").setDefaultValue(by)
        self.osim_model.model.initStateFromProperties(self.osim_model.state)
        self.osim_model.model.getCoordinateSet().get("ball_ty").set_locked(True)
        self.osim_model.model.initStateFromProperties(self.osim_model.state)

        self.osim_model.model.getCoordinateSet().get("ball_tz").set_locked(False)
        self.osim_model.model.initStateFromProperties(self.osim_model.state)
        self.osim_model.model.getCoordinateSet().get("ball_tz").setDefaultValue(bz)
        self.osim_model.model.initStateFromProperties(self.osim_model.state)
        self.osim_model.model.getCoordinateSet().get("ball_tz").set_locked(True)
        self.osim_model.model.initStateFromProperties(self.osim_model.state)
        self.osim_model.reset_manager()
        if not project:
            return self.get_state_desc()
        if obs_as_dict:
            return self.get_observation_dict()
        if self.scaled_state:
            return self.scale_state(self.get_observation())
        else:
            return self.get_observation()
        
    def step(self, action, project=True,obs_as_dict=False):
        self.istep += 1
        self.t+=self.osim_model.stepsize
        observation, reward, done, info = super(EyeHead, self).step(action, project=project, obs_as_dict=obs_as_dict)
        if self.scaled_state:
            observation= self.scale_state(observation)
        self.PrevLACT=self.LACT
        self.PrevRACT=self.RACT
        return observation, reward, done, info

    def get_reward__(self):
        rreward=0
        lreward=0
        if self.devang_flag:
            lreward += 10/(self.left_devang)
            rreward += 10/(self.right_devang)

        if self.lrdist_flag:
            if self.LR_dist>0.06 and self.left_devang>self.right_devang:
                lreward-=2 
            elif self.LR_dist<=0.06 :
                lreward+=2

            if self.LR_dist > 0.06 and self.left_devang < self.right_devang:
                rreward-=2
            elif self.LR_dist <= 0.06:
                rreward += 2
                
        if self.rlpog_flag:
            if self.right_devang>self.left_devang and self.rpog[1]>self.lpog[1]:
                rreward-=2
            elif abs(self.rpog[1]-self.lpog[1]) <= 0.005:
                rreward += 2

            if self.right_devang < self.left_devang and self.rpog[1] < self.lpog[1]:
                lreward-=2
            elif abs(self.rpog[1]-self.lpog[1])<=0.005:
                lreward+=2
            # prevent crossed eyes
            if self.rpog[2]<self.lpog[2] and self.right_devang>self.left_devang :
                rreward-=2
            elif self.rpog[2] >= self.lpog[2]:
                rreward += 2
            
            if self.rpog[2] < self.lpog[2] and self.right_devang < self.left_devang:
                lreward -=2
            elif self.rpog[2] >= self.lpog[2] :
                lreward+=2

        if self.torsion_flag:
            if np.round(abs(self.rCoord[0]),decimals=3)<=np.round(abs(self.RLTorsion),decimals=3)*1.1:
                rreward +=2 #/(abs(abs(self.rCoord[0])-(abs(self.RLTorsion)))+1)
            else:
                rreward-=2
            if np.round(abs(self.lCoord[0]),decimals=3)<=np.round(abs(self.LLTorsion),decimals=3)*1.1:
                lreward += 2#/(abs(abs(self.lCoord[0])-(abs(self.LLTorsion)))+1)
            else:
                lreward-=2

        if self.right_devang <= 2 and\
           self.LR_dist <= 0.06 and\
           self.rpog[2] >= self.lpog[2] and\
           abs(self.rpog[1] - self.lpog[1]) <= 0.005\
           and abs(self.rCoord[0]) <= abs(self.RLTorsion) * 1.1:

            print("rmax")
            rreward = 100

        if self.left_devang <= 2 and self.LR_dist <= 0.06 and self.rpog[2] >= self.lpog[2] and abs(self.rpog[1]-self.lpog[1]) <= 0.01 and  abs(self.lCoord[0])<=abs(self.LLTorsion)*1.1:
            lreward=100
            print("lmax")

        self.lEffort += self.LSumACT*self.osim_model.stepsize
        self.rEffort += self.RSumACT*self.osim_model.stepsize
        lreward -= self.lEffort
        rreward-=self.rEffort
        #print((rreward,lreward))
        return (rreward,lreward)  # ,lrdistreward,lrpogreward,rtrewatd,ltreward)

    def is_done(self):
        if self.istep>=self.max_step:
            return True
        return False

    def get_observation_space_size(self):
        return 27

    def get_observation(self):
        obs_dict = self.get_observation_dict()

        self.lCoord = abs(np.array(obs_dict['joint_pos']['leyeToGround']))
        self.rCoord = abs(np.array(obs_dict['joint_pos']['reyeToGround']))
        ball = np.array(obs_dict['body_pos']['ball'])
        self.ball=ball
        self.lpog = np.array(obs_dict['markers']['LPOG']['pos'])
        self.rpog = np.array(obs_dict['markers']['RPOG']['pos'])
        reye = np.array(obs_dict['markers']['REye']['pos'])
        leye = np.array(obs_dict['markers']['LEye']['pos'])
        self.LR_dist = math.sqrt(np.sum((self.rpog-self.lpog)**2))
        self.L_dist = math.sqrt(np.sum((self.lpog-ball)**2))
        self.R_dist = math.sqrt(np.sum((self.rpog-ball)**2))
        self.RLTorsion = (self.rCoord[1]*self.rCoord[2])/2  # -HV/2
        self.LLTorsion = (self.lCoord[1]*self.lCoord[2])/2  # -HV/2
        self.left_devang = float(Ang_3points(leye, self.lpog, ball))
        self.right_devang = float(Ang_3points(reye, self.rpog, ball))
        self.LSumACT=0; self.RSumACT=0
        self.LACT=[]; self.RACT=[]

        rres = np.array([])
        lres = np.array([])
        rres=np.append(rres,obs_dict['body_pos']['ball'])
        #lres=np.append(lres,obs_dict['body_pos']['ball'])
        for MUS in ['rLR', 'rMR', 'rSR', 'rIR', 'rSO', 'rIO']:
            rres = np.append(rres, obs_dict['muscles'][MUS]['activation'])
            self.RACT.append(obs_dict['muscles'][MUS]['activation'])
            self.RSumACT += obs_dict['muscles'][MUS]['activation']**2
        for MUS in ['lLR', 'lMR', 'lSR', 'lIR', 'lSO', 'lIO']:
            lres = np.append(lres, obs_dict['muscles'][MUS]['activation'])
            self.LACT.append(obs_dict['muscles'][MUS]['activation'])
            self.LSumACT += obs_dict['muscles'][MUS]['activation']**2

        rres=np.append(rres, obs_dict['markers']['RPOG']['pos'])
        lres=np.append(lres, obs_dict['markers']['LPOG']['pos'])
        rres = np.append(rres, obs_dict['joint_pos']['reyeToGround'][0:3])
        lres = np.append(lres, obs_dict['joint_pos']['leyeToGround'][0:3])

        #pdb.set_trace()

        #return np.concatenate((rres,lres))

        rtn_obs_dict = dict(targetpos=np.array(obs_dict['body_pos']['ball']),
                            lpog=np.array(obs_dict['markers']['RPOG']['pos']),
                            rpog=np.array(obs_dict['markers']['LPOG']['pos']),
                            leyeh=np.array(obs_dict['joint_pos']['reyeToGround'][0:3]),
                            reyeh=np.array(obs_dict['joint_pos']['leyeToGround'][0:3]),
                            lmus=np.array([obs_dict['muscles'][MUS]['activation'] for MUS in 'lLR lMR lSR lIR lSO lIO'.split()]),
                            rmus=np.array([obs_dict['muscles'][MUS]['activation'] for MUS in 'rLR rMR rSR rIR rSO rIO'.split()]))

        self.obs_names = [k for k in rtn_obs_dict.keys()]
        self.obs_shapes = [v.shape for _, v in rtn_obs_dict.items()]
        self.act_names = 'rLR rMR rSR rIR rSO rIO lLR lMR lSR lIR lSO lIO'.split()
        
        return rtn_obs_dict
