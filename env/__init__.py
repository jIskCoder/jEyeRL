import pdb
import opensim
import numpy as np
from .jEye_v41 import EyeHead
from .utils import struct


class WrappedEyeHeadF(EyeHead):
    def __init__(self, episode_len=10000, bx=(0.99, 1.01), by=(0., 0.), bz=(0., 0.), **kwargs):
        EyeHead.__init__(self, **kwargs)
        self.args = struct(**locals())

        obs = self.reset()
        self.obs_names = ['obs']

        #pdb.set_trace()
        self.obs_shapes = [(sum(sum(self.obs_shapes, ())),)]
        #self.act_names = ['Act%d' %
        #                  i for i in range(self.action_space.shape[-1])]

        self.observation_space = self.observation_space
        self.action_space = self.action_space

        if episode_len > 0:
            self.time_limit = episode_len
            self.max_step = episode_len

        #self.osim_model.stepsize = .001
        #pdb.set_trace()
        
        if self.visualize:
            from opensim import Vec3
            self.viz = self.osim_model.model.getVisualizer()
            self.viz2 = self.viz.getSimbodyVisualizer()
            self.viz2.setBackgroundColor(Vec3(0, 0, 0))
            self.viz2.setBackgroundType(0)
            self.viz2.setDesiredFrameRate(1000)  # 1/self.osim_model.stepsize)
            self.viz2.setMode(2)

            self.viz2.drawFrameNow(self.osim_model.state)

            def reset_view():
                from numpy import pi
                from opensim import Rotation, Transform
                R = Rotation(pi/2, Vec3(0, 1, 0))
                #self.viz2.zoomCameraToShowAllGeometry()
                self.viz2.setCameraClippingPlanes(0, 1000)
                self.viz2.pointCameraAt(Vec3(0, 1, 0), Vec3(0, 1, 0))
                self.viz2.setCameraFieldOfView(.1*pi/180.)
                self.viz2.setCameraTransform(Transform(R, Vec3(700, 1.05, 0)))

                self.viz2.setShowSimTime(True)
                self.viz2.setShowFrameRate(True)
                self.viz2.setShowFrameNumber(True)
                self.viz2.setShowShadows(False)

            reset_view()

    def get_reward(self):
        obs_dict = self.get_observation_dict()

        rreward=0
        lreward=0

        ball = np.array(obs_dict['body_pos']['ball'])
        lpog = np.array(obs_dict['markers']['LPOG']['pos'])
        rpog = np.array(obs_dict['markers']['RPOG']['pos'])
        reye = np.array(obs_dict['markers']['REye']['pos'])
        leye = np.array(obs_dict['markers']['LEye']['pos'])

        lpogdist = ((ball - lpog)**2).sum()**.5
        rpogdist = ((ball - rpog)**2).sum()**.5
        lrdist = ((lpog - rpog)**2).sum()**.5
        lrydist = abs((lpog[1] - rpog[1]))
        
        cost = ((0, 0),
                (16, lpogdist**2),
                (16, rpogdist**2),
                (32, lrdist**2),
                (64, lrydist**2),
                (64, lpog[2]<rpog[2]),
                (0, 0))

        #print(*map(lambda x: x[-1], cost))
        
        cost = sum(map(lambda x: x[0] * x[1], cost))# / sum(map(lambda x: x[0], cost))
        #cost **= .5

        #print(cost)

        eps = .01
        reward = lrydist < eps

        #pdb.set_trace()

        return 0 + 0*reward - cost/1
        
        
        
    def reset(self,bx=1,by=0,bz=0):
        '''bx = 0*self.args.bx[0] + np.random.uniform(self.args.bx[0], self.args.bx[1])
                                by = 0*self.args.by[0] + np.random.uniform(self.args.by[0], self.args.by[1])   # randn() * self.args.by[1]
                                bz = 0*self.args.bz[0] + np.random.uniform(self.args.bz[0], self.args.bz[1])   # randn() * self.args.bz[1]
        '''
        #print(bx, by, bz)
        #pdb.set_trace()
        obs = [v for _, v in EyeHead.reset(self, bx=bx, by=by, bz=bz).items()]
        self.obs_shapes = [(sum(sum(self.obs_shapes, ())),)]

        #pdb.set_trace()
        return [np.concatenate(obs)]  # [sum([o.tolist() for o in obs], [])]

    def step(self, action):
        state, reward, done, info = EyeHead.step(self, action)
        obs = [v for _, v in state.items()]
        rtn = [np.concatenate(obs)], reward, done, info
        self.obs_shapes = [(sum(sum(self.obs_shapes, ())),)]
        #pdb.set_trace()
        return rtn
        
