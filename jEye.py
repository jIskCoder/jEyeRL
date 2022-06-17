import os
from osim.env import OsimEnv
import numpy as np
from opensim import Vec3
import math
from numpy import pi
from opensim import Rotation, Transform

class EyeHead(OsimEnv):
    
    model_path = os.path.join(os.path.dirname(__file__), './jEye/EyeSkullModel_v2.4.osim')
    '''max_step=0
    LR_dist=0
    L_dist=0
    R_dist=0
    ball=0
    lCoord=[]
    rCoord=[]
    Effort=0
    ACT = 0
    Muscles_ACT=[]
    t=0'''
    def __init__(self, visualize=True, integrator_accuracy=5e-5, report=None,maxstep=100):
        self.Effort=0
        super(EyeHead, self).__init__(visualize=visualize, integrator_accuracy=integrator_accuracy)
        if visualize:
            self.viz2 = self.osim_model.model.getVisualizer().getSimbodyVisualizer()
            self.viz2.setBackgroundColor(Vec3(0, 0, 0))
            self.viz2.setBackgroundType(0)
            self.viz2.setDesiredFrameRate(1000)  # 1/self.osim_model.stepsize)
            self.viz2.setMode(2)
            #self.viz2.drawFrameNow(self.osim_model.state)
            self.reset_view()
        #self.max_step=maxstep
        self.osim_model.stepsize=0.01#0.00148
        #self.rmax_rep=0
    def reset_view(self):
        
        R = Rotation(pi/2, Vec3(0, 1, 0))
        self.viz2.zoomCameraToShowAllGeometry()
        self.viz2.setCameraClippingPlanes(-100, 1000)
        self.viz2.pointCameraAt(Vec3(0,0, 0), Vec3(0, 0, 0))
        self.viz2.setCameraFieldOfView(.05*pi/180.)
        self.viz2.setCameraTransform(Transform(R, Vec3(1000, 0.05, 0)))

        self.viz2.setShowSimTime(True)
        self.viz2.setShowFrameRate(True)
        self.viz2.setShowFrameNumber(True)
        self.viz2.setShowShadows(False)
    
    def reset(self, bx=1,by=0,bz=0,project=True, seed=None, init_pose=None, obs_as_dict=False):
        self.istep=0
        self.t = 0
        self.Effort=0
        self.rmax_rep=0
        
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
        obs_dict = self.get_observation_dict()
        self.Muscles_ACT=[]
        for MUS in ['rLR','rMR','rSR', 'rIR','rSO','rIO']:
            self.Muscles_ACT.append(obs_dict['muscles'][MUS]['activation'])   
          
        return self.get_observation()
    def moveROI(self, bx,by,bz):
        self.osim_model.state = self.osim_model.model.updWorkingState()

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
        return self.get_observation()
        
    def step(self, action, project=True,obs_as_dict=False):
        self.istep+=1
        self.t+=self.osim_model.stepsize
        observation, reward, done, info = super(EyeHead, self).step(action, project=project, obs_as_dict=obs_as_dict)
        return observation, reward, done, info

    def get_reward(self):
        reward=0
        w1=0; w2=0; w3=-1/self.max_step; w4=-2/self.max_step
        lrdist=self.LR_dist**2
        pogdist = ((self.lpog[1]-self.rpog[1])**2)
        pogdist += (self.lpog[2]-self.rpog[2])**2
        dist=self.L_dist**2+self.R_dist**2
        
        dev= (max((self.right_devang),(self.left_devang)))

        reward=(w1*lrdist)+(w2*pogdist)+(w3*dist)+(w4*dev)
        return reward

    def is_done(self):
        if self.istep>=self.max_step:
            return True
        return False

    def get_observation_space_size(self):
        return 2

    def get_observation(self):
        obs_dict = self.get_observation_dict()
        
        self.ROI = np.array(obs_dict['body_pos']['ball'])
        self.lpog = np.array(obs_dict['markers']['LPOG']['pos'])
        self.rpog = np.array(obs_dict['markers']['RPOG']['pos'])
        reye = np.array(obs_dict['markers']['REye']['pos'])
        leye = np.array(obs_dict['markers']['LEye']['pos'])

        res = np.array([])
        res =np.append(res,(np.array(obs_dict['markers']['RPOG']['pos'])-np.array(obs_dict['body_pos']['ball']))[1:3])
        self.LR_dist = math.sqrt(np.sum((self.rpog-self.lpog)**2))
        self.L_dist = math.sqrt(np.sum((self.lpog-self.ROI)**2))
        self.R_dist = math.sqrt(np.sum((self.rpog-self.ROI)**2))
        self.left_devang = float(Ang_3points(leye, self.lpog, self.ROI))
        self.right_devang = float(Ang_3points(reye, self.rpog, self.ROI))
        
        return res

class EyeHeadWrapper(EyeHead):
    def __init__(self, x, y, z, dx,dy,dz,
                 visualize=True, integrator_accuracy=5e-5, report=None, maxstep=100):
        self.ddisplacement = [dx, dy, dz]
        self.displacement = [x, y, z]
        EyeHead.__init__(self, visualize=visualize, integrator_accuracy=integrator_accuracy,
                         maxstep=maxstep)
    def setDis(self,coords):
        self.displacement = coords
    def setdDis(self,coords):
        self.ddisplacement = coords
    def reset(self):

        x = np.random.uniform(
            self.displacement[0]-self.ddisplacement[0], self.ddisplacement[0]+self.displacement[0])
        y = np.random.uniform(
            self.displacement[1]-self.ddisplacement[1], self.ddisplacement[1]+self.displacement[1])
        z = np.random.uniform(
            self.displacement[2]-self.ddisplacement[2], self.ddisplacement[2]+self.displacement[2])
        return EyeHead.reset(self,x, y, z)


class HEyeHeadWrapper(EyeHeadWrapper):
    def __init__(self, x, y, z, dx, dy, dz,visualize=True, integrator_accuracy=5e-5, report=None, maxstep=100, ):
        EyeHeadWrapper.__init__(self,x,y,z,dx,dy,dz, 
                         visualize=visualize, integrator_accuracy=integrator_accuracy,
                         maxstep=maxstep)

    def get_reward(self):
        reward=0
        obs_dict = self.get_observation_dict()
        w1=0; w2=0; w3=-0/self.max_step
        w4=-1/self.max_step
        w5=-0/self.max_step
        w6 = -1/self.max_step
        w7 =-1/self.max_step
        lrdist=self.LR_dist**2
        pogdist = ((self.lpog[1]-self.rpog[1])**2)
        pogdist += (self.lpog[2]-self.rpog[2])**2
        dist=self.L_dist**2+self.R_dist**2
        hdist=((obs_dict['markers']['RPOG']['pos'][2]-obs_dict['body_pos']['ball'][2]))
        vdist = ((obs_dict['markers']['RPOG']['pos']
                  [1]-obs_dict['body_pos']['ball'][1]))
        dev= (max((self.right_devang),(self.left_devang)))
        obs_dict = self.get_observation_dict()
        
        self.ACT = 0
        self.Muscles_ACT = []
        for MUS in ['rLR', 'rMR']:
            self.ACT += obs_dict['muscles'][MUS]['activation']**2
            self.Muscles_ACT.append(obs_dict['muscles'][MUS]['activation'])
        syne = 0
        
        muscs = np.array(self.Muscles_ACT)
        if (muscs[0]+muscs[1]) <= 0.05:
            syne += 0.25
        else:
            if(muscs[0] >= muscs[1] and muscs[1] > 0.05):
                syne += muscs[1]
            elif(muscs[1] > muscs[0] and muscs[0] > 0.05):
                syne += muscs[0]
        self.Effort += self.ACT*self.osim_model.stepsize
        reward = (w1*lrdist)+(w2*pogdist)+(w3*dist)+(w4*dev) + \
            (w5*hdist)+(w6*syne)+(w7*self.Effort)
        return reward

    def get_observation_space_size(self):
        return 3
    def get_observation(self):
        obs_dict = self.get_observation_dict()
        self.ROI = np.array(obs_dict['body_pos']['ball'])
        self.lpog = np.array(obs_dict['markers']['LPOG']['pos'])
        self.rpog = np.array(obs_dict['markers']['RPOG']['pos'])
        reye = np.array(obs_dict['markers']['REye']['pos'])
        leye = np.array(obs_dict['markers']['LEye']['pos'])
        res = np.array([])
        #res = np.append(res, (obs_dict['markers']['RPOG']['pos'][1]-obs_dict['body_pos']['ball'][1]))
        res = np.append(res, (obs_dict['markers']['RPOG']['pos'][2]-obs_dict['body_pos']['ball'][2]))
        for MUS in ['rLR', 'rMR']:
            res=np.append(res,obs_dict['muscles'][MUS]['activation'])
        self.LR_dist = math.sqrt(np.sum((self.rpog-self.lpog)**2))
        self.L_dist = math.sqrt(np.sum((self.lpog-self.ROI)**2))
        self.R_dist = math.sqrt(np.sum((self.rpog-self.ROI)**2))
        self.left_devang = float(Ang_3points(leye, self.lpog, self.ROI))
        self.right_devang = float(Ang_3points(reye, self.rpog, self.ROI))
        return res

class VEyeHeadWrapper(EyeHeadWrapper):
    def __init__(self, x, y, z, dx, dy, dz, visualize=True, integrator_accuracy=5e-5, report=None, maxstep=100):
        EyeHeadWrapper.__init__(self, x, y, z, dx, dy, dz,
                                visualize=visualize, integrator_accuracy=integrator_accuracy,
                                maxstep=maxstep)
    def get_reward(self):
        reward = 0
        obs_dict = self.get_observation_dict()
        w1 = 0
        w2 = 0
        w3 = -0/self.max_step
        w4 = -1/self.max_step
        w5 = -0/self.max_step
        w6 = -1/self.max_step
        w7 = -1/self.max_step
        lrdist = self.LR_dist**2
        pogdist = ((self.lpog[1]-self.rpog[1])**2)
        pogdist += (self.lpog[2]-self.rpog[2])**2
        dist = self.L_dist**2+self.R_dist**2
        hdist = ((obs_dict['markers']['RPOG']['pos']
                  [2]-obs_dict['body_pos']['ball'][2]))
        vdist = ((obs_dict['markers']['RPOG']['pos']
                  [1]-obs_dict['body_pos']['ball'][1]))
        dev = (max((self.right_devang), (self.left_devang)))
        self.ACT = 0
        self.Muscles_ACT = []
        for MUS in ['rSR', 'rIR']:
            self.ACT += obs_dict['muscles'][MUS]['activation']**2
            self.Muscles_ACT.append(obs_dict['muscles'][MUS]['activation'])
        syne = 0
        muscs = np.array(self.Muscles_ACT)
        if (muscs[0]+muscs[1]) <= 0.05:
            syne += 0.25
        else:
            if(muscs[0] > muscs[1] and muscs[1] > 0.05):
                syne += muscs[1]
            elif(muscs[1] > muscs[0] and muscs[0] > 0.05):
                syne += muscs[0]
        '''if (muscs[2]+muscs[3]) <= 0.05:
            syne += 0.25
        else:
            if(muscs[2] > muscs[3] and muscs[3] > 0.05):
                syne += muscs[3]
            elif(muscs[3] > muscs[2] and muscs[2] > 0.05):
                syne += muscs[2]'''
        self.Effort += self.ACT*self.osim_model.stepsize
        reward = (w1*lrdist)+(w2*pogdist)+(w3*dist)+(w4*dev) + \
            (w5*hdist)+(w6*syne)+(w7*self.Effort)
        return reward

    def get_observation_space_size(self):
        return 3
    def get_observation(self):
        obs_dict = self.get_observation_dict()
        self.ROI = np.array(obs_dict['body_pos']['ball'])
        self.lpog = np.array(obs_dict['markers']['LPOG']['pos'])
        self.rpog = np.array(obs_dict['markers']['RPOG']['pos'])
        reye = np.array(obs_dict['markers']['REye']['pos'])
        leye = np.array(obs_dict['markers']['LEye']['pos'])
        res = np.array([])
        res = np.append(res, (obs_dict['markers']['RPOG']['pos'][1]-obs_dict['body_pos']['ball'][1]))
        #res = np.append(res, (obs_dict['markers']['RPOG']['pos'][2]-obs_dict['body_pos']['ball'][2]))
        for MUS in ['rSR', 'rIR']:
            res=np.append(res,obs_dict['muscles'][MUS]['activation'])
        self.LR_dist = math.sqrt(np.sum((self.rpog-self.lpog)**2))
        self.L_dist = math.sqrt(np.sum((self.lpog-self.ROI)**2))
        self.R_dist = math.sqrt(np.sum((self.rpog-self.ROI)**2))
        self.left_devang = float(Ang_3points(leye, self.lpog, self.ROI))
        self.right_devang = float(Ang_3points(reye, self.rpog, self.ROI))
        return res

class HVEyeHeadWrapper(EyeHeadWrapper):
    def __init__(self, x, y, z, dx, dy, dz, visualize=True, integrator_accuracy=5e-5, report=None, maxstep=100):
        EyeHeadWrapper.__init__(self, x, y, z, dx, dy, dz,
                                visualize=visualize, integrator_accuracy=integrator_accuracy,
                                maxstep=maxstep)
    def get_reward(self):
        reward = 0
        obs_dict = self.get_observation_dict()
        w1 = 0
        w2 = 0
        w3 = -0/self.max_step
        w4 = -1/self.max_step
        w5 = -0/self.max_step
        w6 = -0/self.max_step
        w7 = -0/self.max_step
        lrdist = self.LR_dist**2
        pogdist = ((self.lpog[1]-self.rpog[1])**2)
        pogdist += (self.lpog[2]-self.rpog[2])**2
        dist = self.L_dist**2+self.R_dist**2
        hdist = ((obs_dict['markers']['RPOG']['pos']
                  [2]-obs_dict['body_pos']['ball'][2]))
        vdist = ((obs_dict['markers']['RPOG']['pos']
                  [1]-obs_dict['body_pos']['ball'][1]))
        dev = (max((self.right_devang), (self.left_devang)))

        self.ACT = 0
        self.Muscles_ACT = []
        for MUS in ['rLR', 'rMR', 'rSR', 'rIR', 'rSO', 'rIO']:
            self.ACT += obs_dict['muscles'][MUS]['activation']**2
            self.Muscles_ACT.append(obs_dict['muscles'][MUS]['activation'])

        syne = 0
        muscs = np.array(self.Muscles_ACT)
        for i in range(0, 6, 2):
            if (muscs[i]+muscs[i+1]) <= 0.05:
                syne += 0.25
            else:
                if(muscs[i] > muscs[i+1] and muscs[i+1] > 0.05):
                    syne += muscs[i+1]
                elif(muscs[i+1] > muscs[i] and muscs[i] > 0.05):
                    syne += muscs[i]

        self.Effort += self.ACT*self.osim_model.stepsize

        reward = (w1*lrdist)+(w2*pogdist)+(w3*dist)+(w4*dev) + \
            (w5*hdist)+(w6*syne)+(w7*self.Effort)
        return reward
    def get_observation_space_size(self):
        return 6
    def get_observation(self):
        obs_dict = self.get_observation_dict()

        self.ROI = np.array(obs_dict['body_pos']['ball'])
        self.lpog = np.array(obs_dict['markers']['LPOG']['pos'])
        self.rpog = np.array(obs_dict['markers']['RPOG']['pos'])
        reye = np.array(obs_dict['markers']['REye']['pos'])
        leye = np.array(obs_dict['markers']['LEye']['pos'])

        hres = np.array([])
        #hres = np.append(hres, (obs_dict['markers']['RPOG']['pos'][1]-obs_dict['body_pos']['ball'][1]))
        hres = np.append(hres, (obs_dict['markers']['RPOG']['pos'][2]-obs_dict['body_pos']['ball'][2]))
        for MUS in ['rLR', 'rMR']:
            hres=np.append(hres,obs_dict['muscles'][MUS]['activation'])
        vres = np.array([])
        vres = np.append(vres, (obs_dict['markers']['RPOG']['pos'][1]-obs_dict['body_pos']['ball'][1]))
        #vres = np.append(vres, (obs_dict['markers']['RPOG']['pos'][2]-obs_dict['body_pos']['ball'][2]))
        for MUS in ['rSR', 'rIR']:
            vres=np.append(vres,obs_dict['muscles'][MUS]['activation'])
        self.LR_dist = math.sqrt(np.sum((self.rpog-self.lpog)**2))
        self.L_dist = math.sqrt(np.sum((self.lpog-self.ROI)**2))
        self.R_dist = math.sqrt(np.sum((self.rpog-self.ROI)**2))
        self.left_devang = float(Ang_3points(leye, self.lpog, self.ROI))
        self.right_devang = float(Ang_3points(reye, self.rpog, self.ROI))

        return hres,vres

def Ang_3points(origin, p1, p2):
    o1 = p1-origin
    o2 = p2-origin
    cosine_angle = np.dot(o1, o2) / (np.linalg.norm(o1) * np.linalg.norm(o2))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

