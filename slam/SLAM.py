from math import cos, sin
from bresenham2D import bresenham2D
import numpy as np
import matplotlib.pyplot as plt
import load_data as ld
import os, sys, time
import p3_util as ut
from read_data import LIDAR, JOINTS
import probs_utils as prob
import math
import cv2
import transformations as tf
from copy import deepcopy
from mpl_toolkits.mplot3d import Axes3D
import logging
if (sys.version_info > (3, 0)):
    import pickle
else:
    import cPickle as pickle

logger = logging.getLogger()
logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))


def _remove_ground(h_lidar, ray_angle=None, ray_l=None, head_angle=0, h_min=0.2):
    """Filter a ray in lidar scan: remove the ground effect
    using head angle.
    :input
    h_lidar: the height of the lidar w.r.t the ground
    ray_l is a scalar distance from the object detected by the lidar. A number of value
    0.0 meaning that there is no object detected.
    :return
    starting point and ending point of the ray after truncating and an indicator saying that
    whether the last point is occupied or not
    """
    # TODO: truncate 0.1 m as a limit to the min of lidar ray which is accepted
    if ray_l >= 30:
        dmin = cos(head_angle) * 0.001
        dmax = 30  # go to infinity so donnot need to multiply cos(head_angle)
        last_occu = 0

    else:
        try:
            dmin = cos(head_angle) * 0.001
            delta_l = h_min / math.sin(head_angle)
            # print 'Delta_l: ', delta_l
            l2ground = h_lidar / math.sin(head_angle)
            new_l = l2ground - delta_l
            if new_l > ray_l:
                dmax = ray_l * cos(head_angle)
                last_occu = 1
            else:
                dmax = new_l * cos(head_angle)
                last_occu = 0
        except:
            dmin = cos(head_angle) * 0.001
            dmax = cos(head_angle) * ray_l
            last_occu = 1

    return np.array([dmin, dmax, last_occu, ray_angle])


def _ray2world(R_pose, ray_combo, unit=1):
    """Convert ray to world x, y coordinate based on the particle position and orientation
    :input
    R_pos: (3L,) array representing pose of a particle (x, y, theta)
    ray_combo: (4L,) array of the form [[dmin,dmax,last_occu,ray_angle]]
    unit:  how much meter per grid side
    :output
    [[sX,sY,eX,eY],[last_occu]]: x, y position of starting points and ending points of the ray
    and whether the last cell is occupied"""
    world_to_part_rot = tf.twoDTransformation(R_pose[0], R_pose[1], R_pose[2])

    [dmin, dmax, last_occu, ray_angle] = ray_combo
    # starting point of the line
    sx = dmin * cos(ray_angle) / unit
    sy = dmin * sin(ray_angle) / unit
    ex = dmax * cos(ray_angle) / unit
    ey = dmax * sin(ray_angle) / unit
    # print [sx,sy,ex,ey]

    [sX, sY, _] = np.dot(world_to_part_rot, np.array([sx, sy, 1]))
    [eX, eY, _] = np.dot(world_to_part_rot, np.array([ex, ey, 1]))

    return [sX, sY, eX, eY]


def _ray2worldPhysicPos(R_pose, neck_angle, ray_combo):
    """Convert the ending point of a ray to world x, y coordinate and then the indices in MAP array based
    on the neck's angle and the particle position and orientation
    :input
    R_pos: (3L,) array representing physical orientation of a particle (x, y, theta)
    ray_combo: (4L,) array of the form [[dmin,dmax,last_occu,ray_angle]]
    unit:  how much meter per grid side
    :output
    [[sX,sY,eX,eY],[last_occu]]: x, y position of starting points and ending points of the ray
    and whether the last cell is occupied"""
    # rotation matrix that transform body's frame to head's frame (where lidar located in)
    # we need only to take into account neck's angle as head's angle (tilt) has already been considered
    # in removing the ground of every ray.
    body_to_head_rot = tf.twoDTransformation(0, 0, neck_angle)
    world_to_part_rot = tf.twoDTransformation(R_pose[0], R_pose[1], R_pose[2])
    [dmin, dmax, last_occu, ray_angle] = ray_combo
    if last_occu == 0:  # there is no occupied cell
        return None
    # physical position of ending point of the line w.r.t the head of the robot
    ex_h = dmax * cos(ray_angle)
    ey_h = dmax * sin(ray_angle)
    # print [sx,sy,ex,ey]
    # transform this point to obtain physical position in the body's frame
    exy1_r = np.dot(body_to_head_rot, np.array([ex_h, ey_h, 1]))
    # transform this point to obtain physical position in the MAP (global)
    # Rotate these points to obtain physical position in the MAP
    [eX, eY, _] = np.dot(world_to_part_rot, exy1_r)
    return [eX, eY]


def _physicPos2Pos(MAP, pose):
    """ Return the corresponding indices in MAP array, given the physical position"""
    # convert from meters to cells
    [xs0, ys0] = pose
    xis = np.ceil((xs0 - MAP['xmin']) / MAP['res']).astype(np.int16) - 1
    yis = np.ceil((ys0 - MAP['ymin']) / MAP['res']).astype(np.int16) - 1
    return [xis, yis]


def _cellsFrom2Points(twoPoints):
    """Return cells that a line acrossing two points
    :input
    twoPoints = (4L,) array in form: [sX,sY,eX,eY]
    #	(sX, sY)	start point of ray
    #	(eX, eY)	end point of ray
    :return
    2x N array of cells e.i. np.array([[0,1,2,3,4,5,6,7,8,9],[1,2,2,3,3,4,4,5,5,6]])
     """
    [sx, sy, ex, ey] = twoPoints
    # print sx, sy, ex, ey
    cells = bresenham2D(sx, sy, ex, ey)
    return cells

def Twoglobalpos(lidar_hit, joint_angles, body_angles, pose=None, Particles=None):
    neck_angle = joint_angles[0] # yaw wrt body frame
    head_angle = joint_angles[1] # pitch wrt body frame
    roll_gb = body_angles[0]
    pitch_gb = body_angles[1]
    yaw_gb = body_angles[2] # using imu's yaw has better performance than pose's yaw

    # lidar wrt head
    z_hl = 0.15
    H_hl = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,z_hl],[0,0,0,1]]) # no rotation

    # head wrt body
    z_bh = 0.33
    T_bh = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,z_bh],[0,0,0,1]])
    Rz = tf.rot_z_axis(neck_angle)
    Rz = tf.homo_transform(Rz, np.array([0, 0 ,1]))
    Ry = tf.rot_y_axis(head_angle)
    Ry = tf.homo_transform(Ry, np.array([0, 0, 1]))
    R_bh = np.dot(Rz,Ry)
    H_bh = np.dot(T_bh, R_bh)

    if Particles is None: # for mapping
        # body wrt world
        x_gb = pose[0]
        y_gb = pose[1]
        z_gb = 0.93
        T_gb = np.array([[1, 0, 0, x_gb], [0, 1, 0, y_gb], [0, 0, 1, z_gb], [0, 0, 0, 1]])

        Rb2g_yaw = tf.rot_z_axis(yaw_gb)
        Rb2g_pitch = tf.rot_y_axis(pitch_gb)
        Rb2g_roll = tf.rot_x_axis(roll_gb)
        Hb2g_yaw = tf.homo_transform(Rb2g_yaw, np.array([0, 0, 1]))
        Hb2g_pitch = tf.homo_transform(Rb2g_pitch, np.array([0, 0, 1]))
        Hb2g_roll = tf.homo_transform(Rb2g_roll, np.array([0, 0, 1]))
        R_gb = np.dot(np.dot(Hb2g_yaw, Hb2g_pitch), Hb2g_roll)
        H_gb = np.dot(T_gb, R_gb)
        # lidar wrt world
        H_gl = H_gb.dot(H_bh).dot(H_hl)
        lidar_hit = np.vstack((lidar_hit,np.ones((1,lidar_hit.shape[1])))) # 4*n
        world_hit = np.dot(H_gl, lidar_hit)
        # ground check, keep hits not on ground
        not_floor = world_hit[2]>0.1
        world_hit = world_hit[:,not_floor]

        return world_hit[:3,:]

    else: # for particles update
        nums = Particles.shape[1]
        poses = Particles
        particles_hit = []
        lidar_hit = np.vstack((lidar_hit, np.ones((1, lidar_hit.shape[1]))))

        for i in range(nums):
            # body wrt world
            T_gb = np.array([[1, 0, 0, poses[0,i]], [0, 1, 0, poses[1,i]], [0, 0, 1, 0.93], [0, 0, 0, 1]])

            yaw_gb = poses[2,i]
            Rb2g_yaw = tf.rot_z_axis(yaw_gb)
            Rb2g_pitch = tf.rot_y_axis(pitch_gb)
            Rb2g_roll = tf.rot_x_axis(roll_gb)

            Hb2g_yaw = tf.homo_transform(Rb2g_yaw, np.array([0, 0, 1]))
            Hb2g_pitch = tf.homo_transform(Rb2g_pitch, np.array([0, 0, 1]))
            Hb2g_roll = tf.homo_transform(Rb2g_roll, np.array([0, 0, 1]))

            R_gb = np.dot(np.dot(Hb2g_yaw, Hb2g_pitch), Hb2g_roll)
            
            H_gb = np.dot(T_gb, R_gb)

            # lidar wrt world
            H_gl = H_gb.dot(H_bh).dot(H_hl)
            world_hit = np.dot(H_gl, lidar_hit)[:3,:]

            not_floor = world_hit[2] > 0.1
            particles_hit.append(world_hit[:, not_floor])

        return np.transpose(np.asarray(particles_hit), (1,2,0))

def world2resolution(hit, MAP):
    '''
    pixels[0] = ((hit[0] + MAP['sizex'] * 0.5 * MAP['res']) / MAP['res']).astype(np.int)
    pixels[1] = ((-hit[1] + MAP['sizey'] * 0.5 * MAP['res']) / MAP['res']).astype(np.int)
    '''

    '''
    pixels[0] = ((hit[1] + MAP['sizex'] * 0.5 * MAP['res']) / MAP['res']).astype(np.int)
    pixels[1] = ((-hit[0] + MAP['sizey'] * 0.5 * MAP['res']) / MAP['res']).astype(np.int)
    '''

    pixels = np.zeros(hit.shape, dtype=int) #pixels[0] = ((-hit[1] + MAP['sizex'] * 0.5 * MAP['res']) / MAP['res']).astype(np.int)
    pixels[0] = ((hit[1] + MAP['sizex'] * 0.5 * MAP['res']) / MAP['res']).astype(np.int)
    pixels[1] = ((hit[0] + MAP['sizey'] * 0.5 * MAP['res']) / MAP['res']).astype(np.int)
    # boundary
    center = MAP['sizex'] / 2
    in_bound = np.logical_and(np.abs(pixels[0]-center) < center, np.abs(pixels[1]-center) < center)
    pixels = pixels[:,in_bound]
    return pixels

def xyz_pos(scan, angles):
    x = scan * np.cos(angles)
    y = scan * np.sin(angles)
    z = np.zeros(len(angles))
    return np.vstack((x, y, z))
    

class SLAM(object):
    def __init__(self):
        self._characterize_sensor_specs()
    
    def _read_data(self, src_dir, dataset=0, split_name='train'):
        self.dataset_= str(dataset)
        if split_name.lower() not in src_dir:
            src_dir  = src_dir + '/' + split_name
        print('\n------Reading Lidar and Joints (IMU)------')
        self.lidar_  = LIDAR(dataset=self.dataset_, data_folder=src_dir, name=split_name + '_lidar'+ self.dataset_)
        print ('\n------Reading Joints Data------')
        self.joints_ = JOINTS(dataset=self.dataset_, data_folder=src_dir, name=split_name + '_joint'+ self.dataset_)

        self.num_data_ = len(self.lidar_.data_)
        # Position of odometry
        self.odo_indices_ = np.empty((2,self.num_data_),dtype=np.int64)

    def _characterize_sensor_specs(self, p_thresh=None):
        # High of the lidar from the ground (meters)
        self.h_lidar_ = 0.93 + 0.33 + 0.15
        # Accuracy of the lidar
        self.p_true_ = 0.9
        self.p_false_ = 0.1
        
        #TODO: set a threshold value of probability to consider a map's cell occupied  
        self.p_thresh_ = 0.9 if p_thresh is None else p_thresh # > p_thresh => occupied and vice versa
        # Compute the corresponding threshold value of logodd
        self.logodd_thresh_ = prob.log_thresh_from_pdf_thresh(self.p_thresh_)
        

    def _init_particles(self, num_p=0, mov_cov=None, particles=None, weights=None, percent_eff_p_thresh=None):
        # Particles representation
        self.num_p_ = num_p
        #self.percent_eff_p_thresh_ = percent_eff_p_thresh
        self.particles_ = np.zeros((3,self.num_p_),dtype=np.float64) if particles is None else particles
        
        # Weights for particles
        self.weights_ = 1.0/self.num_p_*np.ones(self.num_p_) if weights is None else weights

        # Position of the best particle after update on the map
        self.best_p_indices_ = np.zeros((2,self.num_data_),dtype=np.int64)
        # Best particles
        self.best_p_ = np.empty((3,self.num_data_))
        # Corresponding time stamps of best particles
        self.time_ =  np.empty(self.num_data_)
       
        # Covariance matrix of the movement model
        tiny_mov_cov   = np.array([[1e-8, 0, 0],[0, 1e-8, 0],[0, 0 , 1e-8]])
        self.mov_cov_  = mov_cov if mov_cov is not None else tiny_mov_cov
        # To generate random noise: x, y, z = np.random.multivariate_normal(np.zeros(3), mov_cov, 1).T
        # this return [x], [y], [z]

        # Threshold for resampling the particles
        self.percent_eff_p_thresh_ = percent_eff_p_thresh

    def _init_map(self, map_resolution=0.05):
        '''*Input: resolution of the map - distance between two grid cells (meters)'''
        # Map representation
        MAP= {}
        MAP['res']   = map_resolution #meters
        MAP['xmin']  = -20  #meters
        MAP['ymin']  = -20
        MAP['xmax']  =  20
        MAP['ymax']  =  20
        MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
        MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))

        MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.int8) #DATA TYPE: char or int8
        self.MAP_ = MAP

        self.log_odds_ = np.zeros((self.MAP_['sizex'],self.MAP_['sizey']),dtype = np.float64)
        self.occu_ = np.ones((self.MAP_['sizex'],self.MAP_['sizey']),dtype = np.float64)
        # Number of measurements for each cell
        self.num_m_per_cell_ = np.zeros((self.MAP_['sizex'],self.MAP_['sizey']),dtype = np.uint64)

        self.free = np.log(self.p_true_/self.p_false_)*(-0.2)
        self.occ = np.log(self.p_false_/self.p_true_)*(-0.4)
    
    def _build_first_map(self,t0=0,use_lidar_yaw=True):
        """Build the first map using first lidar"""
        # # TODO: student's input from here
        MAP = self.MAP_
        lidar_data = self.lidar_.data_

        yaw_bias = lidar_data[0]['rpy'][0, 2]
        pose_bias = lidar_data[0]['pose'][0, :2]

        num_beams = lidar_data[0]['scan'].shape[1]
        self.particles_ = np.zeros((3, self.num_p_), dtype=np.float64)
        self.lidar_angles = np.linspace(start=-135 * np.pi / 180, stop=135 * np.pi / 180, num=num_beams).reshape(1, -1)
        for i in range(len(lidar_data)):
            lidar_data[i]['rpy'][0, 2] -= yaw_bias
            lidar_data[i]['pose'][0, :2] -= pose_bias

        self.lidar_.data_ = lidar_data
        self.t0 = 0
        self.best_p_ = np.zeros((3, self.num_data_))
        #==========================================================
        self.t0 = 0

        lidar_inc = self.lidar_.data_
        joint_inc = self.joints_.data_

        # self.lidar_inc = LIDAR()
        # self.joint_inc = JOINTS()

        head_angle = joint_inc['head_angles'][1, t0]
        neck_angle = joint_inc['head_angles'][0, t0]

        lidar_angles = np.linspace(start=-135 * np.pi / 180, stop=135 * np.pi / 180, num= lidar_inc[t0]['scan'].shape[1]).reshape(-1)
        #lidar_angles = lidar_angles.ravel('F') #  transform lidar_angles to 1D

        lidar_scan = lidar_inc[t0]['scan'].reshape(-1)
        #lidar_scan = lidar_angles.ravel('F')  #  transform lidar_scan to 1D

        lidar_pose = lidar_inc[t0]['pose'].reshape(-1)
        #lidar_pose = lidar_pose.ravel('F')

        for i in range(0, lidar_inc[t0]['scan'].shape[1]):
            ray_combo = _remove_ground(h_lidar=self.h_lidar_, ray_angle=lidar_angles[i], ray_l=lidar_scan[i])

            end_point_item = np.array(_ray2worldPhysicPos(R_pose=lidar_pose, neck_angle=neck_angle, ray_combo=ray_combo))
            end_point_map = _physicPos2Pos(MAP=self.MAP_, pose=end_point_item)
            self.log_odds_[end_point_map[1].astype(int), end_point_map[0].astype(int)] = self.occ

            start_point_map = _physicPos2Pos(self.MAP_, lidar_pose[0:2])
            cells = _cellsFrom2Points(twoPoints=np.hstack((start_point_map, end_point_map)))
            self.log_odds_[cells[1].astype(int), cells[0].astype(int)] = self.free

        self.lidar_angles = np.linspace(start=-135 * np.pi / 180, stop=135 * np.pi / 180, num=lidar_inc[t0]['scan'].shape[1]).reshape(1, -1)

        self.lidar_.data_ = lidar_inc

        #End student's input

        self.MAP_ = MAP

    def _calcHit(self, t):
        Pose = self.particles_[:, np.argmax(self.weights_)]
        worldPos = world2resolution(Pose[:2], self.MAP_)


        self.best_p_indices_[1, t] = worldPos[0, 0]
        self.best_p_indices_[0, t] = worldPos[1, 0]

        lidar_data = self.lidar_.data_
        lidar_scan = lidar_data[t]['scan']
        remove_aft_ground = np.logical_and(lidar_scan > 0.1, lidar_scan < 30) # lidar spec
        self.lidar_hit = xyz_pos(lidar_scan[remove_aft_ground], self.lidar_angles[remove_aft_ground])
        joint_idx = np.argmin(np.abs(self.joints_.data_['ts'] - lidar_data[t]['t']))
        self.joint_angles = self.joints_.data_['head_angles'][:,joint_idx]
        world_hit = Twoglobalpos(self.lidar_hit, self.joint_angles, lidar_data[t]['rpy'][0, :], pose=Pose)
        occ = world2resolution(world_hit[:2], self.MAP_)
        #-0.4236
        free_d = self.free
        '''
               self.log_odds_[occ[1], occ[0]] += self.occ - free_d
        '''
        self.log_odds_[occ[1], occ[0]] += self.occ - free_d
        mask = np.zeros(self.log_odds_.shape)
        contour = np.hstack((world2resolution(Pose[:2], self.MAP_).reshape(-1, 1), occ))
        cv2.drawContours(image=mask, contours = [contour.T], contourIdx = -1, color = free_d, thickness=-1)
        self.log_odds_ += mask
        bound = self.num_p_
        self.log_odds_[self.log_odds_ > bound] = bound
        self.log_odds_[self.log_odds_ < -bound] = -bound
        

    def _predict(self,t,use_lidar_yaw=True):
        logging.debug('\n-------- Doing prediction at t = {0}------'.format(t))
        #TODO: student's input from here
        #=================================================
        '''
        self._calcHit(t)
        interval = 1
        curr_xy = self.lidar_.data_[t]['pose'][0,:2]
        curr_theta = self.lidar_.data_[t]['rpy'][0,2]
        prev_xy = self.lidar_.data_[t - interval]['pose'][0,:2]
        prev_theta = self.lidar_.data_[t - interval]['rpy'][0,2]
        R_local = np.array([[np.cos(prev_theta), -np.sin(prev_theta)],
                    [np.sin(prev_theta), np.cos(prev_theta)]])
        d_xy = np.dot(R_local.T, (curr_xy - prev_xy).reshape((-1,1)))
        d_theta = curr_theta - prev_theta
        R_global = np.array([[np.cos(self.particles_[2]), -np.sin(self.particles_[2])],
                    [np.sin(self.particles_[2]), np.cos(self.particles_[2])]])
        
        self.particles_[:2] += np.squeeze(np.einsum('ijk,il->ilk', R_global, d_xy))
        self.particles_[2] += d_theta
        #self.best_p_[2, t] = d_theta
        # self.best_p_[1, t] = worldPos[0, 0]
        # self.best_p_[0, t] = worldPos[1, 0]
        # self.best_p_[2, t] = d_theta

        noise = np.random.multivariate_normal([0,0,0],
                                              np.diag([0.005, 0.005, 0.005]),
                                              size=self.particles_.shape[1])
        self.particles_ += noise.T # slightly incorrect but faster
        '''
        self._calcHit(t)
        Pose = self.particles_[:, np.argmax(self.weights_)]
        worldPos = world2resolution(Pose[:2], self.MAP_)
        '''
                self.best_p_indices_[1, t] = float(worldPos[0, 0] - 400) / 20
        self.best_p_indices_[0, t] = float(worldPos[1, 0] - 400) / 20
        '''
        self.best_p_indices_[1, t] = float(worldPos[0, 0])
        self.best_p_indices_[0, t] = float(worldPos[1, 0])



        interval = 1
        curr_xy = self.lidar_.data_[t]['pose'][0, :2]
        curr_theta = self.lidar_.data_[t]['rpy'][0, 2]
        prev_xy = self.lidar_.data_[t - interval]['pose'][0, :2]
        prev_theta = self.lidar_.data_[t - interval]['rpy'][0, 2]
        R_local = np.array([[np.cos(prev_theta), -np.sin(prev_theta)],
                            [np.sin(prev_theta), np.cos(prev_theta)]])
        d_xy = np.dot(R_local.T, (curr_xy - prev_xy).reshape((-1, 1)))
        d_theta = curr_theta - prev_theta
        R_global = np.array([[np.cos(self.particles_[2]), -np.sin(self.particles_[2])],
                             [np.sin(self.particles_[2]), np.cos(self.particles_[2])]])
        '''
         self.best_p_[1, t] = worldPos[0, 0]
        self.best_p_[0, t] = worldPos[1, 0]
        self.best_p_[2, t] = d_theta
        '''
        self.best_p_[1, t] = float(worldPos[0, 0] - 400) / 20
        self.best_p_[0, t] = float(worldPos[1, 0] - 400) / 20
        self.best_p_[2, t] = d_theta

        i, j, k = R_global.shape
        i, l = d_xy.shape
        particles1 = np.empty((i, l, k))
        for indi in range(i):
            for indl in range(l):
                for indk in range(k):
                    tot = 0
                    for indj in range(j):
                        tot += R_global[indi, indj, indk] * d_xy[indi, indl]
                    particles1[indi, indl, indk] = tot
        # particles =  np.einsum('ijk,il->ilk', R_global, d_xy)
        # print(particles==particles1)
        self.particles_[:2] += np.squeeze(particles1)
        self.particles_[2] += d_theta

        noise = np.random.multivariate_normal([0, 0, 0],
                                              np.diag([0.005, 0.005, 0.005]),
                                              size=self.particles_.shape[1])
        self.particles_ += noise.T  # slightly incorrect but faster

    def _update(self, t, t0=0, fig='on'):
        """Update function where we update the """
        if t == t0:
            self._build_first_map(t0,use_lidar_yaw=True)
            return

        #TODO: student's input from here 
        #=======================================
        # Extract a ray from lidar data
        MAP = self.MAP_
        body_angles = self.lidar_.data_[t]['rpy'][0, :]
        particles_hit = Twoglobalpos(self.lidar_hit, self.joint_angles, body_angles, Particles=self.particles_)
        # get matching between map and particle lidar reading
        corr = np.zeros(self.particles_.shape[1])
        for i in range(self.particles_.shape[1]):
            occ = world2resolution(particles_hit[:2, :, i], self.MAP_)
            corr[i] = np.sum(self.log_odds_[occ[1],occ[0]] > self.logodd_thresh_)

        corr /= 10 # by divide, adding a temperature to the softmax function

        # update particle weights
        log_weights = np.log(self.weights_) + corr
        log_weights -= np.max(log_weights) + np.log(np.sum(np.exp(log_weights - np.max(log_weights))))
        self.weights_ = np.exp(log_weights)

        self.MAP_ = MAP
        return MAP

