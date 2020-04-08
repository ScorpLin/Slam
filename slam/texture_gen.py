import sys
sys.path.insert(0, 'MapUtils')
import load_data as ld
import os, sys, time
import p3_util as ut
import matplotlib.pyplot as plt
import numpy as np
import cPickle as pickle
import time
from scipy import io
import cv2
def gen_rgb_video(img_folder = '../Proj3_2017_Train_rgb',cam_params_folder='..\Proj3_2017_Train_rgb\cameraParam',RGB_file = 'RGB_0',dataset='0'):
    RGB_file = os.path.join(img_folder,RGB_file)
    RGB0 = ld.get_rgb(RGB_file)
    rgb_times = np.array([d['t'][0][0] for d in RGB0])
    state_times, states, map = _get_pose(dataset=dataset)
    img_width = RGB0[0]['imge'].shape[0]
    img_length = RGB0[0]['imge'].shape[1]

    map_video = cv2.VideoWriter('Train_map_' + dataset +'.avi',-1,20,(img_width,img_length))
    rgb_video = cv2.VideoWriter('Train_rgb_' + dataset +'.avi',-1,20,(801,801))

    # for k in range(rgb_times.shape):
    for k in range(10):

        print '\--------- k = {0}--------'.format(k)
        ts = rgb_times[k]
        rgb = RGB0[k]['images']
        # find corresponding time in state
        state_indx = np.argmax(abs(ts - state_times))
        print '- State time:', state_times[state_indx], 'delta_t:', ts - state_times[state_indx]
        map[states[0,k],states[1,k],:] = [70,70,228]

        # write video
        rgb_video.write(rgb)
        map_video.write(map)
    rgb_video.release()
    map_video.release()


def _get_pose(dataset='0', fig='OFF'):
    learned_folder = 'learned_models'
    # state_and_time_file = os.path.join(learned_folder,'Train_state_and_time_' + dataset + '.pickle')
    state_and_time_file = 'Train_state_and_time_' + dataset + '.pickle'
    print 'state_and_time_file:',state_and_time_file
    try:
        with open(state_and_time_file, 'rb') as f:
            slam_inc = pickle.load(f)
    except Exception as e:
        print('Unable to process data from', state_and_time_file, ':', e)
        raise
    num_steps = slam_inc.num_data_
    states = slam_inc.best_p_
    times = slam_inc.time_
    # map = slam_inc.MAP_2_display
    MAP = slam_inc.MAP_
    # recover map to display
    MAP_2_display = 255*np.ones((MAP['map'].shape[0],MAP['map'].shape[1],3),dtype=np.uint8)
    wall_indices = np.where(slam_inc.log_odds_>slam_inc.logodd_thresh_)
    MAP_2_display[list(wall_indices[0]),list(wall_indices[1]),:] = [0,0,0]
    # print MAP_2_display[np.where(self.log_odds_>self.logodd_thresh_)]
    # print 'number of < 1e-1:',len(np.where(abs(self.log_odds_) < 1e-1)[0])
    unexplored_indices = np.where(abs(slam_inc.log_odds_) < 1e-1)
    MAP_2_display[list(unexplored_indices[0]),list(unexplored_indices[1]),:] = [150,150,150]

    lidar = slam_inc.lidar_
    if fig == 'ON':
        plt.figure(1)
        plt.plot(states[0,:],states[1,:],'r')
        plt.gca().invert_yaxis()
    return times, states, MAP_2_display

class GENTEXTURE:
    def __init__(self):
        cam_params_folder = '..\Proj3_2017_Train_rgb\cameraParam'
        # Load extrinsic params of IR w.r.t RGB camera
        ex_params_file = os.path.join(cam_params_folder,'exParams.pkl')
        print 'ex_params_file:',ex_params_file
        ir_params_file = os.path.join(cam_params_folder,'IRcam_Calib_result.pkl')
        rgb_params_file = os.path.join(cam_params_folder,'RGBcamera_Calib_result.pkl')
        try:
            with open(ex_params_file, 'rb') as f:
                ex_params = pickle.load(f)
            with open(ir_params_file, 'rb') as f:
                ir_params = pickle.load(f)
            with open(rgb_params_file, 'rb') as f:
                rgb_params = pickle.load(f)
        except Exception as e:
            print('Unable to process data from', ex_params_file, ':', e)
            raise

        print '\n----------------Obtain camera params-------------------'
        R = ex_params['R']
        T = ex_params['T']
        K_ir = np.array([[ir_params['fc'][0], ir_params['alpha_c'], ir_params['cc'][0]], \
                             [0                  , ir_params['fc'][1] ,  ir_params['cc'][1]], \
                             [0                  ,    0                     ,       0]])
        ir_img_size = ir_params['im_size']
        K_rgb = np.array([[rgb_params['fc'][0], rgb_params['alpha_c'], rgb_params['cc'][0]], \
                              [0                  , rgb_params['fc'][1] ,  rgb_params['cc'][1]], \
                              [0                  ,    0                     ,       0 ]] )
        rgb_img_size = rgb_params['im_size']
        # print 'T:', T
        # print 'R:', R
        # print 'K_ir\n', K_ir
        # print 'K_rgb\n', K_rgb
        # print 'Image sizes: ', ir_img_size, rgb_img_size
        # exit(1 )
        self.T_ = T
        self.R_ = R
        self.K_ir_ = K_ir
        self.K_rgb_ = K_rgb
        self.ir_img_size_ = ir_img_size
        self.rgb_img_size_ = rgb_img_size
        # return T, R, K_ir, K_rgb, ir_img_size, rgb_img_size

    def _get_all_imgs(self,img_folder = '../Proj3_2017_Train_rgb',RGB_file = 'RGB_0',D_file = 'DEPTH_0'):
        """To display RGB images:
        R = rgb_data[k]['image']
		R = np.fliplr(R)
		plt.imshow(R)
		plt.draw()
		plt.pause(0.001)

		To display depth images:
		for k in xrange(len(depth_data)):
		D = depth_data[k]['depth']
		D = np.flip(D,1)
		for r in xrange(len(D)):
			for (c,v) in enumerate(D[r]):
				if (v<=DEPTH_MIN) or (v>=DEPTH_MAX):
					D[r][c] = 0.0

		plt.imshow(D)
		plt.draw()
		plt.pause(0.001)"""
        print '---------------Obtaining all images-------------------'
        RGB_file = os.path.join(img_folder,RGB_file)
        RGB0 = ld.get_rgb(RGB_file)

        D_file = os.path.join(img_folder,D_file)
        D0 = ld.get_depth(D_file)

        ir_times = np.array([d['t'][0][0] for d in D0])
        rgb_times = np.array([d['t'][0][0] for d in RGB0])
        print ir_times - rgb_times
        print ir_times.shape, rgb_times.shape

        # key_names_depth = ['t','width','imu_rpy','id','odom','head_angles','c','sz','vel','rsz','body_height','tr','bpp','name','height','depth']
        self.ir_imgs = D0
        # key_names_rgb = ['t','width','imu_rpy','id','odom','head_angles','c','sz','vel','rsz','body_height','tr','bpp','name','height','image']
        self.rgb_imgs =RGB0
        # self.depth_times = np.array([D0[]])

    # def _ex






# ut.replay_rgb(RGB0)

if __name__ == "__main__":
    # genText(dataset='3')
    # texture = GENTEXTURE()
    # texture._get_all_imgs()
    # gen_rgb_video()
    _get_pose(dataset='1')
