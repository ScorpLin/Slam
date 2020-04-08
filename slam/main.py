#*
#    main.py: the main file used to run the SLAM as well as unit test functions
#    created and maintained by Ty Nguyen
#    tynguyen@seas.upenn.edu
#    Feb 2020
#    RGB video can be found at: 
#       train set 0: https://drive.google.com/open?id=0BwGdUcsOa2DqbTBYS25hanVNR0E 
#       test set: https://drive.google.com/file/d/0BwGdUcsOa2DqT2JKSjZYZlVna2s/view?usp=sharing
#*
# Lidar.pose (x,y,theta=yaw)
# Lidar.ypr  = Joint.ypr
# particle (x,y,theta) 

import numpy as np
import matplotlib.pyplot as plt
import load_data as ld
import os, sys, time
import p3_util as ut
from read_data import LIDAR, JOINTS
import probs_utils as prob
import math
import transformations as tf
from copy import deepcopy
from mpl_toolkits.mplot3d import Axes3D
from SLAM import SLAM
import argparse
import pdb 
import tqdm 
from gen_figures import genMap
import cv2
import logging

if (sys.version_info > (3, 0)):
    import pickle
else:
    import cPickle as pickle
import argparse
logger = logging.getLogger()
logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))
plt.ion()

def test_prediction(src_dir, dataset_id=0, split_name='train', log_dir='logs', is_online_fig=False):
    """ This function is created for you to debug your prediction step.
        It runs the prediction step only.
        After running this function, two figures are generated: 
            One represents the trajectory generated using only lidar odometry inforation 
            One represents the trajectory generated using only prediction step with very small process noise.
        You should expect that the two figures look very similar. 
        We set the motion covaration super small:
    slam_inc.mov_cov_ = np.array([[1e-8, 0, 0],[0, 1e-8, 0],[0, 0 , 1e-8]])
    TODO: run this test_prediction and look at two figures generated and compare to see if they look similarly
    """    
    # Create a SLAM instance
    slam_inc = SLAM()
    
    # Read data
    slam_inc._read_data(src_dir, dataset_id, split_name)

    # *Generate trajectory using only data from odometry
    lidar_data = slam_inc.lidar_.data_
    lidar_all_x = []
    lidar_all_y = []
    lidar_all_yaw = []
    for p in lidar_data:
    	lidar_scan = p['pose'][0]
    	lidar_all_x.append(lidar_scan[0])
    	lidar_all_y.append(lidar_scan[1])
    	lidar_all_yaw.append(lidar_scan[2])
    plt.plot(lidar_all_x,lidar_all_y,'b')
    plt.title('Trajectory using onboard odometry only')
    # You may need to invert yaxis?
    # plt.gca().invert_yaxis()
    plt.pause(1)
    
    # Path to save the generated trajectory figure
    lidar_odom_fig_path = log_dir + '/lidar_dometry_only_trajectory_'+ split_name + '_' + str(dataset_id) + '.jpg'
    plt.savefig(lidar_odom_fig_path)
    plt.close()
    print(">> Save %s"%lidar_odom_fig_path)
    
    # *Generate trajectory using only prediction step 
    # Generate three particles randomly to test the prediction step
    num_p     = 3
    weights   = np.ones(num_p)*1.0/num_p
    particles = np.zeros((3,num_p),dtype=np.float64)
    mov_cov   = np.array([[1e-8, 0, 0],[0, 1e-8, 0],[0, 0 , 1e-8]])

    slam_inc._init_particles(num_p=num_p,particles=particles,weights=weights, mov_cov=mov_cov)

    print('\n--------------Testing Prediction Module ---------------')
    t0 = 0
    delta_t = len(lidar_data)
    slam_inc.particles_[:,0] = slam_inc.lidar_.data_[t0]['pose'][0]
    all_particles = deepcopy(slam_inc.particles_)
    for t in tqdm.tqdm(range(t0, t0 + delta_t)):
        if t > t0:
            slam_inc._predict(t, use_lidar_yaw=False)
            all_particles = np.hstack((all_particles,slam_inc.particles_))
        # Display particles, both old and new
        #print('x:',slam_inc.particles_[0,:])
        if is_online_fig:
            plt.plot(slam_inc.particles_[0,:],slam_inc.particles_[1,:],'*r')
            title = 'Frame ',t
            plt.title(title)
            plt.pause(0.01)
    
    plt.plot(all_particles[0,:],all_particles[1,:],'*c')
    plt.title('Trajectory using prediction only')
    # plt.gca().invert_yaxis()
    # plt.show()
    
    prediction_fig_path = log_dir + '/prediction_only_trajectory_'+ split_name + '_' + str(dataset_id) + '.jpg'
    plt.savefig(prediction_fig_path)
    plt.close()
    print(">> Save %s"%prediction_fig_path)

    print("-----------------------------------\n")
    print(" You should open and compare two figures:\n \t%s\n and\n \t%s"%(lidar_odom_fig_path, prediction_fig_path)) 

def test_update(src_dir, dataset_id=0, split_name='train', log_dir='logs', map_resolution=None, is_online_fig=False):
    """ This function is created for you to debug your update step.
        It creates three particles: np.array([[0.2, 2, 3],[0.4, 2, 5],[0.1, 2.7, 4]]) 
        * Note that it has the shape 3 x num_p so the first particle is [0.2, 0.4, 0.1]
        It builds the first map and updates the 3 particles for 1 time step.  
        After running this function, you should expect that the weight of the first particle is the largest since it is the closest to [0, 0, 0] 
    TODO: run this test_update and look at the result printed. Make sure that the best particle is the first one with weight = 1 
    >>> **** Best particle so far [0.2 0.4 0.1] has weight 1.0
    >>> Final particles:
        np.array([[0.2, 0.4, 0.1],
                  [2. , 2. , 2.7],
                  [3. , 5. , 4. ]]))
    """    
    # Create a SLAM instance
    slam_inc = SLAM()
    
    # Read data
    slam_inc._read_data(src_dir, dataset_id, split_name)
    print('\n--------------Testing Update Module ---------------')
    
    # Generate three particles randomly to test upate
    num_p= 3
    weights = np.ones(num_p)*1.0/num_p
    particles = np.array([[0.2, 2, 3],[0.2,2,3],[0.1,2,3]])
    particles = np.array([[0.2, 2, 3],[0.4, 2, 5],[0.1, 2.7, 4]]) 
    # Initialize particles
    slam_inc._init_particles(num_p=num_p, particles=particles, weights=weights)

    # Initialize the map 
    slam_inc._init_map(map_resolution)

    # print slam_inc.grid.log_odds
    logging.debug('=> Done initialization')
    slam_inc._build_first_map()
    time.sleep(2)


    # Update one step
    slam_inc._update(1,fig='on')
    print(">> Final particles:\n", slam_inc.particles_.T)
    
    print("------------------------------------")
    print("You should compare your printed results with what given in the above comment of the test_update func!")

def particle_SLAM(src_dir, dataset_id=0, split_name='train', running_mode='test_SLAM', log_dir='logs'):
    '''Your main code is here.
    '''
    ###############################################################################################  
    #* Student's input
    #TODO: change the resolution of the map - the distance between two cells (meters)
    map_resolution = 0.05    

    # Number of particles 
    #TODO: change the number of particles
    num_p = 100

    #TODO: change the process' covariance matrix 
    mov_cov = np.array([[1e-8, 0, 0],[0, 1e-8, 0],[0, 0 , 1e-8]])
        
    #TODO: set a threshold value of probability to consider a map's cell occupied  
    p_thresh = 0.9
    
    #TODO: change the threshold of the percentage of effective particles to decide resampling 
    percent_eff_p_thresh = 0.5
    
    #*End student's input
    ###############################################################################################  
    
    # Test prediction
    if running_mode == 'test_prediction':
        test_prediction(src_dir, dataset_id, split_name, log_dir)
        exit(1)
    if running_mode == 'test_update':
        test_update(src_dir, dataset_id, split_name, log_dir, map_resolution)
        exit(1)
    
    # Test SLAM
    # Create a SLAM instance
    slam_inc = SLAM()
    
    # Read data

    slam_inc._read_data(src_dir, dataset_id, split_name)
    num_steps = slam_inc.num_data_
    
    # Characterize the sensors' specifications
    slam_inc._characterize_sensor_specs(p_thresh)

    # Initialize particles
    slam_inc._init_particles(num_p, mov_cov, percent_eff_p_thresh=percent_eff_p_thresh) 

    # Iniitialize the map
    slam_inc._init_map(map_resolution)
   
    # Starting time index
    t0 = 0 

    # Initialize the particle's poses using the lidar measurements at the starting time
    slam_inc.particles_[:,0] = slam_inc.lidar_.data_[t0]['pose'][0]
    # Indicator to notice that map has not been built
    build_first_map = False

    # iterate next time steps
    all_particles = deepcopy(slam_inc.particles_)
    num_resamples = 0
    
    for t in tqdm.tqdm(range(t0, num_steps-t0)):
        # Ignore lidar scans that are obtained before the first IMU
        # if slam_inc.lidar_.data_[t]['t'][0][0] - slam_inc.joints_.data_['ts'][0][0] < 0:
        #     continue
        if not build_first_map:
            slam_inc._build_first_map(t)
            t0 = t
            build_first_map = True
            continue

        # Prediction
        slam_inc._predict(t)

        # Update
        slam_inc._update(t,t0=t0,fig='off')

        # Resample particles if necessary
        num_eff = 1.0/np.sum(np.dot(slam_inc.weights_,slam_inc.weights_))
        logging.debug('>> Number of effective particles: %.2f'%num_eff)
        if num_eff < slam_inc.percent_eff_p_thresh_*slam_inc.num_p_:
            num_resamples += 1
            logging.debug('>> Resampling since this < threshold={0}| Resampling times/t = {1}/{2}'.format(\
                slam_inc.percent_eff_p_thresh_*slam_inc.num_p_, num_resamples, t-t0 + 1))
            [slam_inc.particles_,slam_inc.weights_] = prob.stratified_resampling(\
                slam_inc.particles_,slam_inc.weights_,slam_inc.num_p_)
        
        # Plot the estimated trajectory
        if (t - t0 + 1)%50000 == 0 or t==num_steps-1:
            # Save the result 
            log_file = log_dir + '/SLAM_' + split_name + '_' + str(dataset_id) + '.pkl'
            try:
                with open(log_file, 'wb') as f:
                    pickle.dump(slam_inc,f,pickle.HIGHEST_PROTOCOL)
                print(">> Save the result to: %s"%log_file)
            except Exception as e:
                print('Unable to write data to', log_file, ':', e)
                raise

        
            # Gen map + trajectory
            MAP_2_display = genMap(slam_inc, t)
            MAP_fig_path = log_dir + '/processing_SLAM_map_'+ split_name + '_' + str(t) + '.jpg'
            cv2.imwrite(MAP_fig_path, MAP_2_display)
            plt.title('Estimated Map at time stamp %d/%d'%(t, num_steps - t0 + 1))
            plt.imshow(MAP_2_display)
            plt.pause(0.01)
        
            # logging.debug(">> Save %s"%MAP_fig_path)
            
    # Return best_p which are an array of size 3xnum_data that represents the best particle over the whole time stamp
    return slam_inc.best_p_
    #return slam_inc.best_p_
 
def main():
    parser = argparse.ArgumentParser('main function')
    parser.add_argument('--src_dir',    help="Directory to the data...i.e: data", default='data', type=str)
    parser.add_argument('--log_dir',    help="Directory to save logs",            default='logs', type=str)
    parser.add_argument('--dataset_id', help="Dataset id=0, 1, 2. ..?",           default=0, type=int)
    parser.add_argument('--split_name', help="Train or test split?",              default='train', type=str)
    parser.add_argument('--running_mode', help="Test prediction/Update/SLAM?",    default='test_SLAM', type=str,\
                                        choices=['test_SLAM', 'test_prediction', 'test_update'])

    args   = parser.parse_args()
    
    # Run the particle SLAM 
    
    best_p_array = particle_SLAM(args.src_dir, args.dataset_id, args.split_name, args.running_mode, args.log_dir)
    #==================
    '''
    only best_p_array
    '''
    # This is for TA's 
    #
    
    
    #no return
    
    
    ##################

########################################################
if __name__ == "__main__":
    main()
