import numpy as np
import os, sys, time
import matplotlib.pyplot as plt
import cv2 
if (sys.version_info > (3, 0)):
    import pickle
else:
    import cPickle as pickle
import argparse

def gen_trajectory(args):
    log_dir    = args.log_dir
    dataset_id = args.dataset_id 
    split_name = args.split_name

    # save it to file
    log_file = log_dir + '/SLAM_' + split_name + '_' + str(dataset_id) + '.pkl'
    try:
        with open(log_file, 'rb') as f:
            slam_inc = pickle.load(f)
    except Exception as e:
        print('Unable to process data from', log_file, ':', e)
        raise

    state = slam_inc.best_p_
    plt.figure(1)
    plt.plot(state[0,:],state[1,:],'r')
    # plt.gca().invert_yaxis()
    SLAM_fig_path = log_dir + '/SLAM_trajectory_'+ split_name + '_' + str(dataset_id) + '.jpg'
    plt.savefig(SLAM_fig_path)
    plt.title('Trajectory using update')
    plt.show()

    print(">> Generating map!")
    MAP_2_display = genMap(slam_inc)
    MAP_fig_path = log_dir + '/SLAM_map_'+ split_name + '_' + str(dataset_id) + '.jpg'
    cv2.imwrite(MAP_fig_path, MAP_2_display)

    title = 'Generated Map'
    plt.title(title)
    plt.imshow(MAP_2_display)
    plt.show()
    # # plt.gca().invert_yaxis()
    #plt.close()
    
    
    print(">> Generating logodd visualization!")
    #plot log-odd values
    fig3 = plt.figure(3)
    ax3 = fig3.gca(projection='3d')
    X, Y = np.meshgrid(np.arange(0,slam_inc.MAP_['sizex']),np.arange(0,slam_inc.MAP_['sizey']))
    ax3.plot_surface(X,Y,slam_inc.log_odds_,linewidth=0,cmap=plt.cm.jet, antialiased=False,rstride=1, cstride=1)
    logodd_fig_path = log_dir + '/SLAM_logodd_'+ split_name + '_' + str(dataset_id) + '.jpg'
    plt.savefig(logodd_fig_path)
    plt.show()


def genMap(slam_inc, end_t=None):
    MAP            = slam_inc.MAP_
    best_p         = slam_inc.best_p_
    log_odds       = slam_inc.log_odds_
    logodd_thresh = slam_inc.logodd_thresh_
    best_p_indices = slam_inc.best_p_indices_
    t0             = slam_inc.t0
    if end_t is None:
        end_t          = slam_inc.num_data_ - 1

    MAP_2_display = 255*np.ones((MAP['map'].shape[0],MAP['map'].shape[1],3),dtype=np.uint8)

    wall_indices = np.where(log_odds > logodd_thresh)
    MAP_2_display[list(wall_indices[0]),list(wall_indices[1]),:] = [0,0,0]
    unexplored_indices = np.where(abs(log_odds) < 1e-1)
    MAP_2_display[list(unexplored_indices[0]),list(unexplored_indices[1]),:] = [150,150,150]
    MAP_2_display[best_p_indices[0,t0:end_t],best_p_indices[1,t0:end_t],:] = [70,70,228]

    
    return MAP_2_display
if __name__ == "__main__":
    parser = argparse.ArgumentParser('Gen report function')
    parser.add_argument('--log_dir',    help="Directory to save logs",            default='logs', type=str)
    parser.add_argument('--dataset_id', help="Dataset id=0, 1, 2. ..?",           default=0, type=int)
    parser.add_argument('--split_name', help="Train or test split?",              default='train', type=str)

    args   = parser.parse_args()
    
    gen_trajectory(args)
