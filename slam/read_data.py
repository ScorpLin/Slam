import numpy as np
import matplotlib.pyplot as plt
import sys,os
sys.path.insert(0, 'MapUtils')
from bresenham2D import bresenham2D
import load_data as ld

import time 
import math 
import transformations as tf 
from math import cos, sin 
class JOINTS:
	"""Return data collected from IMU and anything not related to lidar
	return 
	self.data_['ts'][0]: 1 x N array of absolute time values 
	self.data_['pos']: 35xN array of sth we donnot care about 
	self.data_['rpy']: 3x N array of roll, pitch, yaw angles over time 
	self.data_['head_angles']: 2x N array of head angles (neck angle, head angle)
	"""
	def __init__(self,dataset='0',data_folder='data',name=None):
		if name == None:
			joint_file = os.path.join(data_folder,'train_joint'+dataset)
		else:
			joint_file = os.path.join(data_folder,name)
		joint_data = ld.get_joint(joint_file)
		self.num_measures_ = len(joint_data['ts'][0])
		self.data_ = joint_data 
		self.head_angles = self.data_['head_angles'] 
		# head =  self.data_['head_angles'] 
		# fig = plt.figure()
		# ax1 = fig.add_subplot(211)
		# ax1.plot(range(head.shape[1]),head[0,:])
		# ax2 = fig.add_subplot(212)
		# ax2.plot(range(head.shape[1]),head[1,:])
		# plt.show()
	def _get_joint_index(self,joint):
	    jointNames = ['Neck','Head','ShoulderL', 'ArmUpperL', 'LeftShoulderYaw','ArmLowerL','LeftWristYaw','LeftWristRoll','LeftWristYaw2','PelvYL','PelvL','LegUpperL','LegLowerL','AnkleL','FootL','PelvYR','PelvR','LegUpperR','LegLowerR','AnkleR','FootR','ShoulderR', 'ArmUpperR', 'RightShoulderYaw','ArmLowerR','RightWristYaw','RightWristRoll','RightWristYaw2','TorsoPitch','TorsoYaw','l_wrist_grip1','l_wrist_grip2','l_wrist_grip3','r_wrist_grip1','r_wrist_grip2','r_wrist_grip3','ChestLidarPan']
	    joint_idx = 1
	    for (i,jnames) in enumerate(joint):
	        if jnames in jointNames:
	            joint_idx = i
	            break
	    return joint_idx

class LIDAR:
	"""This class return an instance lidar wiht range of theta (in radian), number of measurments,
	relative time (w.r.t previous time step)...
	to retrieve information from lidar, just call
	self.data[i]['scan'] for an 1x1081 array of scans (beam) ([[....]])
	self.data[i]['pose'] for a 1 x 3 (x, y, theta) ([[....]])
	self.data[i]['t'] for an 1 x 1 array of time value  ([[....]])
	self.data[i]['t_s'] for an 1x num_measures_ array of relative time values (in seconds)
	([[....]])
	To obtain a [...] shape, need to access by doing, for example, self.data[i]['scan'][0]  """
	def __init__(self,dataset='0',data_folder='data',name=None):
		if name == None:
			lidar_file = os.path.join(data_folder,'train_lidar' +dataset)
		else:
			lidar_file = os.path.join(data_folder,name)
		lidar_data = ld.get_lidar(lidar_file)
		# substract the beginning value of yaw given by ypr
		# lidar_data[:]['rpy'][0]
		yaw_offset = lidar_data[0]['rpy'][0,2]
		for j in range(len(lidar_data)):
			lidar_data[j]['rpy'][0,2] -= yaw_offset
		self.range_theta_ = np.arange(0,270.25,0.25)*np.pi/float(180)
		self.num_measures_ = len(lidar_data)
		self.data_ = lidar_data
		# self._read_lidar(lidar_data)
		# Limitation of the lidar's ray 
		self.L_MIN = 0.001
		self.L_MAX = 30
		self.res_ = 0.25 # (angular resolution of rada = 0.25 degrees)
	# def _read_lidar(self,lidar_data):
	# 	# lidar_data type: array where each array is a dictionary with a form of 't','pose','res','rpy','scan'
	# 	ts_0 = lidar_data[0]['t'][0][0]
	# 	for i in range(len(lidar_data)):
	# 		for (k,v) in enumerate(lidar_data[i]['scan'][0]):
	# 			if v > 30:
	# 				lidar_data[i]['scan'][0][k] = 0.0
	# 		# Get delta time 
			
	# 		time_step = lidar_data[i]['t'][0] - ts_0
	# 		lidar_data[i]['t_s'] = time_step 
	# 		ts_0 = lidar_data[i]['t'][0]
	# 		# print 'i = {0}, ts = {1}, t_s = {2}'.format(i,lidar_data[i]['t'][0] , lidar_data[i]['t_s'] )
	# 	self.data_ = lidar_data

	def _remove_ground(self,h_lidar,ray_angle=None,ray_l=None,head_angle=0,h_min = 0.2):
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
			dmin = cos(head_angle)*self.L_MIN
			dmax = self.L_MAX # go to infinity so donnot need to multiply cos(head_angle)
			last_occu = 0  
			
		else:
			try:
				dmin = cos(head_angle)*self.L_MIN
				delta_l = h_min/math.sin(head_angle)
				# print 'Delta_l: ', delta_l
				l2ground = h_lidar/math.sin(head_angle)
				new_l = l2ground - delta_l
				if new_l > ray_l:
					dmax = ray_l*cos(head_angle)
					last_occu = 1  
				else:
					dmax = new_l*cos(head_angle)
					last_occu = 0 
			except:
				dmin = cos(head_angle)*self.L_MIN
				dmax = cos(head_angle)*ray_l 
				last_occu = 1

		return np.array([dmin,dmax,last_occu,ray_angle])

	def _ray2world(self,R_pose, ray_combo, unit=1):
		"""Convert ray to world x, y coordinate based on the particle position and orientation
		:input
		R_pos: (3L,) array representing pose of a particle (x, y, theta)
		ray_combo: (4L,) array of the form [[dmin,dmax,last_occu,ray_angle]]
		unit:  how much meter per grid side 
		:output
		[[sX,sY,eX,eY],[last_occu]]: x, y position of starting points and ending points of the ray 
		and whether the last cell is occupied"""
		world_to_part_rot = tf.twoDTransformation(R_pose[0],R_pose[1],R_pose[2])

		[dmin,dmax,last_occu,ray_angle] = ray_combo 
		# starting point of the line 
		sx = dmin*cos(ray_angle)/unit 
		sy = dmin*sin(ray_angle)/unit 
		ex = dmax*cos(ray_angle)/unit 
		ey = dmax*sin(ray_angle)/unit 
		# print [sx,sy,ex,ey]

		[sX, sY, _] = np.dot(world_to_part_rot,np.array([sx,sy,1])) 
		[eX, eY, _] = np.dot(world_to_part_rot,np.array([ex,ey,1])) 

		return [sX,sY,eX,eY]

	def _ray2worldPhysicPos(self,R_pose,neck_angle,ray_combo):
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
		body_to_head_rot = tf.twoDTransformation(0,0,neck_angle)
		world_to_part_rot = tf.twoDTransformation(R_pose[0],R_pose[1],R_pose[2])
		[dmin,dmax,last_occu,ray_angle] = ray_combo
		if last_occu == 0: # there is no occupied cell
			return None
		# physical position of ending point of the line w.r.t the head of the robot
		ex_h = dmax*cos(ray_angle)
		ey_h = dmax*sin(ray_angle)
		# print [sx,sy,ex,ey]
		# transform this point to obtain physical position in the body's frame
		exy1_r = np.dot(body_to_head_rot,np.array([ex_h,ey_h,1]))
		# transform this point to obtain physical position in the MAP (global)
		# Rotate these points to obtain physical position in the MAP
		[eX, eY, _] = np.dot(world_to_part_rot,exy1_r)
		return [eX,eY]

	def _physicPos2Pos(self,MAP,pose):
		""" Return the corresponding indices in MAP array, given the physical position"""
		# convert from meters to cells
		[xs0, ys0] = pose
		xis = np.ceil((xs0 - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
		yis = np.ceil((ys0 - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
		return [xis, yis]

	def _cellsFrom2Points(self,twoPoints):
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



	








# test 
# lidar = LIDAR()
# joint = JOINTS()
def test_remove_ground():
	h_lidar =  1
	lidar_beam = np.array([[range(0,5,1)]])
	# head_angle = 2*math.pi/3
	head_angle = 0.1
	lidar_beam = _remove_ground(h_lidar,lidar_beam,head_angle)
	print('New lidar beam:', lidar_beam)

def test_ray2World():
	# case 1 
	R_pose = np.array([0,0,0])
	ray_combo = [0,10,0,math.pi/3]
	# expect: [0,0,5, 5*sqrt(3) ]
	expect1 = [0,0,5, 5*math.sqrt(3)]
	real1 = _ray2world(R_pose, ray_combo)
	print('-- Case 1')
	print(_ray2world(R_pose, ray_combo))
	print(expect1 )
	# case 2
	R_pose = np.array([1,2,math.pi/3])
	ray_combo = [0,10,0,math.pi/3]
	print( '-- Case 2')
	# expect: [0,0,5, 5*sqrt(3) ]
	expect2 = [1,2,10*cos(math.pi/3)**2 + 1 - 10*sin(math.pi/3)*cos(math.pi/6), \
	               10*sin(math.pi/3)*sin(math.pi/6) + 2 + 10*cos(math.pi/3)*sin(math.pi/3)]
	real2 = _ray2world(R_pose, ray_combo)

	print( _ray2world(R_pose, ray_combo))
	print( expect2)

def test_cellsFrom2Points():
	# case 1 
	R_pose = np.array([0,0,0])
	ray_combo = [0,10,0,math.pi/3]
	# expect: [0,0,5, 5*sqrt(3) ]
	expect1 = [0,0,5, 5*math.sqrt(3)]
	real1 = _ray2world(R_pose, ray_combo)
	print('-- Case 1')
	print(  _cellsFrom2Points(_ray2world(R_pose, ray_combo)))
	print(_cellsFrom2Points(expect1))

def test_bresenham2D():
  import time
  sx = 0
  sy = 1
  print("Testing bresenham2D...")
  r1 = bresenham2D(sx, sy, 10, 5)
  r1_ex = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10],[1,1,2,2,3,3,3,4,4,5,5]])
  r2 = bresenham2D(sx, sy, 9, 6)
  r2_ex = np.array([[0,1,2,3,4,5,6,7,8,9],[1,2,2,3,3,4,4,5,5,6]])	
  if np.logical_and(np.sum(r1 == r1_ex) == np.size(r1_ex),np.sum(r2 == r2_ex) == np.size(r2_ex)):
    print("...Test passed.")
  else:
    print("...Test failed.")

  # Timing for 1000 random rays
  num_rep = 1000
  start_time = time.time()
  for i in range(0,num_rep):
	  x,y = bresenham2D(sx, sy, 500, 200)
  print("1000 raytraces: --- %s seconds ---" % (time.time() - start_time))
if __name__ == '__main__':
	# test_ray2World()
	test_cellsFrom2Points()
	# test_bresenham2D()
