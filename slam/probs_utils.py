#from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import sys,os
sys.path.insert(0, 'MapUtils')
import load_data as ld
import time 
import math 
import transformations as tf

def stratified_resampling(means,weights,N):
    """Return a set of resampled points
    :param
    N: number of points
    means: 3xN array means of points
    weights: (N,) array weights of these points
    :return
    resampled_means: 3xN means of new points
    resampled_weights: 1xN array of new weights"""
    means = np.asarray(means)
    weights = np.asarray(weights)
    c = weights[0]
    j = 0
    u = np.random.uniform(0,1.0/N)
    new_mean = np.zeros(means.shape)
    new_weights = np.zeros(weights.shape)
    # print '-init u, c, j:', u, c, j
    for k in range(N):
        beta = u + float(k)/N
        # print '--beta:', beta
        while beta > c:
            j += 1
            c += weights[j]
        # add point
        new_mean[:,k] = means[:,j]
        new_weights[k] = 1.0/N
    return [new_mean,new_weights]

def test_stratified_resampling():
    weights = np.array([4, 5, 3, 2, 1])
    weights = weights*1.0/sum(weights)
    means = np.array([[4,5,3,2,1],[4,5,3,2,1],[4,5,3,2,1]])
    print( '- orig means:', means)
    print( '- orig weights:', weights)
    print( '----- stratified_resampling(means,weights,N)------:')
    print( stratified_resampling(means,weights,len(weights)))

    weights = np.array([np.random.randint(0,20,1)[0] for i in range(100)])
    means = np.array([weights for i in range(100)])
    weights = weights*1.0/sum(weights)
    start = time.clock()
    print( '----- stratified_resampling(means,weights,N)------:')
    print( stratified_resampling(means,weights,len(weights)))
    print( '----- Running time for N = 100:', time.clock() - start)



def logSumExp(log_weights,max_log_weight):
	delta_log_weight = np.sum(np.exp(log_weights - max_log_weight))
	# print '--delta_log_weight:', delta_log_weight
	return np.log(delta_log_weight)

def simple_update_weights(weights,correlations):
    """Use a simple but quite incorrect way to update weights"""
    weights = weights*correlations
    sum_weights = np.sum(weights)
    weights /= sum_weights
    return weights

def update_weights(weights,correlations):
	"""Return weights based on correlation values"""
	# update weights
	log_weights = np.log(weights)
	# print '--log_weights:', log_weights
	log_weights += correlations
	# print '---- add correlations to obtain', log_weights
	# normalize log_weights
	max_log_weight = np.max(log_weights)
	# print '---- max_log_weight:', max_log_weight
	log_weights = log_weights - max_log_weight - logSumExp(log_weights,max_log_weight)
	# retreive weight values
	return np.exp(log_weights)

def test_simple_update_weights():
	N = 4
	weights = np.ones(N)*1.0/N
	print('--Weights:',weights)
	for i in range(5):
		print ('\n--------Step {0}---------'.format(i))
		correlations = np.random.randint(0,100,N).reshape(N)
		print ('--Correlations:',correlations)
		updated_weights = simple_update_weights(weights,correlations)
		print ('** updated weights:')
		print (updated_weights)
		weights = updated_weights

def test_update_weights():
	N = 4
	weights = np.ones(N)*1.0/N
	print ('--Weights:',weights)
	for i in range(5):
		print ('\n--------Step {0}---------'.format(i))
		correlations = np.random.randint(0,100,N).reshape(N)
		print ('--Correlations:',correlations)
		updated_weights = update_weights(weights,correlations)
		print ('** updated weights:')
		print (updated_weights)
		weights = updated_weights

def rec_pdf_from_log_odds(odd_value):
	return 1 - 1/(1 + np.exp(odd_value))

def log_thresh_from_pdf_thresh(pdf_thresh):
	"""Return a corresponding threshold for log-odds value, given the threshold of
	pdf
	Equation: pdf =  1 - 1/(1 + np.exp(odd_value))
	So 1 + np.exp(odd_value) = 1/(1-pdf)
	np.exp(odd_value) = pdf/(1-pdf)
	odd_value = log(pdf/(1-pdf))"""
	return math.log(pdf_thresh/(1-pdf_thresh))

def mapCorrelation(map,occupied_indices):
  # Ty 11:18 PM March 07 changed. 
  # For each index, add
	return np.sum(map[occupied_indices[0,:],occupied_indices[1,:]])


# odd_value = [5*math.log(1.0/9),math.log(9)]
# print rec_pdf_from_log_odds(odd_value)
if __name__ == "__main__":
	test_update_weights()
    # test_stratified_resampling()
    # test_simple_update_weights()
