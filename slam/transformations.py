import math
import numpy
import numpy as np
from math import cos, sin
from copy import deepcopy

def twoDSmartPlus(x1,x2,type='pose'):
    """Return smart plus of two poses in order (x1 + x2)as defined in particle filter
    :param
    x1,x2: two poses in form of (x,y,theta)
    type:  which type of return you choose. 'pose' to return (x,y,theta) form
                                            ' rot' to return transformation matrix (3x3)
    """
    theta1 = x1[2]
    R_theta1 = twoDRotation(theta1)
    # print '------ Rotation theta1:', R_theta1
    theta2 = x2[2]
    sum_theta = theta2 + theta1
    p1 = x1[0:2]
    p2 = x2[0:2]
    # print 'p2:', p2
    trans_of_u = p1 + np.dot(R_theta1, p2)
    # print '------ transition of u:', trans_of_u-p1
    if type=='pose':
        return np.array([trans_of_u[0], trans_of_u[1],sum_theta])
    # if type == 'rot'
    rot_of_u = twoDRotation(sum_theta)
    return np.array([[rot_of_u[0,0],rot_of_u[0,1],trans_of_u[0]],\
                     [rot_of_u[1,0],rot_of_u[1,1],trans_of_u[1]],\
                     [0            ,   0         ,   1]])

def twoDSmartMinus(x2,x1,type='pose'):
    """Return smart minus of two poses in order (x2 - x1)as defined in particle filter
    :param
    x1,x2: two poses in form of (x,y,theta)
    type:  which type of return you choose. 'pose' to return (x,y,theta) form
                                            ' rot' to return transformation matrix (3x3)
    """
    theta1 = x1[2]
    R_theta1 = twoDRotation(theta1)
    theta2 = x2[2]
    delta_theta = theta2 - theta1
    p1 = x1[0:2]
    p2 = x2[0:2]
    trans_of_u = np.dot(R_theta1.T, (p2-p1))
    if type=='pose':
        return np.array([trans_of_u[0], trans_of_u[1],delta_theta])
    # if type == 'rot'
    rot_of_u = twoDRotation(delta_theta)
    return np.array([[rot_of_u[0,0],rot_of_u[0,1],trans_of_u[0]],\
                     [rot_of_u[1,0],rot_of_u[1,1],trans_of_u[1]],\
                     [0            ,   0         ,   1]])


def twoDRotation(theta):
    """Return rotation matrix of rotation in 2D by theta"""
    return np.array([[cos(theta), -sin(theta)],[sin(theta), cos(theta)]])

def twoDTransformation(x,y,theta):
    """Return transformation matrix of rotation in 2D by theta combining with a translation
    (x, y)"""
    return np.array([[cos(theta), -sin(theta), x],[sin(theta), cos(theta), y],[0,0,1]])
def simple_vector_norm(qv):
    return math.sqrt(abs(np.sum(np.dot(qv,qv))))

def quaternion_exp(q):
    if sum([abs(v) for v in q]) <= 1e-6:
        return np.array([1,0,0,0],dtype=np.float64)
    # q = unit_vector(q)
    qs = q[0]
    qv = q[1:4]


    if sum([abs(v) for v in qv]) <= 1e-6:
        new_qv = [0,0,0]
    else:
        new_qv = [math.exp(qs)*math.sin(simple_vector_norm(qv))/simple_vector_norm(qv)*qv_i for qv_i in qv]
    new_qs = [math.exp(qs)*math.cos(simple_vector_norm(qv))]
    new_quat = new_qs + new_qv
    return np.asarray(new_quat)

def quaternion_log(q):
    # q = unit_vector(q)
    qs = q[0]
    qv = q[1:4]
    if sum([abs(i) for i in q]) <= 1e-6:
        return np.array([0,0,0,0],dtype=np.float64)
    new_qs = np.array([math.log(simple_vector_norm(q))])
    if sum([abs(v) for v in qv]) <= 1e-6:
        new_qv = np.array([0,0,0],dtype=np.float64)
    else:
        new_qv = np.array([math.acos(qs/simple_vector_norm(q))*qv_i/simple_vector_norm(qv) for qv_i in qv])
        # new_qv = [math.atan2(simple_vector_norm(qv),qs)*qv_i/simple_vector_norm(qv) for qv_i in qv]
    new_quat = np.concatenate((new_qs, new_qv))
    return np.asarray(new_quat)

def quaternion_rotate_vector(q):
    """Quaternion to rotation vector"""
    if abs(1 - q[0]**2) < 1e-10:
        return np.array([0,0,0],dtype=np.float64)
    return q[1:4]*2*math.acos(q[0])/math.sqrt(1- q[0]**2)
def quaternion_change_sign_qs(q):
    """Change size of angle but retain the quaternion"""


def quaternion_mean(quat_list,weight_list,q_init=None,num_quat=0,epsilon=1e-3):
    """ Weighted average of quaternions
    :return
    [q_est, qe_set] where q_est is the average and qe_set
    is nx 4 matrix storing set of q_est - q_i's used for calculating
    covariance quaternion"""
    # initial guess
    if q_init == None:
        q_est = quat_list[0]
    else:
        q_est = q_init
    # q_est = np.mean(quat_list,axis=0)
    # q_est = np.array([1,0.2,0,0])
    # print '--- q_est: ', q_est
    T = 50
    if num_quat == 0:
        num_quat = np.sum([1 for _ in quat_list])

    if weight_list == None:
        weight_list = [1.0/num_quat for _ in range(num_quat)]
    # print num_quat
    for t in range(T):
        # print '---------iter {0}-----------'.format(t+1)
        ev = np.empty((num_quat,3))
        e_i_set = np.empty((num_quat,3))
        for i in range(num_quat):
            # print '----i = ', i
            # print quat_list[i]
            inv = quaternion_inverse(q_est)
            qe = quaternion_multiply(quat_list[i],quaternion_inverse(q_est))
            # print 'qe:', qe
            ev_i = 2*quaternion_log(qe)
            # # print 'ev_i:',ev_i
            ev_i = ev_i[1:4]
            if simple_vector_norm(ev_i) < epsilon:
                ev_i = np.array([0,0,0])
            else:
                ev_i *= (-math.pi + math.fmod(simple_vector_norm(ev_i) + math.pi,2*math.pi))/simple_vector_norm(ev_i)
            e_i_set[i] = deepcopy(ev_i)
            ev[i]= weight_list[i]*ev_i
            # print '===== new ev:',ev
        # print '--ev is:', ev
        aver_ev = np.sum(ev,axis=0)
        # print '--ev average:',aver_ev
        qe = np.concatenate(([0],aver_ev/2.0))
        # print 'qe is: ', qe
        # exit(1)
        # update q_est
        exp_qe = quaternion_exp(qe)
        # print 'exp_qe:',exp_qe
        q_est = quaternion_multiply(exp_qe,q_est)
        # q_est = q_est/simple_vector_norm(q_est)
        # print '---Error: ',simple_vector_norm(ev)
        if simple_vector_norm(aver_ev) < epsilon:
            break
        # print '--- q_est: ', q_est
    q_est = np.array([q for q in q_est])
    # if q_est[0] >= 0.9999:
    #     q_est[0] = 1
    #     q_est[1:4] = 0
    # if q_est[0] <= -0.9999:
    #     q_est[0] = 1
    #     q_est[1:4] = 0
    #
    # for i in range(1,4):
    #     if q_est[i] >= 0.9999:
    #         q_est[i] = 1
    #     if q_est[i] <= -0.9999:
    #         q_est[i] = -1
    #     if abs(q_est[i]) <= 1e-6:
    #         q_est[i] = 0
    return [q_est,e_i_set]

def test_quaternion_mean():
    """
    If you look at the two delta quaternions (in angle axis form), they should be of similar magnitude
    and axis that are negative of each other. This indicates the average quaternion is "midway" between
    the two quaternions in the sense that rotating it along an axis clockwise by a certain angle gives you q1,
    and rotating counterclockwise by the same angle gives you q2"""
    print ('------test_quaternion_mean()-----')
    print ('Expectation: q_delta1[0] = q_delta2[0] and q_delta1[1:-1] = -q_delta2[1:-1] \n')
    s = math.sin(math.sqrt(30./180*3.14))
    c = math.cos(math.sqrt(30./180*3.14))

    q1 = np.array([-c,-s,0,0])
    q2 = np.array([-c,s,0,0])
    print ('q1, q2:',q1, q2)
    [qmean,_] =  quaternion_mean(np.array([q1,q2]),weight_list=np.array([0.5,0.5]))

    print ('qmean:',qmean)

    qdelta_1 = quaternion_multiply(q1,quaternion_inverse(qmean))
    qdelta_2 = quaternion_multiply(q2,quaternion_inverse(qmean))

    print ('Q delta: ')
    print (qdelta_1)
    print (qdelta_2)

    # q1 = np.array([-c,-s,s+0.2,0])
    # q2 = np.array([-c,s,0,c-0.1])
    q1 = np.array([-0.8,0.6,0.2,-0.7])
    q2 = np.array([-0.6,0.8,0,0])
    q1 = q1/vector_norm(q1)
    q2 = q2/vector_norm(q2)

    [qmean,_] =  quaternion_mean(np.array([q1,q2]),weight_list=np.array([0.5,0.5]))
    print ('\nq1, q2:',q1, q2)
    print ('qmean:',qmean)

    qdelta_1 = quaternion_multiply(q1,quaternion_inverse(qmean))
    qdelta_2 = quaternion_multiply(q2,quaternion_inverse(qmean))

    print ('Q delta: ')
    print (qdelta_1)
    print (qdelta_2)

def rot_x_axis(phi):
    """Return rotation matrix of a roation around x axis an angle equal to phi"""
    rot_matrix = np.array([[1, 0, 0], \
                           [0, cos(phi), -sin(phi)],\
                           [0, sin(phi), cos(phi)]])
    return rot_matrix

def rot_y_axis(phi):
    """Return rotation matrix of a roation around y axis an angle equal to phi"""
    rot_matrix = np.array([[cos(phi), 0, sin(phi)], \
                           [0, 1, 0],\
                           [-sin(phi),0, cos(phi)]])
    return rot_matrix

def rot_z_axis(phi):
    """Return rotation matrix of a roation around z axis an angle equal to phi"""
    rot_matrix = np.array([[cos(phi), -sin(phi), 0], \
                           [sin(phi), cos(phi), 0],\
                           [0, 0, 1]])
    return rot_matrix

def homo_transform(rot_matrix,p):
    """Return homogeneous transformation matrix given by a roation matrix and a transition
    :param
    rot_matrix: 3x 3 rotation matrix
    p: 1x3 or (3L,) transition"""
    first_part = np.vstack((rot_matrix,np.zeros(3)))
    second_part = np.array([[p[0]],[p[1]],[p[2]],[1]])
    return np.hstack((first_part,second_part))

def test_homo_transform():
    rot_matrix = np.array([[1,0,0],[0,1,0],[0,0,1]])
    p = np.array([2,4,6])
    print(homo_transform(rot_matrix,p))

def mat2euler(R):
    """Return ZYX Euler angles from rotation matrix 3x3
    return z, y, x according to Z, Y, X or yaw, pitch, roll
    """
    S = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = S < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], S)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], S)
        z = 0

    return np.array([z, y, x])

def quat2euler(q):
    """Return ZYX Euler angles from a quaternion
    return z, y, x according to Z, Y, X or yaw, pitch, roll
    """
    rot = quaternion_matrix(q)
    zyx_euler = mat2euler(rot)
    return zyx_euler


def rotation_matrix(angle, direction, point=None):
    """Return matrix to rotate about axis defined by point and direction.
    """
    sina = math.sin(angle)
    cosa = math.cos(angle)
    direction = unit_vector(direction[:3])
    # rotation matrix around unit vector
    R = numpy.diag([cosa, cosa, cosa])
    R += numpy.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += numpy.array([[ 0.0,         -direction[2],  direction[1]],
                      [ direction[2], 0.0,          -direction[0]],
                      [-direction[1], direction[0],  0.0]])
    M = numpy.identity(4)
    M[:3, :3] = R
    if point is not None:
        # rotation not around origin
        point = numpy.array(point[:3], dtype=numpy.float64, copy=False)
        M[:3, 3] = point - numpy.dot(R, point)
    return M


def rotation_from_matrix(matrix):
    """Return rotation angle and axis from rotation matrix.
    """
    R = numpy.array(matrix, dtype=numpy.float64, copy=False)
    R33 = R[:3, :3]
    w, W = numpy.linalg.eig(R33.T)
    i = numpy.where(abs(numpy.real(w) - 1.0) < 1e-8)[0]

    direction = numpy.real(W[:, i[-1]]).squeeze()
    w, Q = numpy.linalg.eig(R)
    i = numpy.where(abs(numpy.real(w) - 1.0) < 1e-8)[0]
    point = numpy.real(Q[:, i[-1]]).squeeze()
    point /= point[3]
    cosa = (numpy.trace(R33) - 1.0) / 2.0
    if abs(direction[2]) > 1e-8:
        sina = (R[1, 0] + (cosa-1.0)*direction[0]*direction[1]) / direction[2]
    elif abs(direction[1]) > 1e-8:
        sina = (R[0, 2] + (cosa-1.0)*direction[0]*direction[2]) / direction[1]
    else:
        sina = (R[2, 1] + (cosa-1.0)*direction[1]*direction[2]) / direction[0]
    angle = math.atan2(sina, cosa)
    return angle, direction, point


def euler_matrix(ai, aj, ak, axes='sxyz'):
    """Return homogeneous rotation matrix from Euler angles and axis sequence.
    ai, aj, ak : Euler's roll, pitch and yaw angles
    axes : One of 24 axis sequences as string or encoded tuple
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    if frame:
        ai, ak = ak, ai
    if parity:
        ai, aj, ak = -ai, -aj, -ak

    si, sj, sk = math.sin(ai), math.sin(aj), math.sin(ak)
    ci, cj, ck = math.cos(ai), math.cos(aj), math.cos(ak)
    cc, cs = ci*ck, ci*sk
    sc, ss = si*ck, si*sk

    M = numpy.identity(4)
    if repetition:
        M[i, i] = cj
        M[i, j] = sj*si
        M[i, k] = sj*ci
        M[j, i] = sj*sk
        M[j, j] = -cj*ss+cc
        M[j, k] = -cj*cs-sc
        M[k, i] = -sj*ck
        M[k, j] = cj*sc+cs
        M[k, k] = cj*cc-ss
    else:
        M[i, i] = cj*ck
        M[i, j] = sj*sc-cs
        M[i, k] = sj*cc+ss
        M[j, i] = cj*sk
        M[j, j] = sj*ss+cc
        M[j, k] = sj*cs-sc
        M[k, i] = -sj
        M[k, j] = cj*si
        M[k, k] = cj*ci
    return M


def euler_from_matrix(matrix, axes='sxyz'):
    """Return Euler angles from rotation matrix for specified axis sequence.
    axes : One of 24 axis sequences as string or encoded tuple
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    M = numpy.array(matrix, dtype=numpy.float64, copy=False)[:3, :3]
    if repetition:
        sy = math.sqrt(M[i, j]*M[i, j] + M[i, k]*M[i, k])
        if sy > _EPS:
            ax = math.atan2( M[i, j],  M[i, k])
            ay = math.atan2( sy,       M[i, i])
            az = math.atan2( M[j, i], -M[k, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2( sy,       M[i, i])
            az = 0.0
    else:
        cy = math.sqrt(M[i, i]*M[i, i] + M[j, i]*M[j, i])
        if cy > _EPS:
            ax = math.atan2( M[k, j],  M[k, k])
            ay = math.atan2(-M[k, i],  cy)
            az = math.atan2( M[j, i],  M[i, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2(-M[k, i],  cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return ax, ay, az


def euler_from_quaternion(quaternion, axes='sxyz'):
    """Return Euler angles from quaternion for specified axis sequence.
    True

    """
    return euler_from_matrix(quaternion_matrix(quaternion), axes)


def quaternion_from_euler(ai, aj, ak, axes='sxyz'):
    """Return quaternion from Euler angles and axis sequence.
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis + 1
    j = _NEXT_AXIS[i+parity-1] + 1
    k = _NEXT_AXIS[i-parity] + 1

    if frame:
        ai, ak = ak, ai
    if parity:
        aj = -aj

    ai /= 2.0
    aj /= 2.0
    ak /= 2.0
    ci = math.cos(ai)
    si = math.sin(ai)
    cj = math.cos(aj)
    sj = math.sin(aj)
    ck = math.cos(ak)
    sk = math.sin(ak)
    cc = ci*ck
    cs = ci*sk
    sc = si*ck
    ss = si*sk

    q = numpy.empty((4, ))
    if repetition:
        q[0] = cj*(cc - ss)
        q[i] = cj*(cs + sc)
        q[j] = sj*(cc + ss)
        q[k] = sj*(cs - sc)
    else:
        q[0] = cj*cc + sj*ss
        q[i] = cj*sc - sj*cs
        q[j] = cj*ss + sj*cc
        q[k] = cj*cs - sj*sc
    if parity:
        q[j] *= -1.0

    return q


def quaternion_about_axis(angle, axis):
    """Return quaternion for rotation about axis.
    """
    q = numpy.array([0.0, axis[0], axis[1], axis[2]])
    qlen = vector_norm(q)
    if qlen > _EPS:
        q *= math.sin(angle/2.0) / qlen
    q[0] = math.cos(angle/2.0)
    return q


def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion.
    """
    q = numpy.array(quaternion, dtype=numpy.float64, copy=True)
    n = numpy.dot(q, q)
    if n < _EPS:
        return numpy.identity(4)
    q *= math.sqrt(2.0 / n)
    q = numpy.outer(q, q)
    return numpy.array([
        [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], 0.0],
        [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], 0.0],
        [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], 0.0],
        [                0.0,                 0.0,                 0.0, 1.0]])


def quaternion_from_matrix(matrix, isprecise=False):
    """Return quaternion from rotation matrix.
    """
    M = numpy.array(matrix, dtype=numpy.float64, copy=False)[:4, :4]
    if isprecise:
        q = numpy.empty((4, ))
        t = numpy.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 1, 2, 3
            if M[1, 1] > M[0, 0]:
                i, j, k = 2, 3, 1
            if M[2, 2] > M[i, i]:
                i, j, k = 3, 1, 2
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = numpy.array([[m00-m11-m22, 0.0,         0.0,         0.0],
                         [m01+m10,     m11-m00-m22, 0.0,         0.0],
                         [m02+m20,     m12+m21,     m22-m00-m11, 0.0],
                         [m21-m12,     m02-m20,     m10-m01,     m00+m11+m22]])
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = numpy.linalg.eigh(K)
        q = V[[3, 0, 1, 2], numpy.argmax(w)]
    if q[0] < 0.0:
        numpy.negative(q, q)
    return q


def quaternion_multiply(quaternion1, quaternion0):
    """Return multiplication of two quaternions"""
    a1, b1, c1, d1 = quaternion0
    a2, b2, c2, d2 = quaternion1
    return numpy.array([a1*a2 - b1*b2 - c1*c2 - d1*d2,
                        a1*b2 + a2*b1 + c1*d2 - c2*d1,
                        a1*c2 + a2*c1 - b1*d2 + b2*d1,
                        a1*d2 + a2*d1 + b1*c2 - b2*c1], dtype=numpy.float64)


def quaternion_conjugate(quaternion):
    """Return conjugate of quaternion.
    """
    q = numpy.array(quaternion, dtype=numpy.float64, copy=True)
    numpy.negative(q[1:], q[1:])
    return q


def quaternion_inverse(quaternion):
    """Return inverse of quaternion.
    """
    q = np.array([quaternion[0],-quaternion[1],-quaternion[2],-quaternion[3]])
    return q/simple_vector_norm(quaternion)**2


def quaternion_slerp(quat0, quat1, fraction, spin=0, shortestpath=True):
    """Return spherical linear interpolation between two quaternions.
    """
    q0 = unit_vector(quat0[:4])
    q1 = unit_vector(quat1[:4])
    if fraction == 0.0:
        return q0
    elif fraction == 1.0:
        return q1
    d = numpy.dot(q0, q1)
    if abs(abs(d) - 1.0) < _EPS:
        return q0
    if shortestpath and d < 0.0:
        # invert rotation
        d = -d
        numpy.negative(q1, q1)
    angle = math.acos(d) + spin * math.pi
    if abs(angle) < _EPS:
        return q0
    isin = 1.0 / math.sin(angle)
    q0 *= math.sin((1.0 - fraction) * angle) * isin
    q1 *= math.sin(fraction * angle) * isin
    q0 += q1
    return q0


# epsilon for testing whether a number is close to zero
_EPS = numpy.finfo(float).eps * 4.0

# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())


def vector_norm(data, axis=None, out=None):
    """Return length, i.e. Euclidean norm, of ndarray along axis.
    """
    data = numpy.array(data, dtype=numpy.float64, copy=True)
    if out is None:
        if data.ndim == 1:
            return math.sqrt(numpy.dot(data, data))
        data *= data
        out = numpy.atleast_1d(numpy.sum(data, axis=axis))
        numpy.sqrt(out, out)
        return out
    else:
        data *= data
        numpy.sum(data, axis=axis, out=out)
        numpy.sqrt(out, out)


def unit_vector(data, axis=None, out=None):
    """Return ndarray normalized by length, i.e. Euclidean norm, along axis.
    """
    if out is None:
        data = numpy.array(data, dtype=numpy.float64, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(numpy.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = numpy.array(data, copy=False)
        data = out
    length = numpy.atleast_1d(numpy.sum(data*data, axis))
    numpy.sqrt(length, length)
    if axis is not None:
        length = numpy.expand_dims(length, axis)
    data /= length
    if out is None:
        return data


def random_vector(size):
    """Return array of random doubles in the half-open interval [0.0, 1.0).
    """
    return numpy.random.random(size)


def vector_product(v0, v1, axis=0):
    """Return vector perpendicular to vectors.
    """
    return numpy.cross(v0, v1, axis=axis)


def angle_between_vectors(v0, v1, directed=True, axis=0):
    """Return angle between vectors.
    """
    v0 = numpy.array(v0, dtype=numpy.float64, copy=False)
    v1 = numpy.array(v1, dtype=numpy.float64, copy=False)
    dot = numpy.sum(v0 * v1, axis=axis)
    dot /= vector_norm(v0, axis=axis) * vector_norm(v1, axis=axis)
    return numpy.arccos(dot if directed else numpy.fabs(dot))


def inverse_matrix(matrix):
    """Return inverse of square transformation matrix"""
    return numpy.linalg.inv(matrix)


def concatenate_matrices(*matrices):
    """Return concatenation of series of transformation matrices.
    """
    M = numpy.identity(4)
    for i in matrices:
        M = numpy.dot(M, i)
    return M


def is_same_transform(matrix0, matrix1):
    """Return True if two matrices perform same transformation.
    """
    matrix0 = numpy.array(matrix0, dtype=numpy.float64, copy=True)
    matrix0 /= matrix0[3, 3]
    matrix1 = numpy.array(matrix1, dtype=numpy.float64, copy=True)
    matrix1 /= matrix1[3, 3]
    return numpy.allclose(matrix0, matrix1)



############################################################################

if __name__ == "__main__":
    print('transformation')
    test_homo_transform()
