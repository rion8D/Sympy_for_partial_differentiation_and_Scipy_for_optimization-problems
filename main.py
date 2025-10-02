import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint 
import scipy.optimize as opt
import sympy 

def diff_objective_func(true_tagPos_dict,order_tag, order_anchor):
	k1 = 1
	layout = [
		(('tag1','tag2'),distanceAB(true_tagPos_dict['tag1'], true_tagPos_dict['tag2'])),
		(('tag2','tag3'),distanceAB(true_tagPos_dict['tag2'], true_tagPos_dict['tag3'])),
		(('tag3','tag1'),distanceAB(true_tagPos_dict['tag3'], true_tagPos_dict['tag1']))
	]
	
	f = sympy.Symbol('f')
	tagPos_dict = {
		'tag1':[sympy.Symbol('tag1_{}'.format(i)) for i in ['x','y','z']],
		'tag2':[sympy.Symbol('tag2_{}'.format(i)) for i in ['x','y','z']],
		'tag3':[sympy.Symbol('tag3_{}'.format(i)) for i in ['x','y','z']]
	}

	anchorPos_dict = {
		'an1':[sympy.Symbol('an1_{}'.format(i)) for i in ['x','y','z']],
		'an2':[sympy.Symbol('an2_{}'.format(i)) for i in ['x','y','z']],
		'an3':[sympy.Symbol('an3_{}'.format(i)) for i in ['x','y','z']],
		'an4':[sympy.Symbol('an4_{}'.format(i)) for i in ['x','y','z']],
	}

	distance_dict ={}
	for tagID in order_tag:
		distance_dict[tagID] = {}
		for anchorID in order_anchor:
			distance_dict[tagID][anchorID] = sympy.Symbol('d_{}_{}'.format(tagID,anchorID))

	for tagID in order_tag:
		tagpos = tagPos_dict[tagID]
		for anchorID in order_anchor:
			anpos = anchorPos_dict[anchorID]
			dist = distance_dict[tagID][anchorID]
			f += (( (tagpos[0]-anpos[0])**2 + (tagpos[1]-anpos[1])**2 + (tagpos[2]-anpos[2])**2 )**(0.5) - dist) **2

	for pair,d in layout:
		t1 = tagPos_dict[pair[0]]
		t2 = tagPos_dict[pair[1]]
		f += k1*(( (t1[0]-t2[0])**2 + (t1[1]-t2[1])**2 + (t1[2]-t2[2])**2 )**(0.5) - d)**2

	#print(f)

	gradParam_list = []
	for tagID in ['tag1','tag2','tag3']:
		for i in range(len(['x','y','z'])):
			gradParam_list.append(sympy.diff(f, tagPos_dict[tagID][i]))

	#pprint(gradParam_list)
	# 偏微分

	return f, gradParam_list, tagPos_dict, anchorPos_dict, distance_dict


def gradient_method3(param, anchorPos_dict, distance_dict, order_tag, order_anchor, layout, g, symbol_tagPos_dict):
	l = []
	c = 0
	for tagID in order_tag:
			for i in range(len(['x','y','z'])):
				l.append((symbol_tagPos_dict[tagID][i], param[c]))
				c+=1
	gg = [float(i.subs(l)) for i in g]
	return np.array(gg)




def distanceAB(A,B):
	a = np.array(A)
	b = np.array(B)
	if a.shape == b.shape and a.ndim == 1:
		return np.linalg.norm(a - b)
	else:
		return None


def objective_fnc_medhotd2(param, anchorPos_dict, distance_dict, order_tag, order_anchor, layout):
	k1 = 1

	tagPos_dict = {
		'tag1':param[0:3],
		'tag2':param[3:6],
		'tag3':param[6:9]
	}

	sum_ = 0
	for tagID in order_tag:
		for anchorID in order_anchor:
			anpos = anchorPos_dict[anchorID]
			dist = distance_dict[tagID][anchorID]
			sum_ += (distanceAB(tagPos_dict[tagID],anpos)-dist)**2
	for pair,d in layout:
		sum_ += k1*(distanceAB(tagPos_dict[pair[0]],tagPos_dict[pair[1]])-d)**2
	return sum_

def objective_fnc_method3(param, anchorPos_dict, distance_dict, order_tag, order_anchor, layout, g, symbol_tagPos_dict):
	k1 = 1

	tag_dict = {
		'tag1':param[0:3],
		'tag2':param[3:6],
		'tag3':param[6:9]
	}

	sum_ = 0
	for tagID in order_tag:
		for anchorID in order_anchor:
			anpos = anchorPos_dict[anchorID]
			dist = distance_dict[tagID][anchorID]
			sum_ += (distanceAB(tag_dict[tagID],anpos)-dist)**2
	for pair,d in layout:
		sum_ += k1*(distanceAB(tag_dict[pair[0]],tag_dict[pair[1]])-d)**2
	return sum_

def rotation_xyz(pointcloud_array, theta_tuple): # deg
	theta_x = np.deg2rad(theta_tuple[0])
	theta_y = np.deg2rad(theta_tuple[1])
	theta_z = np.deg2rad(theta_tuple[2])
	
	rot_x = np.array([[ 1,               0,                0],
					  [ 0, np.cos(theta_x), -np.sin(theta_x)],
					  [ 0, np.sin(theta_x),  np.cos(theta_x)]])

	rot_y = np.array([[ np.cos(theta_y), 0,  np.sin(theta_y)],
					  [               0, 1,                0],
					  [-np.sin(theta_y), 0,  np.cos(theta_y)]])

	rot_z = np.array([[ np.cos(theta_z), -np.sin(theta_z), 0],
					  [ np.sin(theta_z),  np.cos(theta_z), 0],
					  [               0,                0, 1]])

	rot_matrix = np.dot(rot_x, np.dot(rot_y, rot_z))
	rot_pointcloud = np.dot(rot_matrix, pointcloud_array.T).T
	return rot_pointcloud, rot_matrix


def multilateration(anchorPositions_list, distanceBetweenAnchors_list):
	if len(anchorPositions_list) != len(distanceBetweenAnchors_list):
		return [None, None]
	if len(anchorPositions_list) < 3:
		return [None, None]
		
	length = len(anchorPositions_list)
	
	listForAMatrix = []
	listForbMatrix = []
	for baseIndex in range(length):
		basedAnchorPosition = anchorPositions_list[baseIndex]
		basedDistance = distanceBetweenAnchors_list[baseIndex]
		a = [anchorPositions_list[i] for i in range(length) if i != baseIndex]
		d = [distanceBetweenAnchors_list[i] for i in range(length) if i != baseIndex]
		for pos, dist in zip(a, d):
			listForAMatrix.append( [i-j for i,j in zip(pos,basedAnchorPosition)] )
			listForbMatrix.append( [basedDistance**2-dist**2+np.sum([i**2-j**2 for i,j in zip(pos,basedAnchorPosition)])] )
	
	A = np.matrix(listForAMatrix)
	b = np.matrix(listForbMatrix)
	r = np.dot(np.dot(np.linalg.pinv(
		np.dot(np.transpose(A), A)), np.transpose(A)), b)/2
	rlist = r.T.tolist()
	return rlist[0]

def getDistance(true_tagPos_dict, anchorPos_dict, order_tag, order_anchor, rng):
	distance_dict = {}
	for tagID in order_tag:
		m1_anchorPos_list = []
		m1_distance_list = []
		distance_dict[tagID] = {}
		for anchorID in order_anchor:
			anchorPos = anchorPos_dict[anchorID]
			distance_dict[tagID][anchorID] = rng.normal(mean,std) + distanceAB(true_tagPos_dict[tagID], anchorPos)
	return distance_dict


def method1(anchorPos_dict, order_tag, order_anchor, distance_dict):
	# method 1 
	m1_est_tagpos_list = []
	for tagID in order_tag:
		m1_anchorPos_list = []
		m1_distance_list = []
		for anchorID in order_anchor:
			anchorPos = anchorPos_dict[anchorID]
			d = distance_dict[tagID][anchorID]
			m1_distance_list.append(d)
			m1_anchorPos_list.append(anchorPos)
		m1_pos = multilateration(m1_anchorPos_list, m1_distance_list)
		m1_est_tagpos_list.append(m1_pos)

	print('method 1')
	for i in m1_est_tagpos_list:
		print(i)
	return m1_est_tagpos_list

def method2(true_tagPos_dict, anchorPos_dict, order_tag, order_anchor, distance_dict):
	m2_est_dist_dict = {}
	# method 1 (初期値求めるため)
	m1_est_tagpos_list = []
	for tagID in order_tag:
		m1_anchorPos_list = []
		m1_distance_list = []
		if tagID not in m2_est_dist_dict.keys():
			m2_est_dist_dict[tagID] = {}
		for anchorID in order_anchor:
			anchorPos = anchorPos_dict[anchorID]
			d = distance_dict[tagID][anchorID]
			m1_distance_list.append(d)
			m1_anchorPos_list.append(anchorPos)
			m2_est_dist_dict[tagID][anchorID] = d
		m1_pos = multilateration(m1_anchorPos_list, m1_distance_list)
		m1_est_tagpos_list.append(m1_pos)

	# method 2
	layout = [
		(('tag1','tag2'),distanceAB(true_tagPos_dict['tag1'], true_tagPos_dict['tag2'])),
		(('tag2','tag3'),distanceAB(true_tagPos_dict['tag2'], true_tagPos_dict['tag3'])),
		(('tag3','tag1'),distanceAB(true_tagPos_dict['tag3'], true_tagPos_dict['tag1']))
	]
	param = opt.minimize(
		objective_fnc_medhotd2,
		np.array(m1_est_tagpos_list).ravel(),
		method='Powell',
		args=(
			anchorPos_dict,
			m2_est_dist_dict,
			order_tag, 
			order_anchor,
			layout
		)
	)
	m2_est_tagPos_list = [[float(param.x[i+3*j])for i in range(3)] for j in range(3)]
	print('method 2')
	pprint(m2_est_tagPos_list)
	return m2_est_tagPos_list


def histgram(x, min_, max_, width, cumulative=True):
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	t = np.arange(min_, max_, width)
	y = []
	if cumulative:
		for i in t:
			y.append(np.count_nonzero(x<i))
		ax.scatter(t,y,c='black')
	else:
		for i in range(len(t)-1):
			y.append(np.count_nonzero((t[i]<=x) & (x<t[i+1])))
		ax.scatter(t[:-1],y,c='black')

def method3(true_tagPos_dict, anchorPos_dict, order_tag, order_anchor, distance_dict, gradParam_list, symbol_tagPos_dict, symbol_anchorPos_dict, symbol_distance_dict):
	m3_est_dist_dict = {}
	# method 1 (初期値求めるため)
	m1_est_tagpos_list = []
	for tagID in order_tag:
		m1_anchorPos_list = []
		m1_distance_list = []
		m3_est_dist_dict[tagID] = {}
		for anchorID in order_anchor:
			anchorPos = anchorPos_dict[anchorID]
			d = distance_dict[tagID][anchorID]
			m1_distance_list.append(d)
			m1_anchorPos_list.append(anchorPos)
			m3_est_dist_dict[tagID][anchorID] = d
		m1_pos = multilateration(m1_anchorPos_list, m1_distance_list)
		m1_est_tagpos_list.append(m1_pos)

	# method 3
	layout = [
		(('tag1','tag2'),distanceAB(true_tagPos_dict['tag1'], true_tagPos_dict['tag2'])),
		(('tag2','tag3'),distanceAB(true_tagPos_dict['tag2'], true_tagPos_dict['tag3'])),
		(('tag3','tag1'),distanceAB(true_tagPos_dict['tag3'], true_tagPos_dict['tag1']))
	]

	l = []
	for anchorID in order_anchor:
		for i, j in zip(symbol_anchorPos_dict[anchorID], anchorPos_dict[anchorID]):
			l.append((i,j))
	for tagID in order_tag:
		for anchorID in order_anchor:
			l.append((symbol_distance_dict[tagID][anchorID], distance_dict[tagID][anchorID]))
	g = [i.subs(l) for i in gradParam_list]

	param = opt.fmin_bfgs(
		objective_fnc_method3,
		x0 = np.array(m1_est_tagpos_list).ravel(),
		args=(
			anchorPos_dict,
			m3_est_dist_dict,
			order_tag, 
			order_anchor,
			layout,
			g,
			symbol_tagPos_dict
		),
		fprime = gradient_method3
	)
	m3_est_tagPos_list = [[float(param[i+3*j])for i in range(3)] for j in range(3)]
	print('method 3')
	pprint(m3_est_tagPos_list)
	return m3_est_tagPos_list

if __name__ == '__main__':
	seed = 1234
	std = 0.07
	mean = 0

	true_anchorPos_dict = {
		'an1':[0,0,0],
		'an2':[5,0,0.1],
		'an3':[5,5,0.2],
		'an4':[0,5,-0.04]
	}

	true_robotPos = [2.4,3,0.1]
	vect_tag1 = [ 0.325,   0, 0]
	vect_tag2 = [-0.325,   0, 0]
	vect_tag3 = [     0, 0.4, 0]
	rot_pointcloud, rot_matrix = rotation_xyz(np.array([vect_tag1,vect_tag2,vect_tag3]), (0,90,45))
	true_tagPos_dict = {
		'tag1':rot_pointcloud[0]+np.array(true_robotPos),
		'tag2':rot_pointcloud[1]+np.array(true_robotPos),
		'tag3':rot_pointcloud[2]+np.array(true_robotPos)

		#'tag1':np.array(vect_tag1)+np.array(true_robotPos),
		#'tag2':np.array(vect_tag2)+np.array(true_robotPos),
		#'tag3':np.array(vect_tag3)+np.array(true_robotPos),
	}

	order_tag = ['tag1','tag2','tag3']
	order_anchor = ['an1','an2','an3','an4']
	print('true_tagPos_dict')
	pprint(true_tagPos_dict)

	rng = np.random.default_rng(seed)

	# method 3の準備
	f, gradParam_list, symbol_tagPos_dict, symbol_anchorPos_dict, symbol_distance_dict = diff_objective_func(true_tagPos_dict,order_tag, order_anchor)
	#pprint(gradient)


	error1_list = []
	error2_list = []
	error3_list = []

	for c in range(10):
		print(c)

		distance_dict = getDistance(true_tagPos_dict, true_anchorPos_dict, order_tag, order_anchor, rng)
		m1_est_tagpos_list = method1(true_anchorPos_dict, order_tag, order_anchor, distance_dict)
		m2_est_tagpos_list = method2(true_tagPos_dict, true_anchorPos_dict, order_tag, order_anchor, distance_dict)
		m3_est_tagpos_list = method3(true_tagPos_dict, true_anchorPos_dict, order_tag, order_anchor, distance_dict, gradParam_list, symbol_tagPos_dict, symbol_anchorPos_dict, symbol_distance_dict)


		error1 = 0
		for t1, t2 in zip([true_tagPos_dict[i] for i in order_tag], m1_est_tagpos_list):
			error1 += distanceAB(t1,t2)
		error2 = 0
		for t1, t2 in zip([true_tagPos_dict[i] for i in order_tag], m2_est_tagpos_list):
			error2 += distanceAB(t1,t2)
		error3 = 0
		for t1, t2 in zip([true_tagPos_dict[i] for i in order_tag], m3_est_tagpos_list):
			error3 += distanceAB(t1,t2)
		error1_list.append(error1)
		error2_list.append(error2)
		error3_list.append(error3)

		"""
		# plot
		fig1 = plt.figure()
		ax1 = fig1.add_subplot(1,1,1)
		ax1.grid()
		ax1.set_aspect('equal')
		ax1.set_xlim(-1,6)
		ax1.set_ylim(-1,6)

		# true
		X = []
		Y = []
		Z = []
		for tagID, pos in true_tagPos_dict.items():
			X.append(pos[0])
			Y.append(pos[1])
			Z.append(pos[2])
			ax1.text(pos[0],pos[1],tagID)
		ax1.scatter(X,Y, color='black')

		# method3 
		m3_X = []
		m3_Y = []
		m3_Z = []
		for tagID, pos in zip(order_tag, m2_est_tagpos_list):
			m3_X.append(pos[0])
			m3_Y.append(pos[1])
			m3_Z.append(pos[2])
			ax1.text(pos[0],pos[1],tagID)
		ax1.scatter(m3_X,m3_Y, color='blue')	
		plt.show()
		"""

	histgram(error1_list, 0, 5, 0.01)
	histgram(error2_list, 0, 5, 0.01)
	histgram(error3_list, 0, 5, 0.01)
	plt.show()
	exit()

	fig1 = plt.figure()
	ax1 = fig1.add_subplot(1,1,1)
	ax1.grid()
	ax1.set_aspect('equal')
	ax1.set_xlim(-1,6)
	ax1.set_ylim(-1,6)

	# true
	X = []
	Y = []
	Z = []
	for tagID, pos in true_tagPos_dict.items():
		X.append(pos[0])
		Y.append(pos[1])
		Z.append(pos[2])
		ax1.text(pos[0],pos[1],tagID)
	ax1.scatter(X,Y, color='black')

	# method 1
	m1_X = []
	m1_Y = []
	m1_Z = []
	for tagID, pos in zip(order_tag, m1_est_tagpos_list):
		m1_X.append(pos[0])
		m1_Y.append(pos[1])
		m1_Z.append(pos[2])
		ax1.text(pos[0],pos[1],tagID)
	ax1.scatter(m1_X,m1_Y, color='red')	


	# method 2
	m2_X = []
	m2_Y = []
	m2_Z = []
	for tagID, pos in zip(order_tag, m2_est_tagpos_list):
		m2_X.append(pos[0])
		m2_Y.append(pos[1])
		m2_Z.append(pos[2])
		ax1.text(pos[0],pos[1],tagID)
	ax1.scatter(m2_X,m2_Y, color='blue')	

	"""
	fig1 = plt.figure()
	ax1 = fig1.add_subplot(1,1,1, projection='3d')
	xmin,xmax=-1,5
	ymin,ymax=-1,5
	zmin,zmax=-1,5
	ax1.set_xlim3d(xmin, xmax)
	ax1.set_ylim3d(ymin, ymax)
	ax1.set_zlim3d(zmin, zmax)
	ax1.set_xlabel('x-axis')
	ax1.set_ylabel('y-axis')
	ax1.set_zlabel('z-axis')
	ax1.view_init(elev=30, azim=-135)
	ax1.set_box_aspect((xmax-xmin,ymax-ymin,zmax-zmin))
	ax1.scatter3D([i[0] for i in true_tagPos_dict.values()],[i[1] for i in true_tagPos_dict.values()], zs=[i[2] for i in true_tagPos_dict.values()], color='black')
	ax1.scatter3D([i[0] for i in [m1_est_tag1Pos,m1_est_tag2Pos]],[i[1] for i in [m1_est_tag1Pos,m1_est_tag2Pos]], zs=[i[2] for i in [m1_est_tag1Pos,m1_est_tag2Pos]], color='red')
	"""


	plt.show()
