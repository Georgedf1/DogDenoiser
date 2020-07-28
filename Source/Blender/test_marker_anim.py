# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 13:17:20 2020

@author: georg
"""

import bpy
import sys
import numpy as np
import mathutils
import json
import os


def main():
    #Load data
    pathToMainDataset = 'C:/Users/Georgedf/Research/Data'
    pathToDevSource = 'C:/Users/Georgedf/Research/dev/Source'
    dog = 'dog1'
    motion = 'walk'
    
    pathToMotion = os.path.join(pathToMainDataset, dog, 'motion_%s'%motion)
    pathToMarkerObj = os.path.join(pathToDevSource, 'marker_obj', 'marker_obj.obj')
    
    
    #Load markers
    markerFile = os.path.join(pathToMotion, 'motion_capture', 'markers.json')
    #Need to make these utils. when adding back to ImportSknnedMeshMarked
    [markerNames, markers] = GetPointsFromJsonFbx(markerFile)
    markers = MovePointsOutOfMayaCoordSystem(markers)
    markers_num = markers.shape[0]
    
    frame_num = markers.shape[2]
    
    #Load marker object
    OBJ_SCALE = 0.01
    marker_ids = markers_num*[0]
    for marker_num in range(markers_num):
        bpy.ops.import_scene.obj(filepath=pathToMarkerObj, split_mode='OFF')
        marker_ids[marker_num] = bpy.context.selected_objects[0]
        marker_ids[marker_num].scale = ( OBJ_SCALE, OBJ_SCALE, OBJ_SCALE )
    
    #animate the marker objects
    for frame in range(frame_num):
        for marker_key in range(marker_num):
            for i in range(3):
                marker_ids[marker_key].location[i] = markers[marker_key,i,frame]
                marker_ids[marker_key].keyframe_insert(data_path="location", frame=frame, index=i)

    print('done')
        



# convenience functions:
def MovePointsOutOfMayaCoordSystem(points3D, scale=1):

	rot90X = np.array(([1, 0, 0], [0, 0, -1], [0, 1, 0]))
	#the points where generated in Maya -> multiply by 10 and rotate about x-axis by 90 degrees

	points3D = points3D*scale
	if points3D.shape[0] != 3:
		points3D = np.matmul(rot90X, np.transpose(points3D))
		return np.transpose(points3D)
	else:
		points3D = np.matmul(rot90X, points3D)
		return points3D

# convenience function    
def GetPointsFromJsonFbx(filename):
	print('reading %s' % filename)
	j = json.load(open(filename, 'r'))
	# j = json.dumps(j, indent=4, sort_keys=True) #make sure the keys are sorted, include indentation to make it easier to read when printing
	# print(j)

	numPoints = int(round(len(j)/3)) #each entry in j is "___.TranslateX", "__.TranslateY", "____.TranslateZ" -> divide by 3 to get number of points

	namesAll = []
	for entry in j:
		namesAll.append(str(entry))
	namesAll.sort() #sorts a-z
		
	# print(namesAll)
	names = [] #unique, will not contain "____.TranslateX", etc

	#https://stackoverflow.com/questions/2990121/how-do-i-loop-through-a-list-by-twos?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
	if sys.version_info[0] == 2:
		for i in xrange(0,len(namesAll),3): #in steps of 3
			n = namesAll[i]
			# print(n[0:n.find('.')])
			names.append(n[0:n.find('.')])
	else:
		for i in range(0,len(namesAll),3): #in steps of 3
			n = namesAll[i]
			# print(n[0:n.find('.')])
			names.append(n[0:n.find('.')])
	# print(names)


	numFrames = len(j[entry])
	# print('numFrames', numFrames)
	points = np.zeros((numPoints,3,numFrames))
			
	pointIdx = 0

	idx = 0
	for entry in namesAll:
		p = j[entry]
		
		if entry.find('translateX') >= 0:
			points[idx,0,:] = p
		elif entry.find('translateY') >= 0:
			points[idx,1,:] = p
		elif entry.find('translateZ') >= 0:
			points[idx,2,:] = p
			idx = idx+1
		else:
			assert False, 'ERROR reading marker file, entry is not X, Y or Z, entry=%s' % entry
	
	return names, points


if __name__ == '__main__':
    main()