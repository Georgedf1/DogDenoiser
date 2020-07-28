'''
How to install numpy and scipy:

Open Blender, open python console
>>> import sys
>>> sys.exec_prefix
'/path/to/blender/python


Open a windows console as admin:
cd /path/to/blender/python/bin
python -m ensurepip
python -m pip install numpy
python -m pip install scipy
python -m pip install opencv-python


How to run this script:
Open a window in Blender and set it to "Text Editor"
Use the navigator to load this script

'''


import bpy
import sys
import numpy as np
import mathutils
import os
import cv2
import json
from os.path import join, dirname

os.system("cls")

# Add the folder that contains this script to the path, so we can access utils.py
# https://blender.stackexchange.com/questions/14911/get-path-of-my-script-file
for area in bpy.context.screen.areas:
    if area.type == 'TEXT_EDITOR':
        for space in area.spaces:
            if space.type == 'TEXT_EDITOR':
                scriptFolder = dirname(space.text.filepath)
                sys.path.insert(0, scriptFolder)    

from utils import LoadMatFile

def SetValues():
    pathToMainDataset = 'C:/Users/Georgedf/Research/Data'
    dog = 'dog1'
    motion = 'testSeq'
    return pathToMainDataset, dog, motion
    
def main():
    pathToMainDataset, dog, motion = SetValues()
    
    pathToMotion = os.path.join(pathToMainDataset, dog, 'motion_%s'%motion)
    bvhSkelFile = os.path.join(pathToMotion, 'motion_capture', 'skeleton.bvh')
    pathToSkinningWeights = os.path.join(pathToMainDataset, dog, 'meta', 'skinningWeights.mat')
    pathToObj = os.path.join(pathToMainDataset, dog, 'meta', 'neutralMesh.obj')
    pathToDevSource = 'C:/Users/Georgedf/Research/dev/Source'
    pathToMarkerObj = os.path.join(pathToDevSource, 'marker_obj', 'marker_obj.obj')
    
        
    # the mesh and skeleton are defined in millimetres. Import into Blender in metres.
    OBJ_SCALE = 0.01
    BVH_SCALE = 0.01
    
    #----------gf321: import markers and animate--------------
    
    #Make marker parent 'empty' for UI cleanliness/filtering
    bpy.ops.object.add()
    marker_parent = bpy.context.selected_objects[0]
    marker_parent.name = 'Markers'
    
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
        marker_ids[marker_num].parent = marker_parent
    
    #animate the marker objects
    for frame in range(frame_num):
        for marker_key in range(marker_num):
            for i in range(3):
                marker_ids[marker_key].location[i] = markers[marker_key,i,frame]*OBJ_SCALE
                #NOTICE: +2 frame offset to match with Sinead's dog model animation
                marker_ids[marker_key].keyframe_insert(data_path="location", frame=frame+2, index=i)

    
    
    # -------------------back to Sinead's code---------------------------------

    #import bvh
    imported_object = bpy.ops.import_anim.bvh(filepath=bvhSkelFile, filter_glob="*.bvh", target='ARMATURE', global_scale=BVH_SCALE, frame_start=1, use_fps_scale=False, use_cyclic=False, rotate_mode='NATIVE', axis_forward='-Z', axis_up='Y')
    bvh_object = bpy.context.selected_objects[0] ####<--Fix
    # bvh_object.location = mathutils.Vector((0.0, 0.0, 0.0))
    # bvh_object.location = mathutils.Vector((0.0, 0.29094, -0.252148))

    # import obj
    imported_object = bpy.ops.import_scene.obj(filepath=pathToObj, split_mode='OFF')
    mesh_object = bpy.context.selected_objects[0] ####<--Fix
    mesh_object.scale = ( OBJ_SCALE, OBJ_SCALE, OBJ_SCALE )
    # mesh_object.location = mathutils.Vector((0.0, 0.29094, -0.252148))
        
    mesh_object.parent = bvh_object
    mesh_object.modifiers.new(name = 'Skeleton', type = 'ARMATURE')
    mesh_object.modifiers['Skeleton'].object = bvh_object


    bns = GetNamesInArmature(bvh_object)
    # bns = ['Root', 'Spine01', 'Spine02', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftWrist', 'LeftHand', 'LeftFinger', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightWrist', 'RightHand', 'RightFinger', 'Neck01', 'Neck02', 'Neck03', 'Neck04', 'Head', 'Nose', 'LeftEar', 'LeftEarEnd', 'RightEar', 'RightEarEnd', 'LeftLeg', 'LeftLowerLeg', 'LeftAnkle', 'LeftFoot', 'LeftToe', 'RightLeg', 'RightLowerLeg', 'RightAnkle', 'RightFoot', 'RightToe', 'TailBase', 'Tail1', 'Tail2', 'Tail3', 'Tail4', 'Tail5', 'Tail6', 'TailEnd']

    numBones = len(bns) # 42
    # add vertex groups
    for i in range(0, numBones):
        mesh_object.vertex_groups.new(name=bns[i]) #gf321 EDIT: to name=bns[i] from bns[i]



    skinningWeights = LoadMatFile(pathToSkinningWeights)
    numBones = skinningWeights.shape[1] # 43
    
    # Blender doesn't let end effectors (ie, LeftFinger, RightToe, etc) have weights
    # transfer these weights to the parent bone (ie, LeftHand, RightFoot)
    # then select the indices of only non-end effector bones
    endEffectorBones = [8,14,20,22,24,29,34,42]
    jointIsEndEffector = np.zeros((numBones,))
    jointIsEndEffector[endEffectorBones] = 1
    for bn in endEffectorBones:
        skinningWeights[:,bn-1] = skinningWeights[:,bn-1] + skinningWeights[:,bn]
        skinningWeights[:,bn] *= 0
    skinningWeights = skinningWeights[:,jointIsEndEffector==0]
    
    
    print('skinningWeights.shape', skinningWeights.shape, 'numBones', numBones)
    SetSkinningMatrix(mesh_object, skinningWeights, 0)


    # root position at zero: [0.0, -0.29094, 0.252148]

def GetCurrentName(basename):
    # name might be "myMesh.001", so basename here would be "myMesh"
    obs = bpy.data.objects
    name = ''
    for o in obs:
        if basename in o.name:
            name = o.name
            break
    return name
    
def GetVertexGroupsNamesForMeshObject(mesh_object):

    if type(mesh_object) is str:
        mesh_object = GetCurrentName(mesh_object)
        mesh_object = bpy.data.objects[mesh_object]
        
    vgNames = []
    for vg in  mesh_object.vertex_groups:
        vgNames.append(vg.name)
    return vgNames
    
    
def SetSkinningMatrix(mesh_object, skinningMat, verbose=0):
    if type(mesh_object) is str:
        mesh_object = GetCurrentName(mesh_object)
        mesh_object = bpy.data.objects[name]
        
    vertex_groups = mesh_object.vertex_groups
    vertex_groups_names = GetVertexGroupsNamesForMeshObject(mesh_object)

    numBones = len(vertex_groups_names)
    
    me = mesh_object.data
    verts = me.vertices
    numVerts = len(verts)
    
    if type(skinningMat) is np.ndarray:
        print('skinningMat.shape', skinningMat.shape)
        skinningMat = skinningMat.tolist()

    
    for v_idx, s in enumerate(skinningMat):
        # loop through vertices
        # s is list of 43 values
        
        influencedByBones = np.where(np.array(s)>0)[0]
        weights = []
        for boneId, weight in enumerate(s):
            mesh_object.vertex_groups[boneId].add([v_idx], weight, "REPLACE")
            if boneId in influencedByBones: # save to print later
                weights.append(weight)
        
        if verbose:
            print('vertex id', v_idx, ', bones:', influencedByBones, ', weights', weights)
            
    return True
    
def GetNamesInArmature(arm_object):

    if type(arm_object) is str:
        arm_object = GetCurrentName(arm_object)
        arm_object = bpy.data.objects[arm_object]
        
    # if ob.type == 'ARMATURE':
        # armature = ob.data
    armature = arm_object.data
    boneNames = []
    for bone in armature.bones:
        boneNames.append(bone.name)
    return boneNames

#gf321 convenience funcotions borrowed from utils/utils.py
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
#------------------------------------------------
    


if __name__== "__main__":
    main()