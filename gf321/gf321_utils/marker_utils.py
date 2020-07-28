# gf321 marker utility functions (/classes)

import numpy as np
import json

def read_timecodes():
    pass

def read_markers(markerFilepath, withName=False, neutralFirstFrame=True):
    '''
    Reads a json file (of specific format) into a dictionary then converts to array
    Returns said array and the naming order if withName=True
    '''
    with open(markerFilepath) as fp:
        print('Reading JSON File..',markerFilepath)
        marker_dict = json.load(fp)

    marker_num = int(len(marker_dict)/3) #number of markers
    marker_names = [] 
    marker_dict_keys = np.array(list(marker_dict.keys()))
    frame_num = int(len(marker_dict[marker_dict_keys[0]])) #number of frames
    offset = 0
    if neutralFirstFrame:
        frame_num -= 1 #correct frame count for
        offset = 1 #create offset to be used in indexing properly

    marker_array = np.zeros((frame_num, marker_num, 3))
    
    marker_counter = -1
    for ind,key in enumerate(marker_dict_keys):

        dim = ind%3 #dimension 0=x, 1=y, 2=
        if dim == 0:
            marker_counter += 1 #used for indexing
            marker_name = marker_dict_keys[:key.index(".")]
            marker_names.append(marker_name) #record name

        marker_frames_for_key = marker_dict[key]
        marker_array[:,marker_counter,dim] = marker_frames_for_key[offset:]
    
    print('Done')

    if withName:
        return marker_array, marker_names
    else: 
        return marker_array


def make_markers_local(markers_g, local_tfs):
    '''
    NB local wrt root

    Expects input of markers_g (global markers) shape (frame_num, marker_num, 3)
    and local_tfs shape (frame_num,4,4) created by bvh_utils.create_local_tfs or otherwise
    holding the root transform information for each frame and a final row of [0,0,0,1]

    Returns the markers made local in shape (frame_num, marker_num, 3)
    '''

    frame_num = markers_g.shape[0]
    marker_num = markers_g.shape[1]
    
    markers_4 = np.zeros((frame_num,marker_num,4)) #make markers have shape (",",4)
    markers_local = np.zeros((frame_num,marker_num,3))

    markers_4[:,:,:-1] = markers_g
    markers_4[:,:,-1] = 1
    
    for fr in range(frame_num):
        for mk in range(marker_num):
            markers_local[fr][mk] = np.matmul(local_tfs[fr], markers_4[fr][mk])[:-1]
    
    return markers_local


def create_marker_offsets(markers, all_joint_tfs, mean=True):
    '''
    markers input has shape (frame_num, marker_num, 3)
    all_joint_tfs has shape (frame_num, joint_num, 4, 4)  (padded with extra row of [0,0,0,1]) 
    both contain the positions of said markers/joints
    '''

    frame_num = markers.shape[0]
    assert all_joint_tfs.shape[0] == frame_num

    marker_num = markers.shape[1]
    joint_num = all_joint_tfs.shape[1]

    markers_4 = np.zeros((frame_num,marker_num,4))
    markers_4[:,:,:-1] = markers
    markers_4[:,:,-1] = 1

    marker_offsets = np.zeros((frame_num, marker_num, joint_num, 3))

    for fr in range(frame_num):
        for mk in range(marker_num):
            for jt in range(joint_num):
                marker_offsets[fr][mk][jt] = np.matmul(all_joint_tfs[fr][jt], markers_4[fr][mk])[:-1] 
                # [:-1] to take off the 1 added into markers_4

    if mean:
        return marker_offsets.mean(axis=0)
    else:
        return marker_offsets




        

