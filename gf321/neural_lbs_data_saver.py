
import numpy as np
import sys

path_to_files = 'C://Users/Georgedf/Research/dev/gf321/'
sys.path.append(path_to_files)

from gf321_utils import bvh_utils
from gf321_utils import marker_utils

def main():

    motion_names = ['jump','poles','table','trot','walk','testSeq'] #edit to change what's included in dataset
    markers = []
    joints = []

    for name in motion_names:
        bvh_class = bvh_utils.BVHData()
        bvh_class.bvhRead('neural_lbs_data/dog1_' + name + '_skeleton.bvh')

        if joints == []:
            joints = bvh_class.getJointAngles()
            current_joints = joints
            markers = marker_utils.read_markers('neural_lbs_data/dog1_' + name + '_markers.json')
            if name == 'testSeq':
                markers = np.delete(markers,-1,0) #testSeq markers have one too many frames (extra at end) so delete
            current_markers = markers
        else:
            current_joints =bvh_class.getJointAngles()
            joints = np.append(joints, current_joints, axis=0)
            current_markers = marker_utils.read_markers('neural_lbs_data/dog1_' + name + '_markers.json')
            if name == 'testSeq':
                current_markers = np.delete(current_markers,-1,0) #testSeq markers have one too many frames (extra at end) so delete
            markers = np.append(markers, current_markers, axis=0)
    
        print('Shape equal: ',current_markers.shape[0]==current_joints.shape[0])
        if current_markers.shape[0] != current_joints.shape[0]:
            print('FALSE with joints: ',current_joints.shape[0],'and markers: ',current_markers.shape[0])

    np.save('neural_lbs_data/all_joints.npy',joints)
    np.save('neural_lbs_data/all_markers.npy',markers)

    
    # Now mirror, append mirrored ones and save the augmented joint and marker datasets
    joints_mir = np.array(joints) #make copy
    markers_mir = np.array(markers) #"
    
    joints_mir[:,:,0,:] = -joints_mir[:,:,0,:] #mirror in 'x' by making x column of transform negative
    markers_mir[:,:,0] = -markers_mir[:,:,0] #" #NB: markers 'x' is 0 but joints 'x' is 2 as order different

    joints_aug = np.append(joints, joints_mir, axis=0) #append along frames
    markers_aug = np.append(markers, markers_mir, axis=0) #"

    np.save('neural_lbs_data/all_joints_aug.npy',joints_aug)
    np.save('neural_lbs_data/all_markers_aug.npy',markers_aug)


    print('Finished')
    print('Joints shape: ',joints.shape,'. Markers shape: ',markers.shape)
    print('Augmented joints shape: ',joints_aug.shape,'. Augmented markers shape: ',markers_aug.shape)


if __name__ == '__main__':
    main()     
