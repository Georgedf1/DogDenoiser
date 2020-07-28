'''
Experiments showed that there simply wasnt enough training data to learn a skinning function.
This is understandable due to the sheer DoF of the mapping joint_num*3*4 -> marker_num*3 
but it was worth a try :')
'''

import os
import numpy as np
import tensorflow as tf
from gf321_utils import bvh_utils
from gf321_utils import marker_utils

class Neural_LBS():

    def __init__(self):
 
        self.epochs = 60
        self.marker_data = []
        self.joint_data = []
        self.training_split = 0.8 #Splits data into training and validation
        self.tolerance = 1e-6 #tolerance for clipping zeros in standard deviation calculations
        self.data_filepath = 'neural_lbs_data/'
        

    def load_tr_data(self, read_np=True , joints=[], markers=[]):
        '''
        input should be an array of joints of shape (frame_num, joint_num, 3, 4)
        and markers of shape (frame_num, marker_num, 3).
        Joints and markers should be paired (representing the same model) 
        Inputs can be global as this function makes it local
        function then further mirrors the data to augment for training and finally adds to the class' data array
        so that we can make multiple calls of the function on different files

        read_np specifies if we are to read from the presaved numpy files all_markers.npy and all_joints.npy

        if read_np is false joints and markers need to be provided via the keyword arguments
        '''

        #Load npy files and overwrite joints and markers variables
        if read_np:
            if os.path.isfile(self.data_filepath + 'all_joints.npy'):
                joints = np.load(self.data_filepath + 'all_joints.npy')
            else: Exception('all_joints.npy not found')
            if os.path.isfile(self.data_filepath + 'all_markers.npy'):
                markers = np.load(self.data_filepath + 'all_markers.npy')
            else: Exception('all_markers.npy not found')


        #Create useful info for later
        frame_num = joints.shape[0]
        self.marker_num = markers.shape[1]       
        self.joint_num = joints.shape[1]    #dont need for now
        assert markers.shape[0] == frame_num #check

        joints_loc = bvh_utils.make_joints_local(joints)
        root_tfs = bvh_utils.create_local_tfs_root(joints)
        markers_loc = marker_utils.make_markers_local(markers, root_tfs)

        joints_loc_mir = np.array(joints_loc) #make copy
        markers_loc_mir = np.array(markers_loc) #"
        
        joints_loc_mir[:,:,0,:] = -joints_loc_mir[:,:,0,:] #mirror in 'x' by making x column of transform negative
        markers_loc_mir[:,:,0] = -markers_loc_mir[:,:,0] #" #NB: markers 'x' is 0 but joints 'x' is 2 as order different

        joints_aug = np.append(joints_loc, joints_loc_mir, axis=0) #append along frames
        markers_aug = np.append(markers_loc, markers_loc_mir, axis=0) #"

        #Create data and flatten marker data for MSE
        if self.marker_data == []:
            marker_data = np.zeros((2*frame_num, 3*self.marker_num)) #2*frame_num as data is mirrored
            for m in range(self.marker_num):
                for dim in range(3):
                    marker_data[:,(3*m)+dim] = markers_aug[:,m,dim]
            self.marker_data = marker_data #begin adding data. markers is flattened
            self.joint_data = joints_aug #"
        else:
            marker_data = np.zeros((2*frame_num, 3*self.marker_num))
            for m in range(self.marker_num):
                for dim in range(3):
                    marker_data[:,(3*m)+dim] = markers_aug[:,m,dim]
            self.marker_data = np.append(self.marker_data,marker_data, axis=0) #add to list of frames to build full dataset
            self.joint_data = np.append(self.joint_data, joints_aug, axis=0) #"        

    def compile(self):

        tf.keras.backend.clear_session() #reset graph

        #Collect useful info, note marker_num and joint_num have already been collected
        self.frame_num = self.marker_data.shape[0]

        self.input_shape = (self.joint_num, 3, 4)
        self.input_num = 12 * self.joint_num
        self.output_shape = (3*self.marker_num,)
        self.output_num = 3 * self.marker_num

        #Collect means and stds
        self.joint_means = self.joint_data.mean(axis=0)
        self.joint_stds = self.joint_data.std(axis=0)
        self.marker_means = self.marker_data.mean(axis=0)
        self.marker_stds = self.marker_data.std(axis=0)

        #Create masks for safe divison. Only need for joints as markers are position only and do not coincide with root joint
        self.joint_mask_on = np.where(abs(self.joint_stds) < self.tolerance)
        self.joint_mask_off = np.where(abs(self.joint_stds) >= self.tolerance)
        #Make normalised data
        self.marker_data_n = np.zeros(self.marker_data.shape)
        self.joint_data_n = np.zeros(self.joint_data.shape)
        self.marker_data_n = (self.marker_data - self.marker_means)/self.marker_stds 

        self.joint_data_n = np.zeros(self.joint_data.shape)
        mask = np.array(abs(self.joint_stds)<1e-6).nonzero()
        self.joint_stds[mask] = 1 #the trick
        self.joint_data_n = (self.joint_data - self.joint_means) /self.joint_stds

        #Shuffle data
        shuffler = np.random.shuffle(np.arange(self.frame_num)) #create shuffled indices to shuffle with
        self.joint_data_n[:] = self.joint_data_n[shuffler]
        self.marker_data_n[:] = self.marker_data_n[shuffler]

        #Create training data
        self.split_tr = int(self.training_split * self.frame_num)
        self.marker_data_tr = self.marker_data_n[:self.split_tr,:]
        self.joint_data_tr = self.joint_data_n[:self.split_tr,:,:,:]
        #Create validation data
        self.marker_data_val = self.marker_data_n[self.split_tr:,:]
        self.joint_data_val = self.joint_data_n[self.split_tr:,:,:,:]
        

        #Build model
        self.model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape= self.input_shape),
            tf.keras.layers.Dense(self.output_num),
            ])

        #Compile model 
        self.model.compile(optimizer='adam', loss='mse')

    def train(self, verbosity=0):
        #Train model with training data
        print('Training...')
        self.model.fit(self.joint_data_tr,self.marker_data_tr,
            validation_data=(self.joint_data_val, self.marker_data_val), verbose=verbosity, batch_size=32, epochs=self.epochs)
        print('Training loss: ', self.eval(self.joint_data_tr,self.marker_data_tr))
        print('Validation loss: ', self.eval(self.joint_data_val, self.marker_data_val))


    def eval(self, joints_test, markers_test):
        
        frame_num = joints_test.shape[0]
        assert frame_num == markers_test.shape[0]

        #We make the assumption here that if the markers are flattened already then self.eval is being called within self.train
        #If markers are not flat we assume they are global (ie not local or normalised) and so must correct that to evaluate
        
        if len(markers_test.shape) == 3:
            
            marker_num = markers_test.shape[1]

            #make local
            local_tfs = bvh_utils.create_local_tfs_root(joints_test)
            joints_test = bvh_utils.make_joints_local(joints_test)
            markers_test = marker_utils.make_markers_local(markers_test,local_tfs)

            #Flatten markers
            markers_test = markers_test.reshape((frame_num, marker_num * 3 ))

            #Normalise, worrying about clipping so need to use a mask
            joints_test = (joints_test - self.joint_means)/self.joint_stds
            markers_test = (markers_test - self.marker_means)/self.marker_stds

        #Test the model on provided test data
        test_loss = self.model.evaluate(joints_test, markers_test)

        return test_loss

    
    def predict(self, x):
        #Takes in unnormalised, but local, data! so...
        x = (x - self.joint_means) / self.joint_stds #normalise
        pred = self.model.predict(x) #make prediction
        pred_n = pred * self.marker_stds
        pred_n = pred_n + self.marker_means #local markers
        pred_n = np.reshape(pred_n,(pred.shape[0],self.marker_num,3))
        return pred_n
