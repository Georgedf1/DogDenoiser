import os
import numpy as np
import tensorflow as tf 
from gf321_utils import bvh_utils
from gf321_utils import marker_utils

class DogDenoiser():

    def __init__(self):
 
        #Parameters for reading data from file.
        #Ensure these are set properly
        self.data_filepath = 'neural_lbs_data/'
        self.joints_filename = 'all_joints_aug.npy'
        self.markers_filename = 'all_markers_aug.npy'

        #Parameters for neural net
        self.l2_regu = 0.01
        self.layer_num = 5 #hidden layer number
        self.unit_num = 2048 #number of units in a hidden layer
        self.epochs = 10
        self.training_split = 0.8 #Splits data into training and validation
        self.tolerance = 1e-6 #tolerance for clipping zeros in standard deviation calculations



    def load_tr_data(self):
        '''
        Assumes input of mirrored (balanced) global (not local) data
        Inputs should be an array of joints of shape (frame_num, joint_num, 3, 4)
        and markers of shape (frame_num, marker_num, 3).
        Joints and markers should be paired (representing the same motion) 
        Inputs can be global as they are made local here

        Reads from npy files the joint and marker data, so needs to be set up properly in __init__
        '''

        #Load npy files containing joint and marker data
        if os.path.isfile(self.data_filepath + self.joints_filename):
            self.joints = np.load(self.data_filepath + self.joints_filename)
        else: raise Exception('joint npy file not found')
        if os.path.isfile(self.data_filepath + self.markers_filename):
            self.markers = np.load(self.data_filepath + self.markers_filename)
        else: raise Exception('marker npy file not found')

        #Create useful info for later
        self.frame_num = self.joints.shape[0]
        self.marker_num = self.markers.shape[1]       
        self.joint_num = self.joints.shape[1]
        assert self.markers.shape[0] == self.frame_num #check

        #Make local
        joints_loc = bvh_utils.make_joints_local(self.joints)
        root_tfs = bvh_utils.create_local_tfs_root(self.joints)
        markers_loc = marker_utils.make_markers_local(self.markers, root_tfs)

        #Set local data to attributes
        self.joints_loc = joints_loc
        self.markers_loc = markers_loc


    #SORT OUT DATA ABOVE THEN USE TF W/ KERAS LAYER WITHIN A GRAPH. LOOK AT YOUR EGS OR COURSE MATERIALS
############################################

    
    def compile(self):

        tf.keras.backend.clear_session() #reset graph

        self.training = True #activates noise layer when true

        self.input_shape = (self.marker_num, 3)
        self.input_num = 3 * self.marker_num
        self.output_shape = (12*self.joint_num,)
        self.output_num = 12 * self.joint_num

        #Collect means and stds
        self.joint_means = self.joint_loc.mean(axis=0)
        self.joint_stds = self.joint_loc.std(axis=0)
        self.marker_means = self.marker_loc.mean(axis=0)
        self.marker_stds = self.marker_loc.std(axis=0)

        #Create masks for safe divison. NB Only need for joints as markers are position only 
        #   and do not coincide with root joint (thus dont have tiny std)
        self.joint_mask_on = np.where(abs(self.joint_stds) < self.tolerance)
        self.joint_mask_off = np.where(abs(self.joint_stds) >= self.tolerance)
        #Make normalised data...
        self.marker_loc_n = np.zeros(self.marker_loc.shape)
        self.joint_loc_n = np.zeros(self.joint_loc.shape)
        self.marker_loc_n = (self.marker_loc - self.marker_means)/self.marker_stds 
        #...via the mask
        mask = np.array(abs(self.joint_stds)<1e-6).nonzero()
        print(mask)
        print(len(mask))
        print(self.joint_stds.shape)
        self.joint_stds[mask] = 1 #the trick
        self.joint_loc_n = (self.joint_loc - self.joint_means) /self.joint_stds

        #Shuffle loc by creating shuffled indices to shuffle with
        shuffler = np.random.shuffle(np.arange(self.frame_num)) 
        self.joint_loc_n[:] = self.joint_loc_n[shuffler]
        self.marker_loc_n[:] = self.marker_loc_n[shuffler]

        #Create training data
        self.split_tr = int(self.training_split * self.frame_num)
        self.marker_loc_tr = self.marker_loc_n[:self.split_tr,:]
        self.joint_loc_tr = self.joint_loc_n[:self.split_tr,:,:,:]
        #Create validation data
        self.marker_loc_val = self.marker_loc_n[self.split_tr:,:]
        self.joint_loc_val = self.joint_loc_n[self.split_tr:,:,:,:]


        #Neural network: -------------------------------------
        
        #Flatten inputs
        inputs = tf.keras.layers.Input(shape=self.input_shape)
        inputs_noise = NoiseLayer()(inputs)
        inputs_flat = tf.keras.layers.Flatten()(inputs_noise)

        #First layer
        x = tf.keras.layers.Dense(self.unit_num, activation='relu')(inputs_flat)

        #Residual blocks; layer_num many
        for ind in range(self.layer_num):
            y = tf.keras.layers.Dense(self.unit_num)(x) 
            x = tf.keras.layers.Add()([x, y])  # skip connection
            x= tf.keras.layers.ReLU()(x)       

        #Last layer, linear activation for predictions
        last = tf.keras.layers.Dense(self.output_num)(x)
        #Denormalise
        rescale = tf.keras.layers.Multiply()([last,self.joint_stds])
        outputs = tf.keras.layers.Add()([rescale, self.joint_means])
        # ------------------------------------------------------

        #Define model
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        #Compile model
        self.model.compile(optimizer='adagrad', loss='mse')
        



    def train(self, verbosity=1):

        self.training = True #set flag to use noise layer

        #Train model with training data
        self.model.fit(self.marker_data_tr, self.joint_data_tr,
            validation_data=(self.marker_data_val, self.joint_data_val), verbose=verbosity, 
            batch_size=256, epochs=self.epochs)
        
    
    def evaluate(self):
        training = False
        pass 



    #Create NoiseLayer which applies corruption at training and identity at test time
    class NoiseLayer(tf.keras.layers.Layer):

        def __init__(self):
            super().__init__()
            self.trainable = False
            self.dynamic = True
        
        def call(self, inputs, training):
            if training:
                return corrupt(inputs)
            else:
                return inputs


#Corruption function wrapped in a tensorflow op
@tf.function
def corrupt(markers, occlude_std=0.1, shift_std=0.1, max_shift=500):
    '''
    Corruption function following Holden's corruption function.
    
    Expects input markers to have shape (frame_num, marker_num, 3)
    NB markers must be in a 'local' frame
    
    Reads markers and outputes markers with noise(shift) and occlusion(sent to origin)
    Using the default settings from Holden noting that max_shift = 'beta' = 50cm = 500(mm)
    ''' 
    
    frame_num = markers.shape[0]
    marker_num = markers.shape[1]
    
    #Sample probabilities at which to occlude/shift
    occlude_prob = np.random.normal(0, occlude_std, frame_num)
    shift_prob = np.random.normal(0, shift_std, frame_num)
    
    #Sample using clipped probabilities, and reshape appropriately
    occlusions = np.zeros((frame_num,marker_num))
    for fr in range(frame_num):
        occlusions[fr] = np.random.binomial(1, min(abs(occlude_prob[fr]), 2*occlude_std), marker_num)
    occlusions = occlusions.reshape((frame_num, marker_num))
    
    shifts = np.zeros((frame_num,marker_num))
    for fr in range(frame_num):
        shifts[fr] = np.random.binomial(1, min(abs(shift_prob[fr]), 2*shift_std), marker_num)
    shifts = shifts.reshape((frame_num, marker_num))
    
    #Sample the magnitude by which to shift each marker
    shift_mags = np.random.uniform(-max_shift, max_shift, frame_num*marker_num*3)
    shift_mags = shift_mags.reshape((frame_num, marker_num, 3))
    
    #Shift markers by shifting markers by sampled shift magnitude if the relevant entry in shifts is 1
    markers = markers + np.einsum('ij,ijk->ijk', shifts, shift_mags)
    #Occlude markers by sending occluded markers to origin
    markers = np.einsum('ijk,ij->ijk', markers, 1 - occlusions)
    
    return markers

    