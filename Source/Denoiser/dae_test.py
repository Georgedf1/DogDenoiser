# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 20:22:51 2020

@author: georg
"""

import numpy as np
import json
import os


def set_values():
    pathToData = 'C:/Users/georg/OneDrive/Desktop/Research/Data'
    dog = 'dog1'
    motion = 'all' #capture all motion
    motion_types = ['jump','poles','table','testSeq','trot','walk'] #types for dog1
    
    return pathToData, dog, motion, motion_types


def get_markers(pathToData, dog, motion):
    
    '''
    From Data file path, dog, and motion type,
    reads json file and
    returns marker_names and markers as np arrays
    '''
    
    pathToMotion = os.path.join(pathToData, dog, 'motion_%s'%motion)
    markerFile = os.path.join(pathToMotion, 'motion_capture', 'markers.json')
        
    markers_dict = json.load(markerFile)
    
    #marker_names to ensure consistency across motion types in data
    marker_names = np.array(list(markers_dict.keys()))
    print(marker_names)
    
    markers = np.array(list(markers_dict.values()))
    
    return markers


def normalise_markers(x):
    '''
    Normalises along the -1 axis np array with shape (num_markers,frames)
    '''
    x_mean_scaled = np.zeros_like(x)
    x_normed = np.zeros_like(x)
    
    for frame in range(x.shape[1]):
        x_mean_scaled[:,frame] = x[:,frame] - x.mean(axis=-1)
        
    for frame in range(x.shape[1]):
        x_normed[:,frame] = x_mean_scaled[:,frame] / x.std(axis=-1)
        
    return x_normed 

#------------------------------------------------

#Set basic values
pathToData, dog, motion, motion_types = set_values()

#Open data file    
with open(pathToData,'r') as MarkerFile:
    
    #If motion isnt all then only read specific motion type
    if motion != 'all':
        markers = get_markers(pathToData, dog, motion)
    
    #motion = 'all', so obtain all mocap marker data for the given dog
    #Need to ensure motion_types is correct!
    else:
        markers = None
        for motion in motion_types:
            
            marker_set = get_markers(pathToData, dog, motion)
            
            if markers == None:
                markers = marker_set
            else: 
                np.append(markers,marker_set)


    