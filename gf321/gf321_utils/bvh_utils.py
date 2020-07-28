import numpy as np
import sys
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import math
import time
import matplotlib.animation as animation


class Node:

    def __init__(self):
        self.childNodes = []  # list of Node class children
        self.numChannels = 0
        self.channelNames = []
        self.animation = []  # this is rotation information for joints
        self.offset = []  # joint offset from parent
        self.name = "Joint Name"
        self.transMats = []  # Transformation matrices
        self.jointCoords = []  # Joint coordiantes


class BVHData:

    def __init__(self, bvhFileName='Empty', is_first_frame_neutral=True):
        self.root = Node()
        self.lineIter = 0
        self.allLines = []
        self.nodeStack = []
        self.allMotion = []
        self.channelTicker = 0
        self.totalFrames = 0
        self.fileName = bvhFileName
        self.frameTime = 0
        self.animationPreview = []
        self.totalJoints = 0
        self.plotMinMax = []
        self.jointPlots = []
        self.bonePlots = []
        self.first_frame_neutral = is_first_frame_neutral

    def bvhRead(self, bvhFileName):
        '''
        # Open BVH file and read the MOTION first then the HIERARCHY.
        # I'm doing this so that when I create the Node hierarchy I can store the
        # motion data for a Node on the fly in the first pass (and not have to DFS)
        # the hierarchy again later, adding the MOTION data.

        '''
        print('Reading BVH File..', bvhFileName)

        bvhFile = open(bvhFileName)

        self.allLines = bvhFile.read().split("\n")  # Split each line by \n
        numLines = len(self.allLines)
        nodeCount = 0

        ##############################################
        # Read MOTION into a numpy array
        findFRAMEStart = 0  # just a little pointer to where the FRAME/animation data starts
        for motionIter in range(numLines-1):
            line = self.allLines[motionIter].split()

            if(line[0] == 'MOTION'):
                findFRAMEStart = motionIter
                break  # only looking for MOTION line

        # Read MOTION header information
        motionIter = findFRAMEStart
        motionIter += 1  # move to line with frame count

        line = self.allLines[motionIter].split()
        self.totalFrames = int(line[1])  # record number of frames

        motionIter += 1  # move to line with frame time
        line = self.allLines[motionIter].split()
        self.frameTime = float(line[2])  # record frame time

        motionIter += 1  # move to first line of motion
        line = self.allLines[motionIter].split()

        if self.first_frame_neutral:  # if first frame of bvh file is neutral pose, otherwise first line of motion is actual mocap
            self.firstFrame = np.array(line)
            motionIter += 1  # move to first frame of actual mocap
            self.totalFrames -= 1  # ignore first neutral frame

        # Initialise a np array of zeros for the motion data so we can slice it
        # nicely later when reading the hierarchy and add it to each node on the fly
        self.allMotion = np.zeros(
            (self.totalFrames, len(self.allLines[motionIter].split())))
        #print('Shape of MOTION', np.shape(self.allMotion))
        counter = 0

        while(motionIter < numLines-1):
            line = self.allLines[motionIter].split()
            # stack the lines nicely into the array
            self.allMotion[counter, :] = line
            motionIter += 1
            counter += 1

        #############################################
        # Read HIERARCHY into a Node hierarchy
        # Push (append) JOINTS (and End Site) to a stack. When you see a }, then Pop joints off.
        # Always add a new Node as a child to whatever is currently top of the stack.

        # Get current line and split into 'words'
        self.lineIter = 0  # new line counter
        line = self.allLines[self.lineIter].split()

        while(line[0] != 'MOTION'):  # read everything before MOTION (so the hierarchy)

            # Get current line and split into 'words'
            line = self.allLines[self.lineIter].split()

            if line[0] == "ROOT":
                nodeCount += 1
                #print('ROOT', self.lineIter, 'Count', nodeCount)
                # Get root data
                rootNode = Node()
                # Push to top of stack of nodes
                self.nodeStack.append(rootNode)

                rootNode.name = line[1]
                self.lineIter += 2  # Skip the { to OFFSET
                line = self.allLines[self.lineIter].split()

                rootNode.offset = [float(line[1]), float(
                    line[2]), float(line[3])]  # set offset
                self.lineIter += 1  # go to channels line
                line = self.allLines[self.lineIter].split()

                rootNode.numChannels = int(line[1])  # set number of channels
                # +2 to skip CHANNELS and channel number
                for i in range(2, rootNode.numChannels+2):
                    rootNode.channelNames.append(line[i])  # get channel names

                # Slice through the MOTION array getting animation data for the root
                # channelTicker = 0 initially
                rootNode.animation = self.allMotion[:,
                                                    self.channelTicker:self.channelTicker + rootNode.numChannels]
                # keeps track of channels we collect motion data for
                self.channelTicker += rootNode.numChannels

                # Make transformation matrix for this node
                for frame in range(self.totalFrames):
                    # makeTransMat takes arguments (axisAngles, transOffsets, channelNames)
                    rootNode.transMats.append(self.makeTransMat(
                        rootNode.animation[frame, 3:], rootNode.animation[frame, 0:3], rootNode.channelNames))
                    thisMat = rootNode.transMats[frame]
                    rootNode.jointCoords.append(
                        [thisMat[0, 3], thisMat[1, 3], thisMat[2, 3]])

                # Associate this with main root of the BVH class
                self.root = rootNode
                self.totalJoints += 1
                # End root data - ok, so JOINT's start next

            if line[0] == "JOINT":
                nodeCount += 1
                #print('JOINT', line[1], 'Line', self.lineIter, 'Count', nodeCount)
                # Create a joint and add it to the top of the stack.
                # Increment global lineIter as we go so we are
                # in the right place in the BVH file all the time for any call
                self.nodeStack.append(self.addJoint())

            if line[0] == "End":

                # Create an end node
                nodeCount += 1
                #print('JOINT (End Site)', self.lineIter, 'Count', nodeCount)
                self.nodeStack.append(self.addEndSite())

            if line[0] == "}":

                if(self.nodeStack):
                    # Pop the stack, so that new children are added to the correct node
                    nodeCount -= 1
                    #print('Pop Joint - Count', nodeCount)
                    self.nodeStack.pop()
                #else:
                    #print('Stack empty')
                    #print(self.nodeStack)

            self.lineIter += 1

        print('Done')
        # return rootNode (Darren left this here)

    # function is called only when line[0] is JOINT. gf321 checked.
    def addJoint(self):

        # Read JOINT NAME, OFFSET, CHANNEL  and Animation for this joint
        jointNode = Node()
        line = self.allLines[self.lineIter].split()

        jointNode.name = line[1]  # joint name
        self.lineIter += 2  # Skip the { to get to the OFFSET
        line = self.allLines[self.lineIter].split()

        jointNode.offset = [float(line[1]), float(
            line[2]), float(line[3])]  # record offset
        self.lineIter += 1  # go to channel line
        line = self.allLines[self.lineIter].split()

        jointNode.numChannels = int(line[1])  # record number of channels
        for i in range(2, jointNode.numChannels+2):  # +2 to start at first channel name
            jointNode.channelNames.append(line[i])  # record channel names

        # Slice through the MOTION array getting animation data for this joint
        jointNode.animation = self.allMotion[:,
                                             self.channelTicker:self.channelTicker + jointNode.numChannels]
        self.channelTicker += jointNode.numChannels

        # Make transformation matrices for this node (mult by parent transMat)
        for frame in range(self.totalFrames):
            # recursive by nature
            parentTrans = self.nodeStack[-1].transMats[frame]

            if jointNode.numChannels == 3:
                # There is no additional translation of the bone, so just use offset for translation
                jointNode.transMats.append(np.matmul(parentTrans, self.makeTransMat(
                    jointNode.animation[frame, 0:3], jointNode.offset, jointNode.channelNames)))

            else:
                # There is an additional translation of the bone, so just offset + the translation
                #newTrans = [jointNode.offset[i] + jointNode.animation[frame][i] for i in range(len(jointNode.offset))]
                newTrans = [jointNode.offset[i] +
                            0 for i in range(len(jointNode.offset))]
                jointNode.transMats.append(np.matmul(parentTrans, self.makeTransMat(
                    jointNode.animation[frame, 3:], newTrans, jointNode.channelNames)))

            thisMat = jointNode.transMats[-1]
            jointNode.jointCoords.append(
                [thisMat[0, 3], thisMat[1, 3], thisMat[2, 3]])

        # Make this joint a child of whatever is top of the stack
        self.nodeStack[-1].childNodes.append(jointNode)
        self.totalJoints += 1

        return jointNode

    def addEndSite(self):  # gf321 checked.

        # Read JOINT NAME, OFFSET, CHANNEL  and Animation for this joint
        endNode = Node()
        endNode.name = "End Site"
        endNode.numChannels = 0

        self.lineIter += 2  # Skip the { to OFFSET line
        # line is not global, so read again from allLines (which *is* global)
        line = self.allLines[self.lineIter].split()
        endNode.offset = [float(line[1]), float(
            line[2]), float(line[3])]  # record offset

        # Slice through the MOTION array getting animation data for this joint
        endNode.animation = self.allMotion[:,
                                           self.channelTicker:self.channelTicker + endNode.numChannels]
        # (code pretty much copy-pasted from joint and root; animation=[])
        self.channelTicker += endNode.numChannels

        # It does however have coords, whilst joint angles are irrelevant (as no joint)
        # Make transformation matrices for this node (mult by parent transMat)
        for frame in range(self.totalFrames):
            parentTrans = self.nodeStack[-1].transMats[frame]
            endNode.transMats.append(np.matmul(parentTrans, self.makeTransMat(
                [0, 0, 0], endNode.offset, self.root.channelNames)))
            thisMat = endNode.transMats[-1]
            endNode.jointCoords.append(
                [thisMat[0, 3], thisMat[1, 3], thisMat[2, 3]])

        # Make this joint a child of whatever is top of the stack
        self.nodeStack[-1].childNodes.append(endNode)
        self.totalJoints += 1

        return endNode

    def makeRotMat(self, axisAngles):

        # Make a composite rotation matrix from axis angles x,y,z
        # Essentially makeTransMat without the extra row and column
        # Also assumes axisAngles are in order x,y,z
        Rx = np.zeros((3, 3))
        Ry = np.zeros((3, 3))
        Rz = np.zeros((3, 3))

        xRad = math.radians(axisAngles[0])
        yRad = math.radians(axisAngles[1])
        zRad = math.radians(axisAngles[2])

        Rx[0, 0] = 1
        Rx[1, 1] = math.cos(xRad)
        Rx[1, 2] = - math.sin(xRad)
        Rx[2, 1] = math.sin(xRad)
        Rx[2, 2] = math.cos(xRad)

        Ry[0, 0] = math.cos(yRad)
        Ry[0, 2] = math.sin(yRad)
        Ry[1, 1] = 1
        Ry[2, 0] = - math.sin(yRad)
        Ry[2, 2] = math.cos(yRad)

        Rz[0, 0] = math.cos(zRad)
        Rz[0, 1] = - math.sin(zRad)
        Rz[1, 0] = math.sin(zRad)
        Rz[1, 1] = math.cos(zRad)
        Rz[2, 2] = 1

        return np.matmul(Rx, np.matmul(Ry, Rz))

    def makeTransMat(self, axisAngles, transOffsets, channelNames):  # gf321 checked

        # Make a composite rotation matrix from axis angles x,y,z
        # NB! for BVH files, very important that rotation concat follows
        # order specified in channelNames for the current Node

        # Get channel ordering + abbreviate
        if(len(channelNames) == 3):
            xRotPos = channelNames.index('Xrotation')
            yRotPos = channelNames.index('Yrotation')
            zRotPos = channelNames.index('Zrotation')
        else:
            xRotPos = channelNames.index('Xrotation')-3
            yRotPos = channelNames.index('Yrotation')-3
            zRotPos = channelNames.index('Zrotation')-3

        xRad = math.radians(axisAngles[xRotPos])
        yRad = math.radians(axisAngles[yRotPos])
        zRad = math.radians(axisAngles[zRotPos])

        Rx = np.zeros((3, 3))
        Ry = np.zeros((3, 3))
        Rz = np.zeros((3, 3))
        transform = np.zeros((4, 4))

        Rx[0, 0] = 1
        Rx[1, 1] = math.cos(xRad)
        Rx[1, 2] = - math.sin(xRad)
        Rx[2, 1] = math.sin(xRad)
        Rx[2, 2] = math.cos(xRad)

        Ry[0, 0] = math.cos(yRad)
        Ry[0, 2] = math.sin(yRad)
        Ry[1, 1] = 1
        Ry[2, 0] = - math.sin(yRad)
        Ry[2, 2] = math.cos(yRad)

        Rz[0, 0] = math.cos(zRad)
        Rz[0, 1] = - math.sin(zRad)
        Rz[1, 0] = math.sin(zRad)
        Rz[1, 1] = math.cos(zRad)
        Rz[2, 2] = 1

        # Apply rotations in correct order
        # x,y,z
        if(xRotPos == 0) and (yRotPos == 1) and (zRotPos == 2):
            transform[:3, :3] = np.matmul(Rx, np.matmul(Ry, Rz))

        # x,z,y
        if(xRotPos == 0) and (yRotPos == 2) and (zRotPos == 1):
            transform[:3, :3] = np.matmul(Rx, np.matmul(Rz, Ry))

        # y,x,z
        if(xRotPos == 1) and (yRotPos == 0) and (zRotPos == 2):
            transform[:3, :3] = np.matmul(Ry, np.matmul(Rx, Rz))

        # y,z,x
        if(xRotPos == 2) and (yRotPos == 0) and (zRotPos == 1):
            transform[:3, :3] = np.matmul(Ry, np.matmul(Rz, Rx))

        # z,x,y
        if(xRotPos == 1) and (yRotPos == 2) and (zRotPos == 0):
            transform[:3, :3] = np.matmul(Rz, np.matmul(Rx, Ry))

        # z,y,x
        if(xRotPos == 2) and (yRotPos == 1) and (zRotPos == 0):
            transform[:3, :3] = np.matmul(Rz, np.matmul(Ry, Rx))

        # Add translation
        transform[3, 3] = 1
        transform[0:3, 3] = transOffsets

        return transform

    def bvhDraw(self, frameStep=1):
        '''
        # Draw BVH file:
        #
        #   frameStep - for large fps files (>200?) rendering can slow a little so sometimes might be 
        #               useful to have a frameStep parameter
        #
        # First, pre-calculate the joints/bones and place them in a list by recursively 
        # moving through the hiererachy. This is as opposed to estimating 3D positions
        # from the hierarchy for each frame and printing. 
        # Once in a suitable format, FuncAnimation can be used for fast rendering.
        '''

        rootNode = self.root
        frame = 0
        frameStart = 0
        frameEnd = self.totalFrames-1

        # Recursively read the Nodes from the BVH Object and store parent to children connections creating a bone list.
        # This makes for easier drawing
        for frame in range(frameStart, frameEnd, frameStep):

            currentJointCoords = rootNode.jointCoords[frame]

            # If there are children to the root, then start recursion to read the hierarchy
            if len(rootNode.childNodes) > 0:
                for i in range(len(rootNode.childNodes)):
                    self.preCalculateBone(
                        rootNode.childNodes[i], currentJointCoords, frame)

        # Get min and max values for x, y and z axis
        minX, maxX, minY, maxY, minZ, maxZ = 0, 0, 0, 0, 0, 0
        allX, allY, allZ = [], [], []

        for iter in range(len(self.animationPreview)):
            thisBone = self.animationPreview[iter]
            for inner in range(2):
                allX.append(thisBone[0][inner])
                allY.append(thisBone[1][inner])
                allZ.append(thisBone[2][inner])

        minX = min(allX)
        maxX = max(allX)
        minY = min(allY)
        maxY = max(allY)
        minZ = min(allZ)
        maxZ = max(allZ)

        #print('Drawing BVH..')
        self.fig = plt.figure()

        # NB swapping Y and Z as Y is up and not Z
        self.ax = self.fig.add_subplot(projection="3d", xlim=(
            minX, maxX), ylim=(minZ, maxZ), zLim=(minY, maxY))

        # Create plot objects per bone that can be updated with data in the drawSkeleton func (via FuncAnimation)
        # This makes for super fast rendering
        for jointNum in range(self.totalJoints):
            # 3D plots can't contain empty arrays - so have to initialise
            self.jointPlots.append(self.ax.plot3D(
                [0, 0], [0, 0], [0, 0], 'blue'))
            self.bonePlots.append(self.ax.plot3D([0, 0], [0, 0], [0, 0], 'ro'))

        self.ani = animation.FuncAnimation(
            self.fig, self.drawSkeleton, interval=1, repeat=False)
        plt.show()

    def drawSkeleton(self, frame):

        if frame > len(self.animationPreview):
            self.ani.event_source.stop()

        for jointNum in range(self.totalJoints):

            thisJoint = self.animationPreview[(
                frame*self.totalJoints) + jointNum]

            # NB there is no .set_data() for 3 dimensional data. So have to update
            # x and y using .set_data() and then z using set_3d_properties
            self.jointPlots[jointNum][0].set_data(thisJoint[0], thisJoint[2])
            self.jointPlots[jointNum][0].set_3d_properties(thisJoint[1])
            self.bonePlots[jointNum][0].set_data(thisJoint[0], thisJoint[2])
            self.bonePlots[jointNum][0].set_3d_properties(thisJoint[1])

    def preCalculateBone(self, currentNode, lastJointCoords, frame):

        # Put all the bones in a list recursively to make drawing easier
        # While there are children to process, do one, then recurse to process further down the hierarchy
        if len(currentNode.childNodes) > 0:

            currentJoint = currentNode.jointCoords[frame]
            self.animationPreview.append([[lastJointCoords[0], currentJoint[0]], [
                                         lastJointCoords[1], currentJoint[1]], [lastJointCoords[2], currentJoint[2]]])

            for i in range(len(currentNode.childNodes)):
                self.preCalculateBone(
                    currentNode.childNodes[i], currentJoint, frame)
            else:
                return  # exit (return None essentially)
        else:
            currentJoint = currentNode.jointCoords[frame]
            self.animationPreview.append([[lastJointCoords[0], currentJoint[0]], [
                                         lastJointCoords[1], currentJoint[1]], [lastJointCoords[2], currentJoint[2]]])

    
    ################### gf321 functions ##################

    def getJointAngles(self, withJoints=True, withPos=True, withNames=False):
        '''
        Can only use once data has been read using self.bvhRead

        If withJoints=True and withPos=True, returns joint angles and positions in array:

            Returns array of the joint transforms with dimensions n*j*3*4
            where n is the number of frames in motion,
            j is number of joints in skeleton,
            and the 3*4 matrix is the 3*3 matrix of rotation data and extra column of global positions
        
        Otherwise will return either an array of joint angles only joint positions only:
            If withJoints True:
                Returns array of joint angles n*j*3*3
            If withPos True:
                Returns array of joint global positions n*j*3

        If both flags are false raises an exception

        withNames=True will cause function to return both array and names,
        otherwise only array is returned
        
        '''

        #Raise exception if Booleans used incorrectly
        if (not withJoints) and (not withPos):
            raise Exception('Please supply getJointAngles with at least one true optional Boolean (withJoints or withPos or both)')
            
        
        root = self.root
        frameEnd = self.totalFrames - 1

        #set an offset to skip first neutral frame
        if self.first_frame_neutral:
            frameOffset = 1
        else: 
            frameOffset = 0
        
        jointOrder = []

        if withJoints and withPos:
            jointArray = np.zeros((frameEnd, self.totalJoints, 3, 4))
        elif withJoints:
            jointArray = np.zeros((frameEnd, self.totalJoints, 3, 3))
        else:
            jointArray = np.zeros((frameEnd, self.totalJoints, 3))

        def readHierarchyToArray(currentJoint):

            #exit first layer of recursion when back to root
            if len(jointOrder)==self.totalJoints:
                return

            # add joint name to list of jointOrder
            jointOrder.append(currentJoint.name)
            jointIdx = len(jointOrder)-1

            # read transMats of input joint into array,
            for frameIdx in range(frameEnd):

                if withJoints and withPos:
                    #We use offset to avoid the first neutral frame 
                    jointArray[frameIdx][jointIdx] = np.array(currentJoint.transMats)[frameIdx + frameOffset][:][:-1]
                elif withJoints:
                    jointArray[frameIdx][jointIdx] = np.array(currentJoint.transMats)[frameIdx + frameOffset][:-1,:-1]
                else:
                    jointArray[frameIdx][jointIdx] = np.array(currentJoint.jointCoords)[frameIdx + frameOffset][:]

            # read children recursively
            for childJoint in currentJoint.childNodes:
                readHierarchyToArray(childJoint)
            return  # escape

        readHierarchyToArray(root) #run recursion to fill jointArray

        #return the array of joint transforms and joint name order
        if withNames:
            return jointArray, jointOrder
        else: return jointArray


############## gf321 utils methods #################

def create_local_tfs_root(joints_g):
    '''
    Returns local inverse transforms for each frame using the root's global orientation and position
    The returned array has shape (frame_number,4,4) where each transform has form:
    [(   ),tx]
    [( R ),ty]
    [(   ),tz]
    [0,0,0,1 ]
    where R the 3by3 orientation matrix, and tx,ty,tz are the global root position
    '''

    frame_num = joints_g.shape[0]
    
    local_tfs = np.zeros((frame_num,4,4)) #init
    joint_4b4 = np.zeros((frame_num,4,4))
    joint_4b4[:,3,3] = 1
    joint_4b4[:,:3] = joints_g[:,0]
    
    #create local transforms looping over frames
    for fr in range(frame_num):
        joint_4b4_inv = np.linalg.inv(joint_4b4[fr])
        local_tfs[fr] = joint_4b4_inv
    
    return local_tfs


def make_joints_local(joints_g, three_by_four=True):
    
    frame_num = joints_g.shape[0]
    joint_num = joints_g.shape[1]
    
    local_tfs = create_local_tfs_root(joints_g) #using root 
    ones = np.array([0,0,0,1])
    
    joints_g_4b4 = np.zeros((frame_num, joint_num, 4, 4))
    joints_local = np.zeros((frame_num, joint_num, 4, 4))
    
    joints_g_4b4[:,:,-1] = ones
    joints_g_4b4[:,:,:-1] = joints_g
    
    for fr in range(frame_num):
        for jt in range(joint_num):
            joints_local[fr][jt] = np.matmul(local_tfs[fr],joints_g_4b4[fr][jt])

    joints_local_3b4 = np.zeros((frame_num, joint_num, 3, 4))
    joints_local_3b4 = joints_local[:,:,:-1,:]
    
    if three_by_four:
        return joints_local_3b4
    else:
        return joints_local


def local_tfs_each_joint(joints_g):
    '''
    Creates array of all inverse joint transforms to be used in finding
    marker offsets by marker_utils.create_marker_offsets
    takes joints_g of shape (frame_num, joint_num,3,4)

    Returns all_tfs of shape (frame_num, joint_num,4,4)
    using 4,4 for ease of use by other functions
    '''
    frame_num = joints_g.shape[0]
    joint_num = joints_g.shape[1]
    all_tfs = np.zeros((frame_num,joint_num,4,4))
    
    joints_g_4 = np.zeros((frame_num,joint_num,4,4))
    joints_g_4[:,:,:-1] = joints_g
    joints_g_4[:,:,3,3] = 1
    
    for fr in range(frame_num):
        for jt in range(joint_num):
            #compute the inverse transform to bring a vector local to a given joint and frame
            all_tfs[fr][jt] = np.linalg.inv(joints_g_4[fr][jt])
    
    return all_tfs


def get_joint_pos(joints):
    '''
    Returns joint positions in array shape (frame_num,joint_num,3)
    Can take as input shapes (frame_num,joint_num,4,4) or (frame_num,joint_num,3,4) 
    '''
    
    joint_pos = np.zeros((joints.shape[0],joints.shape[1],3))
    
    joint_pos[:,:] = joints[:,:,:3,3]
    
    return joint_pos


def correct_orientation(joints):
    '''
    Gets joints in standard bvh format and puts them in x,y,z orientation with z pointing up.
    The corrected z is positive y in bvh and the corrected y is negative z in bvh.
    '''
    joints_corrected = np.copy(joints)
    joints_corrected[:,:,1] = - joints[:,:,2]
    joints_corrected[:,:,2] = joints[:,:,1]
    
    return joints_corrected

###############################################################









if __name__ == '__main__':
    print('BVHData \'main\' is running the default demo..')
    print('Run as a program, this will run through basic usage.')
    print('System inputs', sys.argv)

    # Default file to load for demo
    #bvhFileName = 'skeleton.bvh'
    #bvhFileName = 'skeleton_motion_jump.bvh'
    bvhFileName = 'dog1_trot_skeleton.bvh'
    bvhObject = BVHData()
    bvhObject.bvhRead(bvhFileName)
    bvhObject.bvhDraw(1)  # takes frameStep parameter
