# gf321

import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib.animation as animation

######################

def plot(positions,frame,plot_pad=0.1):
    '''
    Input positions has to be of shape (frame_num,marker_num/joint_num,3)
    plot_pad pads out the plot with empty space for viewability
    '''
    
    #reshape flattened markers
    if len(positions.shape) ==2:
        positions = positions.reshape((positions.shape[0],int(positions.shape[1]/3),3))
    
    #Compute maxs and mins for plotting neatness
    minX = positions[frame,:,0].min()
    maxX = positions[frame,:,0].max()
    minY = positions[frame,:,2].min()
    maxY = positions[frame,:,2].max()
    minZ = positions[frame,:,1].min()
    maxZ = positions[frame,:,1].max()
    
    print('minmaxs are: ',minX,maxX,minY,maxY,minZ,maxZ)
    
    p = plot_pad*(maxZ-minZ) #pad a little bit according to Z
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', 
                xlim=(minX-p, maxX+p), ylim=(minY-p, maxY+p), zLim=(minZ-p, maxZ+p))
    
    plt.xlabel('x'); plt.ylabel('y')

    ax.plot3D(positions[frame,:,0],positions[frame,:,2],positions[frame,:,1],'bo',markersize=3.5)

######################

class AnimatePlot():

    def __init___(self):
        #For plotting pos1 and pos2 side by side via an offset
        self.plot_offset = [1000,0,0]

    #Assumes the input data hasnt been 'corrected' so is in Z,X,Y format

    #NB: There the code plots a point at the origin for some reason.
    #Most likely is to do with the plot initialisation. Darren's bvhDraw above doesnt do it somehow
    #Although this is no problem for a local plot!
    
    def animated_plot(self,pos1,pos2=np.array([None]),fps=120, offset=False, plot_pad=0,fit_plot=False,to_scale=True):

        '''
        Takes pos1 argument as position of array with shape (frame_num,obj_num,3)
        Optional pos2 argument plots in a different colour
        '''
        if offset: #offset pos2
            pos2[:,:] += self.plot_offset

        frametime = 1/fps
        frame_num = pos1.shape[0]
        joint_num = pos1.shape[1]
        obj_num = joint_num

        if pos2.any() != None:
            assert pos1.shape[0]==pos2.shape[0] #ensure shapes are the same for plotting
            marker_num = pos2.shape[1]
            obj_num += marker_num

        p = plot_pad #for brevity later

        fig = plt.figure()

        pos=pos1
        if pos2.any() != None:
            pos = np.append(pos,pos2,axis=1) #append along the object axis

        if not fit_plot:
            minX = pos[:,:,0].min()
            maxX = pos[:,:,0].max()
            minY = pos[:,:,2].min()
            maxY = pos[:,:,2].max()
            minZ = pos[:,:,1].min()
            maxZ = pos[:,:,1].max()

            mins = min(minX,minY,minZ)
            maxs = max(maxX,maxY,maxZ)


        if not fit_plot:
            if to_scale:
                ax = fig.add_subplot(projection='3d',
                        #xlim=(minX-p, maxs+p), ylim=(minY-p, maxs+p), zLim=(minZ-p, maxs+p))
                        xlim=(mins-p, maxs+p),  ylim=(mins-p, maxs+p), zLim=(minZ-p, minZ+maxs-mins+p))
            else:
                ax = fig.add_subplot(projection='3d',
                        #xlim=(minX-p, maxs+p), ylim=(minY-p, maxs+p), zLim=(minZ-p, maxs+p))
                        xlim=(minX-p, maxX+p), ylim=(minY-p, maxY+p), zLim=(minZ-p, maxZ+p))
        else:
            ax = ax = fig.add_subplot(projection='3d')
            
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')


        plots = []
        def init():
            for obj in range(joint_num):
                plots.append(ax.plot3D([], [], [],'bo',markersize=4))
            if pos2.any() != None:
                for obj in range(marker_num): 
                    plots.append(ax.plot3D([], [], [],'ro',markersize=4))
                                            
        
        def draw(frame):

            for obj in range(obj_num):

                if fit_plot:
                    minX = pos[frame,:,0].min()
                    maxX = pos[frame,:,0].max()
                    minY = pos[frame,:,2].min()
                    maxY = pos[frame,:,2].max()
                    minZ = pos[frame,:,1].min()
                    maxZ = pos[frame,:,1].max()
                    
                    if to_scale:
                        mins = min(minX,minY,minZ)
                        maxs = max(maxX,maxY,maxZ)
                        ax.set_xlim(mins-p,maxs+p)
                        ax.set_ylim(mins-p,maxs+p)
                        ax.set_zlim(mins-p,maxs+p)
                    
                    else:
                        ax.set_xlim(minX-p,maxX+p)
                        ax.set_ylim(minY-p,maxY+p)
                        ax.set_zlim(minZ-p,maxZ+p) 
                
                plots[obj][0].set_data(pos[frame,obj,0], pos[frame,obj,2])
                plots[obj][0].set_3d_properties(pos[frame,obj,1])


        self.anim = animation.FuncAnimation(fig, draw, frames=frame_num, init_func=init, interval=1000*frametime)
        plt.show()

####################################################
