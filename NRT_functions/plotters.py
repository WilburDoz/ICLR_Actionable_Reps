import matplotlib.pyplot as plt
from matplotlib import cycler
import numpy as np
import os
from datetime import datetime 
from pathlib import Path


# Function to save all the figures nicely, written by Tom George
def saveFigure(fig,today, now,saveTitle="",transparent = True, anim=False):
    """saves figure to file, by data (folder) and time (name) 
    Args:
        fig (matplotlib fig object): the figure to be saved
        saveTitle (str, optional): name to be saved as. Current time will be appended to this. Defaults to "".
    """	
    if not os.path.isdir("data/"):
        os.mkdir("data/")

    #today =  datetime.strftime(datetime.now(),'%y%m%d')
    if not os.path.isdir(f"data/{today}/"):
        os.mkdir(f"data/{today}/")
    figdir = f"data/{today}/"
    # path_ = f"{figdir}{saveTitle}_{now}"
    path_ = f"{figdir}{now}/{saveTitle}"
    path = path_
    i=1
    while True:
        if os.path.isfile(path+".png") or os.path.isfile(path+".mp4"):
            path = path_+"_"+str(i)
            i+=1
        else: break
    if anim == True:
        fig.save(path + ".mp4")
    else:
        fig.savefig(path+".png", dpi=400,transparent=transparent,bbox_inches = 'tight')
        
    return path

def neuron_plotter_1D(V, phi, Rows, Columns, dots):
    N = V.shape[0]
    plot_min = min(np.min(V), np.min(V))
    plot_max = max(np.max(V), np.max(V)) 
    
    fig, ax = plt.subplots(Rows, Columns)
    [axi.set_axis_off() for axi in ax.ravel()]
    for neuron in range(N):
        plt.subplot(Rows,Columns,neuron+1)
        if dots:
            plt.plot(phi, V[neuron,:],'*')
        else:
            plt.plot(phi, V[neuron,:])
        plt.ylim([plot_min, plot_max])
        """plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False) # labels along the bottom edge are off
        plt.tick_params(
            axis='y',          # changes apply to the y-axis
            which='both',      # both major and minor ticks are affected
            left=False,      # ticks along the bottom edge are off
            right=False,         # ticks along the top edge are off
            labelleft=False) # labels along the bottom edge are off
        """
    return fig


def freq_plot_1D(W_New, W_Old = 'void'):
    # First check if we're doing two or one plots
    two_plots = 1
    if np.sum(W_Old == 'void') != 0:
        two_plots = 0
    
    print(np.sum(W_New - W_Old))
    M = int((W_New.shape[1]-1)/2)
    freq_cont = np.zeros([M+1,1])
    circ_norms = np.linalg.norm(W_New, axis=0)
    for m in range(1,M+1):
        freq_cont[m] = np.mean(circ_norms[2*m-1:2*m+1])
    freq_cont[0] = circ_norms[0]
    
    if two_plots:
        freq_cont_old = np.zeros([M+1,1])
        circ_norms_old = np.linalg.norm(W_Old, axis=0)
        for m in range(1,M+1):
            freq_cont_old[m] = np.mean(circ_norms_old[2*m-1:2*m+1])
        freq_cont_old[0] = circ_norms_old[0]

    fig = plt.figure()
    plt.subplot(1,2,1)
    plt.title('Frequency Content')
    plt.plot(freq_cont, label='After')
    if two_plots:
        plt.plot(freq_cont_old, label='Before')
        plt.legend()

    plt.subplot(1,2,2)
    plt.plot(circ_norms,label = 'After')
    plt.title('Circle Content')
    if two_plots:
        plt.plot(circ_norms_old, label='Before')
        plt.legend()

    return fig
    
def freq_plot(W, om, thresh_percentage, om_scrunch_threshold, parameters):
  Const_content = np.linalg.norm(W[:,0])
  freq_content = np.linalg.norm(W[:,1:], axis = 0)#np.abs(W[8,1:])
  Freq_content = freq_content[1::2] + freq_content[0::2]
  fig = plt.figure(figsize = (12, 12))

  if parameters["dim"] == 1:
    om_pos = np.abs(om)

    plt.subplot(1,2,1)
    plt.plot(np.hstack([0, om_pos]), np.hstack([Const_content,Freq_content]),'*')

    same_freq_ind = 1
    om_here = om_pos
    freq_here = Freq_content
    while same_freq_ind:
      same_freq = np.logical_and(np.abs(om_here[None,:] - om_here[:,None]) < om_scrunch_threshold, np.abs(om_here[None,:] - om_here[:,None]) > 0)
      if np.sum(same_freq) == 0:
        same_freq_ind = 0
      else:
        om_merged = np.zeros(len(om_here) - 1)
        freq_merged = np.zeros(len(freq_here) - 1)
        merge_ind = np.where(same_freq)
        ind1 = merge_ind[0][0]
        ind2 = merge_ind[1][0]
        om_merged[1:] = np.delete(om_here, [ind1, ind2])
        freq_merged[1:] = np.delete(freq_here, [ind1, ind2])
        om_merged[0] = (om_here[ind1] + om_here[ind2])/2
        freq_merged[0] = (freq_here[ind1] + freq_here[ind2])
        om_here = om_merged
        freq_here = freq_merged

    plt.subplot(1,2,2)
    plt.plot(np.hstack([0, om_here]), np.hstack([Const_content,freq_here]),'*')

  if parameters["dim"] == 2:
    om_pos = np.copy(om)
    om_pos[:,1] = np.abs(om_pos[:,1])
    max_om = np.max(np.linalg.norm(om_pos, axis = 1))

    plt.subplot(2,2,1)
    plt.scatter(om[:,0], om[:,1], s = Freq_content)
    plt.xlim(-max_om*thresh_percentage, max_om*thresh_percentage)
    plt.ylim(-max_om*thresh_percentage, max_om*thresh_percentage)
    plt.title('Freqs')

    same_freq_ind = 1
    om_here = np.copy(om)
    freq_here = np.copy(Freq_content)
    freq_inds = np.arange(len(freq_here))
    while same_freq_ind:
      distances = np.sum(np.power(om_here[None,:,:] - om_here[:,None,:], 2), axis = 2)
      same_freq = np.logical_and(distances < om_scrunch_threshold, distances > 0)
      if np.sum(same_freq) == 0:
        same_freq_ind = 0
      else:
        om_merged = np.zeros([np.shape(om_here)[0] - 1, 2])
        freq_merged = np.zeros(len(freq_here) - 1)
        merge_ind = np.where(same_freq)
        ind1 = merge_ind[0][0]
        ind2 = merge_ind[1][0]
        om_merged[1:,:] = om_here[np.delete(freq_inds, [ind1, ind2]),:]
        freq_merged[1:] = np.delete(freq_here, [ind1, ind2])
        om_merged[0,:] = (om_here[ind1,:] + om_here[ind2,:])/2
        freq_merged[0] = (freq_here[ind1] + freq_here[ind2])
        om_here = om_merged
        freq_here = freq_merged
        freq_inds = np.arange(len(freq_here))

    plt.subplot(2,2,2)
    plt.scatter(om_here[:,0], om_here[:,1], s = freq_here)
    plt.xlim(-max_om*thresh_percentage, max_om*thresh_percentage)
    plt.ylim(-max_om*thresh_percentage, max_om*thresh_percentage)
    plt.title('Freqs Scrunched')

    # one plot where you double every freq's position, once pos once neg
    om_double = np.vstack([om, -om])
    Freq_content_dub = np.tile(Freq_content, 2)

    plt.subplot(2,2,3)
    plt.scatter(om_double[:,0], om_double[:,1], s = Freq_content_dub)
    plt.xlim(-max_om*thresh_percentage, max_om*thresh_percentage)
    plt.ylim(-max_om*thresh_percentage, max_om*thresh_percentage)
    plt.title('Freqs Mirrored')

    # Now do the merging on this one
    same_freq_ind = 1
    om_here = np.copy(om_double)
    freq_here = np.copy(Freq_content_dub)
    freq_inds = np.arange(len(freq_here))
    while same_freq_ind:
      distances = np.sum(np.power(om_here[None,:,:] - om_here[:,None,:], 2), axis = 2)
      same_freq = np.logical_and(distances < om_scrunch_threshold, distances > 0)
      if np.sum(same_freq) == 0:
        same_freq_ind = 0
      else:
        om_merged = np.zeros([np.shape(om_here)[0] - 1, 2])
        freq_merged = np.zeros(len(freq_here) - 1)
        merge_ind = np.where(same_freq)
        ind1 = merge_ind[0][0]
        ind2 = merge_ind[1][0]
        om_merged[1:,:] = om_here[np.delete(freq_inds, [ind1, ind2]),:]
        freq_merged[1:] = np.delete(freq_here, [ind1, ind2])
        om_merged[0,:] = (om_here[ind1,:] + om_here[ind2,:])/2
        freq_merged[0] = (freq_here[ind1] + freq_here[ind2])
        om_here = om_merged
        freq_here = freq_merged
        freq_inds = np.arange(len(freq_here))

    plt.subplot(2,2,4)
    plt.scatter(om_here[:,0], om_here[:,1], s = freq_here)
    plt.xlim(-max_om*thresh_percentage, max_om*thresh_percentage)
    plt.ylim(-max_om*thresh_percentage, max_om*thresh_percentage)
    plt.title('Freqs Mirrored and Scrunched')

  return fig