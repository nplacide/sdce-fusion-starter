# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Data association class with single nearest neighbor association and gating based on Mahalanobis distance
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np
from scipy.stats.distributions import chi2

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import misc.params as params 

class Association:
    '''Data association class with single nearest neighbor association and gating based on Mahalanobis distance'''
    def __init__(self):
        self.association_matrix = np.matrix([])
        self.unassigned_tracks = []
        self.unassigned_meas = []
        
    def associate(self, track_list, meas_list, KF):
             
        ############
        # TODO Step 3: association:
        # - replace association_matrix with the actual association matrix based on Mahalanobis distance (see below) for all tracks and all measurements
        # - update list of unassigned measurements and unassigned tracks
        ############
        
        # the following only works for at most one track and one measurement
        '''
        self.association_matrix = np.matrix([]) # reset matrix
        self.unassigned_tracks = [] # reset lists
        self.unassigned_meas = []

        if len(meas_list) > 0:
            self.unassigned_meas = [0]
        if len(track_list) > 0:
            self.unassigned_tracks = [0]
        if len(meas_list) > 0 and len(track_list) > 0: 
            self.association_matrix = np.matrix([[0]])
        '''
        # Multi tracks and multi measurements implementation
        # Get the number of unassigned tracks
        N = len(track_list)
        #Get the number of unassigned measurement
        M = len(meas_list)
        #Create a list with a range N for the unassigned tracks
        self.unassigned_tracks = list(range(N))
        #Same thing for the unassigned measurements
        self.unassigned_meas = list(range(M))

        # initialize the association matrix with infinity values.
        #Anything not replaced remains infinity
        self.association_matrix = np.inf*np.ones((N,M)) 

        # loop over all tracks and measurements to set up association matrix
        for i in range(N): 
            track = track_list[i]
            for j in range(M):
                meas = meas_list[j]
                #Use MHD()function to implement the Mahalanobis distance between track and measurement
                mhd_dist = self.MHD(track, meas, KF)
                #Use the gating() function to check if measurement lies inside track's gate
                if self.gating(mhd_dist, meas.sensor):
                    #mhd_dist inside track gate -> replace infinity value in association matrix
                    self.association_matrix[i,j] = mhd_dist
        
        ############
        # END student code
        ############ 
                
    def get_closest_track_and_meas(self):
        ############
        # TODO Step 3: find closest track and measurement:
        # - find minimum entry in association matrix
        # - delete row and column
        # - remove corresponding track and measurement from unassigned_tracks and unassigned_meas
        # - return this track and measurement
        ############

        # the following only works for at most one track and one measurement
        '''
        update_track = 0
        update_meas = 0
        
        # remove from list
        self.unassigned_tracks.remove(update_track) 
        self.unassigned_meas.remove(update_meas)
        self.association_matrix = np.matrix([])
        '''
        A = self.association_matrix
		#return np.nan (NaN == Not a Number) where we find infinity in the association matrix
        if np.min(A) == np.inf:
            return np.nan, np.nan

        # We need to find the minimum entries in the association matrix
		# We will need to convert our flat array of flat indices into a tuple of coordinate array
		# We use numpy unravel_index to do that
        itrack, imeas = np.unravel_index(np.argmin(A, axis=None), A.shape) 

        # We delete the corresponding row and column from the matrix
        A = np.delete(A, itrack, 0) 
        A = np.delete(A, imeas, 1)
        self.association_matrix = A

        # update the list of unassigned measurements and unassigned tracks
        update_track = self.unassigned_tracks[itrack] 
        update_meas = self.unassigned_meas[imeas]

        # remove corresponding track and measurement from unassigned_tracks and unassigned_meas
        self.unassigned_tracks.remove(update_track) 
        self.unassigned_meas.remove(update_meas)
        
            
        ############
        # END student code
        ############ 
        return update_track, update_meas     

    def gating(self, MHD, sensor): 
        ############
        # TODO Step 3: return True if measurement lies inside gate, otherwise False
        ############
        #To get the chi square critical value, we need the significance level q and the degrees of freedom df
        q = params.gating_threshold
        df = sensor.dim_meas
        #determine if MHD lies inside gate
        return MHD < chi2.ppf(q, df)    
        
        ############
        # END student code
        ############ 
        
    def MHD(self, track, meas, KF):
        ############
        # TODO Step 3: calculate and return Mahalanobis distance
        ############
        
        # calculate Mahalanobis distance
        H = meas.sensor.get_H(track.x) 
        gamma = KF.gamma(track, meas)
        # Apply the Mahalanobis distance formula
        MHD = gamma.transpose() * np.linalg.inv(KF.S(track,meas,H)) * gamma 
        return MHD
        
        ############
        # END student code
        ############ 
    
    def associate_and_update(self, manager, meas_list, KF):
        # associate measurements and tracks
        self.associate(manager.track_list, meas_list, KF)
    
        # update associated tracks with measurements
        while self.association_matrix.shape[0]>0 and self.association_matrix.shape[1]>0:
            
            # search for next association between a track and a measurement
            ind_track, ind_meas = self.get_closest_track_and_meas()
            if np.isnan(ind_track):
                print('---no more associations---')
                break
            track = manager.track_list[ind_track]
            
            # check visibility, only update tracks in fov    
            if not meas_list[0].sensor.in_fov(track.x):
                continue
            
            # Kalman update
            print('update track', track.id, 'with', meas_list[ind_meas].sensor.name, 'measurement', ind_meas)
            KF.update(track, meas_list[ind_meas])
            
            # update score and track state 
            manager.handle_updated_track(track)
            
            # save updated track
            manager.track_list[ind_track] = track
            
        # run track management 
        manager.manage_tracks(self.unassigned_tracks, self.unassigned_meas, meas_list)
        
        for track in manager.track_list:            
            print('track', track.id, 'score =', track.score)