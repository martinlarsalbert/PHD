import numpy as np
import pandas as pd
from numpy.linalg.linalg import inv, pinv

class KalmanFilter:

    def __init__(
        self,
        Ad: np.ndarray,
        Bd: np.ndarray,
        Cd: np.ndarray,
        Ed: np.ndarray,
        Qd: float,
        Rd: float,
    ) -> pd.DataFrame:
        """Example kalman filter for yaw and yaw rate
        Parameters
        ----------
        Ad : np.ndarray
            2x2 array: discrete time transition matrix
        Bd : np.ndarray
            2x1 array: discrete time input transition matrix
        Cd : np.ndarray
            1x2 array: measurement transition matrix
        Ed : np.ndarray
            2x1 array
        Qd : float
            process noise
        Rd : float
            measurement noise
        Returns
        -------
        pd.DataFrame
            data frame with filtered data
        """
        self.Ad=Ad
        self.Bd=Bd
        self.Cd=Cd
        self.Ed=Ed
        self.Qd=Qd
        self.Rd=Rd
        

    def predict(self, x_hat, P_hat, u, h):
        
        Ad = self.Ad
        Bd = self.Bd
        Ed = self.Ed
        Qd = self.Qd
        
        n_states = len(x_hat)
        self.Phi = Phi = np.eye(n_states) + Ad*h
        #Phi = Ad
        
        
        # Predictor (k+1)
        x_prd = Phi @ x_hat + Bd @ u
        #P_prd = Ad @ P_hat @ Ad.T + Ed * Qd @ Ed.T
        P_prd = Phi @ P_hat @ Phi.T + Qd
        
        return x_prd, P_prd
    
    def update(self, y, P_prd, x_prd):
            
        Cd = self.Cd
        Rd = self.Rd
        n_states = len(x_prd)
        
        epsilon = y - Cd @ x_prd  # Error between meassurement (y) and predicted measurement Cd @ x_prd
        
        # Compute kalman gain:
        S = Cd @ P_prd @ Cd.T + Rd  # System uncertainty
        K = P_prd @ Cd.T @ inv(S)

        # State corrector:
        x_hat = x_prd + K @ epsilon
        
        # corrector
        IKC = np.eye(n_states) - K @ Cd        
        P_hat = IKC * P_prd @ IKC.T + K * Rd @ K.T
        
        return x_hat, P_hat
    
    
    def filter(self,
        x0: np.ndarray,
        P_prd: np.ndarray,
        #h_m: float,
        h: float,
        us: np.ndarray,
        ys: np.ndarray,):
        """_summary_

        Args:
        x0 : np.ndarray
            initial state [yaw, yaw rate]
        P_prd : np.ndarray
            2x2 array: initial covariance matrix
        h_m : float
            time step measurement [s]
        h : float
            time step filter [s]
        us : np.ndarray
            1D array: inputs
        ys : np.ndarray
            1D array: measured yaw
        """
        
        assert ys.ndim==2
        
        N = ys.shape[1]
        n_states = len(x0)
        
        if len(us)!=N:
            us = np.tile(us,[N,1])
        
        # Initialize:
        x_prd=np.zeros((n_states,N))
        x_prd[:,0] = x0
        
        x_hat=np.zeros((n_states,N))
        
        P_hat = P_prd 
        
        for i in range(N-1):
            u = us[i]
            
            x_hat[:,i], P_hat = self.update(y=ys[:,i], P_prd=P_hat, x_prd=x_prd[:,i])
            x_prd[:,i+1],_ = self.predict(x_hat=x_hat[:,i], P_hat=P_hat, u=u, h=h)
        
        i+=1
        x_hat[:,i], P_hat = self.update(y=ys[:,i], P_prd=P_hat, x_prd=x_prd[:,i])
        
        return x_hat
        
    
    def simulate(self, x0: np.ndarray, t:np.ndarray, us: np.ndarray):
        
        N = len(t)
        
        P_hat = np.eye(len(x0))        

        if len(us)!=N:
            us = np.tile(us,[N,1])
        
        x_hat=np.zeros((len(x0),N))
        x_hat[:,0] = x0
        
        for i,t_ in enumerate(t[0:-1]):
            u = us[i]
            h = t[i+1]-t[i]
            x_hat[:,i+1],_ = self.predict(x_hat=x_hat[:,i], P_hat=P_hat, u=u, h=h)
            
        return x_hat
            
            
        