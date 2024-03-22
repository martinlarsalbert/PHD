import numpy as np
import pandas as pd
from numpy.linalg.linalg import inv, pinv

class KalmanFilter:

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        H: np.ndarray,
        E: np.ndarray,
        Q: float,
        R: float,
    ) -> pd.DataFrame:
        """Example kalman filter for yaw and yaw rate
        Parameters
        ----------
        A : np.ndarray
        B : np.ndarray
        H : np.ndarray
            observation model
        Ed : np.ndarray
            2x1 array
        Q : float
            process noise
        R : float
            measurement noise
        Returns
        -------
        pd.DataFrame
            data frame with filtered data
        """
        self.A=A
        self.B=B
        self.H=H
        self.E=E
        self.Q=Q
        self.R=R
        

    def predict(self, x_hat, P_hat, u, h):
        
        A = self.A
        B = self.B
        E = self.E
        Q = self.Q
        Delta = B*h
        Ed = E*h
        Qd = Q*h
        
        n_states = len(x_hat)
        self.Phi = Phi = np.eye(n_states) + A*h
        #Phi = A
        
        
        # Predictor (k+1)
        x_prd = Phi @ x_hat + Delta @ u
        #P_prd = A @ P_hat @ A.T + Ed * Qd @ Ed.T
        #P_prd = Phi @ P_hat @ Phi.T + Qd
        P_prd = Phi @ P_hat @ Phi.T + Ed * Qd @ Ed.T
        
        return x_prd, P_prd
    
    def update(self, y, P_prd, x_prd, h):
            
        H = self.H
        R = self.R
        Rd = R*h
        n_states = len(x_prd)
        
        epsilon = y - H @ x_prd  # Error between meassurement (y) and predicted measurement H @ x_prd
        
        # Compute kalman gain:
        S = H @ P_prd @ H.T + Rd  # System uncertainty
        K = P_prd @ H.T @ inv(S)

        # State corrector:
        x_hat = x_prd + K @ epsilon
        
        # corrector
        IKC = np.eye(n_states) - K @ H        
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
            
            x_hat[:,i], P_hat = self.update(y=ys[:,i], P_prd=P_hat, x_prd=x_prd[:,i], h=h)
            x_prd[:,i+1],_ = self.predict(x_hat=x_hat[:,i], P_hat=P_hat, u=u, h=h)
        
        i+=1
        x_hat[:,i], P_hat = self.update(y=ys[:,i], P_prd=P_hat, x_prd=x_prd[:,i],h=h)
        
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
            
            
        