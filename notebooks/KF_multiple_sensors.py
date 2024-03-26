import numpy as np
import pandas as pd
from numpy.linalg.linalg import inv, pinv

from dataclasses import dataclass


@dataclass
class FilterResult:
    x_prd : np.ndarray
    x_hat : np.ndarray
    K : np.ndarray
    epsilon: np.ndarray

class KalmanFilter:

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        H: np.ndarray,
        Q: float,
        R: float,
        E: np.ndarray=None,
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
        self.Q=Q
        if E is None:
            self.E = np.eye(len(A))  # The entire Q is used
        else:
            self.E=E
                    
        self.R=R
        

    def predict(self, x_hat, P_hat, u, h):
        
        A = self.A
        B = self.B
        E = self.E
        Q = self.Q
        self.Delta = Delta =  B*h
        self.Gamma = Gamma = E*h
          
        n_states = len(x_hat)
        n_inputs = len(u)
        self.Phi = Phi = np.eye(n_states) + A*h
        #Phi = A
        
        
        # Predictor (k+1)
        # State estimate propagation:
        x_prd = Phi @ x_hat
        if n_inputs>0:
            # Add inputs if they exist:
            x_prd+=Delta @ u
            
        # Error covariance propagation:
        #P_prd = Phi @ P_hat @ Phi.T + Gamma * Q @ Gamma.T ## Note Q not Qd!
        Qd = Q*h
        P_prd = Phi @ P_hat @ Phi.T + Qd
        
        return x_prd, P_prd
    
    def update(self, y, P_prd, x_prd, h):
            
        H = self.H
        R = self.R
        Rd = R*h
        n_states = len(x_prd)
        
        epsilon = y - H @ x_prd  # Error between meassurement (y) and predicted measurement H @ x_prd
        
        # Compute kalman gain matrix:
        S = H @ P_prd @ H.T + Rd  # System uncertainty
        K = P_prd @ H.T @ inv(S)

        # State estimate update:
        x_hat = x_prd + K @ epsilon
        
        # Error covariance update:
        IKC = np.eye(n_states) - K @ H        
        P_hat = IKC * P_prd @ IKC.T + K * Rd @ K.T
        
        return x_hat, P_hat, K, epsilon.flatten()
    
    
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
        n_measurement_states = ys.shape[0]
        n_states = len(x0)
        
        if len(us)!=N:
            us = np.tile(us,[N,1])
        
        # Initialize:
        x_prds=np.zeros((n_states,N))
        x_prd = x0
        x_prds[:,0] = x_prd.flatten()
        
        x_hats=np.zeros((n_states,N))
        Ks=np.zeros((N,n_states,n_measurement_states))
        epsilon=np.zeros((n_measurement_states,N))
        
        #P_hat = P_prd 
        
        for i in range(N-1):
            u = us[i]
            
            x_hat, P_hat, K, epsilon[:,i] = self.update(y=ys[:,[i]], P_prd=P_prd, x_prd=x_prd, h=h)
            x_hats[:,i] = x_hat.flatten()
            Ks[i,:,:] = K          
            
            x_prd,P_prd = self.predict(x_hat=x_hat, P_hat=P_hat, u=u, h=h)
            x_prds[:,i+1] = x_prd.flatten()
        
        i+=1
        x_hat, P_hat, K, epsilon[:,i] = self.update(y=ys[:,[i]], P_prd=P_prd, x_prd=x_prd,h=h)
        x_hats[:,i] = x_hat.flatten()
        Ks[i,:,:] = K
        
        result = FilterResult(x_prd=x_prds, x_hat=x_hats, K=Ks, epsilon=epsilon)
        #result['x_prd'] = x_prd
        #result['x_hat'] = x_hat
        #result['K'] = K
        
        return result
        
    
    def simulate(self, x0: np.ndarray, t:np.ndarray, us: np.ndarray):
        
        N = len(t)
        
        P_hat = np.eye(len(x0))        

        if len(us)!=N:
            us = np.tile(us,[N,1])
        
        x_hat=np.zeros((len(x0),N))
        x_hat[:,0] = x0.flatten()
        
        for i,t_ in enumerate(t[0:-1]):
            u = us[i]
            h = t[i+1]-t[i]
            x_hat[:,i+1],_ = self.predict(x_hat=x_hat[:,i], P_hat=P_hat, u=u, h=h)
            
        return x_hat
            
            
        