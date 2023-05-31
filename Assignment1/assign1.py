import numpy as np
import cv2
import os
from scipy.stats import multivariate_normal as mv_norm


class BackgroundSubtractor():
    def __init__(self,num_frames,input_dir,output_dir,N=10,K=4,T=0.7):
        self.num_frames = num_frames
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.img_shp_x = None
        self.img_shp_y = None
        self.img_shp = None
        self.K = K
        self.T = T
        self.N = N
        self.mu = None
        self.sigma = None
        self.B = None
        self.w = None
    
    """
    This is the main function used to run through all the frames for all pixels
    """
    def run(self):
        
        init_weight = [0.01, 0.01, 0.01, 0.01]
        init_sigma = 225
        init_u = np.zeros(3)
        
        if self.input_dir == r'./Candela_m1.10/input/':
            file_list = [os.path.join(self.input_dir, "Candela_m1.10_%06d" % i + ".png") 
                     for i in range(self.num_frames)]
        else:
            file_list = [os.path.join(self.input_dir, "in%06d" % i + ".png") 
                     for i in range(self.num_frames)]
        
        init_img = cv2.imread(file_list[0])
        self.img_shp_x = init_img.shape[0]
        self.img_shp_y = init_img.shape[1]
        self.img_shp = init_img.shape
        
        self.B = np.ones((self.img_shp_x, self.img_shp_y), dtype=int)
        self.sigma = np.array([[[init_sigma for k in range(self.K)] 
                            for j in range(self.img_shp_y)] 
                            for i in range(self.img_shp_x)])
        self.w = np.array([[init_weight for j in range(self.img_shp_y)] 
                           for i in range(self.img_shp_x)])
        self.mu = np.array([[[init_u for k in range(self.K)] 
                             for j in range(self.img_shp_y)]
                             for i in range(self.img_shp_x)])
        
        index = 0
        for f in file_list:
            curr = cv2.imread(f)
            for j in range(self.img_shp_y):
                for i in range(self.img_shp_x):
                    found = -1
                    
                    for k in range(self.K):
                        if self.inGaussian(self.mu[i,j,k], self.sigma[i,j,k], curr[i,j]):
                            found = k
                            break
                            
                    if found == -1:
                        # replace Gaussian with the least weight 
                        wl = np.array([self.w[i,j,k] for k in range(self.K)])
                        idx = np.argmin(wl)
                        self.mu[i,j,idx] = np.array(curr[i,j]).reshape(1, 3)
                        self.sigma[i,j,idx] = np.array(init_sigma)
                        
                    else:
                        # update Gaussians
                        prev_mu = self.mu[i,j,k]
                        prev_sigma = self.sigma[i,j,k]
                        X = curr[i,j].astype(np.float64)

                        self.w[i,j] = (1 - 1/self.N) * self.w[i,j]
                        self.w[i,j,found] += (1/self.N)
                        self.mu[i,j] = (1 - 1/self.N) * self.mu[i,j]
                        self.mu[i,j,found] += (1/self.N)*(curr[i,j]/self.w[i,j,found])
                        delt = X - prev_mu
                        self.sigma[i,j] = (1 - 1/self.N) * self.sigma[i,j]
                        self.sigma[i,j,found] += (1/self.N)*(np.matmul(delt,delt.T)/self.w[i,j,found])
                  
            self.calcBackground(self.T)
            self.saveFrame(curr,index)
            index += 1
            print('Done: {}'.format(f))
                    
        
        
    """
    This function reorders the Gaussians according to w/sigma ratio 
    and calculates B for each pixel which indicates no of Gaussian representing background
    """
    def calcBackground(self,T):
        for i in range(self.img_shp_x):
            for j in range(self.img_shp_y):
                kw = self.w[i,j]
                kn = []
                for k in range(self.K):
                    kn.append(np.linalg.norm(np.sqrt(self.sigma[i,j,k])))
                kn = np.array(kn)
                order = np.argsort(-kw/kn)
                
                self.mu[i,j] = self.mu[i,j,order]
                self.sigma[i,j] = self.sigma[i,j,order]
                self.w[i,j] = self.w[i,j,order]
                self.w[i,j] /= np.linalg.norm(self.w[i,j],1)
                
                wsum = 0
                for k in range(len(order)):
                    wsum += self.w[i,j,k]
                    if wsum > T:
                        self.B[i,j] = k+1
                        break
               
    """
    This function checks whether pixel x is in a Gaussian distribution with params mu,sigma
    """
    def inGaussian(self,mu,sigma,x):
        return np.linalg.norm(mu-x,2)<=2.5*(sigma)**0.5
    
    """
    Helper function to save foreground mask
    """
    def saveFrame(self,img,index):
        fgmask = np.array(img)
        for j in range(self.img_shp_y):
            for i in range(self.img_shp_x):
                fgmask[i,j] = [255,255,255]
                for k in range(self.B[i,j]):
                    if self.inGaussian(self.mu[i,j,k], self.sigma[i,j,k], img[i,j]):
                        fgmask[i,j] = [0,0,0]
                        break
                        
        cv2.imwrite(self.output_dir+'out%05d' % index +'.png', fgmask)
    
    
if __name__ == "__main__":
    input_dir = r'./Candela_m1.10/input/'
    output_dir = r'./Candela_m1.10/output_new/'
    num_frames = 350
    
    try:
        os.mkdir(output_dir)
    except:
        pass
    
    
    bs = BackgroundSubtractor(num_frames,input_dir,output_dir)
    bs.run()
    
    try:
        cmd = "ffmpeg -framerate 10 -pattern_type glob -i '" + output_dir + "/*.png' -c:v libx264 -pix_fmt yuv420p " + output_dir + "/out.avi"
        os.system(cmd)
    except:
        pass
    
    #############################
    # input_dir = r'./CAVIAR1/input/'
    # output_dir = r'./CAVIAR1/output/'
    # num_frames = 610
    
    # try:
    #     os.mkdir(output_dir)
    # except:
    #     pass
    
    # bs = BackgroundSubtractor(num_frames,input_dir,output_dir)
    # bs.run()

    # try:
    #     cmd = "ffmpeg -framerate 10 -pattern_type glob -i '" + output_dir + "/*.png' -c:v libx264 -pix_fmt yuv420p " + output_dir + "/out.avi"
    #     os.system(cmd)
    # except:
    #     pass
    
    # #############################
    # input_dir = r'./HallAndMonitor/input/'
    # output_dir = r'./HallAndMonitor/output/'
    # num_frames = 296
    
    # try:
    #     os.mkdir(output_dir)
    # except:
    #     pass
    
    # bs = BackgroundSubtractor(num_frames,input_dir,output_dir)
    # bs.run()

    # try:
    #     cmd = "ffmpeg -framerate 10 -pattern_type glob -i '" + output_dir + "/*.png' -c:v libx264 -pix_fmt yuv420p " + output_dir + "/out.avi"
    #     os.system(cmd)
    # except:
    #     pass
    
    # #############################
    # input_dir = r'./IBMtest2/input/'
    # output_dir = r'./IBMtest2/output/'
    # num_frames = 90
    
    # try:
    #     os.mkdir(output_dir)
    # except:
    #     pass
   
    # bs = BackgroundSubtractor(num_frames,input_dir,output_dir)
    # bs.run()

    # try:
    #     cmd = "ffmpeg -framerate 10 -pattern_type glob -i '" + output_dir + "/*.png' -c:v libx264 -pix_fmt yuv420p " + output_dir + "/out.avi"
    #     os.system(cmd)
    # except:
    #     pass
    
    # #############################
    # input_dir = r'./HighwayI/input/'
    # output_dir = r'./HighwayI/output/'
    # num_frames = 440
    
    # try:
    #     os.mkdir(output_dir)
    # except:
    #     pass
    
    # bs = BackgroundSubtractor(num_frames,input_dir,output_dir)
    # bs.run()

    # try:
    #     cmd = "ffmpeg -framerate 10 -pattern_type glob -i '" + output_dir + "/*.png' -c:v libx264 -pix_fmt yuv420p " + output_dir + "/out.avi"
    #     os.system(cmd)
    # except:
    #     pass