import cv2
import numpy as np
import os
import sys

input_path = sys.argv[1]
output_path = sys.argv[2]


def intersection_of_points(a,b,c,d):
    diff_x_1 = a[0]-b[0]
    diff_y_1 = a[1]-b[1]

    diff_x_2 = c[0]-d[0]
    diff_y_2 = c[1]-d[1]

    z = diff_x_1*diff_y_2-diff_x_2*diff_y_1
    
    det_1 = a[0]*b[1]-a[1]*b[0]
    det_2 = c[0]*d[1]-c[1]*d[0]

    x_int = (det_1*diff_x_2 - det_2*diff_x_1)/z
    y_int = (det_1*diff_y_2 - det_2*diff_y_1)/z

    return np.array([x_int,y_int])

def read_frames(input_path):
    """
    Reading frames in sorted order
    """
    frames = []
    frame_num = []
    for f in os.listdir(input_path):
        s = f.replace("."," ")
        l = s.split(" ")
        frame_num.append((int(l[0]), f))
    
    frame_num.sort()
#     print(frame_num)
    for _,f in frame_num:
        frames.append(cv2.imread(input_path + f))
        
    return frames

def camera_calibration(input_path, output_path):
    
    if(not os.path.exists(output_path)):
        os.makedirs(output_path)
    
    frames = read_frames(input_path)
    num_pts = 0
    
    corners = []
    ideal_pts = []
    
    for idx, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, crs = cv2.findChessboardCorners(gray, (7,10), None)
        
        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            crs = cv2.cornerSubPix(gray,crs,(11,11),(-1,-1),criteria)
            frame = cv2.drawChessboardCorners(frame, (7,10), crs,ret)
            cv2.imwrite(os.path.join(output_path,"corners_"+str(idx)+".png"),frame)
            crs = crs.reshape(70,2)
            corners.append(crs)
            ip = (intersection_of_points(crs[0],crs[6],crs[63],crs[69]) , intersection_of_points(crs[0],crs[69],crs[6],crs[63]))
            ideal_pts.append(ip)
            num_pts += 1
            
        else:
            print("Unable to detect corners in Frame " + str(idx))
        
    
    param_mat = None
    B = np.zeros((3,3))
    Z = np.zeros((len(ideal_pts), 5), dtype = np.float64)
    i = 0
    for (d1,d2) in ideal_pts:
        (y1,y2) = d2
        (x3,y3) = (1,1)
        (x1,x2) = d1
        tmp = [x1*y1, x1*y3+x3*y1, x2*y2, x2*y3 + x3*y2, x3*y3]
        Z[i] = np.array(tmp , dtype = np.float64)
        i += 1
        
    _, sigma, Vt = np.linalg.svd(Z)
    J = Vt[np.argmin(sigma), :]
    
    
    B[2,2] = J[4]
    B[1,1] = J[2]
    B[0,0] = J[0]
    B[2,0] = J[1]
    B[2,1] = J[3]
    
    B[0,2] = B[2,0]
    B[1,2] = B[2,1]
    
    try:
        _ = np.linalg.cholesky(B)
        param_mat = B

    except np.linalg.LinAlgError as _:
        try:
            B = -1*B
            _ = np.linalg.cholesky(B)
            param_mat = B
            
        except:
            print("The camera cannot be calibrated as both B and -B are not PSD")
            param_mat = None
            return None
            
    if num_pts < 2:
        print("Insufficient number of frames for calibration")
        return None
    
    else:
        """
        Extracting K from B according to Zhang's Paper
        """
        
        B = param_mat
        K = np.zeros((3,3))
        K[2,2] = 1

        uy_dnm = (B[0,0]*B[1,1] - B[0,1]*B[0,1])
        uy_num = (B[0,1]*B[0,2] - B[0,0]*B[1,2])
        uy=  uy_num / uy_dnm 

        lamb_t = (B[0,2]*B[0,2] + uy*(B[0,1]*B[0,2] -B[0,0]*B[1,2])) / B[0,0]
        lamb = B[2,2] - lamb_t

        fx2 = (lamb / B[0,0])
        fx = np.sqrt(fx2)

        fy2 = (lamb*B[0,0] / (B[0,0]*B[1,1] - B[0,1]*B[0,1]))
        fy = np.sqrt(fy2)

        s = (-B[0,1]*fy*fx*fx) / lamb

        ux_t = (B[0,2]*fx*fx) / lamb
        ux = ((s*uy) / (fy)) - ux_t

        K[0,0] = fx
        K[1,1] = fy
        K[0,1] = s
        K[0,2] = ux
        K[1,2] = uy

        return K


def render(K, input_path,output_path):
    if(not os.path.exists(output_path)):
        os.makedirs(output_path)
    
    frames = []
    chess_pts = np.array([[i,j] for j in range(10) for i in range(7)])
    
    frames = read_frames(input_path)
    
    corners = []
    axis_pts = np.array([[3,0,0],[0,3,0],[0,0,-3]], dtype = np.float64)
    cube_pts = np.float32([[0,0,0], [0,2,0], [3,2,0], [3,0,0], [0,0,-2],[0,2,-2],[3,2,-2],[3,0,-2]])
    
    for idx, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, crs = cv2.findChessboardCorners(gray, (7,10), None)
        
        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            crs = cv2.cornerSubPix(gray,crs,(11,11),(-1,-1),criteria)
            frame = cv2.drawChessboardCorners(frame, (7,10), crs,ret)
            crs = crs.reshape(70,2)
            corners.append(crs)
            
            """
            Homography using DLT
            """
            
            A = [] 
            for i in range(0, len(crs)):
                (u,v) = crs[i]
                (x,y) = chess_pts[i]
                A.append(np.array([0,0,0,-x,-y,-1,x*v,y*v,v]))
                A.append(np.array([-x,-y,-1,0,0,0,x*u,y*u,u]))
                
            A = np.array(A)
            _, sig, Vt = np.linalg.svd(A)
            H = Vt[np.argmin(sig), :].reshape((3,3))
            R = np.zeros((3,3))
            t = np.zeros((3,1))
            V = np.matmul(np.linalg.inv(K), H)
            Q = V.view()
            
            r1, r2, t = V[:,0], V[:,1], V[:,2]
            l = (np.linalg.norm(r1) + np.linalg.norm(r2))/2
            r1, r2, t = r1/l , r2/l, t/l
            r3 = np.cross(r1,r2)
            Q[:,2] = r3
            
            u,_,Vt = np.linalg.svd(Q)
            R = np.array(np.matmul(u,Vt))
            t = np.reshape(t,(3,1))
#             print(R,t)
            Rt = np.concatenate((R,t), axis=1)
            cam_mat = np.matmul(K,Rt)
            
            """
            Projecting cube points using camera matrix
            """
            
            axis_dst_pts = []
            cube_dst_pts = []
            con = np.array([1.0])
            
            for i in range(0, len(axis_pts)):
                e = axis_pts[i]
                e = np.concatenate((e, con))
                V = np.array(np.matmul(cam_mat, e))
                V = V/V[2]
                axis_dst_pts.append(V[0:2])
                
            axis_dst_pts = np.array(axis_dst_pts)
            
#             draw_cube(frame,crs,axis_dst_pts,idx)
            
            for i in range(0, len(cube_pts)):
                e = cube_pts[i]
                e = np.concatenate((e, con))
                V = np.array(np.matmul(cam_mat, e))
                V = V/V[2]
                cube_dst_pts.append(V[0:2])
                
            cube_dst_pts = np.array(cube_dst_pts)
            
            draw_cube(frame,crs,axis_dst_pts,cube_dst_pts,idx)
            
        else:
            print("Unable to detect corners in Frame " + str(idx))
    
    
def draw_cube(frame,corners,apts,cpts,idx):
    
    corners = corners.astype(np.int32)
    
    apts = apts.astype(np.int32)
    corner = tuple(corners[0])

    cpts = np.int32(cpts).reshape(-1,2)
    shapes = np.zeros_like(frame, np.uint8)

    cv2.rectangle(shapes, tuple(cpts[0]), tuple(cpts[2]), (255, 5, 25), cv2.FILLED)
    
    frame = cv2.line(frame, tuple(cpts[0]), tuple(cpts[1]), (255, 5, 25), 2)
    frame = cv2.line(frame, tuple(cpts[1]), tuple(cpts[2]), (255, 5, 25), 2)
    frame = cv2.line(frame, tuple(cpts[2]), tuple(cpts[3]), (255, 5, 25), 2)
    frame = cv2.line(frame, tuple(cpts[3]), tuple(cpts[0]), (255, 5, 25), 2)

    for i,j in zip(range(4),range(4,8)):
        frame = cv2.line(frame, tuple(cpts[i]), tuple(cpts[j]), (255, 5, 25), 2)
        
    cv2.rectangle(shapes, tuple(cpts[4]), tuple(cpts[6]), (255, 5, 25), cv2.FILLED)
    frame = cv2.line(frame, tuple(cpts[4]), tuple(cpts[5]), (255, 5, 25), 2)
    frame = cv2.line(frame, tuple(cpts[5]), tuple(cpts[6]), (255, 5, 25), 2)
    frame = cv2.line(frame, tuple(cpts[6]), tuple(cpts[7]), (255, 5, 25), 2)
    frame = cv2.line(frame, tuple(cpts[7]), tuple(cpts[4]), (255, 5, 25), 2)

    out = frame.copy()
    alpha = 0.6
    mask = shapes.astype(bool)
    out[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]
    
    out = cv2.line(out, corner, tuple(apts[2]), (255,255,0), 4)
    out = cv2.line(out, corner, tuple(apts[1]), (0,255,255), 4)
    out = cv2.line(out, corner, tuple(apts[0]), (255,0,255), 4)

    
    cv2.imwrite(os.path.join(output_path,"cube_"+str(idx)+".png"),out)
    
    
    
if __name__ == "__main__":
    K = camera_calibration(input_path, output_path)
    print(K)
    render(K, input_path, output_path)