import numpy as np
import cv2
import os
import sys

def read_frames(input_path):
    """
    Reading frames in sorted order
    """
    frames = []
    frame_num = []
    for f in os.listdir(input_path):
        s = f.replace("."," ")
        l = s.split(" ")
        frame_num.append((int(l[1]), f))
        
    frame_num.sort()
    for _,f in frame_num:
        frames.append(cv2.imread(input_path + f))
        
    return frames

def hessian_corner_detection(input_image, index):
    """
    Corner Detection using Hessian of gradients
    """
    
    sigma = 1.0
    alpha = 0.05
    d = 11
    
    image = cv2.cvtColor(input_image,cv2.COLOR_BGR2GRAY)
#     input_image = np.float32(input_image)
    image_shape = image.shape
    ksize = (0,0)
    blurred = cv2.GaussianBlur(image,ksize,sigma)
#     print((blurred[:15,:15]))
    
    Mxy = [[1,0,-1],[0,0,0],[-1,0,1]]
    kxy = (0.25)*np.array(Mxy)
    Ixy = cv2.filter2D(src=blurred, ddepth=cv2.CV_32F, kernel=kxy)
    Sxy = cv2.GaussianBlur(Ixy,(0,0),sigma)
    
    Myy = [[0,1,0],[0,-2,0],[0,1,0]]
    kyy = (0.5)*np.array(Myy)
    Iyy = cv2.filter2D(src=blurred, ddepth=cv2.CV_32F, kernel=kyy)
    Syy = cv2.GaussianBlur(Iyy,(0,0),sigma)
    
    Mxx = [[0,0,0],[1,-2,1],[0,0,0]]
    kxx = (0.5)*np.array(Mxx)
    Ixx = cv2.filter2D(src=blurred, ddepth=cv2.CV_32F, kernel=kxx)
    Sxx = cv2.GaussianBlur(Ixx,(0,0),sigma)

#     print(Ixx.dtype)
#     print(Sxx.dtype)
    
    det = Sxx * Syy - Sxy * Sxy
    tr = Sxx + Syy
    tr2 = tr * tr
    res = det - alpha*tr2
    
    corners = []
    th = 0.01*np.amax(res)
    
#     marked_image = np.copy(input_image)
   
    for j in range(d,image_shape[1]-d):
        for i in range(d,image_shape[0]-d):
            region = res[i-d:i+d,j-d:j+d]
            if res[i,j] > th:
                x1 = np.amax(region)
                if x1 == res[i,j]:
                    corners.append((i,j))
#                     marked_image[i,j] = np.array([0,0,255])

    print("Number of corners:", len(corners))
#     cv2.imwrite("./dataset2/marked_5_{}.png".format(index),marked_image)

    return corners

def proximity_matching(u, v, corners, frames):
    """
    Matching corner points using proximity and a simple sum-of-squared-difference approach
    """
    corner_pairs = []
    for i in range(len(corners[u])):
        min_dst = np.inf
        min_idx = None
        for j in range(len(corners[v])):
            dst = (corners[u][i][0] - corners[v][j][0])**2 + (corners[u][i][1] - corners[v][j][1])**2
            if dst < min_dst:
                min_dst = dst
                min_idx = j
        corner_pairs.append((i,min_idx))
        
#     print(len(corner_pairs))

    ssd_th = 250
    refined_pairs = []
    for p in corner_pairs:
        x1, y1 = corners[u][p[0]][0], corners[u][p[0]][1]
        x2, y2 = corners[v][p[1]][0], corners[v][p[1]][1]
        patch1 = frames[u][max(0, int(x1)-2):int(x1)+2, max(0, int(y1)-2):int(y1)+2, :]
        patch2 = frames[v][max(0, int(x2)-2):int(x2)+2, max(0, int(y2)-2):int(y2)+2, :]
        ssd = np.sum((patch1 - patch2) ** 2)
        if ssd < ssd_th:
            refined_pairs.append(p)
    
    print("Number of matches b/w frame {} and frame {}:".format(u,v), len(refined_pairs))
#     print([(corners1[p[0]][0], corners1[p[0]][1], corners2[p[1]][0], corners2[p[1]][1]) for p in refined_pairs])
    
    return refined_pairs
        

def get_dimensions(frames, affine_matrices):
    """
    Calculating projected corners for all frames in the first frame and calculating corners based on them
    """
    all_corners = []
    corners = None
    for i in range(0, len(frames)):
        h = frames[i].shape[0]
        w = frames[i].shape[1]
        corners = np.array([[0  ,0  ,1],
                            [w-1,0  ,1],
                            [w-1,h-1,1],
                            [0  ,h-1,1]])
        
        new_corners = np.rint(np.matmul(affine_matrices[i],corners.T).T)

        new_left = np.int32(np.amin(new_corners[:,:1]))
        new_right = np.int32(np.amax(new_corners[:,:1]))
        new_top = np.int32(np.amin(new_corners[:,1:]))
        new_bottom = np.int32(np.amax(new_corners[:,1:]))
        all_corners.append([new_left, new_right, new_top, new_bottom])
        
    all_corners = np.array(all_corners).astype(np.int32)
    final_corners = (np.amin(all_corners[:,0]), np.amax(all_corners[:,1]),np.amin(all_corners[:,2]),np.amax(all_corners[:,3]))
#     print(final_corners)
    H = final_corners[3] - final_corners[2] + 1
    W = final_corners[1] - final_corners[0] + 1
    
    return H,W
    
    

def estimate_affine_matrix(u, v, corners, frames):
    """
    Estimating the Affine matrix of two frames using Projective Geometry and Least Square method.
    """
    matches = proximity_matching(u,v,corners,frames)
    
    src_pts = np.array([ list(corners[u][m[0]])[::-1] for m in matches], dtype=np.float32)
    dst_pts = np.array([ list(corners[v][m[1]])[::-1] for m in matches], dtype=np.float32)
    
    src_pts_hom = np.concatenate([src_pts, np.ones((src_pts.shape[0], 1))], axis=1)
    dst_pts_hom = np.concatenate([dst_pts, np.ones((dst_pts.shape[0], 1))], axis=1)

    A = np.linalg.lstsq(src_pts_hom, dst_pts_hom, rcond=None)[0]
    A = A.T
    A[2] = np.array([0,0,1])
#     print("Affine Transformation b/w frame {} and frame {}:".format(u,v))
#     print(A)
    return A

def remove_borders(image):
    """
    Removing black borders of the generated panorama.
    """
    h, w = image.shape[:2]
    lh, lw = 0, 0
    rh, rw = h, w
    
    is_color = False 
    for j in range(w):
        if np.amax(image[0, j]) > 0:
            is_color = True
            
    if not is_color:
        for i in range(0,h):
            is_color = False 
            for j in range(w):
                if np.amax(image[i, j]) > 0:
                    is_color = True
                    break
            if is_color == False:
                lh += 1
    
    is_color = False 
    for j in range(w):
        if np.amax(image[h-1, j]) > 0:
            is_color = True
    
    if not is_color:
        for i in range(h-1,lh-1,-1):
            is_color = False 
            for j in range(w):
                if np.amax(image[i, j]) > 0:
                    is_color = True
                    break
            if is_color == False:
                rh -= 1
    
    is_color = False
    for j in range(lh,rh):
            if np.amax(image[j, 0]) > 0:
                is_color = True
    
    if not is_color:            
        for i in range(0,w):
            is_color = False 
            for j in range(lh,rh):
                if np.amax(image[j, i]) > 0:
                    is_color = True
                    break
            if is_color == False:
                lw += 1
    
    is_color = False
    for j in range(lh,rh):
            if np.amax(image[j, w-1]) > 0:
                is_color = True
    
    if not is_color:
        for i in range(w-1,lw-1,-1):
            is_color = False 
            for j in range(lh,rh):
                if np.amax(image[j, i]) > 0:
                    is_color = True
                    break
            if is_color == False:
                rw -= 1
            
    print(lh,rh,lw,rw)
    return image[lh:rh, lw:rw]
          
if __name__ == "__main__":
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    dnum = sys.argv[3]
    
    frames = read_frames(input_path)   
    corners = [hessian_corner_detection(frames[i], i) for i in range(len(frames))]
    
    num_frames = len(frames)
    ref_frame = 0
    
    affine_matrices = [None]*num_frames
    affine_matrices[ref_frame] = np.eye(3)
        
    for i in range(ref_frame+1, num_frames):
        affine_matrices[i] = np.matmul(affine_matrices[i-1], estimate_affine_matrix(i,i-1,corners,frames))
#         print("Affine Matrix for frame "+ str(i), affine_matrices[i])
        
    H, W = get_dimensions(frames, affine_matrices)
    print(H,W)
    merged_image = np.zeros((H,W,3)).astype(int)
    warps = [ cv2.warpAffine(frames[i], affine_matrices[i][0:2], (W,H)) for i in range(num_frames)]
    
    for w in warps:
        merged_image = np.bitwise_or(merged_image, w.astype(int))
        
    merged_image = remove_borders(merged_image)
    print("Panorama image shape: ", merged_image.shape)
    
    cv2.imwrite(os.path.join(output_path,"pano_"+ dnum +".png"), merged_image)
    