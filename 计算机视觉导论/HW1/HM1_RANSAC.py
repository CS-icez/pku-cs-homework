import numpy as np
from utils import draw_save_plane_with_points, normalize


if __name__ == "__main__":


    np.random.seed(0)
    # load data, total 130 points inlcuding 100 inliers and 30 outliers
    # to simplify this problem, we provide the number of inliers and outliers here

    noise_points = np.loadtxt("HM1_ransac_points.txt")


    #RANSAC
    # Please formulate the palnace function as:  A*x+B*y+C*z+D=0     

    INLIER = 100
    OUTLIER = 30
    TOTAL = INLIER + OUTLIER
    PROB = 0.001
    SAMPLE_SIZE = 3
    fail_p = 1 - ((INLIER - SAMPLE_SIZE) / TOTAL) ** SAMPLE_SIZE
    t = np.log(PROB) / np.log(fail_p)
    t = int(t) + 1

    sample_time = t # the minimal time that can guarantee the probability of at least one hypothesis does not contain any outliers is larger than 99.9%
    distance_threshold = 0.05

    # sample points group
    indices = np.random.choice(TOTAL, (sample_time, SAMPLE_SIZE), replace=False) # (N, 3) (denote N as `sample_time`)
    samples = noise_points[indices] # (N, 3, 3)

    # estimate the plane with sampled points group
    p0 = samples[:, 0] # (N, 3)
    p1 = samples[:, 1] # (N, 3)
    p2 = samples[:, 2] # (N, 3)

    v1 = p1 - p0 # (N, 3)
    v2 = p2 - p0 # (N, 3)
    normal = np.cross(v1, v2) # (N, 3)
    D = -np.sum(normal * p0, axis=1, keepdims=True) # (N, 1)
    plane = np.hstack((normal, D)) # (N, 4)

    norm = np.linalg.norm(normal, axis=1, keepdims=True) # (N, 1)
    plane = plane / norm # (N, 4)

    #evaluate inliers (with point-to-plance distance < distance_threshold)
    point = np.vstack((noise_points.T, np.ones((1, TOTAL)))) # (4, TOTAL)
    distance = np.abs(plane @ point) # (N, TOTAL)
    inlier_cnt = np.sum(distance < distance_threshold, axis=1) # (N,)

    best_sample = np.argmax(inlier_cnt) # int
    best_plane = plane[best_sample] # (4,)
    inlier_idx = distance[best_sample] < distance_threshold # (TOTAL,)
    inlier = noise_points[inlier_idx] # (M, 3) (denote M as inlier number)

    # minimize the sum of squared perpendicular distances of all inliers with least-squared method 
    A = np.hstack((inlier, np.ones((inlier.shape[0], 1)))) # (M, 4)
    U, S, Vt = np.linalg.svd(A)
    pf = Vt[-1]

    # draw the estimated plane with points and save the results 
    # check the utils.py for more details
    # pf: [A,B,C,D] contains the parameters of palnace function  A*x+B*y+C*z+D=0  
    pf = normalize(pf)
    draw_save_plane_with_points(pf, noise_points,"result/HM1_RANSAC_fig.png") 
    np.savetxt("result/HM1_RANSAC_plane.txt", pf)
    np.savetxt('result/HM1_RANSAC_sample_time.txt', np.array([sample_time]))
