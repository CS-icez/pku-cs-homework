import numpy as np
from utils import read_img, write_img

def padding(img, padding_size, type):
    """
        The function you need to implement for Q1 a).
        Inputs:
            img: array(float)
            padding_size: int
            type: str, zeroPadding/replicatePadding
        Outputs:
            padding_img: array(float)
    """

    h, w = img.shape
    p = padding_size

    if type=="zeroPadding":
        padding_img = np.zeros((h+2*p, w+2*p))
        padding_img[p:p+h, p:p+w] = img # Center.
        return padding_img
    elif type=="replicatePadding":
        padding_img = np.zeros((h+2*p, w+2*p))
        padding_img[p:p+h, p:p+w] = img # Center.
        padding_img[:p, p:p+w] = img[:1, :] # Top.
        padding_img[p+h:, p:p+w] = img[-1:, :] # Bottom.
        padding_img[p:p+h, :p] = img[:, :1] # Left.
        padding_img[p:p+h, p+w:] = img[:, -1:] # Right.
        padding_img[:p, :p] = img[0, 0] # Top left.
        padding_img[:p, p+w:] = img[0, -1] # Top right.
        padding_img[p+h:, :p] = img[-1, 0] # Bottom left.
        padding_img[p+h:, p+w:] = img[-1, -1] # Bottom right.
        return padding_img


def convol_with_Toeplitz_matrix(img, kernel):
    """
        The function you need to implement for Q1 b).
        Inputs:
            img: array(float) 6*6
            kernel: array(float) 3*3
        Outputs:
            output: array(float)
    """

    #zero padding
    K = kernel.shape[0]
    H, W = img.shape
    img_pad = padding(img, (K - 1) // 2, "zeroPadding")
    HP, WP = img_pad.shape

    #build the Toeplitz matrix and compute convolution
    k_pad = np.zeros((K, WP)) # (K, WP)
    k_pad[:K, :K] = kernel

    # Construct circulant matrix. (n,) -> (n_rows, n).
    to_toeplitz = lambda v, n_row: v[np.arange(len(v)) - np.arange(n_row)[:, None]]
    t = np.apply_along_axis(lambda v: to_toeplitz(v, W), 1, k_pad) # (K, W, WP)
    t = np.concatenate((t, np.zeros((HP - K, W, WP))), axis=0) # (HP, W, WP)
    t = to_toeplitz(t, H) # (H, HP, W, WP)
    t = t.transpose(0, 2, 1, 3) # (H, W, HP, WP)
    t = t.reshape(img.size, img_pad.size) # (H*W, HP*WP)

    output = t @ img_pad.flatten() # (H*W, HP*WP) @ (HP*WP, 1) = (H*W, 1)
    output = output.reshape(img.shape) # (H, W)
    
    return output


def convolve(img, kernel):
    """
        The function you need to implement for Q1 c).
        Inputs:
            img: array(float)
            kernel: array(float)
        Outputs:
            output: array(float)
    """
    
    #build the sliding-window convolution here
    H, W = img.shape
    K = kernel.shape[0]

    # Idea: convert each iteration dimension to a tensor dimension.
    x = np.arange(H-K+1).reshape(-1, 1, 1, 1) # (H-K+1, 1, 1, 1)
    y = np.arange(W-K+1).reshape(1, -1, 1, 1) # (1, W-K+1, 1, 1)
    dx = np.arange(K).reshape(1, 1, -1, 1) # (1, 1, K, 1)
    dy = np.arange(K).reshape(1, 1, 1, -1) # (1, 1, 1, K)
    kk = kernel.reshape(1, 1, K, K) # (1, 1, K, K)

    r = x + dx # (H-K+1, 1, K, 1)
    c = y + dy # (1, W-K+1, 1, K)

    windows = img[r, c] # (H-K+1, W-K+1, K, K)
    output = np.sum(windows * kk, axis=(2, 3)) # (H-K+1, W-K+1)

    return output


def Gaussian_filter(img):
    padding_img = padding(img, 1, "replicatePadding")
    gaussian_kernel = np.array([[1/16,1/8,1/16],[1/8,1/4,1/8],[1/16,1/8,1/16]])
    output = convolve(padding_img, gaussian_kernel)
    return output

def Sobel_filter_x(img):
    padding_img = padding(img, 1, "replicatePadding")
    sobel_kernel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    output = convolve(padding_img, sobel_kernel_x)
    return output

def Sobel_filter_y(img):
    padding_img = padding(img, 1, "replicatePadding")
    sobel_kernel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    output = convolve(padding_img, sobel_kernel_y)
    return output



if __name__=="__main__":

    np.random.seed(111)
    input_array=np.random.rand(6,6)
    input_kernel=np.random.rand(3,3)


    # task1: padding
    zero_pad =  padding(input_array,1,"zeroPadding")
    np.savetxt("result/HM1_Convolve_zero_pad.txt",zero_pad)

    replicate_pad = padding(input_array,1,"replicatePadding")
    np.savetxt("result/HM1_Convolve_replicate_pad.txt",replicate_pad)


    #task 2: convolution with Toeplitz matrix
    result_1 = convol_with_Toeplitz_matrix(input_array, input_kernel)
    np.savetxt("result/HM1_Convolve_result_1.txt", result_1)

    #task 3: convolution with sliding-window
    result_2 = convolve(input_array, input_kernel)
    np.savetxt("result/HM1_Convolve_result_2.txt", result_2)

    #task 4/5: Gaussian filter and Sobel filter
    input_img = read_img("Lenna.png")/255

    img_gadient_x = Sobel_filter_x(input_img)
    img_gadient_y = Sobel_filter_y(input_img)
    img_blur = Gaussian_filter(input_img)

    write_img("result/HM1_Convolve_img_gadient_x.png", img_gadient_x*255)
    write_img("result/HM1_Convolve_img_gadient_y.png", img_gadient_y*255)
    write_img("result/HM1_Convolve_img_blur.png", img_blur*255)




    