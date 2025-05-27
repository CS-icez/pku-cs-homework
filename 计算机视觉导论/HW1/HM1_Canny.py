import numpy as np
from HM1_Convolve import Gaussian_filter, Sobel_filter_x, Sobel_filter_y
from utils import read_img, write_img

def compute_gradient_magnitude_direction(x_grad, y_grad):
    """
        The function you need to implement for Q2 a).
        Inputs:
            x_grad: array(float) 
            y_grad: array(float)
        Outputs:
            magnitude_grad: array(float)
            direction_grad: array(float) you may keep the angle of the gradient at each pixel
    """
    magnitude_grad = np.sqrt(x_grad**2 + y_grad**2)
    direction_grad = np.arctan2(y_grad, x_grad)

    return magnitude_grad, direction_grad 



def non_maximal_suppressor(grad_mag, grad_dir):
    """
        The function you need to implement for Q2 b).
        Inputs:
            grad_mag: array(float) 
            grad_dir: array(float)
        Outputs:
            output: array(float)
    """   

    grad_dir = (np.rad2deg(grad_dir) % 180 // 22.5 + 1) // 2 % 4

    # Get all neighbors in a tensor style.
    pad = np.pad(grad_mag, 1, mode='constant')
    t = pad[:-2, 1:-1]
    b = pad[2:, 1:-1]
    l = pad[1:-1, :-2]
    r = pad[1:-1, 2:]
    tl = pad[:-2, :-2]
    tr = pad[:-2, 2:]
    bl = pad[2:, :-2]
    br = pad[2:, 2:]

    NMS_output = np.select(
        # Note that x-axis points rightwards and y-axis points downwards.
        # This can be inferred from the Sobel filter.
        condlist=[
            (grad_dir == 0) & (grad_mag > l) & (grad_mag > r),
            (grad_dir == 1) & (grad_mag > tl) & (grad_mag > br),
            (grad_dir == 2) & (grad_mag > t) & (grad_mag > b),
            (grad_dir == 3) & (grad_mag > tr) & (grad_mag > bl)
        ],
        choicelist=[grad_mag, grad_mag, grad_mag, grad_mag]
    )

    return NMS_output 
            


def hysteresis_thresholding(img) :
    """
        The function you need to implement for Q2 c).
        Inputs:
            img: array(float) 
        Outputs:
            output: array(float)
    """


    #you can adjust the parameters to fit your own implementation 
    low_ratio = 0.06
    high_ratio = 0.18

    low = low_ratio * np.max(img)
    high = high_ratio * np.max(img)

    output = np.where(img > high, 1, 0)
    output = np.pad(output, 1, mode='constant')

    curr_num = np.count_nonzero(output)
    prev_num = 0

    while curr_num > prev_num:
        t = output[:-2, 1:-1]
        b = output[2:, 1:-1]
        l = output[1:-1, :-2]
        r = output[1:-1, 2:]
        tl = output[:-2, :-2]
        tr = output[:-2, 2:]
        bl = output[2:, :-2]
        br = output[2:, 2:]

        cond = (img > low) & (t | b | l | r | tl | tr | bl | br)
        output[1:-1, 1:-1] = np.where(cond, 1, output[1:-1, 1:-1])

        prev_num = curr_num
        curr_num = np.count_nonzero(output)

    output = output[1:-1, 1:-1]
    
    return output 



if __name__=="__main__":

    #Load the input images
    input_img = read_img("Lenna.png")/255

    #Apply gaussian blurring
    blur_img = Gaussian_filter(input_img)

    x_grad = Sobel_filter_x(blur_img)
    y_grad = Sobel_filter_y(blur_img)

    #Compute the magnitude and the direction of gradient
    magnitude_grad, direction_grad = compute_gradient_magnitude_direction(x_grad, y_grad)

    #NMS
    NMS_output = non_maximal_suppressor(magnitude_grad, direction_grad)

    #Edge linking with hysteresis
    output_img = hysteresis_thresholding(NMS_output)
    
    write_img("result/HM1_Canny_result.png", output_img*255)
