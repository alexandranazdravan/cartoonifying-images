import cv2
import numpy as np
import time


def grayscale(image):
    # Convert the image to a NumPy array
    image = np.array(image)

    # Extract the red, green, and blue channels
    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]

    # Convert the image to grayscale using the formula above
    gray = 0.2989 * R + 0.5870 * G + 0.1140 * B

    # Return the grayscale image
    return gray

def blur(image, kernel_size):
    # Create a kernel with the specified size
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)

    # Perform the convolution
    image_blurred = np.zeros(image.shape)
    for i in range(image.shape[0] - kernel_size):
        for j in range(image.shape[1] - kernel_size):
            image_blurred[i][j] = np.sum(image[i:i + kernel_size, j:j + kernel_size] * kernel)

    # Return the blurred image
    return image_blurred

def apply_kernel(image, kernel):
    # Get image dimensions
    image_h, image_w = image.shape[:2]

    # Get kernel dimensions
    kernel_h, kernel_w = kernel.shape[:2]

    # Pad the image with zeros
    # Padding is used in image convolution to ensure that the output image has
    # the same size as the input image. When you convolve an image with a kernel,
    # the output image will typically have a smaller size than the input image because
    # the kernel is applied to each pixel in the image, but the kernel has a finite size
    # and thus only covers a small region of the image.
    pad_h = (kernel_h - 1) // 2
    pad_w = (kernel_w - 1) // 2
    padded_image = np.zeros((image_h + 2 * pad_h, image_w + 2 * pad_w))
    padded_image[pad_h:pad_h + image_h, pad_w:pad_w + image_w] = image

    # Create an empty image to hold the output
    output_image = np.zeros_like(image)

    # Iterate over the image and apply the kernel
    for y in range(image_h):
        for x in range(image_w):
            output_image[y, x] = (padded_image[y:y + kernel_h, x:x + kernel_w] * kernel).sum()

    # Return the output image
    return output_image

def sobel_edge_detector(image, threshold=100):
    # Convert the image to grayscale
    image_gray = grayscale(image)

    # Apply a blur to the grayscale image
    image_blurred = blur(image_gray, kernel_size=3)

    # Create the Sobel kernel
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    grad_x = apply_kernel(image_blurred, kernel_x)
    grad_y = apply_kernel(image_blurred, kernel_y)
    grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    grad_direction = np.arctan2(grad_y, grad_x)

    # Threshold the gradient magnitude
    grad_magnitude[grad_magnitude < threshold] = 0
    grad_magnitude[grad_magnitude >= threshold] = 255

    # Return the gradient magnitude image
    return grad_magnitude, grad_x, grad_y, grad_direction

def canny_detector(img, weak_th=None, strong_th=None):

    # Calculating the gradients
    imgg = sobel_edge_detector(img, 20)
    gx = imgg[1]
    gy = imgg[2]

    # Conversion of Cartesian coordinates to polar
    #  -> mag: an array of magnitudes, one for each Cartesian coordinate;
    #       the magnitude is the distance from the origin (0, 0) in polar coordinates.
    #  -> ang: an array of angles, one for each Cartesian coordinate;
    #       the angle is the angle from the positive x-axis in polar coordinates.
    # The gradient magnitude is the strength of the edge at that pixel, while the gradient
    # direction is the angle at which the edge is oriented. The gradient magnitude and direction
    # are returned as separate arrays, representing the x and y components of the gradient vector
    # at each pixel.
    # The cv2.cartToPolar function is then used to convert these x and y components to magnitude and
    # angle
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)

    # setting the minimum and maximum thresholds
    # for double thresholding
    mag_max = np.max(mag)
    if not weak_th: weak_th = mag_max * 0.1
    if not strong_th: strong_th = mag_max * 0.5

    # getting the dimensions of the input image
    heigh_width = img.shape
    height = heigh_width[0]
    width = heigh_width[1]

    # Looping through every pixel of the grayscale image
    for i_x in range(width):
        for i_y in range(height):

            grad_ang = ang[i_y, i_x]
            grad_ang = abs(grad_ang - 180) if abs(grad_ang) > 180 else abs(grad_ang)

            # Selecting the neighbours of the target pixel according to the gradient direction

            # In the x axis direction
            if grad_ang <= 22.5:
                neighb_1_x, neighb_1_y = i_x - 1, i_y
                neighb_2_x, neighb_2_y = i_x + 1, i_y

            # top right (diagonal-1) direction
            elif grad_ang > 22.5 and grad_ang <= (22.5 + 45):
                neighb_1_x, neighb_1_y = i_x - 1, i_y - 1
                neighb_2_x, neighb_2_y = i_x + 1, i_y + 1

            # In y-axis direction
            elif grad_ang > (22.5 + 45) and grad_ang <= (22.5 + 90):
                neighb_1_x, neighb_1_y = i_x, i_y - 1
                neighb_2_x, neighb_2_y = i_x, i_y + 1

            # top left (diagonal-2) direction
            elif grad_ang > (22.5 + 90) and grad_ang <= (22.5 + 135):
                neighb_1_x, neighb_1_y = i_x - 1, i_y + 1
                neighb_2_x, neighb_2_y = i_x + 1, i_y - 1

            # Now it restarts the cycle
            elif grad_ang > (22.5 + 135) and grad_ang <= (22.5 + 180):
                neighb_1_x, neighb_1_y = i_x - 1, i_y
                neighb_2_x, neighb_2_y = i_x + 1, i_y

            # Non-maximum suppression step
            # The non-maximum suppression step is a post-processing step in the Canny edge detector
            # that helps to thin the detected edges and reduce noise. It works by examining the gradient
            # magnitude and direction at each pixel and comparing it to the gradient magnitude and direction
            # at its neighbors. If the gradient magnitude at a given pixel is not a local maximum compared
            # to its neighbors, then it is suppressed (set to zero).
            # This is done because the gradient magnitude at an edge pixel is typically a local maximum
            # compared to its neighbors. By suppressing the non-maximum gradient magnitudes, we can thin
            # the detected edges and reduce noise.
            if width > neighb_1_x >= 0 and height > neighb_1_y >= 0:
                if mag[i_y, i_x] < mag[neighb_1_y, neighb_1_x]:
                    mag[i_y, i_x] = 0
                    continue

            if width > neighb_2_x >= 0 and height > neighb_2_y >= 0:
                if mag[i_y, i_x] < mag[neighb_2_y, neighb_2_x]:
                    mag[i_y, i_x] = 0

    ids = np.zeros_like(img)

    # double thresholding step
    for i_x in range(width):
        for i_y in range(height):

            grad_mag = mag[i_y, i_x]

            if grad_mag < weak_th:
                mag[i_y, i_x] = 0
            elif strong_th > grad_mag >= weak_th:
                ids[i_y, i_x] = 1
            else:
                ids[i_y, i_x] = 2

    # Dinally returning the magnitude of gradients of edges
    return mag

def median_cut_quantize(sample_img, img_arr):
    # When it reaches the end, color quantize
    r_average = np.mean(img_arr[:, 0])
    g_average = np.mean(img_arr[:, 1])
    b_average = np.mean(img_arr[:, 2])

    for data in img_arr:
        sample_img[data[3]][data[4]] = [r_average, g_average, b_average]


#The median cut algorithm is a method for reducing the number of colors in an image.
# It works by iteratively dividing the color space into two regions based on the median
# value of the pixels in each region, until the desired number of colors is reached.
def split_into_buckets(img, img_arr, depth):
    if len(img_arr) == 0:
        return

    if depth == 0:
        median_cut_quantize(img, img_arr)
        return

    r_range = np.max(img_arr[:, 0]) - np.min(img_arr[:, 0])
    g_range = np.max(img_arr[:, 1]) - np.min(img_arr[:, 1])
    b_range = np.max(img_arr[:, 2]) - np.min(img_arr[:, 2])

    space_with_highest_range = 0

    if g_range >= r_range and g_range >= b_range:
        space_with_highest_range = 1
    elif b_range >= r_range and b_range >= g_range:
        space_with_highest_range = 2
    elif r_range >= b_range and r_range >= g_range:
        space_with_highest_range = 0


    # Sort the image pixels by color space with highest range
    # and find the median and divide the array
    img_arr = img_arr[img_arr[:, space_with_highest_range].argsort()]
    median_index = int((len(img_arr) + 1) / 2)

    # Split the array into two buckets along the median
    split_into_buckets(img, img_arr[0:median_index], depth - 1)
    split_into_buckets(img, img_arr[median_index:], depth - 1)

def median_filter(image, kernel_size):
    # Convert the image to the float32 data type
    image = image.astype(np.float32)

    # Create a padded version of the image
    padded_image = cv2.copyMakeBorder(image, kernel_size//2, kernel_size//2, kernel_size//2, kernel_size//2, cv2.BORDER_REPLICATE)

    # Create an output image of the same size as the input image
    output_image = np.zeros_like(image)

    # Loop over the image pixels
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            # Extract the neighborhood around the pixel
            neighborhood = padded_image[y:y+kernel_size, x:x+kernel_size]

            # Compute the median value of the neighborhood for each color channel
            median_b = np.median(neighborhood[:,:,0])
            median_g = np.median(neighborhood[:,:,1])
            median_r = np.median(neighborhood[:,:,2])

            # Set the pixel value to the median value for each color channel
            output_image[y, x, 0] = median_b
            output_image[y, x, 1] = median_g
            output_image[y, x, 2] = median_r

    # Return the output image
    return output_image

def add_weighted(image1, alpha1, image2, alpha2, beta):

    # Create an output image of the same size as the input images
    output_image = np.zeros_like(image1)

    # Loop over the image pixels
    for y in range(image1.shape[0]):
        for x in range(image1.shape[1]):
            # Compute the weighted sum of the pixel values
            output_image[y, x] = alpha1 * image1[y, x] + alpha2 * image2[y, x] + beta

    # Return the output image
    return output_image

def b_and_w_to_rgb(image):

    # Create an output image of the same size as the input image, with 3 channels
    output_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=image.dtype)

    # Copy the grayscale values to all three channels
    output_image[:,:,0] = image
    output_image[:,:,1] = image
    output_image[:,:,2] = image

    # Return the output image
    return output_image

#The Kuwahara filter is a non-linear image filtering technique that is used to smooth images
# while preserving the edges. It works by dividing the image into overlapping regions (called quadrants),
# and selecting the region with the lowest variance as the output value for each pixel. This effectively
# smooths the image while preserving the edges, because the edges typically have higher variance compared
# to the smooth regions of the image.
#The Kuwahara filter is commonly used in image processing applications such as image denoising, texture
# analysis, and cartoon generation. It is particularly useful in applications where it is important
# to preserve the edges of the image, because it has a good balance between smoothing and edge preservation.

#Steps:
    # 1. the function defines the kernel radius as the kernel size divided by 2. This is used to determine the
    #     range of pixels to include in the kernel region;
    # 2. the function loops over the pixels in the input image, starting from the kernel radius to the end of the
    #     image minus the kernel radius. This ensures that the kernel region is fully contained within the image
    #     boundaries;
    # 3. for each pixel, the function initializes the sum of the squared differences to a large value (np.inf).
    #     This is used to track the minimum sum of the squared differences between the kernel region and the center
    #     pixel;
    # 4. the function then loops over the pixels in the kernel region, starting from the negative kernel radius
    #     to the positive kernel radius;
    # 5. for each kernel pixel, the function checks if any of its elements are outside of the color range.
    #     If any of the elements are outside of the range, the kernel pixel is skipped;
    # 6. if the kernel pixel is not skipped, the function computes the sum of the squared differences
    #     between the kernel pixel and the center pixel.
    # 7. if the sum of the squared differences is less than the current minimum sum, the minimum sum is
    #     updated to the current sum, and the center pixel in the output image is set to the value of the
    #     center pixel in the input image;
    # 8. after all of the kernel pixels have been processed, the function moves on to the next pixel in the
    #     input image;
    # 9. when all of the pixels in the input image have been processed, the function returns the output image.
def kuwahara_channel(image, kernel_size=5, color_range=10):
    # Get the size of the input image
    height, width = image.shape[:2]

    # Create an output image with the same size and type as the input image
    output_image = np.empty_like(image)

    # Loop over the rows and columns of the image
    for y in range(0, height):
        for x in range(0, width):
            # Initialize the sum and count of the pixel values in the kernel
            kernel_sum = 0
            kernel_count = 0

            # Loop over the rows and columns of the kernel
            for ky in range(-kernel_size//2, kernel_size//2 + 1):
                for kx in range(-kernel_size//2, kernel_size//2 + 1):
                    # Check if the kernel index is within the bounds of the image
                    if y + ky >= 0 and y + ky < height and x + kx >= 0 and x + kx < width:
                        # Check if the pixel value is within the color range of the center pixel
                        if abs(image[y + ky, x + kx] - image[y, x]) < color_range:
                            # Add the pixel value to the sum and increment the count
                            kernel_sum += image[y + ky, x + kx]
                            kernel_count += 1

            # Compute the mean of the pixel values in the kernel
            kernel_mean = kernel_sum / kernel_count if kernel_count > 0 else 0

            # Set the output pixel value to the mean
            output_image[y, x] = kernel_mean

    return output_image

def main():
    img = cv2.imread("input/Screenshot 2023-01-05 150923.png")
    # Grayscale and blur done in sobel_edge_detector
    print("Applying canny on the photo")
    image_edges = canny_detector(img)
    print("Canny finished its job!")
    image_inverted = 255 - image_edges
    cv2.imwrite('output/canny_Screenshot_inverted.png', image_inverted)
    print("Edges saved")
    cv2.imwrite('output/canny_Screenshot.png', image_edges)
    print("Edges with black and white reversed saved")
    # cv2.imshow('Invert',image_inverted)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # The reason for flattening the image is to make it easier to split the array into
    # halves and to calculate the median value of the pixels.If the image were not flattened,
    # it would be more difficult to split it into halves and to calculate the median value,
    # because the pixels are arranged in a two-dimensional grid rather than a one-dimensional array.
    median_image = img.copy()
    flattened_img_array = []
    for rindex, rows in enumerate(median_image):
        for cindex, color in enumerate(rows):
            flattened_img_array.append([color[0], color[1], color[2], rindex, cindex])
    flattened_img_array = np.array(flattened_img_array)

    print("Applying the median cut")
    # The 3rd parameter represents how many colors are needed in the power of 2. If the parameter
    # passed is 4 its means 2^4 = 16 colors
    split_into_buckets(median_image, flattened_img_array, 4)
    cv2.imwrite('output/image_median_cut.png', median_image)
    print("Median cut finished its job!")

    # Apply median filter
    print("Applying median filter")
    img_median = median_filter(median_image, 5)
    cv2.imwrite('output/image_median_filter.png', img_median)
    print("Median filter finished its job")

    print("Combining the photo which has its colors reduces with its edges")
    image_inverted = cv2.resize(image_inverted, (img_median.shape[1], img_median.shape[0]))
    # without this : ValueError:  setting an array element with a sequence.
    image_inverted = b_and_w_to_rgb(image_inverted)
    # Combine the images using alpha blending
    combined_image = add_weighted(image_inverted, 0.9, img_median, 1, -225)
    cv2.imwrite("output/combined.png", combined_image)
    print("Alpha blending finished its job! Final photo saved")

    # BONUS
    time.sleep(10)
    img = cv2.imread("output/combined.png")
    b, g, r = cv2.split(img)
    print(f"b = {b}")
    print(f"g = {g}")
    print(f"r = {r}")

    # Clamp the pixel values of each color channel to the range of the data type
    b = np.clip(b, 0, 255)
    g = np.clip(g, 0, 255)
    r = np.clip(r, 0, 255)
    print(f"b (clamped) = {b}")
    print(f"g (clamped) = {g}")
    print(f"r (clamped) = {r}")

    # Apply kuwahara on each color channel
    filtered_b = kuwahara_channel(b, kernel_size=5, color_range=30)
    filtered_g = kuwahara_channel(g, kernel_size=5, color_range=30)
    filtered_r = kuwahara_channel(r, kernel_size=5, color_range=30)


    kuwahara = cv2.merge([filtered_b, filtered_g, filtered_r])
    cv2.imwrite("output/kuwahara.png", kuwahara)

if __name__ == "__main__":
    main()



# This function takes in an image, a list of regions, and the desired number of colors,
# and it returns a version of the image with the number of colors reduced to the desired
# number. The function iteratively divides the regions into smaller subregions until there
# are the desired number of regions, and then it computes the average intensity values of each
# region to create a palette. It maps each pixel in the image to the closest color in the palette
# to produce the output image, and finally it converts the output image to the same format as the
# input image before returning it





