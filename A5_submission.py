'''
###### DO NOT EDIT ######
(Scroll down for start of the assignment)

# MATLAB Code:
# Alexey S. Sokolov a.k.a. nICKEL, Moscow, Russia
# June 2007
# alex.nickel@gmail.com

Zig-zag section
'''

import numpy as np


# Zigzag scan of a matrix

# --INPUT--
# Argument: 2D matrix of any size, not strictly square 

# --OUTPUT--
# Returns: 1-by-(m*n) array, where input matrix is m*n

def zigzag(input):
    # initializing the variables
    # ----------------------------------
    h = 0
    v = 0

    vmin = 0
    hmin = 0

    vmax = input.shape[0]
    hmax = input.shape[1]

    # print(vmax ,hmax )

    i = 0

    output = np.zeros((vmax * hmax))
    # ----------------------------------

    while ((v < vmax) and (h < hmax)):

        if ((h + v) % 2) == 0:  # going up

            if (v == vmin):
                # print(1)
                output[i] = input[v, h]  # if we got to the first line

                if (h == hmax):
                    v = v + 1
                else:
                    h = h + 1

                i = i + 1

            elif ((h == hmax - 1) and (v < vmax)):  # if we got to the last column
                # print(2)
                output[i] = input[v, h]
                v = v + 1
                i = i + 1

            elif ((v > vmin) and (h < hmax - 1)):  # all other cases
                # print(3)
                output[i] = input[v, h]
                v = v - 1
                h = h + 1
                i = i + 1


        else:  # going down

            if ((v == vmax - 1) and (h <= hmax - 1)):  # if we got to the last line
                # print(4)
                output[i] = input[v, h]
                h = h + 1
                i = i + 1

            elif (h == hmin):  # if we got to the first column
                # print(5)
                output[i] = input[v, h]

                if (v == vmax - 1):
                    h = h + 1
                else:
                    v = v + 1

                i = i + 1

            elif ((v < vmax - 1) and (h > hmin)):  # all other cases
                # print(6)
                output[i] = input[v, h]
                v = v + 1
                h = h - 1
                i = i + 1

        if ((v == vmax - 1) and (h == hmax - 1)):  # bottom right element
            # print(7)
            output[i] = input[v, h]
            break

    # print ('v:',v,', h:',h,', i:',i)
    return output


# Inverse zigzag scan of a matrix

# --INPUT--
# Argument: 1-by-m*n array, m & n are vertical & horizontal sizes of output matrix

# --OUTPUT--
# Returns: a 2D matrix of defined sizes with input array items gathered by zigzag

def inverse_zigzag(input, vmax, hmax):
    # print input.shape

    # initializing the variables
    # ----------------------------------
    h = 0
    v = 0

    vmin = 0
    hmin = 0

    output = np.zeros((vmax, hmax))

    i = 0
    # ----------------------------------

    while ((v < vmax) and (h < hmax)):
        # print ('v:',v,', h:',h,', i:',i)
        if ((h + v) % 2) == 0:  # going up

            if (v == vmin):
                # print(1)

                output[v, h] = input[i]  # if we got to the first line

                if (h == hmax):
                    v = v + 1
                else:
                    h = h + 1

                i = i + 1

            elif ((h == hmax - 1) and (v < vmax)):  # if we got to the last column
                # print(2)
                output[v, h] = input[i]
                v = v + 1
                i = i + 1

            elif ((v > vmin) and (h < hmax - 1)):  # all other cases
                # print(3)
                output[v, h] = input[i]
                v = v - 1
                h = h + 1
                i = i + 1


        else:  # going down

            if ((v == vmax - 1) and (h <= hmax - 1)):  # if we got to the last line
                # print(4)
                output[v, h] = input[i]
                h = h + 1
                i = i + 1

            elif (h == hmin):  # if we got to the first column
                # print(5)
                output[v, h] = input[i]
                if (v == vmax - 1):
                    h = h + 1
                else:
                    v = v + 1
                i = i + 1

            elif ((v < vmax - 1) and (h > hmin)):  # all other cases
                output[v, h] = input[i]
                v = v + 1
                h = h - 1
                i = i + 1

        if ((v == vmax - 1) and (h == hmax - 1)):  # bottom right element
            # print(7)
            output[v, h] = input[i]
            break

    return output


'''
######
Assignment 5 starts here
######
'''


def part1_encoder():
    # JPEG encoding

    import numpy as np
    # import scipy
    import matplotlib.pyplot as plt
    from skimage import io
    from scipy.fftpack import dct, idct

    # NOTE: Defining block size
    block_size = 8

    # TODO: Read image using skimage.io as grayscale
    img = io.imread('cameraman.png', as_gray=True)

    plt.imshow(img, cmap='gray')
    plt.title('input image')
    plt.axis('off')
    plt.show()

    '''
    Interesting property: Separability

    The separability property refers to the fact that a 2D DCT can be computed as the product of two 1D DCTs 
    applied along each dimension of the data independently. This means that a 2D DCT can be computed much more 
    efficiently as two 1D DCTs instead of directly computing the 2D transform.
    '''

    # TODO: Function to compute 2D Discrete Cosine Transform (DCT)
    # Apply DCT with type 2 and 'ortho' norm parameters

    def dct2D(img):
        result = np.fft.fft2(img, norm='ortho')
        return result

    # TODO: Get size of image
    h, w = img.shape

    # TODO: Compute number of blocks (of size 8-by-8), cast the numbers to int

    nbh = int(np.ceil(h / 8))
    nbw = int(np.ceil(w / 8))

    # TODO: (If necessary) Pad the image, get size of padded image
    H = int(np.ceil(h / 8) * 8)
    W = int(np.ceil(w / 8) * 8)

    # TODO: Create a numpy zero matrix with size of H,W called padded img
    padded_img = np.zeros(shape=(H, W))

    # TODO: Copy the values of img into padded_img[0:h,0:w]
    padded_img[:h, 0:w] = img

    # TODO: Display padded image
    plt.imshow(np.uint8(padded_img), cmap='gray')
    plt.title('Padded Image')
    plt.axis('off')
    plt.show()

    #
    # # TODO: Create the quantization matrix
    # # Refer to this PDF (https://www.ijg.org/files/Wallace.JPEG.pdf)
    # # Use Fig. 10 (c) (Page 12) as your quantization matrix
    #
    # ###### Your code here ######
    #
    # # TODO: Initialize an empty numpy array to store the quantized blocks
    # quantized_blocks = ###### Your code here ######
    #
    # '''NEW ADDITIONS/MODIFICATIONS'''
    # '''NEW ADDITIONS/MODIFICATIONS'''
    # '''NEW ADDITIONS/MODIFICATIONS'''
    #
    # # TODO: Initialize variables for compression calculations
    # ###### Your code here ######
    #
    # # NOTE: Iterate over blocks
    # for i in range(nbh):
    #
    #     # Compute start and end row indices of the block
    #     row_ind_1 = i * block_size
    #     row_ind_2 = row_ind_1 + block_size
    #
    #     for j in range(nbw):
    #
    #         # Compute start and end column indices of the block
    #         col_ind_1 = j * block_size
    #         col_ind_2 = col_ind_1 + block_size
    #
    #         # TODO: Select current block to process using calculated indices (through splicing)
    #         block = ###### Your code here ######
    #
    #         # TODO: Apply dct2d() to selected block
    #         DCT = ###### Your code here ######
    #
    #         # TODO: Quantization
    #         # Divide each element of DCT block by corresponding element in quantization matrix
    #         quantized_DCT = ###### Your code here ######
    #
    #         # TODO: Reorder DCT coefficients into block (use zigzag function)
    #         reordered = ###### Your code here ######
    #
    #         # TODO: Reshape reordered array to 8-by-8 2D block
    #         reshaped = ###### Your code here ######
    #
    #         # TODO: Copy reshaped matrix into padded_img on current block corresponding indices
    #         ###### Your code here ######
    #
    #         '''NEW ADDITIONS/MODIFICATIONS'''
    #         '''NEW ADDITIONS/MODIFICATIONS'''
    #         '''NEW ADDITIONS/MODIFICATIONS'''
    #
    #         # TODO: Compute pixel locations with non-zero values before and after quantization
    #         # TODO: Compute total number of pixels
    #         ###### Your code here ####
    #
    # plt.imshow(np.uint8(padded_img),cmap='gray')
    # plt.title(###### Title here ######)
    # plt.axis('off')
    # plt.show()
    #
    #
    # # NOTE: Write h, w, block_size and padded_img into .txt files at the end of encoding
    #
    # # TODO: Write padded_img into 'encoded.txt' file
    # # First parameter should be 'encoded.txt'
    # ###### Your code here ######
    #
    # # TODO: write [h, w, block_size] into size.txt
    # # First parameter should be 'size.txt'
    # ###### Your code here ######
    #
    # '''NEW ADDITIONS/MODIFICATIONS'''
    # '''NEW ADDITIONS/MODIFICATIONS'''
    # '''NEW ADDITIONS/MODIFICATIONS'''
    #
    # # TODO: Calculate percentage of pixel locations with non-zero values before and after to measure degree of compression
    #
    # # Print statements as shown in eClass
    # ###### Your code here ######
    #


#
# def part2_decoder():
#     # JPEG decoding
#
#     import numpy as np
#     # import scipy
#     import matplotlib.pyplot as plt
#     from skimage import io
#     from scipy.fftpack import dct,idct
#
#     # NOTE: Defining block size
#     block_size = 8
#
#     # TODO: Function to compute 2D Discrete Cosine Transform (DCT)
#     # Apply IDCT with type 2 and 'ortho' norm parameters
#
#     def idct2D(x):
#             ###### Your code here ######
#             return result
#
#
#     # TODO: Load 'encoded.txt' into padded_img
#     ###### Your code here ######
#
#     # TODO: Load h, w, block_size and padded_img from .txt files
#     ###### Your code here ######
#
#     # TODO: 6. Get size of padded_img, cast to int if needed
#     ###### Your code here ######
#
#     # TODO: Create the quantization matrix (Same as before)
#     # Refer to this PDF (https://www.ijg.org/files/Wallace.JPEG.pdf
#     # Use Fig. 10 (c) (Page 12) as your quantization matrix
#
#     ###### Your code here ######
#
#     # TODO: Compute number of blocks (of size 8-by-8), cast to int
#     nbh = ###### Your code here ###### # (number of blocks in height)
#     nbw = ###### Your code here ###### # (number of blocks in width)
#
#     # TODO: iterate over blocks
#     for i in range(nbh):
#
#             # Compute start and end row indices of the block
#             row_ind_1 = i * block_size
#
#             row_ind_2 = row_ind_1 + block_size
#
#             for j in range(nbw):
#
#                 # Compute start and end column indices of the block
#                 col_ind_1 = j * block_size
#
#                 col_ind_2 = col_ind_1 + block_size
#
#                 # TODO: Select current block to process using calculated indices
#                 block = ###### Your code here ######
#
#                 # TODO: Reshape 8-by-8 2D block to 1D array
#                 reshaped = ###### Your code here ######
#
#                 # TODO: Reorder array into block (use inverse_zigzag function)
#                 reordered = ###### Your code here ######
#
#                 # TODO: De-quantization
#                 # Multiply each element of reordered block by corresponding element in quantization matrix
#                 dequantized_DCT = ###### Your code here ######
#
#                 # TODO: Apply idct2d() to reordered matrix
#                 IDCT = ###### Your code here ######
#
#                 # TODO: Copy IDCT matrix into padded_img on current block corresponding indices
#                 ###### Your code here ######
#
#     # TODO: Remove out-of-range values
#     ###### Your code here ######
#
#     plt.imshow(np.uint8(padded_img),cmap='gray')
#     plt.title(###### Title here ######)
#     plt.axis('off')
#     plt.show()
#
#     # TODO: Get original sized image from padded_img
#     ###### Your code here ######
#
#     plt.imshow(np.uint8(decoded_img),cmap='gray')
#     plt.title(###### Title here ######)
#     plt.axis('off')
#     plt.show()
#


if __name__ == '__main__':
    part1_encoder()
    # part2_decoder()
