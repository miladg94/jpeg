import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import DCTTransformNative as vcadct


def quantize(QP, dct):
    # Quantization
    threshold3 = []
    threshold2 = []
    threshold1 = []
    min = np.min(dct)
    max = np.max(dct)
    if QP == 3:
            # Define thresholds
            gap = (max-min) // 4
            for i in range(3):
                threshold3.append(min + gap)
                min = min + gap
            # Binarize the array
            dct = np.where(dct >= threshold3[2], 0b11,
                        np.where(dct >= threshold3[1], 0b10,
                                np.where(dct >= threshold3[0], 0b01,
                                        np.where(dct >= 0, 0b00, 0b00))))
    elif QP == 2:
            gap = max // 8
            for i in range(7):
                threshold2.append(min + gap)
                min = min + gap
            dct = np.where(dct >= threshold2[6], 0b111,
                        np.where(dct >= threshold2[5], 0b110,
                                np.where(dct >= threshold2[4], 0b101,
                                        np.where(dct >= threshold2[3], 0b100,
                                                np.where(dct >= threshold2[2], 0b011,
                                                        np.where(dct >= threshold2[1], 0b010,
                                                                np.where(dct >= threshold2[0], 0b001,
                                                                        np.where(dct >= 0, 000, 0b000))))))))
    elif QP == 1:
            gap = max // 16
            for i in range(15):
                threshold1.append(min + gap)
                min = min + gap
            dct = np.where(dct >= threshold1[14], 0b1111,
                        np.where(dct >= threshold1[13], 0b1110,
                                np.where(dct >= threshold1[12], 0b1101,
                                        np.where(dct >= threshold1[11], 0b1100,
                                                np.where(dct >= threshold1[10], 0b1011,
                                                        np.where(dct >= threshold1[9], 0b1010,
                                                                np.where(dct >= threshold1[8], 0b1001,
                                                                        np.where(dct >= threshold1[7], 0b1000,
                                                                                np.where(dct >= threshold1[6], 0b0111,
                                                                                        np.where(dct >= threshold1[5], 0b0110,
                                                                                                np.where(dct >= threshold1[4], 0b0101,
                                                                                                        np.where(dct >= threshold1[3], 0b0100,
                                                                                                                np.where(dct >= threshold1[2], 0b0011,
                                                                                                                        np.where(dct >= threshold1[1], 0b0010,
                                                                                                                                np.where(dct >= threshold1[0], 0b0001,
                                                                                                                                        np.where(dct >= 0, 0b0000, 0b0000))))))))))))))))
    dct = dct.flatten()
    return dct


def huffman(data):
    # Count symbol frequencies
    counter = Counter(data)

    # Build Huffman tree
    huffman_tree = defaultdict(str)
    for symbol, frequency in counter.items():
        huffman_tree[symbol] = bin(frequency)[2:]  # Convert frequency to binary string

    # Encode data using Huffman tree
    encoded_data = ''.join(huffman_tree[symbol] for symbol in data)

    return encoded_data, huffman_tree


def compress(y, QP, width, height):
    # Create an empty array for the compressed Y channel
    encoded = []
    trees = []
    block_count = 0

    # Process each 32x32 block in the Y channel
    for i in range(0, height, 32):
        for j in range(0, width, 32):
            # Extract a block
            block = y[i:i + 32, j:j + 32]

            # Apply DCT to the block
            dct = cv2.dct(np.float32(block))

            # Quantization
            dct = quantize(QP, dct)

            # Huffman encoding
            encoded_data, huffman_tree = huffman(dct)
            encoded.append(encoded_data)
            trees.append(huffman_tree)
            block_count += 1

    # Create a compressed Y channel
    compressed_y = np.concatenate((encoded, trees), axis=0)

    return compressed_y


def huffman_decode(encoded_data, huffman_tree):
    decoded_data = ""
    current_code = ""

    for bit in encoded_data:
        current_code += bit
        for symbol, code in huffman_tree.items():
            if current_code == code:
                decoded_data += symbol
                current_code = ""
                break

    return decoded_data


def dequantize(QP, decoded):
    if QP == 3:
        dct = np.where(decoded == 0b11, threshold3[2],
                       np.where(decoded == 0b10, threshold3[1],
                                np.where(decoded == 0b01, threshold3[0], 0)))
    elif QP == 2:
        dct = np.where(decoded == 0b111, threshold2[6],
                       np.where(decoded == 0b110, threshold2[5],
                                np.where(decoded == 0b101, threshold2[4],
                                         np.where(decoded == 0b100, threshold2[3],
                                                  np.where(decoded == 0b011, threshold2[2],
                                                           np.where(decoded == 0b010, threshold2[1],
                                                                    np.where(decoded == 0b001, threshold2[0],
                                                                             0)))))))
    elif QP == 1:
        dct = np.where(decoded == 0b1111, threshold1[14],
                       np.where(decoded == 0b1110, threshold1[13],
                                np.where(decoded == 0b1101, threshold1[12],
                                         np.where(decoded == 0b1100, threshold1[11],
                                                  np.where(decoded == 0b1011, threshold1[10],
                                                           np.where(decoded == 0b1010, threshold1[9],
                                                                    np.where(decoded == 0b1001, threshold1[8],
                                                                             np.where(decoded == 0b1000,
                                                                                      threshold1[7],
                                                                                      np.where(decoded == 0b0111,
                                                                                               threshold1[6],
                                                                                               np.where(
                                                                                                   decoded == 0b0110,
                                                                                                   threshold1[5],
                                                                                                   np.where(
                                                                                                       decoded == 0b0101,
                                                                                                       threshold1[
                                                                                                           4],
                                                                                                       np.where(
                                                                                                           decoded == 0b0100,
                                                                                                           threshold1[
                                                                                                               3],
                                                                                                           np.where(
                                                                                                               decoded == 0b0011,
                                                                                                               threshold1[
                                                                                                                   2],
                                                                                                               np.where(
                                                                                                                   decoded == 0b0010,
                                                                                                                   threshold1[
                                                                                                                       1],
                                                                                                                   np.where(
                                                                                                                       decoded == 0b0001,
                                                                                                                       threshold1[
                                                                                                                           0],
                                                                                                                       0)))))))))))))))
    return dct


def decompress(QP, compressed_y):
    # Access the individual arrays using their keys
    encoded = compressed_y["arr_0"]
    trees = compressed_y["arr_1"]
    decoded = []

    for encoded_data, huffman_tree in zip(encoded, trees):
        # Huffman decoding
        decoded = huffman_decode(encoded_data, huffman_tree)

        # Dequantization
        dct = dequantize(QP, decoded)

        # Apply inverse DCT to the block
        idct = cv2.idct(np.float32(dct))
        decoded.append(idct)

    # Reconstruct the original image by concatenating decoded blocks
    reconstructed_img = np.concatenate(decoded)

    # Reshape the image to its original shape
    reconstructed_img = reconstructed_img.reshape((height, width))

    # Convert the image back to BGR color space
    reconstructed_img_yuv = np.zeros((height, width, 3), dtype=np.uint8)
    reconstructed_img_yuv[:, :, 0] = reconstructed_img
    reconstructed_img = cv2.cvtColor(reconstructed_img_yuv, cv2.COLOR_YUV2BGR)

    return reconstructed_img


def psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    psnr = 20 * np.log10(255 / np.sqrt(mse))
    return psnr


def plot(y, reconstructed_img):
    # Display the original and decompressed Y channels
    plt.figure(figsize=(10, 5))
    plt.imshow(y, cmap='gray')
    plt.title('Original Y Channel')
    plt.axis('off')
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.imshow(reconstructed_img, cmap='gray')
    plt.title('Decompressed Y Channel')
    plt.axis('off')
    plt.show()


def main():
    # Read the original image
    image_path = "C:\original.png"
    img = cv2.imread(image_path)

    # Convert the image to YUV color space
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(img_yuv)

    # Define block size (32x32)
    block_size = 32
    height, width = y.shape

    # Compression
    QP = 3  # Between 1 (min) and 3 (max)
    compressed_y = compress(y, block_size, width, height)

    # Write encoded data to a file
    np.savez('compressed_y.npz', *compressed_y)

    # Load the data from the .npz file
    compressed_y = np.load("compressed_y.npz")

    # Decompression
    img_jpg = decompress(QP, compressed_y)

    # Save the reconstructed image
    cv2.imwrite('decompressed.jpg', img_jpg)

    # Calculate PSNR between original and decompressed images
    psnr_value = psnr(np.array(img), np.array(img_jpg))
    print("PSNR:", psnr_value)

    # Plot the images for comparison
    plot(img, img_jpg)


if __name__ == "__main__":
    main()