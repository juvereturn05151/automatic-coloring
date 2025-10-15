import cv2
import gc
import matplotlib.pyplot as plt

def main():
    #turn rgb image into a gray scale
    image_path = "red_shirt.png"
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) 

    #display the image
    plt.imshow(image, cmap='gray')
    plt.title("Your Input Image")
    plt.axis('off')
    plt.show()

    print("Your Input Image shape: ", image.shape)

    gc.collect()

if __name__ == "__main__":
    main()