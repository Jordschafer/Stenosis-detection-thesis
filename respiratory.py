from skimage import morphology, transform
import matplotlib.pyplot as plt
import numpy as np

max_pixel_r = 4 # define maximum size in pixels of artery
down_sample_rate = 4 # how much do we downsample the image

# the morphological disk that is used for filtering
footprint = morphology.disk(max_pixel_r)

# your code to load the images
# nb_frames = number of frames in video
nb_frames = None

# pixel array with shape [nb_frames, width, height]
width, height = 512, 512
pixel_array = np.zeros((nb_frames, width, height))

# computes the morphological closing of each image in the video (filtering away the vessel)
images_closed = []
for i in range(nb_frames):
    image = pixel_array[i]

    width, height = image.shape

    # downsample image
    image = transform.resize(image, (width // down_sample_rate, height // down_sample_rate))
    closed_image = morphology.closing(image, footprint)
        
    images_closed.append(closed_image)

# center images closed to zero mean
images_closed = np.array(images_closed)
images_c = ((images_closed - np.mean(images_closed)) / np.std(images_closed)).reshape(nb_frames, -1).T #(D, N) where N=nb_frames and D=nb_pixels

# perform PCA
XXT = np.matmul(images_c.T, images_c) #(N, N)
eigen_values_nn, eigen_vectors_nn = np.linalg.eig(XXT) #(N, N)
eigen_values_nn_d = np.diag(eigen_values_nn) #(N, N)
eigen_vectors_dn = np.matmul(np.matmul(images_c, eigen_vectors_nn), np.linalg.inv(eigen_values_nn_d)) #(D, N)
proj_v = np.matmul(images_c.T, eigen_vectors_dn[:, 0]) #(N)

# data point for each frame
xs = np.arange(0, nb_frames)

# PCA will return multiple curves sorted on "importance", the first curve is likely the breathing rate because it causes the most motion in the frames
for i in range(images_c.shape[-1])[:1]:
    # this is the curve
    ys = np.matmul(images_c.T, eigen_vectors_dn[:, i]) #(N)
    
    # normalize
    ys = (ys - np.min(ys)) / (np.max(ys) - np.min(ys))
    plt.plot(xs, ys)