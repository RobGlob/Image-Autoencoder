# Image-Autoencoder

### Working principle(supposedly)

Neural network learn to encode images into a one-dimensional vector of size 128 or 64. And then reconstruct the image from that vector.
This vector is called embedding, it stores the features and characteristics of the object in the image.
This vector can then be used to compare whether or not an object is the same in two pictures.
The cosine distance between the two vectors is a number from 0 to 1, the closer to one the more similar the objects are and vice versa.