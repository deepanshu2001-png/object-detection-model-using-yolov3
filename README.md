Object Detection with TensorFlow GPU


This project implements an object detection solution using TensorFlow with GPU support, leveraging a pre-trained model (YOLOv3) for efficient inference. The setup is optimized for environments with NVIDIA GPUs to ensure fast processing and performance during inference.

Requirements


NVIDIA GPU with CUDA 12.6 support.
Miniconda or Anaconda (to manage the Conda environment).
TensorFlow GPU support.
CUDA, cuDNN, and TensorRT for optimal GPU usage.
Steps to Set Up and Run the Project
Follow these steps to get the project up and running.




Create Conda Environment


The project relies on a specific Conda environment defined in the conda-gpu.yml file. You can create this environment by running:

conda env create -f conda-gpu.yml


This will set up the required environment with the necessary dependencies, including TensorFlow GPU and other relevant packages.

Activate the Conda Environment


Activate the newly created Conda environment for this project:



conda activate yolov3-gpu

Downloading official yolov3 pretrained weights on coco dataset


("https://pjreddie.com/media/files/yolov3.weights")



Load Weights


To perform inference or training, the model requires weights to be loaded. This can be done by running the following script:

python load_weights.py


This script will load the pre-trained weights into the model, preparing it for object detection tasks.

Run the Application


Finally, you can run the application (Flask server) using the following command:

python app.py


This will start a Flask application on port 5000. You can now access the object detection API via http://localhost:5000.


Techniques and Custom Components

Here are some key custom techniques and classes implemented in this repository:

Batch Normalization Customization


In this project, we override TensorFlow's built-in BatchNormalization to ensure it behaves correctly when frozen:

class BatchNormalization(tf.keras.layers.BatchNormalization):
    """
    Make trainable=False freeze BN for real (the og version is sad)
    """
    def call(self, x, training=False):
        if training is None:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)


Transforming Targets for Output


This function processes the target data to match the output of the YOLOv3 model, organizing the bounding boxes and class information in the appropriate format for training.


@tf.function
def transform_targets_for_output(y_true, grid_size, anchor_idxs):
    # (implementation details)


Custom Model Definition


The YOLOv3 model is built with a custom architecture using Darknet-based convolutional layers:
def DarknetConv(x, filters, size, strides=1, batch_norm=True):
    if strides == 1:
        padding = 'same'
    else:
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)  # top left half-padding
        padding = 'valid'
    # Convolution with batch normalization and activation
    x = Conv2D(filters=filters, kernel_size=size, strides=strides, padding=padding,
               use_bias=not batch_norm, kernel_regularizer=l2(0.0005))(x)
    if batch_norm:
        x = BatchNormalization()(x)
        x = LeakyReLU(negative_slope=0.1)(x)
    return x
