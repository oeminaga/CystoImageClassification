# Efficient Real-time Video Frame Classification for Bladder Lesion
------
The scripts are divided into two components:
1. Model training and evaluation
2. Real-time Illumination adapted Bladder Lesion Detection Framework for Cystoscopic Videos.

- To install, we will need python installed; some python packages are required and can be installed by running the following command line:
```
    ./install_packages.sh
```

## NOTES
- The model was developed using Tensorflow/Keras.

- The real-time framework incorporates the use of OpenGL to leverage hardware acceleration for efficient graphical rendering.

- You need to convert the tensorflow format to ONNX format for a better performance and deployment.