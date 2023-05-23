# Efficient Real-time Video Frame Classification for Bladder Lesion
------
The scripts are divided into two components:
1. Model training and evaluation
2. Real-time Illumination adapted Bladder Lesion Detection Framework for Cystoscopy Videos.

- To install, we will need python installed; some python packages are required and can be installed by running the following command line:
```
    ./install_packages.sh
```

- To Run, there are two options:</br>
    1. Process a folder with cystoscopy videos
    ```
    run_exp.sh
    ```

    2. Process in real-time from the video input 
    ```
    run_real_time.sh
    ```
## NOTES
- The real-time framework supports ONNX (pytorch/tensorflow models are convertable to onnx format).

- The real-time framework incorporates the use of OpenGL to leverage hardware acceleration for efficient graphical augmentation.

- You need to convert the tensorflow format to ONNX format for a better performance and deployment.

- CycleGAN was used to optimize the color space for the development set.
------
##### If you have issues, please open a thread
##### Please cite when use this framework in your work


