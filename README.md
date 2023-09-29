# Efficient Real-time Video Frame Classification for Bladder Lesion
------
The scripts are divided into two components:
1. Model training and evaluation
2. Real-time Illumination Adapted Bladder Lesion Detection Framework for Cystoscopy Videos.

- To install, we will need Python installed; some Python packages are required and can be installed by running the following command line:
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
- The real-time framework supports ONNX (Pytorch/TensorFlow models are convertible to ONNX format).

- The real-time framework incorporates the use of OpenGL to leverage hardware acceleration for efficient graphical augmentation.

- You need to convert the TensorFlow format to ONNX format for better performance and deployment.

- CycleGAN was used to optimize the color space for the development set.
------
##### If you have issues, please contact us or open a thread in the issues section.
##### Please cite when use this framework in your work:
Efficient Augmented Intelligence Framework for Bladder Lesion Detection
Okyaz Eminaga, Timothy Jiyong Lee, Mark Laurie, T. Jessie Ge, Vinh La, Jin Long, Axel Semjonow, Martin Bogemann, Hubert Lau, Eugene Shkolyar, Lei Xing, and Joseph C. Liao
JCO Clinical Cancer Informatics 2023 :7 



