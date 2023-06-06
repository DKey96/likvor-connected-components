# Connected Components

The multicore GPU implementation of Connected Components algorithm for image processing.

## Requirements
- Python3.10
- Anaconda

Libraries are listed in `requirements.txt`. To create conda environment with requirements use: `conda create --name <environment_name> --file requirements.txt`
To install the dependencies into already existing conda environment use: `conda install --file requirements.txt`

## How to use
- For running only the one-thread OpenCV implementation of connected components, run `opencv_one_thread_implementation.py`.
- For the GPU multi-thread implementation of connected components, run `multithread_gpu_implementation.py`.
