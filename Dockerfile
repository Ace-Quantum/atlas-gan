# Use an official TensorFlow image with GPU support (if you're using a GPU)
FROM tensorflow/tensorflow:latest-gpu-jupyter

# Alternatively, for CPU only, you can use the following:
# FROM tensorflow/tensorflow:latest-jupyter

# Set environment variables
ENV PYTHONBUFFERED=1

# Install additional Python libraries required for DCGAN
RUN pip install --no-cache-dir \
    numpy \
    matplotlib \
    pillow \
    h5py \
    pandas \
    scikit-learn \
    tqdm \
    opencv-python \
    imageio \
    seaborn

# Install TensorFlow Datasets for easy access to training data (e.g., CIFAR-10)
RUN pip install --no-cache-dir tensorflow-datasets

# Install PyTorch if you'd like to compare models or implement in PyTorch
RUN pip install --no-cache-dir torch torchvision

# Set up the workspace
WORKDIR /workspace

# Expose Jupyter notebook port
EXPOSE 8888

# Add a command to start Jupyter notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root", "--no-browser"]