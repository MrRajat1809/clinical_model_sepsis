# Base PyTorch image with CUDA and cuDNN support
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# Set non-interactive mode for apt-get to prevent build stalls
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies, compiling tools, and graphics libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    curl \
    git \
    libxml2-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libfontconfig1-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libfreetype6-dev \
    libpng-dev \
    libtiff5-dev \
    libjpeg-dev \
    pandoc \
    r-base \
    && rm -rf /var/lib/apt/lists/*

# Set working directory for the container
WORKDIR /workspace

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements.txt

# Install R packages for data manipulation, clinical analysis, and visualization
RUN Rscript -e "options(download.file.method = 'libcurl'); install.packages(c('tidyverse', 'survminer', 'survival', 'ggpubr', 'svglite', 'IRkernel', 'tableone', 'circlize'), repos='http://cran.rstudio.com/')"

# Register the R kernel with Jupyter
RUN Rscript -e "IRkernel::installspec(user = FALSE)"

# Expose the default Jupyter port
EXPOSE 8888

# Initialize JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]