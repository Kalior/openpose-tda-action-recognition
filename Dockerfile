FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04

# Install general dependencies
RUN apt-get update -y && apt-get upgrade -y
RUN apt-get install -y \
    wget \
    unzip \
    lsof \
    apt-utils \
    lsb-core \
    libatlas-base-dev \
    libopencv-dev \
    python-opencv \
    python-pip \
    build-essential \
    libprotobuf-dev \
    libleveldb-dev \
    libsnappy-dev \
    libhdf5-serial-dev \
    protobuf-compiler \
    cmake \
    libgflags-dev \
    libgoogle-glog-dev \
    liblmdb-dev

RUN apt-get install --no-install-recommends -y libboost-all-dev

# Install Caffe
ENV CAFFE_ROOT=/caffe
COPY --from=bvlc/caffe:gpu /opt/caffe $CAFFE_ROOT
RUN ln -s $CAFFE_ROOT/build/include/caffe/proto $CAFFE_ROOT/include/caffe/proto
COPY --from=bvlc/caffe:gpu /usr/local/lib/libnccl.so.1.3.5 /usr/local/lib/libnccl.so.1.3.5

ENV PYCAFFE_ROOT $CAFFE_ROOT/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig

# Install openpose
RUN wget https://github.com/CMU-Perceptual-Computing-Lab/openpose/archive/master.zip; \
    unzip master.zip; rm master.zip; mv openpose-master openpose

WORKDIR openpose/build
RUN cmake \
    -DBUILD_CAFFE=OFF \
    -DCaffe_INCLUDE_DIRS=$CAFFE_ROOT/include \
    -DCaffe_LIBS=$CAFFE_ROOT/build/lib/libcaffe.so \
    -DDOWNLOAD_BODY_25_MODEL=OFF \
    -DDOWNLOAD_BODY_COCO_MODEL=ON \
    -DDOWNLOAD_FACE_MODEL=OFF \
    -DDOWNLOAD_HAND_MODEL=OFF \
    -DDOWNLOAD_BODY_MPI_MODEL=OFF \
    -DBUILD_PYTHON=ON \
    -DGPU_MODE=CUDA \
    ..

RUN make -j`nproc`
RUN make install

ENV OPENPOSEPYTHON=/usr/local/python/
ENV PYTHONPATH $OPENPOSEPYTHON:$PYTHONPATH

RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:jonathonf/python-3.6
RUN apt-get update
RUN apt-get install -y \
    python3.6 \
    python3.6-dev \
    python3-pip \
    python3-pytest \
    python3-tk

RUN apt-get update -y && apt-get upgrade -y
RUN apt-get install -y \
    curl \
    make \
    g++ \
    graphviz \
    doxygen \
    perl \
    libboost-all-dev \
    libeigen3-dev \
    libgmp3-dev \
    libmpfr-dev \
    libtbb-dev \
    locales \
    libfreetype6-dev \
    pkg-config \
    software-properties-common

RUN python3.6 -m pip install pip --upgrade
RUN python3.6 -m pip install \
    numpy \
    scikit-learn \
    cython \
    sphinx \
    sphinxcontrib-bibtex \
    matplotlib

# Install sklearn_tda
RUN apt-get install -y \
    git-core

RUN git clone https://github.com/MathieuCarriere/sklearn_tda
WORKDIR sklearn_tda

# These changes maybe should be submitted to sklearn_tda
RUN sed -i "s|from vectors import|from .vectors import|g" sklearn_tda/code.py
RUN sed -i "s|from kernels import|from .kernels import|g" sklearn_tda/code.py
RUN sed -i "s|from hera_wasserstein import|from .hera_wasserstein import|g" sklearn_tda/code.py
RUN sed -i "s|from hera_bottleneck import|from .hera_bottleneck import|g" sklearn_tda/code.py

RUN python3.6 -m pip install .


# Install GCAL
WORKDIR /gcal
RUN curl -LO "https://github.com/CGAL/cgal/releases/download/releases%2FCGAL-4.12/CGAL-4.12.tar.xz" \
    && tar xf CGAL-4.12.tar.xz && cd CGAL-4.12 \
    && cmake -DCGAL_HEADER_ONLY=ON -DCMAKE_BUILD_TYPE=Release . && make all install

# Install Gudhi
WORKDIR /gudhi

RUN curl -LO "https://gforge.inria.fr/frs/download.php/file/37579/2018-06-14-13-32-49_GUDHI_2.2.0.tar.gz" \
    && tar xf 2018-06-14-13-32-49_GUDHI_2.2.0.tar.gz && cd 2018-06-14-13-32-49_GUDHI_2.2.0 \
    && mkdir build && cd build && cmake -DPython_ADDITIONAL_VERSIONS=3 -DPYTHON_EXECUTABLE=/usr/bin/python3.6 ..

WORKDIR 2018-06-14-13-32-49_GUDHI_2.2.0/build

ENV GUDHIPATH /gudhi/2018-06-14-13-32-49_GUDHI_2.2.0/build/cython
ENV PYTHONPATH $GUDHIPATH:$PYTHONPATH

RUN make all test install


WORKDIR /requirements
COPY requirements.txt requirements.txt

RUN python3.6 -m pip install pip --upgrade
RUN python3.6 -m pip install -r requirements.txt

COPY requirements.txt requirements.txt
RUN python3.6 -m pip install -r requirements.txt

WORKDIR /tda-action-recognition
