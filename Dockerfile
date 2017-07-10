FROM nvidia/cuda:7.5-cudnn5-devel

MAINTAINER Nikki Aldeborgh <nikki.aldeborgh@digitalglobe.com>

RUN apt-get -y update && apt-get -y \
    install python \
    build-essential \
    python-software-properties \
    software-properties-common \
    ipython \
    python-pip \
    python-scipy \
    python-numpy \
    python-dev \
    python-six \
    gdal-bin \
    python-gdal \
    libgdal-dev \
    libspatialite-dev \
    sqlite3 \
    libatlas-base-dev \
    gfortran \
    libyaml-dev \
    libpq-dev \
    libcurl4-gnutls-dev \
    libproj-dev \
    libxml2-dev \
    libgeos-dev \
    libnetcdf-dev \
    libpoppler-dev \
    libspatialite-dev \
    libhdf4-alt-dev \
    libhdf5-serial-dev \
    vim \
    wget \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN pip install gdal numpy h5py geojson sklearn keras==1.2.2 geojsontools
RUN pip install --upgrade six
RUN pip install Theano==0.9.0

COPY ./bin /
COPY .theanorc /root/.theanorc
COPY keras.json /root/.keras/keras.json
