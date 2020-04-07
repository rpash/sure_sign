#!/bin/bash

# Install the latest version of OpenCV with the non-free contrib modules

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null && pwd )"

cd /tmp
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git

cd opencv
mkdir -p build && cd build
cmake -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
      -DOPENCV_ENABLE_NONFREE=ON ..

num_procs_avail=$(($(nproc)-1))
make -j$((num_procs_avail > 1 ? num_procs_avail : 1))

sudo make install

# Add opencv to virtualenv
VENV_DIR=${DIR}/env
echo $VENV_DIR
if [ -d $VENV_DIR ]; then
      python_version="$(python3 --version | sed 's/.*\(3\.[0-9]\).*/\1/')"
      python_so="/usr/local/lib/python${python_version}/site-packages/cv2"
      venv_so="${VENV_DIR}/lib/python${python_version}/site-packages/cv2"
      ln -s $python_so $venv_so
fi
