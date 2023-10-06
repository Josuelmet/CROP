#!/bin/bash
# Taken from https://colab.research.google.com/github/count0/colab-gt/blob/master/colab-gt.ipynb
# If this doesn't work, here's a potential workaround: https://stackoverflow.com/questions/69404659/troubles-with-graph-tool-installations-in-google-colab
echo "deb http://downloads.skewed.de/apt jammy main" >> /etc/apt/sources.list
apt-key adv --keyserver keyserver.ubuntu.com --recv-key 612DEFB798507F25
apt-get update
apt-get install python3-graph-tool python3-matplotlib python3-cairo

# Colab uses a Python install that deviates from the system's! Bad collab! We need some workarounds.
apt purge python3-cairo
apt install libcairo2-dev pkg-config python3-dev
pip install --force-reinstall pycairo
pip install zstandard


git clone https://github.com/AhmedImtiazPrio/splinecam.git

pip install networkx
pip install python-igraph>=0.10
pip install tqdm
pip install livelossplot

pip uninstall torch torchvision -y
pip install --pre torch==1.12+cu116 torchvision -f https://download.pytorch.org/whl/torch_stable.html