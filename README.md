# 🐑🐑🐑 🐕 kernlearn

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Learn interaction kernels that govern collective behaviour

Repository to accompany the thesis: _Understanding Collective Behvaiour Through Machine Learning_.

## Installation

`pip install git+https://github.com/crimbs/kernlearn.git`

## Fish Schooling Dataset

The dataset can be downloaded from the ScholarsArchive@OSU [here](https://ir.library.oregonstate.edu/concern/datasets/zk51vq07c).

> The data is a JSON format file containing the position, velocity, and fish identifier data for 300 golden shiners in a shallow (depth of 4.5 to 5 cm) rectangular water tank (2.1 by 1.2 meters). There are 5000 individual frames (samples of position and velocity) corresponding to video taken at a rate of 30 frames/s and analysed to extract individual fish’s trajectories. The fields px, py, vx, vy correspond to the x- and y-components of each detected fish’s position (at center of mass) and velocity. The onfish field gives each fish in the frame a unique identifier. When an individual can no longer be distinguished, its identifier is retired; once the fish is again being tracked it is assigned a new ID. This data is a subsample of the frames acquired by the creators in the study <https://doi.org/10.1073/pnas.1107583108>, refer to it and its supplementary information for more details.
