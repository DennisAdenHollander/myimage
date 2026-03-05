# ROMAN

Welcome to the adapted version of ROMAN.
ROMAN is a view-invariant global localization method that maps open-set objects and uses the geometry, shape, and semantics of objects to find the transformation between a current pose and previously created object map.
This enables loop closure between robots even when a scene is observed from *opposite views.*



## Original paper

The original ROMAN paper can be found [here](https://www.roboticsproceedings.org/rss21/p029.pdf):


## System Overview


ROMAN has three modules: mapping, data association, and
pose graph optimization. The front-end mapping pipeline tracks
segments across RGB-D images to generate segment maps. The data
association module incorporates semantics and shape geometry attributes from submaps along with gravity as a prior into the ROMAN
alignment module to align maps and detect loop closures. These loop
closures and VIO are then used for pose graph optimization.

The `roman` package has a Python submodule corresponding to each pipeline module. Code for creating open-set object maps can be found in `roman.map`. Code for finding loop closures via data association of object maps can be found in `roman.align`. Finally, code for interfacing ROMAN with Kimera-RPGO for pose graph optimization can be found in `roman.offline_rpgo`.

## Dependencies

Installing all needed dependencies can be done by building and running the docker file. For this just use `build_docker.sh` and `run_docker.sh`.

## Code

The relevant code to the executed internship can be found in `/workspace/roman/roman` in the folder `/map` are the files `fastsam_wrapper.py`, `maskdino_wrapper.py`, `maskdino_wrapper_count.py`, `run.py`, `run_count.py` and `run_maskdino.py`. These files are where most of the changes made compared to the original can be found.


## Reproduce the experiments

Follow the steps below to reproduce the experiments.

### 1. Build and Run the Docker Environment

Install all required dependencies by building and running the provided Docker container:

```
./build_docker.sh
./run_docker.sh
```

### 2. Download and Place the Dataset

The experiments use data from the Kimera-Multi dataset.

#### 1. Request access to the dataset from the repository: [Kimera Multi Data Repo](https://github.com/MIT-SPARK/Kimera-Multi-Data)
#### 2. Place the data in `/workspace/roman/test_data`

Make sure the folder structure matches the expected dataset layout.

### 3. Configure the Robots and Parameters

Model parameters are located in:
```
/workspace/roman/params/Odom
/workspace/roman/params/VIO
```

Each experiment folder contains multiple .yaml configuration files.

Important configuration files: 

#### - `data.yaml`
Defines which robots are used in the experiment and where their data is located.
Ensure the dataset paths are correct.

#### - `fastsam.yaml`
Contains parameters for the ROMAN front-end, including:
- plane filtering
- class filtering
- confidence thresholds

Other `.yaml files` contain model-specific parameters.

### 4. Run Experiments

All experiment scripts are located in: `/workspace/roman/roman_mount/runs`

To execute all models on the same dataset, run: `./All_runs.sh`

Individual experiment scripts can be found in:
```
runs/VIO
runs/Odom
```
Each `.sh` script runs a single model configuration.

### 5. Running on Large Datasets

Each run script includes a parameter:
```
--max-time
```

This parameter splits the dataset into segments of `--max-time` seconds.
This allows large datasets to be processed in manageable chunks so the runs remain executable.

### 6. Counting Robot Classes

To count the detected classes for each robot frame, run:
```
./run_count.sh
```
Note:
This script uses the robot configuration defined in:
```
VIO/MaskDINO
```
If you change the robots used in the experiments, the counting script must be updated accordingly.

 


