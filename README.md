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

## Running

After building the image, the runs that have been executed for this research can be found in `workspace/roman/roman_mount/runs`. When you want to run all models on the same data, run `All_runs.sh`. All individual runs can be found in either `VIO` or `Odom`. Every `.sh` file that is present there runs one model.

In `run_count.sh` the classes of each robot is counted for each frame. Note: the parameters inside `VIO/MaskDINO` are used to execute this file, if you change the robots inside this folder that partake in the test, they will not be counted using this `.sh` file.

The data is used from the Kimera-Multi dataset. This data can be downloaded after requesting access to the data on the [Kimera Multi Data Repo](https://github.com/MIT-SPARK/Kimera-Multi-Data).
The data should be stored in `/workspace/roman/test_data`.

The model parameters can be found in `workspace/roman/params/Odom` or `workspace/roman/params/VIO`. These parameters are made ready for use like in the executed experiments.  If you open the folders until you see all of the `.yaml` files, inside `data.yaml` you can change which robots you want to use the data of (make sure the paths are correct). Inside `fastsam.yaml` the parameters linked to the ROMAN front-end are located here, this includes plane filtering, class filtering, and confidence filtering. The other `.yaml` files speak for their own.

Evaluation can be done by using `/workspace/roman/evaluation/evaluate.py` for the precision and recall of the place recognition, `/workspace/roman/roman/offline_rgpo/evaluate.py` for the ATE RMSE and `/workspace/roman/evaluation/avg_fastsam_times.py` for the reported runtime of each front-end. 

While running each model or all models, there is a parameter present called `--max-time` that seperates the large amount of data in parts of --max-time seconds such that the run is executable.

## Code

The relevant code to the executed internship can be found in `/workspace/roman/roman` in the folder `/map` are the files `fastsam_wrapper.py`, `maskdino_wrapper.py`, `maskdino_wrapper_count.py`, `run.py`, `run_count.py` and `run_maskdino.py`. These files are where most of the changes made compared to the original can be found.


