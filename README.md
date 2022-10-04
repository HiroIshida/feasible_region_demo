```bash
mkdir -p ~/feasible_region_ws/src
cd ~/feasible_region_ws/src
git clone https://github.com/HiroIshida/feasible_region_demo.git
wstool init
wstool merge feasible_region_demo/.rosinstall
wstool update
rosdep install --from-paths . --ignore-src -y -r
cd ~/feasible_region_ws
catkin build
```

