FROM osrf/ros:noetic-desktop-full

# install developper tools:
RUN apt update 
RUN apt install -y build-essential sudo git vim tmux wget

# install gui tools: firefox adn xdg-utils for browse files
RUN apt install -y xdg-utils firefox

# networking stuff:
RUN apt install -y openssh-server net-tools

RUN \
  useradd h-ishida && \
  echo "h-ishida ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/h-ishida && \
  chmod 0440 /etc/sudoers.d/h-ishida && \
  mkdir -p /home/h-ishida && \
  chown h-ishida:h-ishida /home/h-ishida && \
  chsh -s /bin/bash h-ishida

RUN echo 'root:root' | chpasswd
RUN echo 'h-ishida:h-ishida' | chpasswd

WORKDIR /home/h-ishida
USER h-ishida

# set up ssh
# https://github.com/IMC3ofC/db2express-c.docker/issues/12
RUN sudo mkdir /var/run/sshd
RUN sudo sed -i 's/\#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
EXPOSE 2233

# setup ssh keys
COPY --chown=h-ishida .ssh /home/h-ishida/.ssh

RUN sudo apt-get update && sudo apt-get install python3-pip -y

# setup my own dotfiles (PyYAML is required to intsall)
RUN git clone https://github.com/HiroIshida/dotfiles2.git
RUN pip3 install PyYAML
RUN sudo apt install curl -y
RUN cd dotfiles2 && ./install --light && exec bash

# install rospackage
RUN sudo apt-get install ros-noetic-jsk-pcl-* -y
RUN sudo apt-get install ros-noetic-jsk-tools -y

# crate workspace
SHELL ["/bin/bash", "-c"]
RUN \
    sudo apt-get install python3-osrf-pycommon python3-catkin-tools && \
    source /opt/ros/noetic/setup.bash && \
    mkdir -p ~/catkin_ws/src && \
    cd ~/catkin_ws/src && git clone git@github.com:HiroIshida/feasible_region_demo.git && \
    cd ~/catkin_ws && catkin build

CMD bash
