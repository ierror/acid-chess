#!/bin/bash

# for Ubuntu 22.04

SSH_HOST=$1
SSH_USER="root"
PROJECT_ROOT="/root/ac"
SSH_CMD="ssh $SSH_USER@$SSH_HOST"
INSTALL_NVIDIA_DRIVERS=0
SCRIPT_DIR=$(dirname $0)

die() {
    message=$1
    [ -z "$message" ] && message="Died"
    echo "$message at ${BASH_SOURCE[1]}:${FUNCNAME[1]} line ${BASH_LINENO[0]}." >&2
    exit 1
}

run_command () {
  $SSH_CMD "$1"
  if [ $? -ne 0 ];
  then
      die "$1 failed";
  fi
}

[ -z "$SSH_HOST" ] && die "Server can't be empty. Call me with server as first arg: $0 <SERVER>"

run_command  "sudo apt update"
run_command  "sudo apt -y install python3-pip tmux libgl1-mesa-glx tmux build-essential pkg-config cmake cmake-qt-gui ninja-build valgrind ubuntu-drivers-common"
run_command  "sudo pip install pipenv"

rsync -vaP -e "ssh " $SCRIPT_DIR/../../Pipfile* $SSH_USER@$SSH_HOST:$PROJECT_ROOT
rsync -vaP  --exclude 'data/models/' --exclude '*.pkl' --exclude '*.ckpt*' --exclude '.git/' --exclude '*.*boards.model*' --exclude '*.pth' --exclude 'tmp/*' --exclude '*.mov' --exclude '*.model' --exclude '*-checkpoint.ipynb' -e "ssh " $SCRIPT_DIR/../../ $SSH_USER@$SSH_HOST:$PROJECT_ROOT &

if [ $INSTALL_NVIDIA_DRIVERS -eq 1 ]
then
  run_command  "sudo apt -y install linux-headers-\$(uname -r)"
  run_command  "sudo wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin"
  run_command  "sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600"
  run_command  "cd /mnt; sudo wget https://developer.download.nvidia.com/compute/cuda/12.0.1/local_installers/cuda-repo-ubuntu2204-12-0-local_12.0.1-525.85.12-1_amd64.deb"
  run_command  "cd /mnt; sudo dpkg -i cuda-repo-ubuntu2204-12-0-local_12.0.1-525.85.12-1_amd64.deb"
  run_command  "cd /mnt; sudo cp /var/cuda-repo-ubuntu2204-12-0-local/cuda-*-keyring.gpg /usr/share/keyrings/"
  run_command  "sudo apt-get update"
  run_command  "sudo apt-get -y install cuda"
fi

run_command  "cd $PROJECT_ROOT; sudo pipenv requirements --dev | grep -Ev "^pyobjc" | grep -Ev "click" | grep -Ev "^pyside6"  > req.txt"
run_command  "sudo pip install --upgrade pip"
run_command  "cd $PROJECT_ROOT; sudo pip install --use-pep517 --ignore-installed -r req.txt"


fg

echo "ssh $SSH_USER@$SSH_HOST -L 127.0.0.1:8001:127.0.0.1:8888"
