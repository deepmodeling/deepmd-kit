# .bashrc

# User specific aliases and functions

alias rm='rm -i'
alias cp='cp -i'
alias mv='mv -i'

# Source global definitions
if [ -f /etc/bashrc ]; then
	. /etc/bashrc
fi
export http_proxy=http://child-prc.intel.com:913
export https_proxy=http://child-prc.intel.com:913

export PADDLEPADDLE_TP_CACHE="/home/guest/tp_cache"
#export LD_LIBRARY_PATH=/usr/local/lib/python3.8/dist-packages/deepmd_kit-1.2.3.dev530+g8d8f289.d20220512-py3.8-linux-x86_64.egg/deepmd/op:$LD_LIBRARY_PATH
#export LIBRARY_PATH=/usr/local/lib/python3.8/dist-packages/deepmd_kit-1.2.3.dev530+g8d8f289.d20220512-py3.8-linux-x86_64.egg/deepmd/op:$LIBRARY_PATH
#export DEEP_MD_PATH=/usr/local/lib/python3.8/dist-packages/deepmd_kit-1.2.3.dev530+g8d8f289.d20220512-py3.8-linux-x86_64.egg/deepmd/op
#export LD_LIBRARY_PATH=/usr/local/lib/python3.8/dist-packages/deepmd_kit-1.2.3.dev491+g1a06aa4-py3.8-linux-x86_64.egg/deepmd/op:$LD_LIBRARY_PATH
#export LIBRARY_PATH=/usr/local/lib/python3.8/dist-packages/deepmd_kit-1.2.3.dev491+g1a06aa4-py3.8-linux-x86_64.egg/deepmd/op:$LIBRARY_PATH
#export DEEP_MD_PATH=/usr/local/lib/python3.8/dist-packages/deepmd_kit-1.2.3.dev491+g1a06aa4-py3.8-linux-x86_64.egg/deepmd/op
#
#1.2.3.dev530+g8d8f289.d20220512
export tensorflow_root=/home/tensorflowroot
export deepmd_root=/home/deepmdroot
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/root/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/root/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/root/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/root/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
export PATH=/home/jessie/cmake-3.21.0-linux-x86_64/bin:$PATH

export PATH=/home/lammps-stable_29Oct2020/src:$PATH
#export LD_LIBRARY_PATH=/home/paddle-deepmd/source/build/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/deepmd-kit/source/build/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/Paddle/build/paddle_inference_install_dir/paddle/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/Paddle/build/paddle_inference_install_dir/third_party/install/mkldnn/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/Paddle/build/paddle_inference_install_dir/third_party/install/mklml/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/Paddle/build/paddle/fluid/pybind/:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/home/paddle-deepmd/source/build:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/deepmd-kit/source/build:$LD_LIBRARY_PATH

