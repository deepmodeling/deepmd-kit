FROM dp-harbor-registry.us-east-1.cr.aliyuncs.com/eflops/pytorch2.0:py3.10
WORKDIR /data/deepmd-pytorch
RUN apt update && apt install -y --no-install-recommends openssh-server sudo
RUN sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/g' /etc/ssh/sshd_config
RUN pip install ipykernel
COPY . .
RUN pip install .
