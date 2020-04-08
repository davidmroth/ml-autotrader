from python:3

copy requirements.txt /tmp
run pip install -r /tmp/requirements.txt
run apt purge git -y

run echo "alias ll='ls -lah --group-directories-first --color=auto'" > /root/.bashrc
workdir /app
cmd ["/bin/bash"]
