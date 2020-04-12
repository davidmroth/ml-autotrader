from python:3

ENV TZ=America/Chicago

# Update timezone
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Update Python dependencies
copy requirements.txt /tmp
run pip install -r /tmp/requirements.txt

# Install OS dependencies
run apt update
run apt install graphviz
run apt purge git -y

# User conveniences
run echo "alias ll='ls -lah --group-directories-first --color=auto'" > /root/.bashrc

# Set working directory
workdir /app

# Default command
cmd ["/bin/bash"]
