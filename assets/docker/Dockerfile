from python:3

ENV TZ=America/Chicago

# Update timezone
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# TA-Lib
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
  tar -xvzf ta-lib-0.4.0-src.tar.gz && \
  cd ta-lib/ && \
  ./configure --prefix=/usr && \
  make && \
  make install

RUN rm -R ta-lib ta-lib-0.4.0-src.tar.gz

# Update Python dependencies
copy requirements.txt /tmp
run pip install -r /tmp/requirements.txt

# Install OS dependencies
run apt-get update
run apt-get install -y graphviz
run apt-get purge -y git

# User conveniences
run echo "alias ll='ls -lah --group-directories-first --color=auto'" > /root/.bashrc

# Set working directory
workdir /app

# Default command
cmd ["/bin/bash"]
