FROM python:3.7

# These lines install the minimum required to
# run the data-download script. By doing this
# early, hopefully we avoid running this very
# slow couple of lines very often, because it
# would start to get a bit frustrating if you
# had to re-download for every build and test
# cycle.
RUN pip3 install fs torchvision click scipy
RUN mkdir /app
COPY cache-datasets.py /root/cache-datasets.py
RUN python /root/cache-datasets.py

RUN pip3 install torchvision torch click nose fs scipy numpy attrs pylint flake8 mypy
RUN pip3 install matplotlib
COPY . /app
WORKDIR /app
RUN python setup.py develop
RUN make check
