FROM python:3.7

RUN pip3 install torchvision click nose
RUN pip3 install fs scipy numpy attrs
RUN mkdir /app
COPY cache-datasets.py /root/cache-datasets.py
RUN python /root/cache-datasets.py
COPY . /app
WORKDIR /app
RUN python setup.py develop
RUN nosetests
