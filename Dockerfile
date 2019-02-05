FROM tiangolo/uwsgi-nginx-flask:python3.7 AS core
COPY ./app/requirements.txt /app/
COPY ./app/fix.sh /app/
RUN pip install -r /app/requirements.txt
RUN /app/fix.sh

FROM core
COPY ./app /app
RUN git config --global user.email "sandbox@kronos.com"
RUN git config --global user.name "SandBox"