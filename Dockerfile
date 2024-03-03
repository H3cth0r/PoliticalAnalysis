FROM python:3.12

WORKDIR /python-docker

ENV HOST 0.0.0.0

COPY requirements.txt requirements.txt
RUN pip3 install --upgrade pip && \
    pip3 install -r requirements.txt

COPY . .

EXPOSE 8080

CMD [ "python", "app.py"]
