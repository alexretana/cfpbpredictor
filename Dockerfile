FROM ecoron/python36-sklearn:latest

WORKDIR /usr/src/app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

ENTRYPOINT ["python"]

CMD ["flask_app.py"]