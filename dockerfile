FROM python:3.9

WORKDIR /app

COPY setup.sh .
RUN sh setup.sh

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY index .
RUN ls
COPY app.py .

# run script
CMD ["python", "app.py"]
