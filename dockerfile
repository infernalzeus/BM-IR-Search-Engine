FROM python:3.9

WORKDIR /app

COPY requirements.txt .
COPY setup.sh .
RUN sh setup.sh

COPY . .

# run script
CMD ["python", "app.py", "--port", "5005"]
