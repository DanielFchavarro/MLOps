FROM python:3.9
COPY . .
RUN pip3 install --no-cache-dir --upgrade -r requirements.txt
CMD ["uvicorn", "code.main:app", "--host", "0.0.0.0", "--port", "80"]