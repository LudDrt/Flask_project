# syntax=docker/dockerfile:1
FROM python:latest

# Accept arguments
ARG arg_pwd

# Environment variables
ENV DB_HOST="mdb"
ENV DB_USER="root"
ENV DB_PASSWORD=$arg_pwd
ENV DB="little_flask"

# copy the requirements file into the image
COPY ./requirements.txt /src/

# switch wirking directory
WORKDIR /src

# Install app dependencies
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y
RUN pip install -r requirements.txt

# copying files
COPY ./static/* /src/static/
COPY ./templates/* /src/templates/
COPY ./database.py /src/
COPY ./P3_Ludo.py /src/

# Expose port
EXPOSE 5000

# Starting the application
CMD ["python3", "P3_Ludo.py"]
