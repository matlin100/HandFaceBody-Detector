# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory in the container
WORKDIR /Users/yechezkelmatlin/PycharmProjects/track me/src/app.py

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r ../requirements.txt

# Make port 4000 available to the world outside this container
EXPOSE 4000

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["python", "app.py"]
