# Use the official Python image as the base image
FROM python:3.8

# Set the working directory inside the container
WORKDIR /pythonProject

# Copy the Python script and other necessary files into the container
COPY Code01.py .


# Install required dependencies
RUN pip install streamlit pandas seaborn matplotlib scikit-learn

# Expose port 8501
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run","--server.address","0.0.0.0", "Code01.py"]
