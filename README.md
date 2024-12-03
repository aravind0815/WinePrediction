# Wine Quality Prediction
## CS 643 - Cloud Computing

## Overview
This project focuses on predicting wine quality using Apache Spark, leveraging AWS technologies for a distributed computing environment. The solution utilizes Spark’s machine learning capabilities to train and evaluate a Random Forest model on a dataset of wine quality. The pipeline is implemented and executed on a cluster of EC2 instances, with an optional Dockerized setup for enhanced portability.

---

## Getting Started with AWS
### AWS Academy Setup
1. **Registration**: Access the AWS Academy course and register using your NJIT email.
2. **Account Setup**: Create an AWS account using the invitation link provided in the course.
3. **Access Details**: Retrieve your AWS access key, secret key, and session token from the AWS Academy Learner Lab.
4. **SSH Key**: Secure the PEM file (`key_pair.pem`) for accessing EC2 instances.

---

## Environment Setup on AWS
### EMR and EC2 Configuration
1. **Create a Cluster**:
   - Go to the EMR console and click **Create Cluster**.
   - Select applications like **Hadoop** and **Spark** during setup.
   - Choose instance configurations:
     - **1 Master Node**
     - **4 Task Nodes**
   - Assign the EC2 key pair (`key_pair.pem`) for SSH access.
2. **Cluster Configuration**:
   - Name the cluster and select manual termination.
   - Enable logging and debugging for better monitoring (optional).
   - Use default IAM roles (`EMR_EC2_DefaultRole` and `EMR_AutoScaling_DefaultRole`).
3. **S3 Bucket**:
   - Create an S3 bucket (e.g., `s3://wineprecdit`) to store datasets and scripts.
4. **SSH into Cluster**:
   ```bash
   ssh -i ~/key_pair.pem hadoop@<master-node-dns>
   ```

---

## Execution of Machine Learning Pipeline

### Steps Performed by the Script:
1. **Spark Session Initialization**: Sets up a Spark session to execute the pipeline.
2. **Data Loading**: Reads training and validation datasets from the S3 bucket.
3. **Data Preprocessing**: Normalizes numeric columns and removes unnecessary quotes.
4. **Feature Engineering**:
   - Assembles feature columns for the prediction model.
   - Indexes the `quality` column as the target label.
5. **Model Training**:
   - Configures and trains a Random Forest Classifier.
   - Evaluates the model on training data using accuracy and F1 score metrics.
6. **Hyperparameter Tuning**:
   - Performs cross-validation to fine-tune parameters.
7. **Final Evaluation**:
   - Applies the best model on the validation dataset.
   - Outputs the final accuracy and weighted F1 score.

---

### Execution Without Docker
1. SSH into the EMR cluster:
   ```bash
   ssh -i ~/key_pair.pem hadoop@<master-node-dns>
   ```
2. Switch to the root user:
   ```bash
   sudo su
   ```
3. Install necessary dependencies:
   ```bash
   pip install numpy --user
   ```
4. Train the model:
   ```bash
   spark-submit s3://wineprecdit/training.py
   ```
5. Validate predictions:
   ```bash
   spark-submit s3://winecluster/prediction.py s3://winecluster/ValidationDataset.csv
   ```

---

## Dockerized Execution

### Docker Setup
1. Start Docker on the EC2 instance:
   ```bash
   sudo systemctl start docker
   sudo systemctl enable docker
   ```
2. Pull Docker images:
   ```bash
   sudo docker pull aravind0815/winecluster:train
   sudo docker pull aravind0815/winecluster:predict
   ```

### Running Docker Containers
1. Train the model using the training image:
   ```bash
   sudo docker run -v /home/ec2-user/:/job aravind0815/winecluster:train
   ```
2. Validate predictions using the prediction image:
   ```bash
   sudo docker run -v /home/ec2-user/:/job aravind0815/winecluster:predict ValidationDataset.csv
   ```

---

## Results
### Without Docker
- **Accuracy**: 0.96875
- **Weighted F1 Score**: 0.954190
---

## Docker Configuration and Commands
- **Build Image**:
   ```bash
   docker build -t aravind0815/winecluster:latest .
   ```
- **Push to Docker Hub**:
   ```bash
   docker push aravind0815/winecluster:latest
   ```
- **Run Docker Container**:
   ```bash
   sudo docker run -v /home/ec2-user/:/job aravind0815/winecluster:latest
   ```

---

## Cluster Creation Guide
1. Navigate to the EMR console and click **Create Cluster**.
2. Provide a name and select **manual termination** for better control.
3. Use **1 core node** and **4 task nodes** for distributed processing.
4. Assign the PEM file for security configuration.
5. Upload scripts and datasets to an S3 bucket (e.g., `s3://winecluster`).

---

## Project Repository
**GitHub:** [Wine Prediction Repository](https://github.com/aravind0815/WinePrediction)

---

### Docker Configuration and Usage

- **Build Docker Image**:
   ```bash
   docker build -t aravind0815/winecluster:prediction .
   ```

- **Push to Docker Hub**:
   ```bash
   docker push aravind0815/winecluster:prediction
   ```

- **Manage Docker Services**:
   - Start, enable, and verify Docker using `systemctl`:
     ```bash
     sudo systemctl start docker
     sudo systemctl enable docker
     ```

- **Run Docker Container**:
   ```bash
   sudo docker run aravind0815/winecluster:prediction
   ```

### Additional Docker Information
- **Docker Image Details**: [Visit Docker Hub](https://hub.docker.com/repository/docker/aravind0815/winecluster/general)

---

### Final Model Performance
- **Accuracy**: 0.967
- **F1 Score**: 0.954

---

### **Student Details**
- **Name**: Aravind Kalyan Sivakumar
- **UCID**: as4588
- **Contact**: as4588@njit.edu
