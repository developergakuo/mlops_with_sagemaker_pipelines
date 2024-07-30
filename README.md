## MLOps with SageMaker Pipelines
This repository contains two Jupyter notebooks that demonstrate the use of Amazon SageMaker Pipelines for building and deploying machine learning models. The two pipelines covered are:

- Training Pipeline
- Inference Pipeline

### Training Pipeline
The training pipeline is responsible for the following steps:

- Data Processing: Preprocesses the input data.
- Model Training: Trains the machine learning model using the processed data.
- Model Evaluation: Evaluates the performance of the trained model.
- Model Registration: Registers the trained and evaluated model into the SageMaker Model Registry.

Notebook: `sagemaker-pipelines-train-pipeline.ipynb`

#### Key Steps:
- Environment Setup: Initializes the necessary SageMaker session and roles.
- Data Processing Step: Uses SKLearnProcessor to preprocess the data.
- Training Step: Trains the model using an Estimator.
- Evaluation Step: Evaluates the trained model and calculates metrics.
- Model Registration Step: Registers the model in the SageMaker Model Registry with associated metrics.


### Inference Pipeline
The inference pipeline is responsible for the following steps:

- Model Retrieval: Retrieves the latest approved model from the SageMaker Model Registry.
- Batch Transform Job: Uses the retrieved model to perform batch inference on a given dataset.
Notebook: `sagemaker-pipelines-inference-pipeline.ipynb`

#### Key Steps:
- Environment Setup: Initializes the necessary SageMaker session and roles.
- Model Retrieval: Lists and retrieves the latest approved model package.
- Model Deployment: Deploys the model to a SageMaker endpoint (if needed for real-time inference).
- Batch Transform Step: Defines a batch transform job that processes input data in S3 and outputs predictions to S3.

### Usage
To use these pipelines, follow the steps below:

- Clone the Repository:

`git clone https://github.com/developergakuo/mlops_with_sagemaker_pipelines.git`
- `cd mlops_with_sagemaker_pipelines`
- Open the Notebooks:
   Open the Jupyter notebooks in your preferred environment (e.g., Jupyter Notebook, JupyterLab, or SageMaker Studio).

- Run the Training Pipeline:
Execute the cells in `sagemaker-pipelines-train-pipeline.ipynb` to preprocess data, train the model, evaluate it, and register the model.

- Run the Inference Pipeline:
Execute the cells in `sagemaker-pipelines-inference-pipeline.ipynb` to retrieve the registered model and perform batch inference.

#### Notes
Ensure that you have the necessary AWS credentials and permissions to create and manage SageMaker resources.
Modify the S3 paths and other parameters as needed to fit your specific use case and data locations.
