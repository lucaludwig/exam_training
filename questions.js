const quizData = [
    {
        "question": "A company plans to deploy an ML model for production inference on an Amazon SageMaker endpoint. The average inference payload size will vary from 100 MB to 300 MB. Inference requests must be processed in 60 minutes or less.\nWhich SageMaker inference option will meet these requirements?",
        "options": [
            "A. Serverless inference",
            "B. Asynchronous inference",
            "C. Real-time inference",
            "D. Batch transform"
        ],
        "answer": "B"
    },
    {
        "question": "An ML engineer needs to encrypt all data in transit when an ML training job runs. The ML engineer must ensure that encryption in transit is applied to processes that Amazon SageMaker uses during the training job.\nWhich solution will meet these requirements?",
        "options": [
            "A. Encrypt communication between nodes for batch processing.",
            "B. Encrypt communication between nodes in a training cluster.",
            "C. Specify an AWS Key Management Service (AWS KMS) key during creation of the training job request.",
            "D. Specify an AWS Key Management Service (AWS KMS) key during creation of the SageMaker domain."
        ],
        "answer": "B"
    },
    {
        "question": "An ML engineer is developing a fraud detection model by using the Amazon SageMaker XGBoost algorithm. The model classifies transactions as either fraudulent or legitimate. During testing, the model excels at identifying fraud in the training dataset. However, the model is inefficient at identifying fraud in new and unseen transactions.\nWhat should the ML engineer do to improve the fraud detection for new transactions?",
        "options": [
            "A. Increase the learning rate.",
            "B. Remove some irrelevant features from the training dataset.",
            "C. Increase the value of the max_depth hyperparameter.",
            "D. Decrease the value of the max_depth hyperparameter."
        ],
        "answer": "D"
    },
    {
        "question": "A company is using ML to predict the presence of a specific weed in a farmer's field. The company is using the Amazon SageMaker linear learner built-in algorithm with a value of multiclass_classifier for the predictor_type hyperparameter.\nWhat should the company do to MINIMIZE false positives?",
        "options": [
            "A. Set the value of the weight decay hyperparameter to zero.",
            "B. Increase the number of training epochs.",
            "C. Increase the value of the target_precision hyperparameter.",
            "D. Change the value of the predictor_type hyperparameter to regressor."
        ],
        "answer": "C"
    },
    {
        "question": "A company has an ML model that needs to run one time each night to predict stock values. The model input is 3 MB of data that is collected during the current day. The model produces the predictions for the next day. The prediction process takes less than 1 minute to finish running.\nHow should the company deploy the model on Amazon SageMaker to meet these requirements?",
        "options": [
            "A. Use a multi-model serverless endpoint. Enable caching.",
            "B. Use an asynchronous inference endpoint. Set the InitialInstanceCount parameter to 0.",
            "C. Use a real-time endpoint. Configure an auto scaling policy to scale the model to 0 when the model is not in use.",
            "D. Use a serverless inference endpoint. Set the MaxConcurrency parameter to 1."
        ],
        "answer": "D"
    },
    {
        "question": "An ML engineer has developed a binary classification model outside of Amazon SageMaker. The ML engineer needs to make the model accessible to a SageMaker Canvas user for additional tuning.\nThe model artifacts are stored in an Amazon S3 bucket. The ML engineer and the Canvas user are part of the same SageMaker domain.\nWhich combination of requirements must be met so that the ML engineer can share the model with the Canvas user? (Choose two.)",
        "options": [
            "A. The ML engineer and the Canvas user must be in separate SageMaker domains.",
            "B. The Canvas user must have permissions to access the S3 bucket where the model artifacts are stored.",
            "C. The model must be registered in the SageMaker Model Registry.",
            "D. The ML engineer must host the model on AWS Marketplace.",
            "E. The ML engineer must deploy the model to a SageMaker endpoint."
        ],
        "answer": "BC"
    },
    {
        "question": "A company has a large, unstructured dataset. The dataset includes many duplicate records across several key attributes.\nWhich solution on AWS will detect duplicates in the dataset with the LEAST code development?",
        "options": [
            "A. Use Amazon Mechanical Turk jobs to detect duplicates.",
            "B. Use Amazon QuickSight ML Insights to build a custom deduplication model.",
            "C. Use Amazon SageMaker Data Wrangler to pre-process and detect duplicates.",
            "D. Use the AWS Glue FindMatches transform to detect duplicates."
        ],
        "answer": "D"
    },
    {
        "question": "A company has deployed an XGBoost prediction model in production to predict if a customer is likely to cancel a subscription. The company uses Amazon SageMaker Model Monitor to detect deviations in the F1 score.\nDuring a baseline analysis of model quality, the company recorded a threshold for the F1 score.\nAfter several months of no change, the model's F1 score decreases significantly.\nWhat could be the reason for the reduced F1 score?",
        "options": [
            "A. Concept drift occurred in the underlying customer data that was used for predictions.",
            "B. The model was not sufficiently complex to capture all the patterns in the original baseline data.",
            "C. The original baseline data had a data quality issue of missing values.",
            "D. Incorrect ground truth labels were provided to Model Monitor during the calculation of the baseline."
        ],
        "answer": "A"
    },
    {
        "question": "A company uses Amazon SageMaker Studio to develop an ML model. The company has a single SageMaker Studio domain. An ML engineer needs to implement a solution that provides an automated alert when SageMaker compute costs reach a specific threshold.\nWhich solution will meet these requirements?",
        "options": [
            "A. Add resource tagging by editing the SageMaker user profile in the SageMaker domain. Configure AWS Cost Explorer to send an alert when the threshold is reached.",
            "B. Add resource tagging by editing the SageMaker user profile in the SageMaker domain. Configure AWS Budgets to send an alert when the threshold is reached.",
            "C. Add resource tagging by editing each user's IAM profile. Configure AWS Cost Explorer to send an alert when the threshold is reached.",
            "D. Add resource tagging by editing each user's IAM profile. Configure AWS Budgets to send an alert when the threshold is reached."
        ],
        "answer": "B"
    },
    {
        "question": "A company is using an Amazon Redshift database as its single data source. Some of the data is sensitive.\nA data scientist needs to use some of the sensitive data from the database. An ML engineer must give the data scientist access to the data without transforming the source data and without storing anonymized data in the database.\nWhich solution will meet these requirements with the LEAST implementation effort?",
        "options": [
            "A. Configure dynamic data masking policies to control how sensitive data is shared with the data scientist at query time.",
            "B. Create a materialized view with masking logic on top of the database. Grant the necessary read permissions to the data scientist.",
            "C. Unload the Amazon Redshift data to Amazon S3. Use Amazon Athena to create schema-on-read with masking logic. Share the view with the data scientist.",
            "D. Unload the Amazon Redshift data to Amazon S3. Create an AWS Glue job to anonymize the data. Share the dataset with the data scientist."
        ],
        "answer": "A"
    },
    {
        "question": "A company wants to build a real-time analytics application that uses streaming data from social media. An ML engineer must implement a solution that ingests and transforms 5 GB of data each minute. The solution also must load the data into a data store that supports fast queries for the real-time analytics. Which solution will meet these requirements?",
        "options": [
            "A. Use Amazon EventBridge to ingest the social media data. Use AWS Glue to transform the data. Store the transformed data in Amazon ElastiCache (Memcached).",
            "B. Use Amazon Simple Queue Service (Amazon SQS) to ingest the social media data. Use AWS Lambda to transform the data. Store the transformed data in Amazon S3.",
            "C. Use Amazon Simple Notification Service (Amazon SNS) to ingest the social media data. Use Amazon EMR to transform the data. Store the transformed data in Amazon RDS.",
            "D. Use Amazon Kinesis Data Streams to ingest the social media data. Use Amazon Managed Service for Apache Flink to transform the data. Store the transformed data in Amazon DynamoDB."
        ],
        "answer": "D",
        "explanation": "Amazon Kinesis Data Streams is designed for high-throughput ingestion of streaming data such as social media feeds. Amazon Managed Service for Apache Flink enables real-time transformations on that data. Amazon DynamoDB provides low-latency reads and writes, making it suitable for fast queries in real-time analytics. This combination fully meets the scale and speed requirements."
    },
    {
        "question": "Hotspot Question\nAn ML engineer must choose the appropriate Amazon SageMaker algorithm to solve specific AI problems.\nSelect the correct SageMaker built-in algorithm from the following list for each use case.\nEach algorithm should be selected one time.\n- Random Cut Forest (RCF) algorithm\n- Semantic segmentation algorithm\n- Sequence-to-Sequence (seq2seq) algorithm\n\n1. Summarize the text of a research paper.\n2. Scan every pixel of an image to help self-driving cars identify objects in their path.\n3. Identify abnormal data points in a dataset.",
        "options": [
            "A. 1: Sequence-to-Sequence, 2: Semantic segmentation, 3: Random Cut Forest",
            "B. 1: Semantic segmentation, 2: Random Cut Forest, 3: Sequence-to-Sequence",
            "C. 1: Random Cut Forest, 2: Sequence-to-Sequence, 3: Semantic segmentation"
        ],
        "answer": "A"
    },
    {
        "question": "A company has an application that uses different APIs to generate embeddings for input text. The company needs to implement a solution to automatically rotate the API tokens every 3 months.\nWhich solution will meet this requirement?",
        "options": [
            "A. Store the tokens in AWS Secrets Manager. Create an AWS Lambda function to perform the rotation.",
            "B. Store the tokens in AWS Systems Manager Parameter Store. Create an AWS Lambda function to perform the rotation.",
            "C. Store the tokens in AWS Key Management Service (AWS KMS). Use an AWS managed key to perform the rotation.",
            "D. Store the tokens in AWS Key Management Service (AWS KMS). Use an AWS owned key to perform the rotation."
        ],
        "answer": "A"
    },
    {
        "question": "A company needs to give its ML engineers appropriate access to training data. The ML engineers must access training data from only their own business group. The ML engineers must not be allowed to access training data from other business groups.\nThe company uses a single AWS account and stores all the training data in Amazon S3 buckets.\nAll ML model training occurs in Amazon SageMaker.\nWhich solution will provide the ML engineers with the appropriate access?",
        "options": [
            "A. Enable S3 bucket versioning.",
            "B. Configure S3 Object Lock settings for each user.",
            "C. Add cross-origin resource sharing (CORS) policies to the S3 buckets.",
            "D. Create IAM policies. Attach the policies to IAM users or IAM roles."
        ],
        "answer": "D"
    },
    {
        "question": "A company is planning to create an internal-only chat interface to help employees handle customer queries. Currently, the employees need to refer to a massive knowledge base of internal documents to address customer issues. The new solution must be serverless. Which combination of steps will meet these requirements?",
        "options": [
            "A. Set up Amazon Bedrock with the Anthropic Claude foundation model.",
            "B. Set up Amazon SageMaker JumpStart with the Llama foundation model.",
            "C. Use Amazon EC2 instances with Amazon API Gateway to invoke the model API.",
            "D. Use AWS Lambda functions with Amazon API Gateway to invoke the model API.",
            "E. Use an Amazon S3 bucket to store vector database dumps and embeddings.",
            "F. Use Amazon RDS for MySQL to store vector database dumps and embeddings."
        ],
        "answer": "ADE",
        "explanation": "To build a serverless internal chat interface, you can use Amazon Bedrock with a foundation model like Claude, invoke the model API through AWS Lambda with Amazon API Gateway, and store embeddings in a vector database format using Amazon S3. This avoids server management, ensures scalability, and leverages serverless components end-to-end."
    },
    {
        "question": "A company is planning to use Amazon SageMaker to make classification ratings that are based on images. The company has 6 GB of training data that is stored on an Amazon FSx for NetApp ONTAP system virtual machine (SVM). The SVM is in the same VPC as SageMaker.\nAn ML engineer must make the training data accessible for ML models that are in the SageMaker environment.\nWhich solution will meet these requirements?",
        "options": [
            "A. Mount the FSx for ONTAP file system as a volume to the SageMaker Instance.",
            "B. Create an Amazon S3 bucket. Use Mountpoint for Amazon S3 to link the S3 bucket to the FSx for ONTAP file system.",
            "C. Create a catalog connection from SageMaker Data Wrangler to the FSx for ONTAP file system.",
            "D. Create a direct connection from SageMaker Data Wrangler to the FSx for ONTAP file system."
        ],
        "answer": "A"
    },
    {
        "question": "An ML engineer needs to merge and transform data from two sources to retrain an existing ML model. One data source consists of .csv files that are stored in an Amazon S3 bucket. Each .csv file consists of millions of records. The other data source is an Amazon Aurora DB cluster.\nThe result of the merge process must be written to a second S3 bucket. The ML engineer needs to perform this merge-and-transform task every week.\nWhich solution will meet these requirements with the LEAST operational overhead?",
        "options": [
            "A. Create a transient Amazon EMR cluster every week. Use the cluster to run an Apache Spark job to merge and transform the data.",
            "B. Create a weekly AWS Glue job that uses the Apache Spark engine. Use DynamicFrame native operations to merge and transform the data.",
            "C. Create an AWS Lambda function that runs Apache Spark code every week to merge and transform the data. Configure the Lambda function to connect to the initial S3 bucket and the DB cluster.",
            "D. Create an AWS Batch job that runs Apache Spark code on Amazon EC2 instances every week.Configure the Spark code to save the data from the EC2 instances to the second S3 bucket."
        ],
        "answer": "B"
    },
    {
        "question": "A company is developing a new online application to gather information from customers. An ML engineer has developed a new ML model that will determine a score for each customer. The model will use the score to determine which product to display to the customer. The ML engineer needs to minimize response-time latency for the model. How should the ML engineer deploy the application in Amazon SageMaker to meet these requirements?",
        "options": [
            "A. Configure batch transform.",
            "B. Configure a real-time inference endpoint.",
            "C. Configure a serverless inference endpoint.",
            "D. Configure an asynchronous inference endpoint."
        ],
        "answer": "B",
        "explanation": "To minimize response-time latency, the ML model should be deployed to a real-time inference endpoint in Amazon SageMaker. This provides low-latency predictions by keeping the model loaded and ready to handle incoming requests, which is critical for an online application serving customers in real time."
    },
    {
        "question": "An ML engineer is developing a classification model. The ML engineer needs to use custom libraries in processing jobs, training jobs, and pipelines in Amazon SageMaker. Which solution will provide this functionality with the LEAST implementation effort?",
        "options": [
            "A. Manually install the libraries in the SageMaker containers.",
            "B. Build a custom Docker container that includes the required libraries. Host the container in Amazon Elastic Container Registry (Amazon ECR). Use the ECR image in the SageMaker jobs and pipelines.",
            "C. Create a SageMaker notebook instance to host the jobs. Create an AWS Lambda function to install the libraries on the notebook instance when the notebook instance starts. Configure the SageMaker jobs and pipelines to run on the notebook instance.",
            "D. Run code for the libraries externally on Amazon EC2 instances. Store the results in Amazon S3.Import the results into the SageMaker jobs and pipelines."
        ],
        "answer": "B",
        "explanation": "Building a custom Docker container with the required libraries and hosting it in Amazon ECR allows SageMaker jobs, training, and pipelines to consistently use the same environment. This approach minimizes manual setup, ensures portability, and provides the least ongoing implementation effort compared to repeatedly installing or managing libraries separately."
    },
    {
        "question": "A company has a large collection of chat recordings from customer interactions after a product release. An ML engineer needs to create an ML model to analyze the chat data. The ML engineer needs to determine the success of the product by reviewing customer sentiments about the product.\nWhich action should the ML engineer take to complete the evaluation in the LEAST amount of time?",
        "options": [
            "A. Use Amazon Rekognition to analyze sentiments of the chat conversations.",
            "B. Train a Naive Bayes classifier to analyze sentiments of the chat conversations.",
            "C. Use Amazon Comprehend to analyze sentiments of the chat conversations.",
            "D. Use random forests to classify sentiments of the chat conversations."
        ],
        "answer": "C"
    },
    {
        "question": "Case Study\nA company is building a web-based AI application by using Amazon SageMaker. The application will provide the following capabilities and features: ML experimentation, training, a central model registry, model deployment, and model monitoring.\nThe application must ensure secure and isolated use of training data during the ML lifecycle. The training data is stored in Amazon S3.\nThe company needs to run an on-demand workflow to monitor bias drift for models that are deployed to real-time endpoints from the application.\nWhich action will meet this requirement?",
        "options": [
            "A. Configure the application to invoke an AWS Lambda function that runs a SageMaker Clarify job.",
            "B. Invoke an AWS Lambda function to pull the sagemaker-model-monitor-analyzer built-in SageMaker image.",
            "C. Use AWS Glue Data Quality to monitor bias.",
            "D. Use SageMaker notebooks to compare the bias."
        ],
        "answer": "A"
    },
    {
        "question": "A company is training a large language model (LLM) by using on-premises infrastructure. A live conversational engine uses the LLM to help customers find real-time insights in credit card data.\nAn ML engineer must implement a solution to train and deploy the LLM on Amazon SageMaker.\nWhich solution will meet these requirements?",
        "options": [
            "A. Use SageMaker Training Compiler to train the LLM. Deploy the LLM by using SageMaker real-time inference.",
            "B. Use SageMaker with deep learning containers for large model inference to train the LLM. Deploy the LLM by using SageMaker real-time inference.",
            "C. Use SageMaker Notebook Jobs to train the LLM. Deploy the LLM by using SageMaker Asynchronous Inference.",
            "D. Use SageMaker Studio to train the LLM. Deploy the LLM by using SageMaker batch transform."
        ],
        "answer": "A",
        "explanation": "SageMaker Training Compiler accelerates training of large models like LLMs by optimizing GPU utilization, making it suitable for efficient large-scale training. For deployment of a live conversational engine that requires real-time responses, the correct choice is a SageMaker real-time inference endpoint. This combination meets both training and deployment requirements effectively."
    },
    {
        "question": "A data scientist is evaluating different binary classification models. A false positive result is 5 times more expensive (from a business perspective) than a false negative result.\nThe models should be evaluated based on the following criteria:\n1) Must have a recall rate of at least 80%\n2) Must have a false positive rate of 10% or less\n3) Must minimize business costs\nAfter creating each binary classification model, the data scientist generates the corresponding confusion matrix.\nWhich confusion matrix represents the model that satisfies the requirements?",
        "options": [
            "A. TN = 91, FP = 9, FN = 22, TP = 78",
            "B. TN = 99, FP = 1, FN = 21, TP = 79",
            "C. TN = 96, FP = 4, FN = 10, TP = 90",
            "D. TN = 98, FP = 2, FN = 18, TP = 82"
        ],
        "answer": "D",
        "explanation": "Recall = TP / (TP + FN). False Positive Rate (FPR) = FP / (FP + TN). Cost = 5 * FP + FN.\nModel D:\nRecall = 82 / (82 + 18) = 0.82 (82% >= 80%)\nFPR = 2 / (2 + 98) = 0.02 (2% <= 10%)\nCost = 5 * 2 + 18 = 28 (Lowest cost compared to others)"
    },
    {
        "question": "A company has historical data that shows whether customers needed long-term support from company staff. The company needs to develop an ML model to predict whether new customers will require long-term support.\nWhich modeling approach should the company use to meet this requirement?",
        "options": [
            "A. Anomaly detection",
            "B. Linear regression",
            "C. Logistic regression",
            "D. Semantic segmentation"
        ],
        "answer": "C"
    },
    {
        "question": "A credit card company has a fraud detection model in production on an Amazon SageMaker endpoint. The company develops a new version of the model. The company needs to assess the new model's performance by using live data and without affecting production end users.\nWhich solution will meet these requirements?",
        "options": [
            "A. Set up SageMaker Debugger and create a custom rule.",
            "B. Set up blue/green deployments with all-at-once traffic shifting.",
            "C. Set up blue/green deployments with canary traffic shifting.",
            "D. Set up shadow testing with a shadow variant of the new model."
        ],
        "answer": "D"
    },
    {
        "question": "A company is planning to use Amazon Redshift ML in its primary AWS account. The source data is in an Amazon S3 bucket in a secondary account.\nAn ML engineer needs to set up an ML pipeline in the primary account to access the S3 bucket in the secondary account. The solution must not require public IPv4 addresses.\nWhich solution will meet these requirements?",
        "options": [
            "A. Provision a Redshift cluster and Amazon SageMaker Studio in a VPC with no public access enabled in the primary account. Create a VPC peering connection between the accounts. Update the VPC route tables to remove the route to 0.0.0.0/0.",
            "B. Provision a Redshift cluster and Amazon SageMaker Studio in a VPC with no public access enabled in the primary account. Create an AWS Direct Connect connection and a transit gateway. Associate the VPCs from both accounts with the transit gateway. Update the VPC route tables to remove the route to 0.0.0.0/0.",
            "C. Provision a Redshift cluster and Amazon SageMaker Studio in a VPC in the primary account. Create an AWS Site-to-Site VPN connection with two encrypted IPsec tunnels between the accounts. Set up interface VPC endpoints for Amazon S3.",
            "D. Provision a Redshift cluster and Amazon SageMaker Studio in a VPC in the primary account.Create an S3 gateway endpoint. Update the S3 bucket policy to allow IAM principals from the primary account. Set up interface VPC endpoints for SageMaker and Amazon Redshift."
        ],
        "answer": "D"
    },
    {
        "question": "An ML engineer is training a simple neural network model. The ML engineer tracks the performance of the model over time on a validation dataset. The model's performance improves substantially at first and then degrades after a specific number of epochs.\nWhich solutions will mitigate this problem? (Choose two.)",
        "options": [
            "A. Enable early stopping on the model.",
            "B. Increase dropout in the layers.",
            "C. Increase the number of layers.",
            "D. Increase the number of neurons.",
            "E. Investigate and reduce the sources of model bias."
        ],
        "answer": "AB"
    },
    {
        "question": "An ML engineer needs to use data with Amazon SageMaker Canvas to train an ML model. The data is stored in Amazon S3 and is complex in structure. The ML engineer must use a file format that minimizes processing time for the data.\nWhich file format will meet these requirements?",
        "options": [
            "A. CSV files compressed with Snappy",
            "B. JSON objects in JSONL format",
            "C. JSON files compressed with gzip",
            "D. Apache Parquet files"
        ],
        "answer": "D"
    },
    {
        "question": "An ML engineer has an Amazon Comprehend custom model in Account A in the us-east-1 Region. The ML engineer needs to copy the model to Account B in the same Region.\nWhich solution will meet this requirement with the LEAST development effort?",
        "options": [
            "A. Use Amazon S3 to make a copy of the model. Transfer the copy to Account B.",
            "B. Create a resource-based IAM policy. Use the Amazon Comprehend ImportModel API operation to copy the model to Account B.",
            "C. Use AWS DataSync to replicate the model from Account A to Account B.",
            "D. Create an AWS Site-to-Site VPN connection between Account A and Account B to transfer the model."
        ],
        "answer": "B"
    },
    {
        "question": "A machine learning engineer is preparing a data frame for a supervised learning task with the Amazon SageMaker Linear Learner algorithm. The ML engineer notices the target label classes are highly imbalanced and multiple feature columns contain missing values. The proportion of missing values across the entire data frame is less than 5%.\nWhat should the ML engineer do to minimize bias due to missing values?",
        "options": [
            "A. Replace each missing value by the mean or median across non-missing values in same row.",
            "B. Delete observations that contain missing values because these represent less than 5% of the data.",
            "C. Replace each missing value by the mean or median across non-missing values in the same column.",
            "D. For each feature, approximate the missing values using supervised learning based on other features."
        ],
        "answer": "D",
        "explanation": "Use supervised learning to predict missing values based on the values of other features. Different supervised learning approaches might have different performances, but any properly implemented supervised learning approach should provide the same or better approximation than mean or median approximation, as proposed in responses A and C. Supervised learning applied to the imputation of missing values is an active field of research."
    },
    {
        "question": "A company needs an AWS solution that will automatically create versions of ML models as the models are created.\nWhich solution will meet this requirement?",
        "options": [
            "A. Amazon Elastic Container Registry (Amazon ECR)",
            "B. Model packages from Amazon SageMaker Marketplace",
            "C. Amazon SageMaker ML Lineage Tracking",
            "D. Amazon SageMaker Model Registry"
        ],
        "answer": "D"
    },
    {
        "question": "A company uses 10 Reserved Instances of accelerated instance types to serve the current version of an ML model. An ML engineer needs to deploy a new version of the model to an Amazon SageMaker real-time inference endpoint.\nThe solution must use the original 10 instances to serve both versions of the model. The solution also must include one additional Reserved Instance that is available to use in the deployment process. The transition between versions must occur with no downtime or service interruptions.\nWhich solution will meet these requirements?",
        "options": [
            "A. Configure a blue/green deployment with all-at-once traffic shifting.",
            "B. Configure a blue/green deployment with canary traffic shifting and a size of 10%.",
            "C. Configure a shadow test with a traffic sampling percentage of 10%.",
            "D. Configure a rolling deployment with a rolling batch size of 1."
        ],
        "answer": "D"
    },
    {
        "question": "Hotspot Question\nAn ML engineer is building a generative AI application on Amazon Bedrock by using large language models (LLMs).\nSelect the correct generative AI term from the following list for each description.\n- Text representation of basic units of data processed by LLMs\n- High-dimensional vectors that contain the semantic meaning of text\n- Enrichment of information from additional data sources to improve a generated response",
        "options": [
            "A. Token, Embedding, Retrieval Augmented Generation (RAG)",
            "B. Embedding, Token, Temperature",
            "C. Temperature, RAG, Token",
            "D. RAG, Embedding, Token"
        ],
        "answer": "A"
    },
    {
        "question": "A company is using Amazon SageMaker to create ML models. The company's data scientists need fine-grained control of the ML workflows that they orchestrate. The data scientists also need the ability to visualize SageMaker jobs and workflows as a directed acyclic graph (DAG). The data scientists must keep a running history of model discovery experiments and must establish model governance for auditing and compliance verifications.\nWhich solution will meet these requirements?",
        "options": [
            "A. Use AWS CodePipeline and its integration with SageMaker Studio to manage the entire ML workflows. Use SageMaker ML Lineage Tracking for the running history of experiments and for auditing and compliance verifications.",
            "B. Use AWS CodePipeline and its integration with SageMaker Experiments to manage the entire ML workflows. Use SageMaker Experiments for the running history of experiments and for auditing and compliance verifications.",
            "C. Use SageMaker Pipelines and its integration with SageMaker Studio to manage the entire ML workflows. Use SageMaker ML Lineage Tracking for the running history of experiments and for auditing and compliance verifications.",
            "D. Use SageMaker Pipelines and its integration with SageMaker Experiments to manage the entire ML workflows. Use SageMaker Experiments for the running history of experiments and for auditing and compliance verifications."
        ],
        "answer": "C"
    },
    {
        "question": "A company runs an ML model on Amazon SageMaker. The company uses an automatic process that makes API calls to create training jobs for the model. The company has new compliance rules that prohibit the collection of aggregated metadata from training jobs. Which solution will prevent SageMaker from collecting metadata from the training jobs?",
        "options": [
            "A. Opt out of metadata tracking for any training job that is submitted.",
            "B. Ensure that training jobs are running in a private subnet in a custom VPC.",
            "C. Encrypt the training data with an AWS Key Management Service (AWS KMS) customer managed key.",
            "D. Reconfigure the training jobs to use only AWS Nitro instances."
        ],
        "answer": "A",
        "explanation": "Amazon SageMaker automatically collects training job metadata, but you can opt out of metadata tracking when submitting a training job. This disables collection of aggregated metadata, ensuring compliance with rules that prohibit metadata collection."
    },
    {
        "question": "A company has deployed an ML model that detects fraudulent credit card transactions in real time in a banking application. The model uses Amazon SageMaker Asynchronous Inference. Consumers are reporting delays in receiving the inference results.\nAn ML engineer needs to implement a solution to improve the inference performance. The solution also must provide a notification when a deviation in model quality occurs.\nWhich solution will meet these requirements?",
        "options": [
            "A. Use SageMaker real-time inference for inference. Use SageMaker Model Monitor for notifications about model quality.",
            "B. Use SageMaker batch transform for inference. Use SageMaker Model Monitor for notifications about model quality.",
            "C. Use SageMaker Serverless Inference for inference. Use SageMaker Inference Recommender for notifications about model quality.",
            "D. Keep using SageMaker Asynchronous Inference for inference. Use SageMaker Inference Recommender for notifications about model quality."
        ],
        "answer": "A"
    },
    {
        "question": "A company has developed a new ML model. The company requires online model validation on 10% of the traffic before the company fully releases the model in production. The company uses an Amazon SageMaker endpoint behind an Application Load Balancer (ALB) to serve the model.\nWhich solution will set up the required online validation with the LEAST operational overhead?",
        "options": [
            "A. Use production variants to add the new model to the existing SageMaker endpoint. Set the variant weight to 0.1 for the new model. Monitor the number of invocations by using Amazon CloudWatch.",
            "B. Use production variants to add the new model to the existing SageMaker endpoint. Set the variant weight to 1 for the new model. Monitor the number of invocations by using Amazon CloudWatch.",
            "C. Create a new SageMaker endpoint. Use production variants to add the new model to the new endpoint. Monitor the number of invocations by using Amazon CloudWatch.",
            "D. Configure the ALB to route 10% of the traffic to the new model at the existing SageMaker endpoint. Monitor the number of invocations by using AWS CloudTrail."
        ],
        "answer": "A"
    },
    {
        "question": "Case Study\nAn ML engineer is developing a fraud detection model on AWS. The training dataset includes transaction logs, customer profiles, and tables from an on-premises MySQL database. The transaction logs and customer profiles are stored in Amazon S3.\nThe dataset has a class imbalance that affects the learning of the model's algorithm. Additionally, many of the features have interdependencies. The algorithm is not capturing all the desired underlying patterns in the data.\nBefore the ML engineer trains the model, the ML engineer must resolve the issue of the imbalanced data.\nWhich solution will meet this requirement with the LEAST operational effort?",
        "options": [
            "A. Use Amazon Athena to identify patterns that contribute to the imbalance. Adjust the dataset accordingly.",
            "B. Use Amazon SageMaker Studio Classic built-in algorithms to process the imbalanced dataset.",
            "C. Use AWS Glue DataBrew built-in features to oversample the minority class.",
            "D. Use the Amazon SageMaker Data Wrangler balance data operation to oversample the minority class."
        ],
        "answer": "D"
    },
    {
        "question": "A company needs to develop an ML model. The model must identify an item in an image and must provide the location of the item.\nWhich Amazon SageMaker algorithm will meet these requirements?",
        "options": [
            "A. Image classification",
            "B. XGBoost",
            "C. Object detection",
            "D. K-nearest neighbors (k-NN)"
        ],
        "answer": "C"
    },
    {
        "question": "A company is running ML models on premises by using custom Python scripts and proprietary datasets. The company is using PyTorch. The model building requires unique domain knowledge.\nThe company needs to move the models to AWS.\nWhich solution will meet these requirements with the LEAST effort?",
        "options": [
            "A. Use SageMaker built-in algorithms to train the proprietary datasets.",
            "B. Use SageMaker script mode and premade images for ML frameworks.",
            "C. Build a container on AWS that includes custom packages and a choice of ML frameworks.",
            "D. Purchase similar production models through AWS Marketplace."
        ],
        "answer": "B"
    },
    {
        "question": "A machine learning team has several large CSV datasets in Amazon S3. Historically, models built with the Amazon SageMaker Linear Learner algorithm have taken hours to train on similar-sized datasets. The team's leaders need to accelerate the training process.\nWhat can a machine learning specialist do to address this concern?",
        "options": [
            "A. Use Amazon SageMaker Pipe mode.",
            "B. Use Amazon Machine Learning to train the models.",
            "C. Use Amazon Kinesis to stream the data to Amazon SageMaker.",
            "D. Use AWS Glue to transform the CSV dataset to the JSON format."
        ],
        "answer": "A",
        "explanation": "Amazon SageMaker Pipe mode streams the data directly to the container, which improves the performance of training jobs. In Pipe mode, your training job streams data directly from Amazon S3. Streaming can provide faster start times for training jobs and better throughput. With Pipe mode, you also reduce the size of the Amazon EBS volumes for your training instances."
    },
    {
        "question": "A company is using an Amazon S3 bucket to collect data that will be used for ML workflows. The company needs to use AWS Glue DataBrew to clean and normalize the data. Which solution will meet these requirements?",
        "options": [
            "A. Create a DataBrew dataset by using the S3 path. Clean and normalize the data by using a DataBrew profile job.",
            "B. Create a DataBrew dataset by using the S3 path. Clean and normalize the data by using a DataBrew recipe job.",
            "C. Create a DataBrew dataset by using a Java Database Connectivity (JDBC) driver to connect to the S3 bucket. Clean and normalize the data by using a DataBrew profile job.",
            "D. Create a DataBrew dataset by using a Java Database Connectivity (JDBC) driver to connect to the S3 bucket. Clean and normalize the data by using a DataBrew recipe job."
        ],
        "answer": "B",
        "explanation": "The correct solution is to create a DataBrew dataset using the S3 path and then clean and normalize the data with a DataBrew recipe job. Recipes define and apply transformations to the data, while profile jobs are used only for data analysis and profiling, not cleaning."
    },
    {
        "question": "Hotspot Question\nA company stores historical data in .csv files in Amazon S3. Only some of the rows and columns in the .csv files are populated. The columns are not labeled. An ML engineer needs to prepare and store the data so that the company can use the data to train ML models.\nSelect and order the correct steps from the following list to perform this task. Each step should be selected one time or not at all. (Select and order three.)\n- Create an Amazon SageMaker batch transform job for data cleaning and feature engineering.\n- Store the resulting data back in Amazon S3.\n- Use Amazon Athena to infer the schemas and available columns.\n- Use AWS Glue crawlers to infer the schemas and available columns.\n- Use AWS Glue DataBrew for data cleaning and feature engineering.",
        "options": [
            "A. 1: Use AWS Glue crawlers..., 2: Use AWS Glue DataBrew..., 3: Store the resulting data...",
            "B. 1: Use Amazon Athena..., 2: Create an Amazon SageMaker batch transform..., 3: Store the resulting data...",
            "C. 1: Use AWS Glue crawlers..., 2: Create an Amazon SageMaker batch transform..., 3: Store the resulting data..."
        ],
        "answer": "A"
    },
    {
        "question": "A company has trained and deployed an ML model by using Amazon SageMaker. The company needs to implement a solution to record and monitor all the API call events for the SageMaker endpoint. The solution also must provide a notification when the number of API call events breaches a threshold.\nWhich solution will meet these requirements?",
        "options": [
            "A. Use SageMaker Debugger to track the inferences and to report metrics. Create a custom rule to provide a notification when the threshold is breached.",
            "B. Use SageMaker Debugger to track the inferences and to report metrics. Use the tensor_variance built-in rule to provide a notification when the threshold is breached.",
            "C. Log all the endpoint invocation API events by using AWS CloudTrail. Use an Amazon CloudWatch dashboard for monitoring. Set up a CloudWatch alarm to provide notification when the threshold is breached.",
            "D. Add the Invocations metric to an Amazon CloudWatch dashboard for monitoring. Set up a CloudWatch alarm to provide notification when the threshold is breached."
        ],
        "answer": "C"
    },
    {
        "question": "An ML engineer needs to deploy a trained model that is based on a genetic algorithm. The algorithm solves a complex problem and can take several minutes to generate predictions. When the model is deployed, the model needs to access large amounts of data to process requests. The requests can involve as much as 100 MB of data.\nWhich deployment solution will meet these requirements with the LEAST operational overhead?",
        "options": [
            "A. Deploy the model to Amazon EC2 instances in an Auto Scaling group behind an Application Load Balancer.",
            "B. Deploy the model to an Amazon SageMaker real-time inference endpoint.",
            "C. Deploy the model to an Amazon SageMaker Asynchronous Inference endpoint.",
            "D. Package the model as a container. Deploy the model to Amazon Elastic Container Service (Amazon ECS) on Amazon EC2 instances."
        ],
        "answer": "C",
        "explanation": "SageMaker Asynchronous Inference is designed for models with long processing times and large payloads. It can handle input data up to 1 GB and avoids holding open connections during long inference runs, reducing operational overhead compared to managing EC2 or ECS infrastructure."
    },
    {
        "question": "A company is creating an application that will recommend products for customers to purchase. The application will make API calls to Amazon Q Business. The company must ensure that responses from Amazon Q Business do not include the name of the company's main competitor.\nWhich solution will meet this requirement?",
        "options": [
            "A. Configure the competitor's name as a blocked phrase in Amazon Q Business.",
            "B. Configure an Amazon Q Business retriever to exclude the competitor's name.",
            "C. Configure an Amazon Kendra retriever for Amazon Q Business to build indexes that exclude the competitor's name.",
            "D. Configure document attribute boosting in Amazon Q Business to deprioritize the competitor's name."
        ],
        "answer": "A"
    },
    {
        "question": "A company has trained an ML model in Amazon SageMaker. The company needs to host the model to provide inferences in a production environment. The model must be highly available and must respond with minimum latency. The size of each request will be between 1 KB and 3 MB. The model will receive unpredictable bursts of requests during the day. The inferences must adapt proportionally to the changes in demand.\nHow should the company deploy the model into production to meet these requirements?",
        "options": [
            "A. Create a SageMaker real-time inference endpoint. Configure auto scaling. Configure the endpoint to present the existing model.",
            "B. Deploy the model on an Amazon Elastic Container Service (Amazon ECS) cluster. Use ECS scheduled scaling that is based on the CPU of the ECS cluster.",
            "C. Install SageMaker Operator on an Amazon Elastic Kubernetes Service (Amazon EKS) cluster. Deploy the model in Amazon EKS. Set horizontal pod auto scaling to scale replicas based on the memory metric.",
            "D. Use Spot Instances with a Spot Fleet behind an Application Load Balancer (ALB) for inferences.Use the ALBRequestCountPerTarget metric as the metric for auto scaling."
        ],
        "answer": "A"
    },
    {
        "question": "Case Study\nAn ML engineer is developing a fraud detection model on AWS. The training dataset includes transaction logs, customer profiles, and tables from an on-premises MySQL database. The transaction logs and customer profiles are stored in Amazon S3. The dataset has a class imbalance that affects the learning of the model's algorithm. Additionally, many of the features have interdependencies. The algorithm is not capturing all the desired underlying patterns in the data.\nWhich AWS service or feature can aggregate the data from the various data sources?",
        "options": [
            "A. Amazon EMR Spark jobs",
            "B. Amazon Kinesis Data Streams",
            "C. Amazon DynamoDB",
            "D. AWS Lake Formation"
        ],
        "answer": "D"
    },
    {
        "question": "A company is exploring generative AI and wants to add a new product feature. An ML engineer is making API calls from existing Amazon EC2 instances to Amazon Bedrock. The EC2 instances are in a private subnet and must remain private during the implementation. The EC2 instances have an assigned security group that allows access to all IP addresses in the private subnet.\nWhat should the ML engineer do to establish a connection between the EC2 instances and Amazon Bedrock?",
        "options": [
            "A. Modify the security group to allow inbound and outbound traffic to and from Amazon Bedrock.",
            "B. Use AWS PrivateLink to access Amazon Bedrock through an interface VPC endpoint.",
            "C. Configure Amazon Bedrock to use the private subnet where the EC2 instances are deployed.",
            "D. Link the existing VPC to Amazon Bedrock by using an AWS Direct Connect connection."
        ],
        "answer": "B",
        "explanation": "Since the EC2 instances are in a private subnet and must not have public internet access, the correct solution is to use AWS PrivateLink with an interface VPC endpoint for Amazon Bedrock. This allows private connectivity from the VPC to the Bedrock service without exposing traffic to the public internet."
    },
    {
        "question": "An ML engineer wants an Amazon SageMaker notebook to automatically stop running after 1 hour of idle time. How can the ML engineer accomplish this goal?",
        "options": [
            "A. Create a lifecycle configuration in SageMaker. Copy the auto-stop-idle script from GitHub to the Start Notebook section.",
            "B. Create a lifecycle configuration in SageMaker. Copy the auto-stop-idle script from GitHub to the Create Notebook section.",
            "C. Track the notebook's CPU metric by using Amazon CloudWatch Logs. Invoke an AWS Lambda function from CloudWatch Logs to shut down the notebook instance if CPU utilization becomes zero.",
            "D. Track the notebook's memory metric by using Amazon CloudWatch Logs. Invoke an AWS Lambda function from CloudWatch Logs to shut down the notebook instance if memory utilization becomes zero."
        ],
        "answer": "A",
        "explanation": "The correct approach is to use a SageMaker lifecycle configuration and place the auto-stop-idle script in the Start Notebook section. This ensures the notebook runs the monitoring script on startup, which checks for idle time and automatically stops the notebook after the defined threshold (1 hour in this case)."
    },
    {
        "question": "A company has used Amazon SageMaker to deploy a predictive ML model in production. The company is using SageMaker Model Monitor on the model. After a model update, an ML engineer notices data quality issues in the Model Monitor checks.\nWhat should the ML engineer do to mitigate the data quality issues that Model Monitor has identified?",
        "options": [
            "A. Adjust the model's parameters and hyperparameters.",
            "B. Initiate a manual Model Monitor job that uses the most recent production data.",
            "C. Create a new baseline from the latest dataset. Update Model Monitor to use the new baseline for evaluations.",
            "D. Include additional data in the existing training set for the model. Retrain and redeploy the model."
        ],
        "answer": "C"
    },
    {
        "question": "An ML engineer is using Amazon SageMaker to train a deep learning model that requires distributed training. After some training attempts, the ML engineer observes that the instances are not performing as expected. The ML engineer identifies communication overhead between the training instances.\nWhat should the ML engineer do to MINIMIZE the communication overhead between the instances?",
        "options": [
            "A. Place the instances in the same VPC subnet. Store the data in a different AWS Region from where the instances are deployed.",
            "B. Place the instances in the same VPC subnet but in different Availability Zones. Store the data in a different AWS Region from where the instances are deployed.",
            "C. Place the instances in the same VPC subnet. Store the data in the same AWS Region and Availability Zone where the instances are deployed.",
            "D. Place the instances in the same VPC subnet. Store the data in the same AWS Region but in a different Availability Zone from where the instances are deployed."
        ],
        "answer": "C"
    },
    {
        "question": "A company is using Amazon SageMaker to develop ML models. The company stores sensitive training data in an Amazon S3 bucket. The model training must have network isolation from the internet.\nWhich solution will meet this requirement?",
        "options": [
            "A. Run the SageMaker training jobs in private subnets. Create a NAT gateway. Route traffic for training through the NAT gateway.",
            "B. Run the SageMaker training jobs in private subnets. Create an S3 gateway VPC endpoint. Route traffic for training through the S3 gateway VPC endpoint.",
            "C. Run the SageMaker training jobs in public subnets that have an attached security group. In the security group, use inbound rules to limit traffic from the internet. Encrypt SageMaker instance storage by using server-side encryption with AWS KMS keys (SSE-KMS).",
            "D. Encrypt traffic to Amazon S3 by using a bucket policy that includes a value of True for the aws:SecureTransport condition key. Use default at-rest encryption for Amazon S3. Encrypt SageMaker instance storage by using server-side encryption with AWS KMS keys (SSE KMS)."
        ],
        "answer": "B"
    },
    {
        "question": "A company is using ML to predict the presence of a specific weed in a farmer's field. The\ncompany is using the Amazon SageMaker linear learner built-in algorithm with a value of\nmulticlass_dassifier for the predictorjype hyperparameter.\nWhat should the company do to MINIMIZE false positives?",
        "options": [
            "B. Increase the number of training epochs.",
            "C. Increase the value of the target_precision hyperparameter.\n2\n\n\fIT Certification Guaranteed, The Easy Way!",
            "D. Change the value of the predictorjype hyperparameter to regressor."
        ],
        "answer": "C"
    },
    {
        "question": "8\n\n\fIT Certification Guaranteed, The Easy Way!\n\nAn ML engineer is developing a classification model. The ML engineer needs to use custom\nlibraries in processing jobs, training jobs, and pipelines in Amazon SageMaker. Which\nsolution will provide this functionality with the LEAST implementation effort?",
        "options": [
            "B. Build a custom Docker container that includes the required libraries. Host the container in\nAmazon Elastic Container Registry (Amazon ECR). Use the ECR image in the SageMaker\njobs and pipelines.",
            "C. Create a SageMaker notebook instance to host the jobs. Create an AWS Lambda function\nto install the libraries on the notebook instance when the notebook instance starts. Configure\nthe SageMaker jobs and pipelines to run on the notebook instance.",
            "D. Run code for the libraries externally on Amazon EC2 instances. Store the results in\nAmazon S3.Import the results into the SageMaker jobs and pipelines."
        ],
        "answer": "B",
        "explanation": "Building a custom Docker container with the required libraries and hosting it in Amazon ECR\nallows SageMaker jobs, training, and pipelines to consistently use the same environment.\nThis approach minimizes manual setup, ensures portability, and provides the least ongoing\nimplementation effort compared to repeatedly installing or managing libraries separately."
    },
    {
        "question": "A company is planning to use Amazon Redshift ML in its primary AWS account. The source\ndata is in an Amazon S3 bucket in a secondary account.\nAn ML engineer needs to set up an ML pipeline in the primary account to access the S3\n\n11\n\n\fIT Certification Guaranteed, The Easy Way!\n\nbucket in the secondary account. The solution must not require public IPv4 addresses.\nWhich solution will meet these requirements?",
        "options": [
            "B. Provision a Redshift cluster and Amazon SageMaker Studio in a VPC with no public\naccess enabled in the primary account. Create an AWS Direct Connect connection and a\ntransit gateway.\nAssociate the VPCs from both accounts with the transit gateway. Update the VPC route\ntables to remove the route to 0.0.0.0/0.",
            "C. Provision a Redshift cluster and Amazon SageMaker Studio in a VPC in the primary\naccount.\nCreate an AWS Site-to-Site VPN connection with two encrypted IPsec tunnels between the\naccounts. Set up interface VPC endpoints for Amazon S3.",
            "D. Provision a Redshift cluster and Amazon SageMaker Studio in a VPC in the primary\naccount.Create an S3 gateway endpoint. Update the S3 bucket policy to allow IAM principals\nfrom the primary account. Set up interface VPC endpoints for SageMaker and Amazon\nRedshift."
        ],
        "answer": "D"
    },
    {
        "question": "12\n\n\fIT Certification Guaranteed, The Easy Way!\n\nAn ML engineer has an Amazon Comprehend custom model in Account A in the us-east-1\nRegion. The ML engineer needs to copy the model to Account B in the same Region.\nWhich solution will meet this requirement with the LEAST development effort?",
        "options": [
            "B. Create a resource-based IAM policy. Use the Amazon Comprehend ImportModel API\noperation to copy the model to Account B.",
            "C. Use AWS DataSync to replicate the model from Account A to Account B.",
            "D. Create an AWS Site-to-Site VPN connection between Account A and Account B to\ntransfer the model."
        ],
        "answer": "B"
    },
    {
        "question": "13\n\n\fIT Certification Guaranteed, The Easy Way!\n\nA company uses 10 Reserved Instances of accelerated instance types to serve the current\nversion of an ML model. An ML engineer needs to deploy a new version of the model to an\nAmazon SageMaker real-time inference endpoint.\nThe solution must use the original 10 instances to serve both versions of the model. The\nsolution also must include one additional Reserved Instance that is available to use in the\ndeployment process. The transition between versions must occur with no downtime or\nservice interruptions.\nWhich solution will meet these requirements?",
        "options": [
            "B. Configure a blue/green deployment with canary traffic shifting and a size of 10%.",
            "C. Configure a shadow test with a traffic sampling percentage of 10%.",
            "D. Configure a rolling deployment with a rolling batch size of 1."
        ],
        "answer": "D"
    },
    {
        "question": "A machine learning team has several large CSV datasets in Amazon S3. Historically, models\n17\n\n\fIT Certification Guaranteed, The Easy Way!\n\nbuilt with the Amazon SageMaker Linear Learner algorithm have taken hours to train on\nsimilar-sized datasets. The team's leaders need to accelerate the training process.\nWhat can a machine learning specialist do to address this concern?",
        "options": [
            "B. Use Amazon Machine Learning to train the models.",
            "C. Use Amazon Kinesis to stream the data to Amazon SageMaker.",
            "D. Use AWS Glue to transform the CSV dataset to the JSON format."
        ],
        "answer": "A",
        "explanation": "Amazon SageMaker Pipe mode streams the data directly to the container, which improves\nthe performance of training jobs. In Pipe mode, your training job streams data directly from\nAmazon S3. Streaming can provide faster start times for training jobs and better throughput.\nWith Pipe mode, you also reduce the size of the Amazon EBS volumes for your training\ninstances."
    },
    {
        "question": "A company has trained and deployed an ML model by using Amazon SageMaker. The\ncompany needs to implement a solution to record and monitor all the API call events for the\n19\n\n\fIT Certification Guaranteed, The Easy Way!\n\nSageMaker endpoint. The solution also must provide a notification when the number of API\ncall events breaches a threshold.\nWhich solution will meet these requirements?",
        "options": [
            "B. Use SageMaker Debugger to track the inferences and to report metrics. Use the\ntensor_variance built-in rule to provide a notification when the threshold is breached.",
            "C. Log all the endpoint invocation API events by using AWS CloudTrail. Use an Amazon\nCloudWatch dashboard for monitoring. Set up a CloudWatch alarm to provide notification\nwhen the threshold is breached.",
            "D. Add the Invocations metric to an Amazon CloudWatch dashboard for monitoring. Set up a\nCloudWatch alarm to provide notification when the threshold is breached."
        ],
        "answer": "C"
    },
    {
        "question": "A company is exploring generative AI and wants to add a new product feature. An ML\nengineer is making API calls from existing Amazon EC2 instances to Amazon Bedrock. The\n\n21\n\n\fIT Certification Guaranteed, The Easy Way!\n\nEC2 instances are in a private subnet and must remain private during the implementation.\nThe EC2 instances have an assigned security group that allows access to all IP addresses in\nthe private subnet.\nWhat should the ML engineer do to establish a connection between the EC2 instances and\nAmazon Bedrock?",
        "options": [
            "B. Use AWS PrivateLink to access Amazon Bedrock through an interface VPC endpoint.",
            "C. Configure Amazon Bedrock to use the private subnet where the EC2 instances are\ndeployed.",
            "D. Link the existing VPC to Amazon Bedrock by using an AWS Direct Connect connection."
        ],
        "answer": "B",
        "explanation": "Since the EC2 instances are in a private subnet and must not have public internet access,\nthe correct solution is to use AWS PrivateLink with an interface VPC endpoint for Amazon\nBedrock.\nThis allows private connectivity from the VPC to the Bedrock service without exposing traffic\nto the public internet."
    },
    {
        "question": "A company has used Amazon SageMaker to deploy a predictive ML model in production.\nThe company is using SageMaker Model Monitor on the model. After a model update, an ML\nengineer notices data quality issues in the Model Monitor checks.\nWhat should the ML engineer do to mitigate the data quality issues that Model Monitor has\nidentified?\n22\n\n\fIT Certification Guaranteed, The Easy Way!",
        "options": [
            "B. Initiate a manual Model Monitor job that uses the most recent production data.",
            "C. Create a new baseline from the latest dataset. Update Model Monitor to use the new\nbaseline for evaluations.",
            "D. Include additional data in the existing training set for the model. Retrain and redeploy the\nmodel."
        ],
        "answer": "C"
    },
    {
        "question": "23\n\n\fIT Certification Guaranteed, The Easy Way!\n\nCase Study\nAn ML engineer is developing a fraud detection model on AWS. The training dataset includes\ntransaction logs, customer profiles, and tables from an on-premises MySQL database. The\ntransaction logs and customer profiles are stored in Amazon S3.\nThe dataset has a class imbalance that affects the learning of the model's algorithm.\nAdditionally, many of the features have interdependencies. The algorithm is not capturing all\nthe desired underlying patterns in the data.\nAfter the data is aggregated, the ML engineer must implement a solution to automatically\ndetect anomalies in the data and to visualize the result.\nWhich solution will meet these requirements?",
        "options": [
            "B. Use Amazon Redshift Spectrum to automatically detect the anomalies. Use Amazon\nQuickSight to visualize the result.",
            "C. Use Amazon SageMaker Data Wrangler to automatically detect the anomalies and to\nvisualize the result.",
            "D. Use AWS Batch to automatically detect the anomalies. Use Amazon QuickSight to\nvisualize the result."
        ],
        "answer": "C"
    },
    {
        "question": "A term frequency-inverse document frequency (tf-idf) matrix using both unigrams and\nbigrams is built from a text corpus consisting of the following two sentences:\n1. Please call the number below.\n2. Please do not call us.\nWhat are the dimensions of the tf-idf matrix?",
        "options": [
            "B. (2, 8)",
            "C. (2, 10)",
            "D. (8, 10)"
        ],
        "answer": "A",
        "explanation": "There are 2 sentences, 8 unique unigrams, and 8 unique bigrams, so the result would be\n(2,16).\nThe phrases are \"Please call the number below\" and \"Please do not call us.\" Each word\nindividually (unigram) is \"Please,\" \"call,\" \"the,\" \"number,\" \"below,\" \"do,\" \"not,\" and \"us.\" The\nunique bigrams are \"Please call,\" \"call the,\" \"the number,\" \"number below,\" \"Please do,\" \"do\nnot,\"\n\"not call,\" and \"call us.\""
    },
    {
        "question": "An ML engineer has deployed an Amazon SageMaker model to a serverless endpoint in\nproduction. The model is invoked by the InvokeEndpoint API operation.\nThe model's latency in production is higher than the baseline latency in the test environment.\nThe ML engineer thinks that the increase in latency is because of model startup time.\nWhat should the ML engineer do to confirm or deny this hypothesis?\n24\n\n\fIT Certification Guaranteed, The Easy Way!",
        "options": [
            "B. Schedule a SageMaker Model Monitor job with Amazon CloudWatch metrics enabled.",
            "C. Enable Amazon CloudWatch metrics. Observe the ModelSetupTime metric in the\nSageMaker namespace.",
            "D. Enable Amazon CloudWatch metrics. Observe the ModelLoadingWaitTime metric in the\nSageMaker namespace."
        ],
        "answer": "D"
    },
    {
        "question": "A company is using Amazon SageMaker and millions of files to train an ML model. Each file\nis several megabytes in size. The files are stored in an Amazon S3 bucket. The company\nneeds to improve training performance.\nWhich solution will meet these requirements in the LEAST amount of time?",
        "options": [
            "B. Create an Amazon FSx for Lustre file system. Link the file system to the existing S3\nbucket.\nAdjust the training job to read from the file system.",
            "C. Create an Amazon Elastic File System (Amazon EFS) file system. Transfer the existing\ndata to the file system. Adjust the training job to read from the file system.",
            "D. Create an Amazon ElastiCache (Redis OSS) cluster. Link the Redis OSS cluster to the\nexisting S3 bucket. Stream the data from the Redis OSS cluster directly to the training job."
        ],
        "answer": "B"
    },
    {
        "question": "An ML engineer needs to train a supervised deep learning model. The available dataset is a\nlarge number of unlabeled images that only employees should access. The ML engineer\nneeds to implement a solution that labels the dataset with the highest possible accuracy.\nWhich combination of steps should the ML engineer take to meet these requirements?\n(Choose two.)",
        "options": [
            "B. Train the deep learning model directly on the raw data. Let the model infer the labels by\nitself.",
            "C. Use Amazon SageMaker Ground Truth to create an annotation job that specifies the\nlabeling task and requirements.",
            "D. Set up workforce teams to access a private workforce to run and review the annotation job\ncreated by Amazon SageMaker Ground Truth.",
            "E. Use Amazon Mechanical Turk to complete the annotation job created by Amazon\nSageMaker Ground Truth."
        ],
        "answer": "CD",
        "explanation": "To achieve the highest labeling accuracy with controlled employee-only access, the ML\nengineer should use Amazon SageMaker Ground Truth to define the annotation job and then\nassign it to a private workforce of employees for labeling and review. This ensures high25\n\n\fIT Certification Guaranteed, The Easy Way!\n\nquality, secure labeling restricted to authorized personnel."
    },
    {
        "question": "A company needs to use Amazon SageMaker to train a model on more than 300 GB of data.\nThe training data is composed of files that are 200 MB in size. The data is stored in Amazon\nS3 Standard storage and feeds a dashboard tool. Which SageMaker training ingestion\nmechanism is the MOST cost-effective solution for this scenario?",
        "options": [
            "B. Amazon FSx for Lustre file system",
            "C. Amazon S3 in fast file mode while using S3 Express One Zone",
            "D. Amazon S3 in fast file mode without using S3 Express One Zone"
        ],
        "answer": "D",
        "explanation": "For large-scale training data already stored in Amazon S3, the most cost-effective solution is\nto use SageMaker's S3 fast file mode without S3 Express One Zone. Fast file mode enables\nstreaming directly from S3 without duplicating the dataset onto local storage, reducing startup\ntime and storage cost. Using S3 Express One Zone would increase cost, so standard fast file\nmode is the most economical choice."
    },
    {
        "question": "A medical company is using AWS to build a tool to recommend treatments for patients. The\ncompany has obtained health records and self-reported textual information in English from\npatients. The company needs to use this information to gain insight about the patients.\nWhich solution will meet this requirement with the LEAST development effort?",
        "options": [
            "B. Use Amazon Comprehend Medical to summarize the data.",
            "C. Use Amazon Kendra to create a quick-search tool to query the data.",
            "D. Use the Amazon SageMaker Sequence-to-Sequence (seq2seq) algorithm to create a text\nsummary from the data."
        ],
        "answer": "B"
    },
    {
        "question": "A company has AWS Glue data processing jobs that are orchestrated by an AWS Glue\nworkflow.\nThe AWS Glue jobs can run on a schedule or can be launched manually.\nThe company is developing pipelines in Amazon SageMaker Pipelines for ML model\ndevelopment. The pipelines will use the output of the AWS Glue jobs during the data\nprocessing phase of model development. An ML engineer needs to implement a solution that\nintegrates the AWS Glue jobs with the pipelines.\nWhich solution will meet these requirements with the LEAST operational overhead?",
        "options": [
            "B. Use processing steps in SageMaker Pipelines. Configure inputs that point to the Amazon\nResource Names (ARNs) of the AWS Glue jobs.\n\n26\n\n\fIT Certification Guaranteed, The Easy Way!",
            "C. Use Callback steps in SageMaker Pipelines to start the AWS Glue workflow and to stop\nthe pipelines until the AWS Glue jobs finish running.",
            "D. Use Amazon EventBridge to invoke the pipelines and the AWS Glue jobs in the desired\norder."
        ],
        "answer": "C"
    },
    {
        "question": "A company runs training jobs on Amazon SageMaker by using a compute optimized\ninstance.\nDemand for training runs will remain constant for the next 55 weeks. The instance needs to\nrun for 35 hours each week. The company needs to reduce its model training costs.\nWhich solution will meet these requirements?",
        "options": [
            "B. Use SageMaker Edge Manager for the training. Specify the instance requirement in the\nedge device configuration. Run the training.",
            "C. Use the heterogeneous cluster feature of SageMaker Training. Configure the\ninstance_type, instance_count, and instance_groups arguments to run training jobs.",
            "D. Opt in to a SageMaker Savings Plan with a 1-year term and an All Upfront payment. Run a\nSageMaker Training job on the instance."
        ],
        "answer": "D"
    },
    {
        "question": "A medical company needs to store clinical data. The data includes personally identifiable\ninformation (PII) and protected health information (PHI).\nAn ML engineer needs to implement a solution to ensure that the PII and PHI are not used to\ntrain ML models.\nWhich solution will meet these requirements?",
        "options": [
            "B. Upload the clinical data to an Amazon Redshift database. Use built-in SQL stored\nprocedures to automatically classify and mask the PII and PHI before the data is used for\nmodel training.",
            "C. Use Amazon Comprehend to detect and mask the PII before the data is used for model\ntraining.\nUse Amazon Comprehend Medical to detect and mask the PHI before the data is used for\nmodel training.",
            "D. Create an AWS Lambda function to encrypt the PII and PHI. Program the Lambda\nfunction to save the encrypted data to an Amazon S3 bucket for model training."
        ],
        "answer": "C"
    },
    {
        "question": "An ML engineer needs to ensure that a dataset complies with regulations for personally\nidentifiable information (PII). The ML engineer will use the data to train an ML model on\n\n27\n\n\fIT Certification Guaranteed, The Easy Way!\n\nAmazon SageMaker instances. SageMaker must not use any of the PII.\nWhich solution will meet these requirements in the MOST operationally efficient way?",
        "options": [
            "B. Use the Amazon Comprehend DetectPiiEntities API call to redact the PII from the data.\nStore the data in an Amazon Elastic File System (Amazon EFS) file system. Mount the EFS\nfile system to the SageMaker instances for model training.",
            "C. Use AWS Glue DataBrew to cleanse the dataset of PII. Store the data in an Amazon\nElastic File System (Amazon EFS) file system. Mount the EFS file system to the SageMaker\ninstances for model training.",
            "D. Use Amazon Macie for automatic discovery of PII in the data. Remove the PII. Store the\ndata in an Amazon S3 bucket. Mount the S3 bucket to the SageMaker instances for model\ntraining."
        ],
        "answer": "A"
    },
    {
        "question": "A company is building an ML model by using Amazon SageMaker, AWS owned libraries, and\nopen source libraries. The company must ensure that SageMaker does not collect metadata\nabout usage and errors during training. Which solution will meet these requirements?",
        "options": [
            "B. Add an IAM role to the SageMaker domain to deny Amazon CloudWatch the permission to\nreport metadata.",
            "C. Turn off the setting in the SageMaker domain to share metadata for console jobs. Opt out\nof metadata collection for each training job that is submitted through the AWS CLI or AWS\nSDKs.",
            "D. Set a parameter to opt out of metadata collection for each training job that is submitted\nthrough the AWS CLI, Boto3, or the SageMaker Python SDK."
        ],
        "answer": "C",
        "explanation": "To prevent metadata collection in Amazon SageMaker, you must disable the metadata\nsharing setting in the SageMaker domain for console jobs and explicitly opt out of metadata\ncollection for each training job submitted through the AWS CLI or SDKs. This ensures that\nneither usage nor error metadata is collected during training."
    },
    {
        "question": "A company uses Amazon SageMaker for its ML workloads. The company's ML engineer\nreceives a 50 MB Apache Parquet data file to build a fraud detection model. The file includes\nseveral correlated columns that are not required.\nWhat should the ML engineer do to drop the unnecessary columns in the file with the LEAST\neffort?",
        "options": [
            "B. Create an Apache Spark job that uses a custom processing script on Amazon EMR.",
            "C. Create a SageMaker processing job by calling the SageMaker Python SDK.",
            "D. Create a data flow in SageMaker Data Wrangler. Configure a transform step."
        ],
        "answer": "D"
    },
    {
        "question": "An IoT company uses Amazon SageMaker to train and test an XGBoost model for object\ndetection. ML engineers need to monitor performance metrics when they train the model with\nvariants in hyperparameters. The ML engineers also need to send Short Message Service\n(SMS) text messages after training is complete.\nWhich solution will meet these requirements?",
        "options": [
            "B. Use Amazon CloudWatch to monitor performance metrics. Use Amazon Simple\nNotification Service (Amazon SNS) for message delivery.",
            "C. Use AWS CloudTrail to monitor performance metrics. Use Amazon Simple Queue Service\n(Amazon SQS) for message delivery.",
            "D. Use AWS CloudTrail to monitor performance metrics. Use Amazon Simple Notification\nService (Amazon SNS) for message delivery."
        ],
        "answer": "B"
    },
    {
        "question": "An ML engineer needs to deploy four ML models in an Amazon SageMaker inference\npipeline.\nThe models were built with different frameworks. The ML engineer also needs to give clients\nthe ability to use the invoke_endpoint call to perform inference for each model. Which\nsolution will meet these requirements MOST cost-effectively?",
        "options": [
            "B. Create a SageMaker multi-container endpoint.",
            "C. Create multiple SageMaker single-model endpoints.",
            "D. Run a SparkML job to generate multiple endpoints."
        ],
        "answer": "B",
        "explanation": "A SageMaker multi-container endpoint allows deployment of multiple models built with\ndifferent frameworks in a single endpoint. Each container can host a model with its required\nframework, and clients can use the same invoke_endpoint call while specifying the target\ncontainer. This meets the requirement for framework diversity and is more cost-effective than\nrunning separate single-model endpoints."
    },
    {
        "question": "A company has collected customer comments on its products, rating them as safe or unsafe,\nusing decision trees. The training dataset has the following features: id, date, full review, full\nreview summary, and a binary safe/unsafe tag. During training, any data sample with missing\nfeatures was dropped. In a few instances, the test set was found to be missing the full review\ntext field.\nFor this use case, which is the most effective course of action to address test data samples\nwith missing features?",
        "options": [
            "B. Copy the summary text fields and use them to fill in the missing full review text fields, and\nthen run through the test set.",
            "C. Use an algorithm that handles missing data better than decision trees.",
            "D. Generate synthetic data to fill in the fields that are missing data, and then run through the\ntest set."
        ],
        "answer": "B",
        "explanation": "In this case, a full review summary usually contains the most descriptive phrases of the entire\n31\n\n\fIT Certification Guaranteed, The Easy Way!\n\nreview and is a valid stand-in for the missing full review text field."
    },
    {
        "question": "A company receives daily .csv files about customer interactions with its ML model. The\ncompany stores the files in Amazon S3 and uses the files to retrain the model. An ML\nengineer needs to implement a solution to mask credit card numbers in the files before the\nmodel is retrained.\nWhich solution will meet this requirement with the LEAST development effort?",
        "options": [
            "B. Create Apache Spark code to run on an AWS Glue job. Use the Sensitive Data Detection\nfunctionality in AWS Glue to find and mask sensitive data.",
            "C. Create Apache Spark code to run on an AWS Glue job. Program the code to perform a\nregex operation to find and mask sensitive data.",
            "D. Create Apache Spark code to run on an Amazon EC2 instance. Program the code to\nperform an operation to find and mask sensitive data."
        ],
        "answer": "B"
    },
    {
        "question": "A company wants to develop an ML model by using tabular data from its customers. The\ndata contains meaningful ordered features with sensitive information that should not be\ndiscarded. An ML engineer must ensure that the sensitive data is masked before another\nteam starts to build the model.\nWhich solution will meet these requirements?",
        "options": [
            "B. Prepare the data by using AWS Glue DataBrew.",
            "C. Run an AWS Batch job to change the sensitive data to random values.",
            "D. Run an Amazon EMR job to change the sensitive data to random values."
        ],
        "answer": "B"
    },
    {
        "question": "A company is building a real-time data processing pipeline for an ecommerce application.\nThe application generates a high volume of clickstream data that must be ingested,\nprocessed, and visualized in near real time. The company needs a solution that supports\nSQL for data processing and Jupyter notebooks for interactive analysis.\nWhich solution will meet these requirements?",
        "options": [
            "B. Use Amazon Kinesis Data Streams to ingest the data. Use Amazon Data Firehose to\ntransform the data. Use Amazon Athena to process the data. Use Amazon QuickSight to\nvisualize the data.",
            "C. Use Amazon Managed Streaming for Apache Kafka (Amazon MSK) to ingest the data.\nUse AWS Glue with PySpark to process the data. Store the processed data in Amazon S3.\n\n32\n\n\fIT Certification Guaranteed, The Easy Way!\n\nUse Amazon QuickSight to visualize the data.",
            "D. Use Amazon Managed Streaming for Apache Kafka (Amazon MSK) to ingest the data.\nUse Amazon Managed Service for Apache Flink to process the data. Use the built-in Flink\ndashboard to visualize the data."
        ],
        "answer": "D"
    },
    {
        "question": "A company needs to perform feature engineering, aggregation, and data preparation. After\nthe features are produced, the company must implement a solution on AWS to process and\nstore the features. Which solution will meet these requirements?",
        "options": [
            "B. Use Amazon SageMaker Model Monitor to automatically ingest and transform the data.\nCreate an Amazon S3 bucket to store the features in JSON format.",
            "C. Use Amazon Managed Service for Apache Flink to transform the data and to ingest the\ndata directly into Amazon SageMaker Feature Store. Use Feature Store to manage and store\nthe features.",
            "D. Use an Amazon SageMaker batch transform job to analyze, transform, and ingest the\ndata.Create an Amazon DynamoDB table to store the features."
        ],
        "answer": "A",
        "explanation": "Amazon SageMaker Feature Processing (via processing jobs) is used to perform feature\nengineering and data preparation. The engineered features can then be ingested into\nSageMaker Feature Store, which is a purpose-built service to manage and store ML features\nfor reuse across training and inference. This combination directly addresses the company's\nrequirements."
    },
    {
        "question": "A company that has hundreds of data scientists is using Amazon SageMaker to create ML\nmodels. The models are in model groups in the SageMaker Model Registry.\nThe data scientists are grouped into three categories: computer vision, natural language\nprocessing (NLP), and speech recognition. An ML engineer needs to implement a solution to\norganize the existing models into these groups to improve model discoverability at scale. The\nsolution must not affect the integrity of the model artifacts and their existing groupings.\nWhich solution will meet these requirements?",
        "options": [
            "B. Create a model group for each category. Move the existing models into these category\nmodel groups.",
            "C. Use SageMaker ML Lineage Tracking to automatically identify and tag which model\ngroups should contain the models.",
            "D. Create a Model Registry collection for each of the three categories. Move the existing\nmodel groups into the collections."
        ],
        "answer": "D"
    },
    {
        "question": "An ML engineer receives datasets that contain missing values, duplicates, and extreme\noutliers.\nThe ML engineer must consolidate these datasets into a single data frame and must prepare\nthe data for ML.\nWhich solution will meet these requirements?",
        "options": [
            "B. Use Amazon SageMaker Ground Truth to import the datasets and to consolidate them into\na single data frame. Use the human-in-the-loop capability to prepare the data.",
            "C. Manually import and merge the datasets. Consolidate the datasets into a single data\nframe. Use Amazon Q Developer to generate code snippets that will prepare the data.",
            "D. Manually import and merge the datasets. Consolidate the datasets into a single data\nframe. Use Amazon SageMaker data labeling to prepare the data."
        ],
        "answer": "A"
    },
    {
        "question": "An ML engineer trained an ML model on Amazon SageMaker to detect automobile accidents\nfrom dosed-circuit TV footage. The ML engineer used SageMaker Data Wrangler to create a\ntraining dataset of images of accidents and non-accidents.\nThe model performed well during training and validation. However, the model is\nunderperforming in production because of variations in the quality of the images from various\n\n35\n\n\fIT Certification Guaranteed, The Easy Way!\n\ncameras.\nWhich solution will improve the model's accuracy in the LEAST amount of time?",
        "options": [
            "B. Recreate the training dataset by using the Data Wrangler corrupt image transform. Specify\nthe impulse noise option.",
            "C. Recreate the training dataset by using the Data Wrangler enhance image contrast\ntransform.\nSpecify the Gamma contrast option.",
            "D. Recreate the training dataset by using the Data Wrangler resize image transform. Crop all\nimages to the same size."
        ],
        "answer": "C"
    },
    {
        "question": "A company regularly receives new training data from the vendor of an ML model. The vendor\ndelivers cleaned and prepared data to the company's Amazon S3 bucket every 3-4 days.\nThe company has an Amazon SageMaker pipeline to retrain the model. An ML engineer\nneeds to implement a solution to run the pipeline when new data is uploaded to the S3\nbucket.\nWhich solution will meet these requirements with the LEAST operational effort?",
        "options": [
            "B. Create an Amazon EventBridge rule that has an event pattern that matches the S3 upload\n.\nConfigure the pipeline as the target of the rule.",
            "C. Create an S3 Lifecycle rule to transfer the data to the SageMaker training instance and to\ninitiate training.",
            "D. Create an AWS Lambda function that scans the S3 bucket. Program the Lambda function\nto initiate the pipeline when new data is uploaded."
        ],
        "answer": "B"
    },
    {
        "question": "A company has a binary classification model in production. An ML engineer needs to develop\na new version of the model.\nThe new model version must maximize correct predictions of positive labels and negative\nlabels.\nThe ML engineer must use a metric to recalibrate the model to meet these requirements.\nWhich metric should the ML engineer use for the model recalibration?",
        "options": [
            "B. Precision",
            "C. Recall",
            "D. Specificity"
        ],
        "answer": "A"
    },
    {
        "question": "A company wants to launch a new internal generative AI interface to answer user questions.\nThe interface will be based on a popular open source large language model (LLM). Which\ncombination of steps will deploy the interface with the LEAST operational overhead? (Choose\ntwo.)",
        "options": [
            "B. Download the LLM as a .zip file. Deploy the LLM on a GPU-based Amazon EC2 instance.",
            "C. Create a frontend HTML interface that uses an Amazon API Gateway WebSocket API with\nAWS Lambda functions to handle the user interaction.",
            "D. Use Amazon QuickSight to create a UI to handle the user interaction.",
            "E. Use Amazon Lex to create a UI to handle the user interaction."
        ],
        "answer": "AC",
        "explanation": "The least operational overhead comes from using Amazon SageMaker JumpStart to quickly\ndeploy the open source LLM without needing to manage infrastructure, and building a\nlightweight frontend HTML interface with API Gateway WebSocket API and Lambda to\nhandle user interactions efficiently. This avoids the manual setup of EC2 or unrelated\nservices like QuickSight or Lex."
    },
    {
        "question": "An ML engineer is training an ML model to identify people's health risk based on 20 features\nand\n1 target. The target class has two values:\n- Likely to have health risk (positive class)\n- Unlikely to have health risk (negative class)\nThe age range of people in the dataset is 30 years old to 60 years old. Age is one of the\nfeatures.\nThe ML engineer analyzes the features. For the positive class, the difference in proportions\nof labels (DPL) value is (+0.9) for the age range of 40 to 45 compared with all other age\nranges.\nWhat should the ML engineer do to correct this data imbalance?",
        "options": [
            "B. Undersample the positive class for the age range of 40 to 45.",
            "C. Undersample the positive class for all age ranges except 40 to 45.",
            "D. Oversample the negative class for all age ranges except 40 to 45."
        ],
        "answer": "B",
        "explanation": "A DPL of +0.9 indicates that the positive class is heavily overrepresented in the 40-45 age\nrange compared to other age ranges. To correct this imbalance, the solution is to\nundersample the positive class within the 40-45 range, reducing its dominance and improving\nfairness in the dataset."
    },
    {
        "question": "A data scientist is working on optimizing a model during the training process by varying\n\n37\n\n\fIT Certification Guaranteed, The Easy Way!\n\nmultiple parameters. The data scientist observes that, during multiple runs with identical\nparameters, the loss function converges to different, yet stable, values.\nWhat should the data scientist do to improve the training process?",
        "options": [
            "B. Decrease the learning rate. Reduce the batch size.",
            "C. Decrease the learning rate. Keep the batch size the same.",
            "D. Do not change the learning rate. Increase the batch size."
        ],
        "answer": "B",
        "explanation": "It is most likely that the loss function is very curvy and has multiple local minima where the\ntraining is getting stuck. Decreasing the batch size would help the data scientist stochastically\nget out of the local minima saddles. Decreasing the learning rate would prevent overshooting\nthe global loss function minimum."
    },
    {
        "question": "A company wants to use Amazon SageMaker to host an ML model that runs on CPU for realtime predictions. The model will have intermittent traffic during business hours and will have\nperiods of no traffic after business hours. The company needs a solution that will serve\ninference requests in the most cost-effective manner. Which hosting option will meet these\nrequirements?",
        "options": [
            "B. Deploy the model to a SageMaker Serverless Inference endpoint. Configure increased\nprovisioned concurrency during business hours.",
            "C. Deploy the model to a SageMaker Asynchronous Inference endpoint. Configure an auto\nscaling policy that scales in to zero outside business hours.",
            "D. Deploy the model to a SageMaker real-time endpoint. Create a scheduled AWS Lambda\nfunction that activates the endpoint during business hours only."
        ],
        "answer": "B",
        "explanation": "SageMaker Serverless Inference is the most cost-effective option for models with intermittent\ntraffic. It automatically scales down to zero when idle, so no cost is incurred outside business\nhours. Configuring provisioned concurrency during business hours ensures low-latency\nresponses when traffic is expected."
    },
    {
        "question": "A company has several teams that have developed separate prediction models on their own\nlaptops. The teams developed the models by using Python with scikit-learn and TensorFlow\nframeworks.\nThe company must rebuild the models and must integrate the models into an ML\ninfrastructure that the company manages by using Amazon SageMaker. The company also\nmust incorporate the models into a model registry.\nWhich solution will meet these requirements with the LEAST operational overhead?",
        "options": [
            "B. Import the models into the SageMaker Model Registry. Use SageMaker to run the\nimported models.",
            "C. Use code from the laptops to create containers for the models. Use the bring your own\ncontainer (BYOC) functionality of SageMaker to import and use the models. Register the\nmodels in the SageMaker Model Registry.",
            "D. Import the Python-based models into SageMaker. Rebuild the scikit-learn and TensorFlow\nmodels in SageMaker. Register all the models in the SageMaker Model Registry."
        ],
        "answer": "D",
        "explanation": "The least operational overhead comes from directly importing the scikit-learn and TensorFlow\nmodels into SageMaker, rebuilding them using the respective prebuilt SageMaker\nframeworks, and then registering them in the SageMaker Model Registry. This leverages\nmanaged framework containers provided by SageMaker, avoids custom container\nmanagement, and integrates seamlessly with the registry."
    },
    {
        "question": "A company runs Amazon SageMaker ML models that use accelerated instances. The models\nrequire real-time responses. Each model has different scaling requirements. The company\nmust not allow a cold start for the models.\nWhich solution will meet these requirements?",
        "options": [
            "B. Create a SageMaker Asynchronous Inference endpoint for each model. Create an auto\nscaling policy for each endpoint.",
            "C. Create a SageMaker endpoint. Create an inference component for each model. In the\ninference component settings, specify the newly created endpoint. Create an auto scaling\npolicy for each inference component. Set the parameter for the minimum number of copies to\nat least 1.",
            "D. Create an Amazon S3 bucket. Store all the model artifacts in the S3 bucket. Create a\nSageMaker multi-model endpoint. Point the endpoint to the S3 bucket. Create an auto\nscaling policy for the endpoint. Set the parameter for the minimum number of copies to at\nleast 1."
        ],
        "answer": "C"
    },
    {
        "question": "A company has a conversational AI assistant that sends requests through Amazon Bedrock\nto an Anthropic Claude large language model (LLM). Users report that when they ask similar\nquestions multiple times, they sometimes receive different answers. An ML engineer needs\nto improve the responses to be more consistent and less random.\nWhich solution will meet these requirements?",
        "options": [
            "B. Increase the temperature parameter. Decrease the top_k parameter.\n\n39\n\n\fIT Certification Guaranteed, The Easy Way!",
            "C. Decrease the temperature parameter. Increase the top_k parameter.",
            "D. Decrease the temperature parameter and the top_k parameter."
        ],
        "answer": "D"
    },
    {
        "question": "An ML engineer wants to use a set of survey responses as training data for an ML classifier.\nAll the survey responses are either \"yes\" or \"no.\" The ML engineer needs to convert the\nresponses into a feature that will produce better model training results. The ML engineer\nmust not increase the dimensionality of the dataset.\nWhich methods will meet these requirements? (Choose two.)",
        "options": [
            "B. Label encoding",
            "C. One-hot encoding",
            "D. Statistical imputation",
            "E. Tokenization"
        ],
        "answer": "AB",
        "explanation": "Both binary encoding and label encoding convert categorical yes/no responses into\nnumerical values without increasing dimensionality. For example, mapping yes → 1 and no →\n0. Unlike one-hot encoding, which would add extra dimensions, these methods keep the\ndataset compact and effective for training."
    },
    {
        "question": "A financial company receives a high volume of real-time market data streams from an\nexternal provider. The streams consist of thousands of JSON records every second.\nThe company needs to implement a scalable solution on AWS to identify anomalous data\npoints.\nWhich solution will meet these requirements with the LEAST operational overhead?",
        "options": [
            "B. Ingest real-time data into Amazon Kinesis data streams. Deploy an Amazon SageMaker\nendpoint for real-time outlier detection. Create an AWS Lambda function to detect anomalies.\nUse the data streams to invoke the Lambda function.",
            "C. Ingest real-time data into Apache Kafka on Amazon EC2 instances. Deploy an Amazon\nSageMaker endpoint for real-time outlier detection. Create an AWS Lambda function to\ndetect anomalies. Use the data streams to invoke the Lambda function.",
            "D. Send real-time data to an Amazon Simple Queue Service (Amazon SQS) FIFO queue.\nCreate an AWS Lambda function to consume the queue messages. Program the Lambda\nfunction to start an AWS Glue extract, transform, and load (ETL) job for batch processing and\nanomaly detection."
        ],
        "answer": "A"
    },
    {
        "question": "40\n\n\fIT Certification Guaranteed, The Easy Way!\n\nAn ML engineer notices class imbalance in an image classification training job.\nWhat should the ML engineer do to resolve this issue?",
        "options": [
            "B. Transform some of the images in the dataset.",
            "C. Apply random oversampling on the dataset.",
            "D. Apply random data splitting on the dataset."
        ],
        "answer": "C"
    },
    {
        "question": "A company has an Amazon S3 bucket that contains 1 TB of files from different sources. The\nS3 bucket contains the following file types in the same S3 folder: CSV, JSON, XLSX, and\nApache Parquet.\nAn ML engineer must implement a solution that uses AWS Glue DataBrew to process the\ndata.\nThe ML engineer also must store the final output in Amazon S3 so that AWS Glue can\nconsume the output in the future.\nWhich solution will meet these requirements?",
        "options": [
            "B. Use DataBrew to process the existing S3 folder. Store the output in AWS Glue Parquet\nformat.",
            "C. Separate the data into a different folder for each file type. Use DataBrew to process each\nfolder individually. Store the output in Apache Parquet format.",
            "D. Separate the data into a different folder for each file type. Use DataBrew to process each\nfolder individually. Store the output in AWS Glue Parquet format."
        ],
        "answer": "C"
    },
    {
        "question": "A company is setting up a system to manage all of the datasets it stores in Amazon S3. The\ncompany would like to automate running transformation jobs on the data and maintaining a\ncatalog of the metadata concerning the datasets. The solution should require the least\namount of setup and maintenance.\nWhich solution will allow the company to achieve its goals?",
        "options": [
            "B. Create an AWS Glue crawler to populate the AWS Glue Data Catalog. Then, author an\nAWS Glue ETL job, and set up a schedule for data transformation jobs.",
            "C. Create an Amazon EMR cluster with Apache Spark installed. Then, create an Apache Hive\nmetastore and a script to run transformation jobs on a schedule.",
            "D. Create an Amazon SageMaker Jupyter notebook instance that transforms the data. Then,\ncreate an Apache Hive metastore and a script to run transformation jobs on a schedule."
        ],
        "answer": "B",
        "explanation": "AWS Glue is the correct answer because this option requires the least amount of setup and\n41\n\n\fIT Certification Guaranteed, The Easy Way!\n\nmaintenance since it is serverless, and it does not require management of the infrastructure."
    },
    {
        "question": "An ML engineer needs to implement a solution to host a trained ML model. The rate of\nrequests to the model will be inconsistent throughout the day.\nThe ML engineer needs a scalable solution that minimizes costs when the model is not in\nuse.\nThe solution also must maintain the model's capacity to respond to requests during times of\npeak usage.\nWhich solution will meet these requirements?",
        "options": [
            "B. Deploy the model on an Amazon Elastic Container Service (Amazon ECS) cluster that\nuses AWS Fargate. Set a static number of tasks to handle requests during times of peak\nusage.",
            "C. Deploy the model to an Amazon SageMaker endpoint. Deploy multiple copies of the model\nto the endpoint. Create an Application Load Balancer to route traffic between the different\ncopies of the model at the endpoint.",
            "D. Deploy the model to an Amazon SageMaker endpoint. Create SageMaker endpoint auto\nscaling policies that are based on Amazon CloudWatch metrics to adjust the number of\ninstances dynamically."
        ],
        "answer": "D"
    },
    {
        "question": "A company is working on an ML project that will include Amazon SageMaker notebook\ninstances.\nAn ML engineer must ensure that the SageMaker notebook instances do not allow root\naccess.\nWhich solution will prevent the deployment of notebook instances that allow root access?",
        "options": [
            "B. Use AWS Key Management Service (AWS KMS) keys to stop deployments of SageMaker\nnotebook instances that allow root access.",
            "C. Monitor resource creation by using Amazon EventBridge events. Create an AWS Lambda\nfunction that deletes all deployed SageMaker notebook instances that allow root access.",
            "D. Monitor resource creation by using AWS CloudFormation events. Create an AWS Lambda\nfunction that deletes all deployed SageMaker notebook instances that allow root access."
        ],
        "answer": "A"
    },
    {
        "question": "A company wants to provide services to help other businesses label images. The company\nwants its labeling specialists to complete human labeling tasks on AWS. How should the\ncompany register the labeling specialists to receive tasks on AWS?",
        "options": [
            "B. Create and use an internal workforce in Amazon SageMaker Ground Truth.",
            "C. Create and use Amazon Mechanical Turk entities in an Amazon SageMaker human loop.",
            "D. Use the Amazon Mechanical Turk website."
        ],
        "answer": "B",
        "explanation": "To enable labeling specialists within the company to perform tasks, the correct solution is to\ncreate and use an internal workforce in Amazon SageMaker Ground Truth. This allows the\ncompany to securely register and manage its own labeling team to receive and complete\nhuman labeling tasks."
    },
    {
        "question": "A company is using an AWS Lambda function to monitor the metrics from an ML model. An\nML engineer needs to implement a solution to send an email message when the metrics\nbreach a threshold.\nWhich solution will meet this requirement?",
        "options": [
            "B. Log the metrics from the Lambda function to Amazon CloudFront. Configure an Amazon\nCloudWatch alarm to send the email message.",
            "C. Log the metrics from the Lambda function to Amazon CloudWatch. Configure a\nCloudWatch alarm to send the email message.",
            "D. Log the metrics from the Lambda function to Amazon CloudWatch. Configure an Amazon\nCloudFront rule to send the email message.\n44\n\n\fIT Certification Guaranteed, The Easy Way!"
        ],
        "answer": "C"
    },
    {
        "question": "An ML engineer is deploying a trained model to an Amazon SageMaker endpoint. The ML\nengineer needs to receive alerts when data quality issues occur in production. Which solution\nwill meet this requirement?",
        "options": [
            "B. Integrate the SageMaker endpoint with a SageMaker Clarify processing job. Configure an\nAmazon CloudWatch alarm to provide alerts.",
            "C. Configure a monitoring job in SageMaker Model Monitor. Integrate Model Monitor with\nAmazon CloudWatch to provide alerts.",
            "D. Configure a data flow in SageMaker Data Wrangler. Integrate Data Wrangler with Amazon\nCloudWatch to provide alerts."
        ],
        "answer": "C",
        "explanation": "SageMaker Model Monitor is designed to continuously monitor deployed models for data\nquality issues such as data drift or violations of data constraints. By integrating Model Monitor\nwith Amazon CloudWatch, alerts can be automatically triggered when data quality issues\noccur in production."
    },
    {
        "question": "A company deployed an ML model that uses the XGBoost algorithm to predict product\nfailures.\nThe model is hosted on an Amazon SageMaker endpoint and is trained on normal operating\ndata.\nAn AWS Lambda function provides the predictions to the company's application.\nAn ML engineer must implement a solution that uses incoming live data to detect decreased\nmodel accuracy over time.\nWhich solution will meet these requirements?",
        "options": [
            "B. Modify the Lambda function to calculate model drift by using real-time inference data and\nmodel predictions. Program the Lambda function to send alerts.",
            "C. Schedule a monitoring job in SageMaker Model Monitor. Use the job to detect drift by\nanalyzing the live data against a baseline of the training data statistics and constraints.",
            "D. Schedule a monitoring job in SageMaker Debugger. Use the job to detect drift by\nanalyzing the live data against a baseline of the training data statistics and constraints."
        ],
        "answer": "C"
    },
    {
        "question": "A company's ML engineer has deployed an ML model for sentiment analysis to an Amazon\nSageMaker endpoint. The ML engineer needs to explain to company stakeholders how the\nmodel makes predictions.\nWhich solution will provide an explanation for the model's predictions?\n45\n\n\fIT Certification Guaranteed, The Easy Way!",
        "options": [
            "B. Use SageMaker Clarify on the deployed model.",
            "C. Show the distribution of inferences from A/?testing in Amazon CloudWatch.",
            "D. Add a shadow endpoint. Analyze prediction differences on samples."
        ],
        "answer": "B"
    },
    {
        "question": "An ML engineer needs to create data ingestion pipelines and ML model deployment pipelines\non AWS. All the raw data is stored in Amazon S3 buckets.\nWhich solution will meet these requirements?",
        "options": [
            "B. Use AWS Glue to create the data ingestion pipelines. Use Amazon SageMaker Studio\nClassic to create the model deployment pipelines.",
            "C. Use Amazon Redshift ML to create the data ingestion pipelines. Use Amazon SageMaker\nStudio Classic to create the model deployment pipelines.",
            "D. Use Amazon Athena to create the data ingestion pipelines. Use an Amazon SageMaker\nnotebook to create the model deployment pipelines."
        ],
        "answer": "B"
    },
    {
        "question": "A company has an ML model that generates text descriptions based on images that\ncustomers upload to the company's website. The images can be up to 50 MB in total size.\nAn ML engineer decides to store the images in an Amazon S3 bucket. The ML engineer must\nimplement a processing solution that can scale to accommodate changes in demand.\nWhich solution will meet these requirements with the LEAST operational overhead?",
        "options": [
            "B. Create an Amazon SageMaker Asynchronous Inference endpoint and a scaling policy.\nRun a script to make an inference request for each image.",
            "C. Create an Amazon Elastic Kubernetes Service (Amazon EKS) cluster that uses Karpenter\nfor auto scaling. Host the model on the EKS cluster. Run a script to make an inference\nrequest for each image.",
            "D. Create an AWS Batch job that uses an Amazon Elastic Container Service (Amazon ECS)\ncluster.Specify a list of images to process for each AWS Batch job."
        ],
        "answer": "B"
    },
    {
        "question": "A data scientist uses logistic regression to build a fraud detection model. While the model\naccuracy is 99%, 90% of the fraud cases are not detected by the model.\nWhat action will definitively help the model detect more than 10% of fraud cases?",
        "options": [
            "B. Decreasing the class probability threshold",
            "C. Using regularization to reduce overfitting",
            "D. Using oversampling to balance the dataset"
        ],
        "answer": "B",
        "explanation": "Decreasing the class probability threshold makes the model more sensitive and, therefore,\nmarks more cases as the positive class, which is fraud in this case. This will increase the\nlikelihood of fraud detection. However, it comes at the price of lowering precision."
    },
    {
        "question": "An insurance company needs to automate claim compliance reviews because human\nreviews are expensive and error-prone. The company has a large set of claims and a\ncompliance label for each. Each claim consists of a few sentences in English, many of which\n48\n\n\fIT Certification Guaranteed, The Easy Way!\n\ncontain complex related information. Management would like to use Amazon SageMaker\nbuilt-in algorithms to design a machine learning supervised model that can be trained to read\neach claim and predict if the claim is compliant or not.\nWhich approach should be used to extract features from the claims to be used as inputs for\nthe downstream supervised task?",
        "options": [
            "B. Apply Amazon SageMaker BlazingText in Word2Vec mode to claims in the training set.\nSend the derived features space as inputs for the downstream supervised task.",
            "C. Apply Amazon SageMaker BlazingText in classification mode to labeled claims in the\ntraining set to derive features for the claims that correspond to the compliant and noncompliant labels, respectively.",
            "D. Apply Amazon SageMaker Object2Vec to claims in the training set. Send the derived\nfeatures space as inputs for the downstream supervised task."
        ],
        "answer": "D",
        "explanation": "Amazon SageMaker Object2Vec generalizes the Word2Vec embedding technique for words\nto more complex objects, such as sentences and paragraphs. Since the supervised learning\ntask is at the level of whole claims, for which there are labels, and no labels are available at\nthe word level, Object2Vec needs be used instead of Word2Vec."
    },
    {
        "question": "A company has implemented a data ingestion pipeline for sales transactions from its\necommerce website. The company uses Amazon Data Firehose to ingest data into Amazon\nOpenSearch Service. The buffer interval of the Firehose stream is set for 60 seconds. An\nOpenSearch linear model generates real-time sales forecasts based on the data and\npresents the data in an OpenSearch dashboard.\nThe company needs to optimize the data ingestion pipeline to support sub-second latency for\nthe real-time dashboard.\nWhich change to the architecture will meet these requirements?",
        "options": [
            "B. Replace the Firehose stream with an AWS DataSync task. Configure the task with\nenhanced fan- out consumers.",
            "C. Increase the buffer interval of the Firehose stream from 60 seconds to 120 seconds.",
            "D. Replace the Firehose stream with an Amazon Simple Queue Service (Amazon SQS)\nqueue."
        ],
        "answer": "A"
    },
    {
        "question": "An ML engineer needs to deploy ML models to get inferences from large datasets in an\nasynchronous manner. The ML engineer also needs to implement scheduled monitoring of\nthe data quality of the models. The ML engineer must receive alerts when changes in data\nquality occur.\n49\n\n\fIT Certification Guaranteed, The Easy Way!\n\nWhich solution will meet these requirements?",
        "options": [
            "B. Deploy the models by using scheduled AWS Batch jobs. Use AWS CloudTrail to monitor\nthe data quality and to send alerts.",
            "C. Deploy the models by using Amazon Elastic Container Service (Amazon ECS) on AWS\nFargate.\nUse Amazon EventBridge to monitor the data quality and to send alerts.",
            "D. Deploy the models by using Amazon SageMaker batch transform. Use SageMaker Model\nMonitor to monitor the data quality and to send alerts."
        ],
        "answer": "D"
    },
    {
        "question": "A company uses Amazon SageMaker for its ML process. A compliance audit discovers that\nan Amazon S3 bucket for training data uses server-side encryption with S3 managed keys\n(SSE- S3).\nThe company requires customer managed keys. An ML engineer changes the S3 bucket to\nuse server-side encryption with AWS KMS keys (SSE-KMS). The ML engineer makes no\nother configuration changes.\nAfter the change to the encryption settings, SageMaker training jobs start to fail with\nAccessDenied errors.\nWhat should the ML engineer do to resolve this problem?",
        "options": [
            "B. Update the S3 bucket policy that is attached to the S3 bucket. Set the value of the\naws:SecureTransport condition key to True.",
            "C. Update the IAM policy that is attached to the execution role for the training jobs. Include\nthe kms:Encrypt and kms:Decrypt permissions.",
            "D. Update the IAM policy that is attached to the user that created the training jobs. Include\nthe kms:CreateGrant permission."
        ],
        "answer": "C"
    },
    {
        "question": "A company stores time-series data about user clicks in an Amazon S3 bucket. The raw data\nconsists of millions of rows of user activity every day. ML engineers access the data to\ndevelop their ML models.\nThe ML engineers need to generate daily reports and analyze click trends over the past 3\ndays by using Amazon Athena. The company must retain the data for 30 days before\narchiving the data.\nWhich solution will provide the HIGHEST performance for data retrieval?",
        "options": [
            "B. Create AWS Lambda functions to copy the time-series data into separate S3 buckets.\nApply S3 Lifecycle policies to archive data that is older than 30 days to S3 Glacier Flexible\n\n50\n\n\fIT Certification Guaranteed, The Easy Way!\n\nRetrieval.",
            "C. Organize the time-series data into partitions by date prefix in the S3 bucket. Apply S3\nLifecycle policies to archive partitions that are older than 30 days to S3 Glacier Flexible\nRetrieval.",
            "D. Put each day's time-series data into its own S3 bucket. Use S3 Lifecycle policies to\narchive S3 buckets that hold data that is older than 30 days to S3 Glacier Flexible Retrieval."
        ],
        "answer": "C"
    },
    {
        "question": "An ML engineer is evaluating several ML models and must choose one model to use in\nproduction. The cost of false negative predictions by the models is much higher than the cost\nof false positive predictions.\nWhich metric finding should the ML engineer prioritize the MOST when choosing the model?",
        "options": [
            "B. High precision",
            "C. Low recall",
            "D. High recall"
        ],
        "answer": "D"
    },
    {
        "question": "A company wants to improve the sustainability of its ML operations.\nWhich actions will reduce the energy usage and computational resources that are associated\nwith the company's training jobs? (Choose two.)",
        "options": [
            "B. Use Amazon SageMaker Ground Truth for data labeling.",
            "C. Deploy models by using AWS Lambda functions.",
            "D. Use AWS Trainium instances for training.",
            "E. Use PyTorch or TensorFlow with the distributed training option."
        ],
        "answer": "AD"
    },
    {
        "question": "A company needs to create a central catalog for all the company's ML models. The models\nare in AWS accounts where the company developed the models initially. The models are\nhosted in Amazon Elastic Container Registry (Amazon ECR) repositories.\nWhich solution will meet these requirements?",
        "options": [
            "B. Create a new AWS account with a new ECR repository as the central catalog. Configure\nECR cross-account replication between the initial ECR repositories and the central catalog.",
            "C. Use the Amazon SageMaker Model Registry to create a model group for models hosted in\nAmazon ECR. Create a new AWS account. In the new account, use the SageMaker Model\nRegistry as the central catalog. Attach a cross-account resource policy to each model group\nin the initial AWS accounts.\n51\n\n\fIT Certification Guaranteed, The Easy Way!",
            "D. Use an AWS Glue Data Catalog to store the models. Run an AWS Glue crawler to migrate\nthe models from the ECR repositories to the Data Catalog. Configure cross-account access\nto the Data Catalog."
        ],
        "answer": "C"
    },
    {
        "question": "A company has an ML model that is deployed to an Amazon SageMaker endpoint for realtime inference. The company needs to deploy a new model. The company must compare the\nnew model's performance to the currently deployed model's performance before shifting all\ntraffic to the new model. Which solution will meet these requirements with the LEAST\noperational effort?",
        "options": [
            "B. Deploy the new model to a separate endpoint. Use Amazon CloudFront to distribute traffic\nbetween the two endpoints.",
            "C. Deploy the new model as a shadow variant on the same endpoint as the current model.\nRoute a portion of live traffic to the shadow model for evaluation.",
            "D. Use AWS Lambda functions with custom logic to route traffic between the current model\nand the new model."
        ],
        "answer": "C",
        "explanation": "SageMaker supports shadow variant deployments, which allow a new model to run alongside\nthe current one on the same endpoint. A portion of live traffic is mirrored to the shadow model\nfor evaluation, while only the current model's output is returned to users. This provides the\nrequired comparison with minimal operational effort, avoiding the need for custom trafficsplitting solutions."
    },
    {
        "question": "A company runs an Amazon SageMaker domain in a public subnet of a newly created VPC.\nThe network is configured properly, and ML engineers can access the SageMaker domain.\nRecently, the company discovered suspicious traffic to the domain from a specific IP\naddress.\nThe company needs to block traffic from the specific IP address.\nWhich update to the network configuration will meet this requirement?",
        "options": [
            "B. Create a network ACL inbound rule to deny traffic from the specific IP address. Assign the\nrule to the default network Ad for the subnet where the domain is located.",
            "C. Create a shadow variant for the domain. Configure SageMaker Inference Recommender\nto send traffic from the specific IP address to the shadow endpoint.",
            "D. Create a VPC route table to deny inbound traffic from the specific IP address. Assign the\nroute table to the domain."
        ],
        "answer": "B"
    },
    {
        "question": "A medical company ingests streams of data from devices that monitor patients' vital signs.\nThe company uses Amazon SageMaker and plans to prepare ML models to predict adverse\nevents for patients. The dataset is large with thousands of features.\nAn ML engineer needs to run several hundred training iterations with different sets of\nfeatures, different algorithms, and many potential parameters. The ML engineer must\nimplement a solution to log the characteristics and results of each training iteration.\nWhich solution will meet these requirements with the LEAST implementation effort?",
        "options": [
            "B. Write the characteristics of each iteration to logs in Amazon S3. Use AWS Glue and\nAmazon Athena to search the logs.",
            "C. Use the SageMaker Model Registry to track the characteristics and results of each\niteration.",
            "D. Use SageMaker Experiments to track the characteristics and results of each iteration."
        ],
        "answer": "D",
        "explanation": "SageMaker Experiments is specifically designed to track and organize ML experiments,\nincluding characteristics such as features, algorithms, parameters, and results. It provides\nexperiment tracking with minimal implementation effort, making it the best fit for logging and\ncomparing multiple training iterations."
    },
    {
        "question": "An ML engineer needs to use AWS services to identify and extract meaningful unique\nkeywords from documents.\nWhich solution will meet these requirements with the LEAST operational overhead?",
        "options": [
            "B. Use Amazon SageMaker and the BlazingText algorithm. Apply custom pre-processing\nsteps for stemming and removal of stop words. Calculate term frequency-inverse document\nfrequency (TF- IDF) scores to identify and extract relevant keywords.",
            "C. Store the documents in an Amazon S3 bucket. Create AWS Lambda functions to process\nthe documents and to run Python scripts for stemming and removal of stop words. Use\nbigram and trigram techniques to identify and extract relevant keywords.",
            "D. Use Amazon Comprehend custom entity recognition and key phrase extraction to identify\nand extract relevant keywords."
        ],
        "answer": "D"
    },
    {
        "question": "A company is interested in building a fraud detection model. Currently, the data scientist does\nnot have a sufficient amount of information due to the low number of fraud cases.\nWhich method is MOST likely to detect the GREATEST number of valid fraud cases?",
        "options": [
            "B. Undersampling",
            "C. Oversampling using SMOTE",
            "D. Class weight adjustment"
        ],
        "answer": "C",
        "explanation": "With datasets that are not fully populated, the Synthetic Minority Over-sampling Technique\n(SMOTE. adds new information by adding synthetic data points to the minority class. This\ntechnique would be the most effective in this scenario."
    },
    {
        "question": "A manufacturing company uses an ML model to determine whether products meet a\nstandard for quality. The model produces an output of \"Passed\" or \"Failed.\" Robots separate\nthe products into the two categories by using the model to analyze photos on the assembly\nline.\nWhich metrics should the company use to evaluate the model's performance? (Choose two.)",
        "options": [
            "B. Root mean square error (RMSE) and mean absolute percentage error (MAPE)",
            "C. Accuracy and F1 score",
            "D. Bilingual Evaluation Understudy (BLEU) score",
            "E. Perplexity"
        ],
        "answer": "AC"
    },
    {
        "question": "An ML engineer needs to use AWS CloudFormation to create an ML model that an Amazon\nSageMaker endpoint will host.\nWhich resource should the ML engineer declare in the CloudFormation template to meet this\nrequirement?",
        "options": [
            "B. AWS::SageMaker::Endpoint",
            "C. AWS::SageMaker::NotebookInstance",
            "D. AWS::SageMaker::Pipeline"
        ],
        "answer": "A"
    },
    {
        "question": "A company wants to predict the success of advertising campaigns by considering the color\nscheme of each advertisement. An ML engineer is preparing data for a neural network\nmodel.\nThe dataset includes color information as categorical data.\nWhich technique for feature engineering should the ML engineer use for the model?",
        "options": [
            "B. Implement padding to ensure that all color feature vectors have the same length.",
            "C. Perform dimensionality reduction on the color categories.",
            "D. One-hot encode the color categories to transform the color scheme feature into a binary\n56\n\n\fIT Certification Guaranteed, The Easy Way!\n\nmatrix."
        ],
        "answer": "D"
    },
    {
        "question": "Case Study\nAn ML engineer is developing a fraud detection model on AWS. The training dataset includes\ntransaction logs, customer profiles, and tables from an on-premises MySQL database. The\ntransaction logs and customer profiles are stored in Amazon S3.\nThe dataset has a class imbalance that affects the learning of the model's algorithm.\nAdditionally, many of the features have interdependencies. The algorithm is not capturing all\nthe desired underlying patterns in the data.\nThe training dataset includes categorical data and numerical data. The ML engineer must\nprepare the training dataset to maximize the accuracy of the model.\nWhich action will meet this requirement with the LEAST operational overhead?",
        "options": [
            "B. Use AWS Glue to transform the numerical data into categorical data.",
            "C. Use Amazon SageMaker Data Wrangler to transform the categorical data into numerical\ndata.",
            "D. Use Amazon SageMaker Data Wrangler to transform the numerical data into categorical\ndata."
        ],
        "answer": "C"
    },
    {
        "question": "An ML engineer needs to use an Amazon EMR cluster to process large volumes of data in\nbatches. Any data loss is unacceptable.\nWhich instance purchasing option will meet these requirements MOST cost-effectively?",
        "options": [
            "B. Run the primary node, core nodes, and task nodes on Spot Instances.",
            "C. Run the primary node on an On-Demand Instance. Run the core nodes and task nodes on\nSpot Instances.",
            "D. Run the primary node and core nodes on On-Demand Instances. Run the task nodes on\nSpot Instances."
        ],
        "answer": "D"
    },
    {
        "question": "A company is developing a new ML model that uses the XGBoost algorithm. The company\nwill train the model on data that is stored in an Amazon S3 bucket. The data is in a nested\nJSON format.\nAn ML engineer needs to convert the JSON files into a tabular format.\nWhich solution will meet this requirement with the LEAST operational overhead?",
        "options": [
            "B. Write custom Scala code to convert the files. Use Amazon EMR Serverless to run the\nScala code.\n\n57\n\n\fIT Certification Guaranteed, The Easy Way!",
            "C. Create an AWS Lambda function that uses a Python runtime and invokes the reduce()\nfunction to convert the files. Invoke the Lambda function.",
            "D. Create an Amazon Athena database that is based on the JSON files. Use the Athena\nflatten function to convert the data."
        ],
        "answer": "A",
        "explanation": "The AWS Glue PySpark Relationalize transform is purpose-built to convert nested JSON into\ntabular format with minimal operational overhead. It automates the flattening process without\nrequiring custom code or complex infrastructure, making it the most efficient solution for\npreparing the data for XGBoost training."
    },
    {
        "question": "An ML engineer needs to process thousands of existing CSV objects and new CSV objects\nthat are uploaded. The CSV objects are stored in a central Amazon S3 bucket and have the\nsame number of columns. One of the columns is a transaction date. The ML engineer must\nquery the data based on the transaction date.\nWhich solution will meet these requirements with the LEAST operational overhead?",
        "options": [
            "B. Create a new S3 bucket for processed data. Set up S3 replication from the central S3\nbucket to the new S3 bucket. Use S3 Object Lambda to query the objects based on\ntransaction date.",
            "C. Create a new S3 bucket for processed data. Use AWS Glue for Apache Spark to create a\njob to query the CSV objects based on transaction date. Configure the job to store the results\nin the new S3 bucket. Query the objects from the new S3 bucket.",
            "D. Create a new S3 bucket for processed data. Use Amazon Data Firehose to transfer the\ndata from the central S3 bucket to the new S3 bucket. Configure Firehose to run an AWS\nLambda function to query the data based on transaction date."
        ],
        "answer": "A"
    },
    {
        "question": "A company has a team of data scientists who use Amazon SageMaker notebook instances to\ntest ML models. When the data scientists need new permissions, the company attaches the\npermissions to each individual role that was created during the creation of the SageMaker\nnotebook instance.\nThe company needs to centralize management of the team's permissions.\nWhich solution will meet this requirement?",
        "options": [
            "B. Create a single IAM group. Add the data scientists to the group. Associate the group with\neach notebook instance that the team uses.",
            "C. Create a single IAM user. Attach the AdministratorAccess AWS managed IAM policy to the\nuser.\n\n58\n\n\fIT Certification Guaranteed, The Easy Way!\n\nConfigure each notebook instance to use the IAM user.",
            "D. Create a single IAM group. Add the data scientists to the group. Create an IAM role.\nAttach the AdministratorAccess AWS managed IAM policy to the role. Associate the role with\nthe group.Associate the group with each notebook instance that the team uses."
        ],
        "answer": "A"
    },
    {
        "question": "An advertising company uses AWS Lake Formation to manage a data lake. The data lake\ncontains structured data and unstructured data. The company's ML engineers are assigned\nto specific advertisement campaigns.\nThe ML engineers must interact with the data through Amazon Athena and by browsing the\ndata directly in an Amazon S3 bucket. The ML engineers must have access to only the\nresources that are specific to their assigned advertisement campaigns.\nWhich solution will meet these requirements in the MOST operationally efficient way?",
        "options": [
            "B. Store users and campaign information in an Amazon DynamoDB table. Configure\nDynamoDB Streams to invoke an AWS Lambda function to update S3 bucket policies.",
            "C. Use Lake Formation to authorize AWS Glue to access the S3 bucket. Configure Lake\nFormation tags to map ML engineers to their campaigns.",
            "D. Configure S3 bucket policies to restrict access to the S3 bucket based on the ML\nengineers' campaigns."
        ],
        "answer": "C"
    },
    {
        "question": "A company wants to reduce the cost of its containerized ML applications. The applications\nuse ML models that run on Amazon EC2 instances, AWS Lambda functions, and an Amazon\nElastic Container Service (Amazon ECS) cluster. The EC2 workloads and ECS workloads\nuse Amazon Elastic Block Store (Amazon EBS) volumes to save predictions and artifacts.\nAn ML engineer must identify resources that are being used inefficiently. The ML engineer\nalso must generate recommendations to reduce the cost of these resources.\nWhich solution will meet these requirements with the LEAST development effort?",
        "options": [
            "B. Add cost allocation tags to the resources. Activate the tags in AWS Billing and Cost\nManagement.",
            "C. Check AWS CloudTrail event history for the creation of the resources.",
            "D. Run AWS Compute Optimizer."
        ],
        "answer": "D"
    },
    {
        "question": "Case Study\nA company is building a web-based AI application by using Amazon SageMaker. The\napplication will provide the following capabilities and features: ML experimentation, training, a\ncentral model registry, model deployment, and model monitoring.\n\n59\n\n\fIT Certification Guaranteed, The Easy Way!\n\nThe application must ensure secure and isolated use of training data during the ML lifecycle.\nThe training data is stored in Amazon S3.\nThe company must implement a manual approval-based workflow to ensure that only\napproved models can be deployed to production endpoints.\nWhich solution will meet this requirement?",
        "options": [
            "B. Use SageMaker ML Lineage Tracking on the central model registry. Create tracking\nentities for the approval process.",
            "C. Use SageMaker Model Monitor to evaluate the performance of the model and to manage\nthe approval.",
            "D. Use SageMaker Pipelines. When a model version is registered, use the AWS SDK to\nchange the approval status to \"Approved.\""
        ],
        "answer": "D"
    },
    {
        "question": "Case Study\nAn ML engineer is developing a fraud detection model on AWS. The training dataset includes\ntransaction logs, customer profiles, and tables from an on-premises MySQL database. The\ntransaction logs and customer profiles are stored in Amazon S3.\nThe dataset has a class imbalance that affects the learning of the model's algorithm.\nAdditionally, many of the features have interdependencies. The algorithm is not capturing all\nthe desired underlying patterns in the data.\nThe ML engineer needs to use an Amazon SageMaker built-in algorithm to train the model.\nWhich algorithm should the ML engineer use to meet this requirement?",
        "options": [
            "B. Linear learner",
            "C. K-means clustering",
            "D. Neural Topic Model (NTM)"
        ],
        "answer": "A"
    },
    {
        "question": "A company needs to run a batch data-processing job on Amazon EC2 instances. The job will\nrun during the weekend and will take 90 minutes to finish running. The processing can handle\ninterruptions. The company will run the job every weekend for the next 6 months.\nWhich EC2 instance purchasing option will meet these requirements MOST cost-effectively?",
        "options": [
            "B. Reserved Instances",
            "C. On-Demand Instances",
            "D. Dedicated Instances"
        ],
        "answer": "A"
    },
    {
        "question": "A company uses a hybrid cloud environment. A model that is deployed on premises uses\ndata in Amazon 53 to provide customers with a live conversational engine.\n\n60\n\n\fIT Certification Guaranteed, The Easy Way!\n\nThe model is using sensitive data. An ML engineer needs to implement a solution to identify\nand remove the sensitive data.\nWhich solution will meet these requirements with the LEAST operational overhead?",
        "options": [
            "B. Deploy the model on an Amazon Elastic Container Service (Amazon ECS) cluster that\nuses AWS Fargate. Create an AWS Batch job to identify and remove the sensitive data.",
            "C. Use Amazon Macie to identify the sensitive data. Create a set of AWS Lambda functions\nto remove the sensitive data.",
            "D. Use Amazon Comprehend to identify the sensitive data. Launch Amazon EC2 instances to\nremove the sensitive data."
        ],
        "answer": "C"
    },
    {
        "question": "A company must install a custom script on any newly created Amazon SageMaker notebook\ninstances.\nWhich solution will meet this requirement with the LEAST operational overhead?",
        "options": [
            "B. Create a custom Amazon Elastic Container Registry (Amazon ECR) image that contains\nthe custom script. Push the ECR image to a Docker registry. Attach the Docker image to a\nSageMaker Studio domain. Select the kernel to run as part of the SageMaker notebook.",
            "C. Create a custom package index repository. Use AWS CodeArtifact to manage the\ninstallation of the custom script. Set up AWS PrivateLink endpoints to connect CodeArtifact to\nthe SageMaker instance. Install the script.",
            "D. Store the custom script in Amazon S3. Create an AWS Lambda function to install the\ncustom script on new SageMaker notebooks. Configure Amazon EventBridge to invoke the\nLambda function when a new SageMaker notebook is initialized."
        ],
        "answer": "A"
    },
    {
        "question": "A company stores training data as a .csv file in an Amazon S3 bucket. The company must\nencrypt the data and must control which applications have access to the encryption key.\nWhich solution will meet these requirements?",
        "options": [
            "B. Create a new API key by using the Amazon API Gateway CreateApiKey API operation.\nUse the AWS CLI with a reference to the new API key to encrypt the file.",
            "C. Create a new IAM role. Attach a policy that allows the AWS Key Management Service\n(AWS KMS) GenerateDataKey action. Use the role to encrypt the file.",
            "D. Create a new AWS Key Management Service (AWS KMS) key. Use the AWS Encryption\nCLI with a reference to the new KMS key to encrypt the file."
        ],
        "answer": "D",
        "explanation": "The correct approach is to create a new AWS KMS key and use the AWS Encryption CLI to\nencrypt the file with that key. This ensures the data in S3 is encrypted, and access to the\nencryption key can be controlled through KMS key policies and IAM permissions, meeting\nboth encryption and access control requirements."
    },
    {
        "question": "A company is building a deep learning model on Amazon SageMaker. The company uses a\nlarge amount of data as the training dataset. The company needs to optimize the model's\nhyperparameters to minimize the loss function on the validation dataset.\nWhich hyperparameter tuning strategy will accomplish this goal with the LEAST computation\ntime?",
        "options": [
            "B. Grid search",
            "C. Bayesian optimization",
            "D. Random search"
        ],
        "answer": "A"
    },
    {
        "question": "A company has an ML model that uses historical transaction data to predict customer\nbehavior.\nAn ML engineer is optimizing the model in Amazon SageMaker to enhance the model's\npredictive accuracy. The ML engineer must examine the input data and the resulting\npredictions to identify trends that could skew the model's performance across different\ndemographics.\nWhich solution will provide this level of analysis?",
        "options": [
            "B. Create AWS Glue DataBrew recipes to correct the data based on statistics from the model\noutput.",
            "C. Use SageMaker Clarify to evaluate the model and training data for underlying patterns that\nmight affect accuracy.",
            "D. Create AWS Lambda functions to automate data pre-processing and to ensure consistent\nquality of input data for the model."
        ],
        "answer": "C"
    },
    {
        "question": "A company needs to use Retrieval Augmented Generation (RAG) to supplement an open\nsource large language model (LLM) that runs on Amazon Bedrock. The company's data for\nRAG is a set of documents in an Amazon S3 bucket. The documents consist of .csv files and\n.docx files.\nWhich solution will meet these requirements with the LEAST operational overhead?",
        "options": [
            "B. Convert the data into vectors. Store the data in an Amazon Neptune database. Connect\nthe database to Amazon Bedrock. Call the Amazon Bedrock API to perform RAG queries.",
            "C. Fine-tune an existing LLM by using an AutoML job in Amazon SageMaker. Configure the\nS3 bucket as a data source for the AutoML job. Deploy the LLM to a SageMaker endpoint.\nUse the endpoint to perform RAG queries.",
            "D. Create a knowledge base for Amazon Bedrock. Configure a data source that references\nthe S3 bucket. Use the Amazon Bedrock API to perform RAG queries."
        ],
        "answer": "D"
    },
    {
        "question": "A company shares Amazon SageMaker Studio notebooks that are accessible through a\nVPN.\nThe company must enforce access controls to prevent malicious actors from exploiting\npresigned URLs to access the notebooks.\nWhich solution will meet these requirements?",
        "options": [
            "B. Set up Studio client VPC validation by using the aws:sourceVpc IAM policy condition.",
            "C. Set up Studio client role endpoint validation by using the aws:PrimaryTag IAM policy\ncondition.",
            "D. Set up Studio client user endpoint validation by using the aws:PrincipalTag IAM policy\ncondition."
        ],
        "answer": "A"
    },
    {
        "question": "A company is gathering audio, video, and text data in various languages. The company\nneeds to use a large language model (LLM) to summarize the gathered data that is in\nSpanish.\nWhich solution will meet these requirements in the LEAST amount of time?",
        "options": [
            "B. Use Amazon Transcribe and Amazon Translate to convert the data into English text. Use\nAmazon Bedrock with the Jurassic model to summarize the text.",
            "C. Use Amazon Rekognition and Amazon Translate to convert the data into English text. Use\nAmazon Bedrock with the Anthropic Claude model to summarize the text.",
            "D. Use Amazon Comprehend and Amazon Translate to convert the data into English text.\nUse Amazon Bedrock with the Stable Diffusion model to summarize the text."
        ],
        "answer": "B"
    },
    {
        "question": "A company needs to extract entities from a PDF document to build a classifier model.\nWhich solution will extract and store the entities in the LEAST amount of time?",
        "options": [
            "B. Use an open source AI optical character recognition (OCR) tool on Amazon SageMaker to\nextract the entities. Store the output in Amazon S3.\n63\n\n\fIT Certification Guaranteed, The Easy Way!",
            "C. Use Amazon Textract to extract the entities. Use Amazon Comprehend to convert the\nentities to text. Store the output in Amazon S3.",
            "D. Use Amazon Textract integrated with Amazon Augmented AI (Amazon A2I) to extract the\nentities.Store the output in Amazon S3."
        ],
        "answer": "C"
    },
    {
        "question": "A company is using Amazon EMR. The company has a large dataset in Amazon S3 that\nneeds to be ingested into Amazon SageMaker Feature Store. The dataset contains historical\ndata and real-time streaming data.\nThe company must ensure that the Feature Store online store is updated with the most\nrecent data as soon as the data becomes available. The company also must maintain a\ncomplete Feature Store offline store for batch processing.\nWhich solution will meet these requirements?",
        "options": [
            "B. Use the PutRecord API in Feature Store Runtime to ingest all the data into the offline\nstore.",
            "C. Use the Feature Store Spark connector to ingest the data as Spark DataFrames with the\nonline store and offline store enabled.",
            "D. Use the Feature Store Spark connector to ingest the data as Spark DataFrames with only\nthe online store enabled."
        ],
        "answer": "C",
        "explanation": "The SageMaker Feature Store Spark connector allows ingestion of large-scale data from\nAmazon EMR into Feature Store as Spark DataFrames. Enabling both the online store\nensures real-time updates for the latest data, while the offline store maintains the full\nhistorical dataset for batch analytics. This setup meets both low-latency and historical\nprocessing requirements."
    },
    {
        "question": "A company is planning to create several ML prediction models. The training data is stored in\nAmazon S3. The entire dataset is more than 5 TB in size and consists of CSV, JSON,\nApache Parquet, and simple text files.\nThe data must be processed in several consecutive steps. The steps include complex\nmanipulations that can take hours to finish running. Some of the processing involves natural\nlanguage processing (NLP) transformations. The entire process must be automated.\nWhich solution will meet these requirements?",
        "options": [
            "B. Use Amazon SageMaker notebooks for each data processing step. Automate the process\nby using Amazon EventBridge.",
            "C. Process data at each step by using AWS Lambda functions. Automate the process by\nusing AWS Step Functions and Amazon EventBridge.\n\n64\n\n\fIT Certification Guaranteed, The Easy Way!",
            "D. Use Amazon SageMaker Pipelines to create a pipeline of data processing steps.\nAutomate the pipeline by using Amazon EventBridge."
        ],
        "answer": "D"
    },
    {
        "question": "A company has an existing Amazon SageMaker model (v1) on a production endpoint. The\ncompany develops a new model version (v2) and needs to test v2 in production before\nsubstituting v2 for v1.\nThe company needs to implement a solution to minimize the risk of v2 generating incorrect\noutput in production. The solution must prevent any disruption of production traffic during the\nchange to v2.\nWhich solution will meet these requirements?",
        "options": [
            "B. Create a second production variant for v2. Assign 10% of the traffic to v2 and 90% of the\ntraffic to v1. Collect all the output of v2 in an Amazon S3 bucket. If v2 performs as expected,\nswitch all the traffic to v2.",
            "C. Deploy v2 to a new endpoint. Turn on data capturing for the production endpoint. Write a\nscript to pass 100% of input data to v2. If v2 performs as expected, deactivate the v1\nendpoint and direct the traffic to v2.",
            "D. Deploy v2 into a shadow variant that samples 100% of the inference requests. Collect all\nthe output in an Amazon S3 bucket. If v2 performs as expected, promote v2 to production."
        ],
        "answer": "D",
        "explanation": "A shadow variant allows the new model (v2) to receive a copy of 100% of production traffic\nwhile only v1's outputs are returned to users. This enables safe side-by-side evaluation of v2\nwithout impacting production responses, minimizing risk and ensuring no disruption of live\ntraffic until v2 is validated and promoted."
    },
    {
        "question": "Case Study\nA company is building a web-based AI application by using Amazon SageMaker. The\napplication will provide the following capabilities and features: ML experimentation, training, a\ncentral model registry, model deployment, and model monitoring.\nThe application must ensure secure and isolated use of training data during the ML lifecycle.\nThe training data is stored in Amazon S3.\nThe company is experimenting with consecutive training jobs.\nHow can the company MINIMIZE infrastructure startup times for these jobs?",
        "options": [
            "B. Use SageMaker managed warm pools.",
            "C. Use SageMaker Training Compiler.",
            "D. Use the SageMaker distributed data parallelism (SMDDP) library."
        ],
        "answer": "B"
    },
    {
        "question": "A company needs to host a custom ML model to perform forecast analysis. The forecast\nanalysis will occur with predictable and sustained load during the same 2-hour period every\nday.\nMultiple invocations during the analysis period will require quick responses. The company\nneeds AWS to manage the underlying infrastructure and any auto scaling activities.\nWhich solution will meet these requirements?",
        "options": [
            "B. Configure an Auto Scaling group of Amazon EC2 instances to use scheduled scaling.",
            "C. Use Amazon SageMaker Serverless Inference with provisioned concurrency.",
            "D. Run the model on an Amazon Elastic Kubernetes Service (Amazon EKS) cluster on\nAmazon EC2 with pod auto scaling."
        ],
        "answer": "C"
    },
    {
        "question": "A company has a Retrieval Augmented Generation (RAG) application that uses a vector\ndatabase to store embeddings of documents. The company must migrate the application to\nAWS and must implement a solution that provides semantic search of text files. The\ncompany has already migrated the text repository to an Amazon S3 bucket.\nWhich solution will meet these requirements?",
        "options": [
            "B. Use a custom Amazon SageMaker notebook to run a custom script to generate\nembeddings. Use SageMaker Feature Store to store the embeddings. Use SQL queries to\nperform the semantic searches.",
            "C. Use the Amazon Kendra S3 connector to ingest the documents from the S3 bucket into\nAmazon Kendra. Query Amazon Kendra to perform the semantic searches.",
            "D. Use an Amazon Textract asynchronous job to ingest the documents from the S3 bucket.\nQuery Amazon Textract to perform the semantic searches."
        ],
        "answer": "C"
    },
    {
        "question": "Case Study\nA company is building a web-based AI application by using Amazon SageMaker. The\napplication will provide the following capabilities and features: ML experimentation, training, a\ncentral model registry, model deployment, and model monitoring.\nThe application must ensure secure and isolated use of training data during the ML lifecycle.\nThe training data is stored in Amazon S3.\nThe company needs to use the central model registry to manage different versions of models\nin the application.\nWhich action will meet this requirement with the LEAST operational overhead?",
        "options": [
            "B. Use Amazon Elastic Container Registry (Amazon ECR) and unique tags for each model\n\n66\n\n\fIT Certification Guaranteed, The Easy Way!\n\nversion.",
            "C. Use the SageMaker Model Registry and model groups to catalog the models.",
            "D. Use the SageMaker Model Registry and unique tags for each model version."
        ],
        "answer": "C"
    },
    {
        "question": "A company is planning to use an Amazon SageMaker prebuilt algorithm to create a\nrecommendation model. The algorithm must be able to make predictions on high-dimensional\nsparse data. Which SageMaker algorithm should the company choose for the\nrecommendation model?",
        "options": [
            "B. Factorization Machines",
            "C. Principal component analysis (PCA)",
            "D. Sequence-to-Sequence (seq2seq)"
        ],
        "answer": "B",
        "explanation": "The Factorization Machines algorithm in SageMaker is specifically designed for\nrecommendation systems and works well with high-dimensional sparse data such as useritem interactions. It efficiently models variable interactions and is the best choice for building\na recommendation model in this scenario."
    },
    {
        "question": "An ML engineer needs to use metrics to assess the quality of a time-series forecasting\nmodel.\nWhich metrics apply to this model? (Choose two.)",
        "options": [
            "B. LogLoss",
            "C. Root mean square error (RMSE)",
            "D. InferenceLatency",
            "E. Average weighted quantile loss (wQL)"
        ],
        "answer": "CE"
    },
    {
        "question": "An ML engineer needs to use Amazon SageMaker to fine-tune a large language model (LLM)\nfor text summarization. The ML engineer must follow a low-code no-code (LCNC) approach.\nWhich solution will meet these requirements?",
        "options": [
            "B. Use SageMaker Autopilot to fine-tune an LLM that is deployed by a custom API endpoint.",
            "C. Use SageMaker Autopilot to fine-tune an LLM that is deployed on Amazon EC2 instances.",
            "D. Use SageMaker Autopilot to fine-tune an LLM that is deployed by SageMaker JumpStart."
        ],
        "answer": "D"
    },
    {
        "question": "A company regularly receives new training data from the vendor of an ML model. The vendor\n67\n\n\fIT Certification Guaranteed, The Easy Way!\n\ndelivers cleaned and prepared data to the company's Amazon S3 bucket every 3-4 days.\nThe company has an Amazon SageMaker pipeline to retrain the model. An ML engineer\nneeds to implement a solution to run the pipeline when new data is uploaded to the S3\nbucket.\nWhich solution will meet these requirements with the LEAST operational effort?",
        "options": [
            "B. Create an AWS Lambda function that scans the S3 bucket. Program the Lambda function\nto initiate the pipeline when new data is uploaded.",
            "C. Create an Amazon EventBridge rule that has an event pattern that matches the S3 upload.\nConfigure the pipeline as the target of the rule.",
            "D. Use Amazon Managed Workflows for Apache Airflow (Amazon MWAA) to orchestrate the\npipeline when new data is uploaded."
        ],
        "answer": "C"
    },
    {
        "question": "An ML engineer has trained a neural network by using stochastic gradient descent (SGD).\nThe neural network performs poorly on the test set. The values for training loss and validation\nloss remain high and show an oscillating pattern. The values decrease for a few epochs and\nthen increase for a few epochs before repeating the same cycle.\nWhat should the ML engineer do to improve the training process?",
        "options": [
            "B. Increase the size of the test set.",
            "C. Increase the learning rate.",
            "D. Decrease the learning rate."
        ],
        "answer": "D"
    },
    {
        "question": "An ML engineer normalized training data by using min-max normalization in AWS Glue\nDataBrew. The ML engineer must normalize the production inference data in the same way\nas the training data before passing the production inference data to the model for predictions.\nWhich solution will meet this requirement?",
        "options": [
            "B. Keep the min-max normalization statistics from the training set. Use these values to\nnormalize the production samples.",
            "C. Calculate a new set of min-max normalization statistics from a batch of production\nsamples. Use these values to normalize all the production samples.",
            "D. Calculate a new set of min-max normalization statistics from each production sample. Use\nthese values to normalize all the production samples."
        ],
        "answer": "B"
    },
    {
        "question": "A company uses Amazon Athena to query a dataset in Amazon S3. The dataset has a target\nvariable that the company wants to predict.\n68\n\n\fIT Certification Guaranteed, The Easy Way!\n\nThe company needs to use the dataset in a solution to determine if a model can predict the\ntarget variable.\nWhich solution will provide this information with the LEAST development effort?",
        "options": [
            "B. Implement custom scripts to perform data pre-processing, multiple linear regression, and\nperformance evaluation. Run the scripts on Amazon EC2 instances.",
            "C. Configure Amazon Macie to analyze the dataset and to create a model. Report the\nmodel's achieved performance.",
            "D. Select a model from Amazon Bedrock. Tune the model with the data. Report the model's\nachieved performance."
        ],
        "answer": "A"
    },
    {
        "question": "An ML engineer is using a training job to fine-tune a deep learning model in Amazon\nSageMaker Studio. The ML engineer previously used the same pre-trained model with a\nsimilar dataset. The ML engineer expects vanishing gradient, underutilized GPU, and\noverfitting problems.\nThe ML engineer needs to implement a solution to detect these issues and to react in\npredefined ways when the issues occur. The solution also must provide comprehensive realtime metrics during the training.\nWhich solution will meet these requirements with the LEAST operational overhead?",
        "options": [
            "B. Use Amazon CloudWatch default metrics to gain insights about the training job. Use the\nmetrics to invoke an AWS Lambda function to initiate the predefined actions.",
            "C. Expand the metrics in Amazon CloudWatch to include the gradients in each training step.\nUse the metrics to invoke an AWS Lambda function to initiate the predefined actions.",
            "D. Use SageMaker Debugger built-in rules to monitor the training job. Configure the rules to\ninitiate the predefined actions."
        ],
        "answer": "D"
    },
    {
        "question": "An ML engineer needs to use an ML model to predict the price of apartments in a specific\nlocation.\nWhich metric should the ML engineer use to evaluate the model's performance?",
        "options": [
            "B. Area Under the ROC Curve (AUC)",
            "C. F1 score",
            "D. Mean absolute error (MAE)"
        ],
        "answer": "D"
    },
    {
        "question": "Case Study\nAn ML engineer is developing a fraud detection model on AWS. The training dataset includes\ntransaction logs, customer profiles, and tables from an on-premises MySQL database. The\ntransaction logs and customer profiles are stored in Amazon S3.\nThe dataset has a class imbalance that affects the learning of the model's algorithm.\nAdditionally, many of the features have interdependencies. The algorithm is not capturing all\nthe desired underlying patterns in the data.\nAfter the data is aggregated, the ML engineer must implement a solution to automatically\ndetect anomalies in the data and to visualize the result.\nWhich solution will meet these requirements?",
        "options": [
            "A. Use Amazon Athena to automatically detect the anomalies and to visualize the result.\nB. Use Amazon Redshift Spectrum to automatically detect the anomalies. Use Amazon\nQuickSight to visualize the result.\nC. Use Amazon SageMaker Data Wrangler to automatically detect the anomalies and to\nvisualize the result.\nD. Use AWS Batch to automatically detect the anomalies. Use Amazon QuickSight to\nvisualize the result."
        ],
        "answer": "C"
    },
    {
        "question": "An ML engineer has deployed an Amazon SageMaker model to a serverless endpoint in\nproduction. The model is invoked by the InvokeEndpoint API operation.\nThe model's latency in production is higher than the baseline latency in the test environment.\nThe ML engineer thinks that the increase in latency is because of model startup time.\nWhat should the ML engineer do to confirm or deny this hypothesis?",
        "options": [
            "A. Schedule a SageMaker Model Monitor job. Observe metrics about model quality.\nB. Schedule a SageMaker Model Monitor job with Amazon CloudWatch metrics enabled.\nC. Enable Amazon CloudWatch metrics. Observe the ModelSetupTime metric in the\nSageMaker namespace.\nD. Enable Amazon CloudWatch metrics. Observe the ModelLoadingWaitTime metric in the\nSageMaker namespace."
        ],
        "answer": "D"
    },
    {
        "question": "An ML engineer needs to ensure that a dataset complies with regulations for personally\nidentifiable information (PII). The ML engineer will use the data to train an ML model on\nAmazon SageMaker instances. SageMaker must not use any of the PII.\nWhich solution will meet these requirements in the MOST operationally efficient way?",
        "options": [
            "A. Use the Amazon Comprehend DetectPiiEntities API call to redact the PII from the data.\nStore the data in an Amazon S3 bucket. Access the S3 bucket from the SageMaker\ninstances for model training.\nB. Use the Amazon Comprehend DetectPiiEntities API call to redact the PII from the data.\nStore the data in an Amazon Elastic File System (Amazon EFS) file system. Mount the EFS\nfile system to the SageMaker instances for model training.\nC. Use AWS Glue DataBrew to cleanse the dataset of PII. Store the data in an Amazon\nElastic File System (Amazon EFS) file system. Mount the EFS file system to the SageMaker\ninstances for model training.\nD. Use Amazon Macie for automatic discovery of PII in the data. Remove the PII. Store the\ndata in an Amazon S3 bucket. Mount the S3 bucket to the SageMaker instances for model\ntraining."
        ],
        "answer": "A"
    },
    {
        "question": "An ML engineer trained an ML model on Amazon SageMaker to detect automobile accidents\nfrom dosed-circuit TV footage. The ML engineer used SageMaker Data Wrangler to create a\ntraining dataset of images of accidents and non-accidents.\nThe model performed well during training and validation. However, the model is\nunderperforming in production because of variations in the quality of the images from various\ncameras.\nWhich solution will improve the model's accuracy in the LEAST amount of time?",
        "options": [
            "A. Collect more images from all the cameras. Use Data Wrangler to prepare a new training\ndataset.\nB. Recreate the training dataset by using the Data Wrangler corrupt image transform. Specify\nthe impulse noise option.\nC. Recreate the training dataset by using the Data Wrangler enhance image contrast\ntransform.\nSpecify the Gamma contrast option.\nD. Recreate the training dataset by using the Data Wrangler resize image transform. Crop all\nimages to the same size."
        ],
        "answer": "C"
    },
    {
        "question": "A data scientist is working on optimizing a model during the training process by varying\nmultiple parameters. The data scientist observes that, during multiple runs with identical\nparameters, the loss function converges to different, yet stable, values.\nWhat should the data scientist do to improve the training process?",
        "options": [
            "A. Increase the learning rate. Keep the batch size the same.\nB. Decrease the learning rate. Reduce the batch size.\nC. Decrease the learning rate. Keep the batch size the same.\nD. Do not change the learning rate. Increase the batch size."
        ],
        "answer": "B",
        "explanation": "It is most likely that the loss function is very curvy and has multiple local minima where the\ntraining is getting stuck. Decreasing the batch size would help the data scientist stochastically\nget out of the local minima saddles. Decreasing the learning rate would prevent overshooting\nthe global loss function minimum."
    },
    {
        "question": "An ML engineer notices class imbalance in an image classification training job.\nWhat should the ML engineer do to resolve this issue?",
        "options": [
            "A. Reduce the size of the dataset.\nB. Transform some of the images in the dataset.\nC. Apply random oversampling on the dataset.\nD. Apply random data splitting on the dataset."
        ],
        "answer": "C"
    },
    {
        "question": "A company's ML engineer has deployed an ML model for sentiment analysis to an Amazon\nSageMaker endpoint. The ML engineer needs to explain to company stakeholders how the\nmodel makes predictions.\nWhich solution will provide an explanation for the model's predictions?",
        "options": [
            "A. Use SageMaker Model Monitor on the deployed model.\nB. Use SageMaker Clarify on the deployed model.\nC. Show the distribution of inferences from A/?testing in Amazon CloudWatch.\nD. Add a shadow endpoint. Analyze prediction differences on samples."
        ],
        "answer": "B"
    },
    {
        "question": "An insurance company needs to automate claim compliance reviews because human\nreviews are expensive and error-prone. The company has a large set of claims and a\ncompliance label for each. Each claim consists of a few sentences in English, many of which\ncontain complex related information. Management would like to use Amazon SageMaker\nbuilt-in algorithms to design a machine learning supervised model that can be trained to read\neach claim and predict if the claim is compliant or not.\nWhich approach should be used to extract features from the claims to be used as inputs for\nthe downstream supervised task?",
        "options": [
            "A. Derive a dictionary of tokens from claims in the entire dataset. Apply one-hot encoding to\ntokens found in each claim of the training set. Send the derived features space as inputs to\nan Amazon SageMaker built- in supervised learning algorithm.\nB. Apply Amazon SageMaker BlazingText in Word2Vec mode to claims in the training set.\nSend the derived features space as inputs for the downstream supervised task.\nC. Apply Amazon SageMaker BlazingText in classification mode to labeled claims in the\ntraining set to derive features for the claims that correspond to the compliant and noncompliant labels, respectively.\nD. Apply Amazon SageMaker Object2Vec to claims in the training set. Send the derived\nfeatures space as inputs for the downstream supervised task."
        ],
        "answer": "D",
        "explanation": "Amazon SageMaker Object2Vec generalizes the Word2Vec embedding technique for words\nto more complex objects, such as sentences and paragraphs. Since the supervised learning\ntask is at the level of whole claims, for which there are labels, and no labels are available at\nthe word level, Object2Vec needs be used instead of Word2Vec."
    },
    {
        "question": "An ML engineer needs to deploy ML models to get inferences from large datasets in an\nasynchronous manner. The ML engineer also needs to implement scheduled monitoring of\nthe data quality of the models. The ML engineer must receive alerts when changes in data\nquality occur.\nWhich solution will meet these requirements?",
        "options": [
            "A. Deploy the models by using scheduled AWS Glue jobs. Use Amazon CloudWatch alarms\nto monitor the data quality and to send alerts.\nB. Deploy the models by using scheduled AWS Batch jobs. Use AWS CloudTrail to monitor\nthe data quality and to send alerts.\nC. Deploy the models by using Amazon Elastic Container Service (Amazon ECS) on AWS\nFargate.\nUse Amazon EventBridge to monitor the data quality and to send alerts.\nD. Deploy the models by using Amazon SageMaker batch transform. Use SageMaker Model\nMonitor to monitor the data quality and to send alerts."
        ],
        "answer": "D"
    },
    {
        "question": "Case Study\nA company is building a web-based AI application by using Amazon SageMaker. The\napplication will provide the following capabilities and features: ML experimentation, training, a\ncentral model registry, model deployment, and model monitoring.\nThe application must ensure secure and isolated use of training data during the ML lifecycle.\nThe training data is stored in Amazon S3.\nThe company must implement a manual approval-based workflow to ensure that only\napproved models can be deployed to production endpoints.\nWhich solution will meet this requirement?",
        "options": [
            "A. Use SageMaker Experiments to facilitate the approval process during model registration.\nB. Use SageMaker ML Lineage Tracking on the central model registry. Create tracking\nentities for the approval process.\nC. Use SageMaker Model Monitor to evaluate the performance of the model and to manage\nthe approval.\nD. Use SageMaker Pipelines. When a model version is registered, use the AWS SDK to\nchange the approval status to \"Approved.\""
        ],
        "answer": "D"
    },
    {
        "question": "A company uses a hybrid cloud environment. A model that is deployed on premises uses\ndata in Amazon 53 to provide customers with a live conversational engine.\nThe model is using sensitive data. An ML engineer needs to implement a solution to identify\nand remove the sensitive data.\nWhich solution will meet these requirements with the LEAST operational overhead?",
        "options": [
            "A. Deploy the model on Amazon SageMaker. Create a set of AWS Lambda functions to\nidentify and remove the sensitive data.\nB. Deploy the model on an Amazon Elastic Container Service (Amazon ECS) cluster that\nuses AWS Fargate. Create an AWS Batch job to identify and remove the sensitive data.\nC. Use Amazon Macie to identify the sensitive data. Create a set of AWS Lambda functions\nto remove the sensitive data.\nD. Use Amazon Comprehend to identify the sensitive data. Launch Amazon EC2 instances to\nremove the sensitive data."
        ],
        "answer": "C"
    },
    {
        "question": "A company uses Amazon Athena to query a dataset in Amazon S3. The dataset has a target\nvariable that the company wants to predict.\nThe company needs to use the dataset in a solution to determine if a model can predict the\ntarget variable.\nWhich solution will provide this information with the LEAST development effort?",
        "options": [
            "A. Create a new model by using Amazon SageMaker Autopilot. Report the model's achieved\nperformance.\nB. Implement custom scripts to perform data pre-processing, multiple linear regression, and\nperformance evaluation. Run the scripts on Amazon EC2 instances.\nC. Configure Amazon Macie to analyze the dataset and to create a model. Report the\nmodel's achieved performance.\nD. Select a model from Amazon Bedrock. Tune the model with the data. Report the model's\nachieved performance."
        ],
        "answer": "A"
    }
];