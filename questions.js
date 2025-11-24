const quizData = [
    {
        "question": "A company has an ML model that generates text descriptions based on images that customers upload to the company's website. The images can be up to 50 MB in total size.\nAn ML engineer decides to store the images in an Amazon S3 bucket. The ML engineer must implement a processing solution that can scale to accommodate changes in demand.\nWhich solution will meet these requirements with the LEAST operational overhead?",
        "options": [
            "A. Create an Amazon SageMaker batch transform job to process all the images in the S3 bucket.",
            "B. Create an Amazon SageMaker Asynchronous Inference endpoint and a scaling policy. Run a script to make an inference request for each image.",
            "C. Create an Amazon Elastic Kubernetes Service (Amazon EKS) cluster that uses Karpenter for auto scaling. Host the model on the EKS cluster. Run a script to make an inference request for each image.",
            "D. Create an AWS Batch job that uses an Amazon Elastic Container Service (Amazon ECS) cluster.Specify a list of images to process for each AWS Batch job.",
        ],
        "answer": "B",
        "explanation": "SageMaker Asynchronous Inference is designed for processing large payloads, such as images up to 50 MB, and can handle requests that do not require an immediate response. It scales automatically based on the demand, minimizing operational overhead while ensuring cost-efficiency. A script can be used to send inference requests for each image, and the results can be retrieved asynchronously. This approach is ideal for accommodating varying levels of traffic with minimal manual intervention."
    },
    {
        "question": "A company has developed a new ML model. The company requires online model validation on 10% of the traffic before the company fully releases the model in production. The company uses an Amazon SageMaker endpoint behind an Application Load Balancer (ALB) to serve the model.\nWhich solution will set up the required online validation with the LEAST operational overhead?",
        "options": [
            "A. Use production variants to add the new model to the existing SageMaker endpoint. Set the variant weight to 0.1 for the new model. Monitor the number of invocations by using Amazon CloudWatch.",
            "B. Use production variants to add the new model to the existing SageMaker endpoint. Set the variant weight to 1 for the new model. Monitor the number of invocations by using Amazon CloudWatch.",
            "C. Create a new SageMaker endpoint. Use production variants to add the new model to the new endpoint. Monitor the number of invocations by using Amazon CloudWatch.",
            "D. Configure the ALB to route 10% of the traffic to the new model at the existing SageMaker endpoint.Monitor the number of invocations by using AWS CloudTrail."
        ],
        "answer": "A",
        "explanation": "Scenario:The company wants to perform online validation of a new ML model on 10% of the traffic before fully deploying the model in production. The setup must have minimal operational overhead.\nWhy Use SageMaker Production Variants?\n* Built-In Traffic Splitting:Amazon SageMaker endpoints support production variants, allowing multiple models to run on a single endpoint. You can direct a percentage of incoming traffic to each variant by adjusting the variant weights.\n* Ease of Management:Using production variants eliminates the need for additional infrastructure like separate endpoints or custom ALB configurations."
    },
    {
        "question": "A company has a Retrieval Augmented Generation (RAG) application that uses a vector database to store embeddings of documents. The company must migrate the application to AWS and must implement a solution that provides semantic search of text files. The company has already migrated the text repository to an Amazon S3 bucket.\nWhich solution will meet these requirements?",
        "options": [
            "A. Use an AWS Batch job to process the files and generate embeddings. Use AWS Glue to store the embeddings. Use SQL queries to perform the semantic searches.",
            "B. Use a custom Amazon SageMaker notebook to run a custom script to generate embeddings. Use SageMaker Feature Store to store the embeddings. Use SQL queries to perform the semantic searches.",
            "C. Use the Amazon Kendra S3 connector to ingest the documents from the S3 bucket into Amazon Kendra. Query Amazon Kendra to perform the semantic searches.",
            "D. Use an Amazon Textract asynchronous job to ingest the documents from the S3 bucket. Query Amazon Textract to perform the semantic searches."
        ],
        "answer": "C",
        "explanation": "Amazon Kendra is an Al-powered search service designed for semantic search use cases. It allows ingestion of documents from an Amazon S3 bucket using theAmazon Kendra S3 connector. Once the documents are ingested, Kendra enables semantic searches with its built-in capabilities, removing the need to manually generate embeddings or manage a vector database. This approach is efficient, requires minimal operational effort, and meets the requirements for a Retrieval Augmented Generation (RAG) application."
    },
    {
        "question": "An ML engineer is using Amazon SageMaker to train a deep learning model that requires distributed training. After some training attempts, the ML engineer observes that the instances are not performing as expected. The ML engineer identifies communication overhead between the training instances. What should the ML engineer do to MINIMIZE the communication overhead between the instances?",
        "options": [
            "A. Place the instances in the same VPC subnet. Store the data in a different AWS Region from where the instances are deployed.",
            "B. Place the instances in the same VPC subnet but in different Availability Zones. Store the data in a different AWS Region from where the instances are deployed.",
            "C. Place the instances in the same VPC subnet. Store the data in the same AWS Region and Availability Zone where the instances are deployed.",
            "D. Place the instances in the same VPC subnet. Store the data in the same AWS Region but in a different Availability Zone from where the instances are deployed."
        ],
        "answer": "C",
        "explanation": "To minimize communication overhead during distributed training:\n1. Same VPC Subnet: Ensures low-latency communication between training instances by keeping the network traffic within a single subnet.\n2. Same AWS Region and Availability Zone: Reduces network latency further because cross-AZ communication incurs additional latency and costs.\n3. Data in the Same Region and AZ: Ensures that the training data is accessed with minimal latency, improving performance during training.\nThis configuration optimizes communication efficiency and minimizes overhead."
    },
    {
        "question": "A company stores historical data in .csv files in Amazon S3. Only some of the rows and columns in the .csv files are populated. The columns are not labeled. An ML engineer needs to prepare and store the data so that the company can use the data to train ML models.\nSelect and order the correct steps from the following list to perform this task. Each step should be selected one time or not at all. (Select and order three.)",
        "options": [
            "A. Step 1: Use AWS Glue crawlers to infer the schemas and available columns. Step 2: Use AWS Glue DataBrew for data cleaning and feature engineering. Step 3: Store the resulting data back in Amazon S3.",
            "B. Step 1: Use Amazon Athena to infer the schemas and available columns. Step 2: Use AWS Glue DataBrew for data cleaning and feature engineering. Step 3: Store the resulting data back in Amazon S3.",
            "C. Step 1: Create an Amazon SageMaker batch transform job for data cleaning and feature engineering. Step 2: Store the resulting data back in Amazon S3. Step 3: Use AWS Glue crawlers to infer the schemas and available columns."
        ],
        "answer": "A",
        "explanation": "Step 1: Use AWS Glue crawlers to infer the schemas and available columns.Step 2: Use AWS Glue DataBrew for data cleaning and feature engineering.Step 3: Store the resulting data back in Amazon S3."
    },
    {
        "question": "A company has historical data that shows whether customers needed long-term support from company staff. The company needs to develop an ML model to predict whether new customers will require long-term support.\nWhich modeling approach should the company use to meet this requirement?",
        "options": [
            "A. Anomaly detection",
            "B. Linear regression",
            "C. Logistic regression",
            "D. Semantic segmentation"
        ],
        "answer": "C",
        "explanation": "Logistic regression is a suitable modeling approach for this requirement because it is designed for binary classification problems, such as predicting whether a customer will require long-term support ('yes' or 'no'). It calculates the probability of a particular class and is widely used for tasks like this where the outcome is categorical."
    },
    {
        "question": "An ML engineer is developing a fraud detection model on AWS. The training dataset includes transaction logs, customer profiles, and tables from an on-premises MySQL database. The transaction logs and customer profiles are stored in Amazon S3. The dataset has a class imbalance that affects the learning of the model's algorithm. Additionally, many of the features have interdependencies. The algorithm is not capturing all the desired underlying patterns in the data. The ML engineer needs to use an Amazon SageMaker built-in algorithm to train the model. Which algorithm should the ML engineer use to meet this requirement?",
        "options": [
            "A. LightGBM",
            "B. Linear learner",
            "C. k-means clustering",
            "D. Neural Topic Model (NTM)"
        ],
        "answer": "B",
        "explanation": "Why Linear Learner?\n* SageMaker'sLinear Learneralgorithm is well-suited for binary classification problems such as fraud detection. It handles class imbalance effectively by incorporating built-in options forweight balancing across classes.\n* Linear Learner can capture patterns in the data while being computationally efficient."
    },
    {
        "question": "A company has used Amazon SageMaker to deploy a predictive ML model in production. The company is using SageMaker Model Monitor on the model. After a model update, an ML engineer notices data quality issues in the Model Monitor checks.\nWhat should the ML engineer do to mitigate the data quality issues that Model Monitor has identified?",
        "options": [
            "A. Adjust the model's parameters and hyperparameters.",
            "B. Initiate a manual Model Monitor job that uses the most recent production data.",
            "C. Create a new baseline from the latest dataset. Update Model Monitor to use the new baseline for evaluations.",
            "D. Include additional data in the existing training set for the model. Retrain and redeploy the model."
        ],
        "answer": "C",
        "explanation": "When Model Monitor identifies data quality issues, it might be due to a shift in the data distribution compared to the original baseline. By creating a new baseline using the most recent production data and updating Model Monitor to evaluate against this baseline, the ML engineer ensures that the monitoring is aligned with the current data patterns. This approach mitigates false positives and reflects the updated data characteristics without immediately retraining the model."
    },
    {
        "question": "A company has a conversational Al assistant that sends requests through Amazon Bedrock to an Anthropic Claude large language model (LLM). Users report that when they ask similar questions multiple times, they sometimes receive different answers. An ML engineer needs to improve the responses to be more consistent and less random.\nWhich solution will meet these requirements?",
        "options": [
            "A. Increase the temperature parameter and the top_k parameter.",
            "B. Increase the temperature parameter. Decrease the top_k parameter.",
            "C. Decrease the temperature parameter. Increase the top_k parameter.",
            "D. Decrease the temperature parameter and the top_k parameter."
        ],
        "answer": "D",
        "explanation": "Thetemperatureparameter controls the randomness in the model's responses. Lowering the temperature makes the model produce more deterministic and consistent answers. Thetop_kparameter limits the number of tokens considered for generating the next word. Reducing top_k further constrains the model's options, ensuring more predictable responses. By decreasing both parameters, the responses become more focused and consistent, reducing variability in similar queries."
    },
    {
        "question": "An ML engineer is evaluating several ML models and must choose one model to use in production. The cost of false negative predictions by the models is much higher than the cost of false positive predictions.\nWhich metric finding should the ML engineer prioritize the MOST when choosing the model?",
        "options": [
            "A. Low precision",
            "B. High precision",
            "C. Low recall",
            "D. High recall"
        ],
        "answer": "D",
        "explanation": "Recall measures the ability of a model to correctly identify all positive cases (true positives) out of all actual positives, minimizing false negatives. Since the cost of false negatives is much higher than falsepositives in this scenario, the ML engineer should prioritize models with high recall to reduce the likelihood of missing positive cases."
    },
    {
        "question": "A company has hundreds of data scientists is using Amazon SageMaker to create ML models. The models are in model groups in the SageMaker Model Registry. The data scientists are grouped into three categories: computer vision, natural language processing (NLP), and speech recognition. An ML engineer needs to implement a solution to organize the existing models into these groups to improve model discoverability at scale. The solution must not affect the integrity of the model artifacts and their existing groupings.\nWhich solution will meet these requirements?",
        "options": [
            "A. Create a custom tag for each of the three categories. Add the tags to the model packages in the SageMaker Model Registry.",
            "B. Create a model group for each category. Move the existing models into these category model groups.",
            "C. Use SageMaker ML Lineage Tracking to automatically identify and tag which model groups should contain the models.",
            "D. Create a Model Registry collection for each of the three categories. Move the existing model groups into the collections."
        ],
        "answer": "A",
        "explanation": "Using custom tags allows you to organize and categorize models in the SageMaker Model Registry without altering their existing groupings or affecting the integrity of the model artifacts. Tags are a lightweight and scalable way to improve model discoverability at scale, enabling the data scientists to filter and identify models by category (e.g., computer vision, NLP, speech recognition). This approach meets the requirements efficiently without introducing structural changes to the existing model registry setup."
    },
    {
        "question": "An ML engineer needs to use Amazon SageMaker to fine-tune a large language model (LLM) for text summarization. The ML engineer must follow a low-code no-code (LCNC) approach.\nWhich solution will meet these requirements?",
        "options": [
            "A. Use SageMaker Studio to fine-tune an LLM that is deployed on Amazon EC2 instances.",
            "B. Use SageMaker Autopilot to fine-tune an LLM that is deployed by a custom API endpoint.",
            "C. Use SageMaker Autopilot to fine-tune an LLM that is deployed on Amazon EC2 instances.",
            "D. Use SageMaker Autopilot to fine-tune an LLM that is deployed by SageMaker JumpStart."
        ],
        "answer": "D",
        "explanation": "SageMaker JumpStart provides access to pre-trained models, including large language models (LLMs), which can be easily deployed and fine-tuned with a low-code/no-code (LCNC) approach. Using SageMaker Autopilot with JumpStart simplifies the fine-tuning process by automating model optimization and reducing the need for extensive coding, making it the ideal solution for this requirement."
    },
    {
        "question": "An ML engineer normalized training data by using min-max normalization in AWS Glue DataBrew. The ML engineer must normalize the production inference data in the same way as the training data before passing the production inference data to the model for predictions.\nWhich solution will meet this requirement?",
        "options": [
            "A. Apply statistics from a well-known dataset to normalize the production samples.",
            "B. Keep the min-max normalization statistics from the training set. Use these values to normalize the production samples.",
            "C. Calculate a new set of min-max normalization statistics from a batch of production samples. Use these values to normalize all the production samples.",
            "D. Calculate a new set of min-max normalization statistics from each production sample. Use these values to normalize all the production samples."
        ],
        "answer": "B",
        "explanation": "To ensure consistency between training and inference, themin-max normalization statistics (min and max values)calculated during training must be retained and applied to normalize production inference data. Using the same statistics ensures that the model receives data in the same scale and distribution as it did during training, avoiding discrepancies that could degrade model performance."
    },
    {
        "question": "A company needs to give its ML engineers appropriate access to training data. The ML engineers must access training data from only their own business group. The ML engineers must not be allowed to access training data from other business groups. The company uses a single AWS account and stores all the training data in Amazon S3 buckets. All ML model training occurs in Amazon SageMaker.\nWhich solution will provide the ML engineers with the appropriate access?",
        "options": [
            "A. Enable S3 bucket versioning.",
            "B. Configure S3 Object Lock settings for each user.",
            "C. Add cross-origin resource sharing (CORS) policies to the S3 buckets.",
            "D. Create IAM policies. Attach the policies to IAM users or IAM roles."
        ],
        "answer": "D",
        "explanation": "By creating IAM policies with specific permissions, you can restrict access to Amazon S3 buckets or objects based on the user's business group. These policies can be attached to IAM users or IAM roles associated with the ML engineers, ensuring that each engineer can only access training data belonging to their group. This approach is secure, scalable, and aligns with AWS best practices for access control."
    },
    {
        "question": "A company is using Amazon Redshift ML in its primary AWS account. The source data is in an Amazon S3 bucket in a secondary account. An ML engineer needs to set up an ML pipeline in the primary account to access the S3 bucket in the secondary account. The solution must not require public IPv4 addresses.\nWhich solution will meet these requirements?",
        "options": [
            "A. Provision a Redshift cluster and Amazon SageMaker Studio in a VPC with no public access enabled in the primary account. Create a VPC peering connection between the accounts. Update the VPC route tables to remove the route to 0.0.0.0/0.",
            "B. Provision a Redshift cluster and Amazon SageMaker Studio in a VPC with no public access enabled in the primary account. Create an AWS Direct Connect connection and a transit gateway. Associate the VPCs from both accounts with the transit gateway. Update the VPC route tables to remove the route to 0.0.0.0/0.",
            "C. Provision a Redshift cluster and Amazon SageMaker Studio in a VPC in the primary account. Create an AWS Site-to-Site VPN connection with two encrypted IPsec tunnels between the accounts. Set up interface VPC endpoints for Amazon S3.",
            "D. Provision a Redshift cluster and Amazon SageMaker Studio in a VPC in the primary account. Create an S3 gateway endpoint. Update the S3 bucket policy to allow IAM principals from the primary account.Set up interface VPC endpoints for SageMaker and Amazon Redshift."
        ],
        "answer": "D",
        "explanation": "S3 Gateway Endpoint: Allows private access to S3 from within a VPC without requiring a public IPv4 address, ensuring that data transfer between the primary and secondary accounts is secure and private. Bucket Policy Update: The S3 bucket policy in the secondary account must explicitly allow access from the primary account's IAM principals to provide the necessary permissions. Interface VPC Endpoints: Required for private communication between the VPC and Amazon SageMaker and Amazon Redshift services, ensuring the solution operates without public internet access."
    },
    {
        "question": "A company needs to run a batch data-processing job on Amazon EC2 instances. The job will run during the weekend and will take 90 minutes to finish running. The processing can handle interruptions. The company will run the job every weekend for the next 6 months.\nWhich EC2 instance purchasing option will meet these requirements MOST cost-effectively?",
        "options": [
            "A. Spot Instances",
            "B. Reserved Instances",
            "C. On-Demand Instances",
            "D. Dedicated Instances"
        ],
        "answer": "A",
        "explanation": "Scenario:The company needs to run a batch job for 90 minutes every weekend over the next 6 months. The processing can handle interruptions, and cost-effectiveness is a priority.\nWhy Spot Instances?\n* Cost-Effective:Spot Instances provide up to 90% savings compared to On-Demand Instances, making them the most cost-effective option for batch processing.\n* Interruption Tolerance:Since the processing can tolerate interruptions, Spot Instances are suitable for this workload."
    },
    {
        "question": "An ML engineer is developing a fraud detection model by using the Amazon SageMaker XGBoost algorithm. The model classifies transactions as either fraudulent or legitimate. During testing, the model excels at identifying fraud in the training dataset. However, the model is inefficient at identifying fraud in new and unseen transactions.\nWhat should the ML engineer do to improve the fraud detection for new transactions?",
        "options": [
            "A. Increase the learning rate.",
            "B. Remove some irrelevant features from the training dataset.",
            "C. Increase the value of the max_depth hyperparameter.",
            "D. Decrease the value of the max_depth hyperparameter."
        ],
        "answer": "D",
        "explanation": "A high max_depth value in XGBoost can lead to overfitting, where the model learns the training dataset too well but fails to generalize to new and unseen data. By decreasing the max_depth, the model becomes less complex, reducing overfitting and improving its ability to detect fraud in new transactions. This adjustment helps the model focus on general patterns rather than memorizing specific details in the training data."
    },
    {
        "question": "An ML engineer needs to process thousands of existing CSV objects and new CSV objects that are uploaded. The CSV objects are stored in a central Amazon S3 bucket and have the same number of columns. One of the columns is a transaction date. The ML engineer must query the data based on the transaction date.\nWhich solution will meet these requirements with the LEAST operational overhead?",
        "options": [
            "A. Use an Amazon Athena CREATE TABLE AS SELECT (CTAS) statement to create a table based on the transaction date from data in the central S3 bucket. Query the objects from the table.",
            "B. Create a new S3 bucket for processed data. Set up S3 replication from the central S3 bucket to the new S3 bucket. Use S3 Object Lambda to query the objects based on transaction date.",
            "C. Create a new S3 bucket for processed data. Use AWS Glue for Apache Spark to create a job to query the CSV objects based on transaction date. Configure the job to store the results in the new S3 bucket. Query the objects from the new S3 bucket.",
            "D. Create a new S3 bucket for processed data. Use Amazon Data Firehose to transfer the data from the central S3 bucket to the new S3 bucket. Configure Firehose to run an AWS Lambda function to query the data based on transaction date."
        ],
        "answer": "A",
        "explanation": "Scenario:The ML engineer needs a low-overhead solution to query thousands of existing and new CSV objects stored in Amazon S3 based on a transaction date.\nWhy Athena?\n* Serverless:Amazon Athena is a serverless query service that allows direct querying of data stored in S3 using standard SQL, reducing operational overhead.\n* Ease of Use:By using the CTAS statement, the engineer can create a table with optimized partitions based on the transaction date. Partitioning improves query performance and minimizes costs by scanning only relevant data."
    },
    {
        "question": "An ML engineer is developing a fraud detection model on AWS. The training dataset includes transaction logs, customer profiles, and tables from an on-premises MySQL database. The transaction logs and customer profiles are stored in Amazon S3. The dataset has a class imbalance that affects the learning of the model's algorithm. Additionally, many of the features have interdependencies. The algorithm is not capturing all the desired underlying patterns in the data.\nWhich AWS service or feature can aggregate the data from the various data sources?",
        "options": [
            "A. Amazon EMR Spark jobs",
            "B. Amazon Kinesis Data Streams",
            "C. Amazon DynamoDB",
            "D. AWS Lake Formation"
        ],
        "answer": "D",
        "explanation": "Problem Description:\n* The dataset includes multiple data sources: Transaction logs and customer profiles in Amazon S3, and tables in an on-premises MySQL database.\n* The solution requiresdata aggregationfrom diverse sources for centralized processing.\n* Why AWS Lake Formation? AWS Lake Formationis designed to simplify the process of aggregating, cataloging, and securing data from various sources, including S3, relational databases, and other on-premises systems."
    },
    {
        "question": "A company has an application that uses different APIs to generate embeddings for input text. The company needs to implement a solution to automatically rotate the API tokens every 3 months.\nWhich solution will meet this requirement?",
        "options": [
            "A. Store the tokens in AWS Secrets Manager. Create an AWS Lambda function to perform the rotation.",
            "B. Store the tokens in AWS Systems Manager Parameter Store. Create an AWS Lambda function to perform the rotation.",
            "C. Store the tokens in AWS Key Management Service (AWS KMS). Use an AWS managed key to perform the rotation.",
            "D. Store the tokens in AWS Key Management Service (AWS KMS). Use an AWS owned key to perform the rotation."
        ],
        "answer": "A",
        "explanation": "AWS Secrets Manager is designed for securely storing, managing, and automatically rotating secrets, including API tokens. By configuring a Lambda function for custom rotation logic, the solution can automatically rotate the API tokens every 3 months as required. Secrets Manager simplifies secret management and integrates seamlessly with other AWS services, making it the ideal choice for this use case."
    },
    {
        "question": "A credit card company has a fraud detection model in production on an Amazon SageMaker endpoint. The company develops a new version of the model. The company needs to assess the new model's performance by using live data and without affecting production end users.\nWhich solution will meet these requirements?",
        "options": [
            "A. Set up SageMaker Debugger and create a custom rule.",
            "B. Set up blue/green deployments with all-at-once traffic shifting.",
            "C. Set up blue/green deployments with canary traffic shifting.",
            "D. Set up shadow testing with a shadow variant of the new model."
        ],
        "answer": "D",
        "explanation": "Shadow testing allows you to send a copy of live production traffic to a shadow variant of the new model while keeping the existing production model unaffected. This enables you to evaluate the performance of the new model in real-time with live data without impacting end users. SageMaker endpoints support this setup by allowing traffic mirroring to the shadow variant, making it an ideal solution for assessing the new model's performance."
    },
    {
        "question": "A company is using an ML model to predict the presence of a specific weed in a farmer's field. The company is using the Amazon SageMaker linear learner built-in algorithm with a value of multiclass_dassifier for the predictorjype hyperparameter.\nWhat should the company do to MINIMIZE false positives?",
        "options": [
            "A. Set the value of the weight decay hyperparameter to zero.",
            "B. Increase the number of training epochs.",
            "C. Increase the value of the target_precision hyperparameter.",
            "D. Change the value of the predictorjype hyperparameter to regressor."
        ],
        "answer": "C",
        "explanation": "Thetarget_precisionhyperparameter in the Amazon SageMaker linear learner controls the trade-off between precision and recall for the model. Increasing the target_precision prioritizes minimizing false positives by making the model more cautious in its predictions. This approach is effective for use cases where false positives have higher consequences than false negatives."
    },
    {
        "question": "An ML engineer has deployed an ML model for sentiment analysis to an Amazon SageMaker endpoint. The ML engineer needs to explain to company stakeholders how the model makes predictions.\nWhich solution will provide an explanation for the model's predictions?",
        "options": [
            "A. Use SageMaker Model Monitor on the deployed model.",
            "B. Use SageMaker Clarify on the deployed model.",
            "C. Show the distribution of inferences from A/# testing in Amazon CloudWatch.",
            "D. Add a shadow endpoint. Analyze prediction differences on samples."
        ],
        "answer": "B",
        "explanation": "SageMaker Clarify is designed to provide explainability for ML models. It can analyze feature importance and explain how input features influence the model's predictions. By using Clarify with the deployed SageMaker model, the ML engineer can generate insights and present them to stakeholders to explain the sentiment analysis predictions effectively."
    },
    {
        "question": "An ML engineer needs to use an ML model to predict the price of apartments in a specific location. Which metric should the ML engineer use to evaluate the model's performance?",
        "options": [
            "A. Accuracy",
            "B. Area Under the ROC Curve (AUC)",
            "C. F1 score",
            "D. Mean absolute error (MAE)"
        ],
        "answer": "D",
        "explanation": "When predicting continuous variables, such as apartment prices, it's essential to evaluate the model's performance using appropriate regression metrics. The Mean Absolute Error (MAE) is a widely used metric for this purpose. Understanding Mean Absolute Error (MAE): MAE measures the average magnitude of errors in a set of predictions, without considering their direction. It calculates the average absolute difference between predicted values and actual values, providing a straightforward interpretation of prediction accuracy."
    },
    {
        "question": "A company wants to host an ML model on Amazon SageMaker. An ML engineer is configuring a continuous integration and continuous delivery (CI/CD) pipeline in AWS CodePipeline to deploy the model. The pipeline must run automatically when new training data for the model is uploaded to an Amazon S3 bucket.\nSelect and order the pipeline's correct steps from the following list to perform this task. Each step should be selected one time or not at all. (Select and order three.)",
        "options": [
            "A. Step 1: An S3 event notification invokes the pipeline when new data is uploaded. Step 2: SageMaker retrains the model by using the data in the S3 bucket. Step 3: The pipeline deploys the model to a SageMaker endpoint.",
            "B. Step 1: An S3 Lifecycle rule invokes the pipeline when new data is uploaded. Step 2: SageMaker retrains the model by using the data in the S3 bucket. Step 3: The pipeline deploys the model to a SageMaker endpoint.",
            "C. Step 1: An S3 event notification invokes the pipeline when new data is uploaded. Step 2: The pipeline deploys the model to a SageMaker endpoint. Step 3: SageMaker retrains the model by using the data in the S3 bucket."
        ],
        "answer": "A",
        "explanation": "Step 1: An S3 event notification invokes the pipeline when new data is uploaded.Step 2: SageMaker retrains the model by using the data in the S3 bucket.Step 3: The pipeline deploys the model to a SageMaker endpoint.\nOrder Summary:\n* An S3 event notification invokes the pipeline when new data is uploaded.\n* SageMaker retrains the model by using the data in the S3 bucket.\n* The pipeline deploys the model to a SageMaker endpoint.\nThis configuration ensures an automated, efficient, and scalable CI/CD pipeline for continuous retraining and deployment of the ML model in Amazon SageMaker."
    },
    {
        "question": "An ML engineer is developing a fraud detection model on AWS. The training dataset includes transaction logs, customer profiles, and tables from an on-premises MySQL database. The transaction logs and customer profiles are stored in Amazon S3. The dataset has a class imbalance that affects the learning of the model's algorithm. Additionally, many of the features have interdependencies. The algorithm is not capturing all the desired underlying patterns in the data. After the data is aggregated, the ML engineer must implement a solution to automatically detect anomalies in the data and to visualize the result.\nWhich solution will meet these requirements?",
        "options": [
            "A. Use Amazon Athena to automatically detect the anomalies and to visualize the result.",
            "B. Use Amazon Redshift Spectrum to automatically detect the anomalies. Use Amazon QuickSight to visualize the result.",
            "C. Use Amazon SageMaker Data Wrangler to automatically detect the anomalies and to visualize the result.",
            "D. Use AWS Batch to automatically detect the anomalies. Use Amazon QuickSight to visualize the result."
        ],
        "answer": "C",
        "explanation": "Amazon SageMaker Data Wrangler is a comprehensive tool that streamlines the process of data preparation and offers built-in capabilities for anomaly detection and visualization. Key Features of SageMaker Data Wrangler: * Data Importation: Connects seamlessly to various data sources, including Amazon S3 and on-premises databases, facilitating the aggregation of transaction logs, customer profiles, and MySQL tables."
    },
    {
        "question": "A company needs to use Amazon Athena to query a dataset in Amazon S3. The dataset has a target variable that the company wants to predict. The company needs to use the dataset in a solution to determine if a model can predict the target variable.\nWhich solution will provide this information with the LEAST development effort?",
        "options": [
            "A. Create a new model by using Amazon SageMaker Autopilot. Report the model's achieved performance.",
            "B. Implement custom scripts to perform data pre-processing, multiple linear regression, and performance evaluation. Run the scripts on Amazon EC2 instances.",
            "C. Configure Amazon Macie to analyze the dataset and to create a model. Report the model's achieved performance.",
            "D. Select a model from Amazon Bedrock. Tune the model with the data. Report the model's achieved performance."
        ],
        "answer": "A",
        "explanation": "Amazon SageMaker Autopilot automates the process of building, training, and tuning machine learning models. It provides insights into whether the target variable can be effectively predicted by evaluating the model's performance metrics. This solution requires minimal development effort as SageMaker Autopilot handles data preprocessing, algorithm selection, and hyperparameter optimization automatically, making it the most efficient choice for this scenario."
    },
    {
        "question": "A company uses Amazon SageMaker for its ML workloads. The company's ML engineer receives a 50 MB Apache Parquet data file to build a fraud detection model. The file includes several correlated columns that are not required.\nWhat should the ML engineer do to drop the unnecessary columns in the file with the LEAST effort?",
        "options": [
            "A. Download the file to a local workstation. Perform one-hot encoding by using a custom Python script.",
            "B. Create an Apache Spark job that uses a custom processing script on Amazon EMR.",
            "C. Create a SageMaker processing job by calling the SageMaker Python SDK.",
            "D. Create a data flow in SageMaker Data Wrangler. Configure a transform step."
        ],
        "answer": "D",
        "explanation": "SageMaker Data Wrangler provides a no-code/low-code interface for preparing and transforming data, including dropping unnecessary columns. By creating a data flow and configuring a transform step, the ML engineer can easily remove correlated or unneeded columns from the Parquet file with minimal effort. This approach avoids the need for custom coding or managing additional infrastructure."
    },
    {
        "question": "A company has a team of data scientists who use Amazon SageMaker notebook instances to test ML models. When the data scientists need new permissions, the company attaches the permissions to each individual role that was created during the creation of the SageMaker notebook instance. The company needs to centralize management of the team's permissions.\nWhich solution will meet this requirement?",
        "options": [
            "A. Create a single IAM role that has the necessary permissions. Attach the role to each notebook instance that the team uses.",
            "B. Create a single IAM group. Add the data scientists to the group. Associate the group with each notebook instance that the team uses.",
            "C. Create a single IAM user. Attach the AdministratorAccess AWS managed IAM policy to the user. Configure each notebook instance to use the IAM user.",
            "D. Create a single IAM group. Add the data scientists to the group. Create an IAM role. Attach the AdministratorAccess AWS managed IAM policy to the role. Associate the role with the group.Associate the group with each notebook instance that the team uses."
        ],
        "answer": "A",
        "explanation": "Managing permissions for multiple Amazon SageMaker notebook instances can become complex when handled individually. To centralize and streamline permission management, AWS recommends creating a single IAM role with the necessary permissions and attaching this role to each notebook instance used by the data science team."
    },
    {
        "question": "An ML engineer has trained a neural network by using stochastic gradient descent (SGD). The neural network performs poorly on the test set. The values for training loss and validation loss remain high and show an oscillating pattern. The values decrease for a few epochs and then increase for a few epochs before repeating the same cycle.\nWhat should the ML engineer do to improve the training process?",
        "options": [
            "A. Introduce early stopping.",
            "B. Increase the size of the test set.",
            "C. Increase the learning rate.",
            "D. Decrease the learning rate."
        ],
        "answer": "D",
        "explanation": "In training neural networks using Stochastic Gradient Descent (SGD), the learning rate is a critical hyperparameter that influences the convergence behavior of the model. Observing oscillations in training and validation loss suggests that the learning rate may be too high, causing the optimization process to overshoot minima in the loss landscape."
    },
    {
        "question": "An ML engineer has an Amazon Comprehend custom model in Account A in the us-east-1 Region. The ML engineer needs to copy the model to Account B in the same Region.\nWhich solution will meet this requirement with the LEAST development effort?",
        "options": [
            "A. Use Amazon S3 to make a copy of the model. Transfer the copy to Account B.",
            "B. Create a resource-based IAM policy. Use the Amazon Comprehend ImportModel API operation to copy the model to Account B.",
            "C. Use AWS DataSync to replicate the model from Account A to Account B.",
            "D. Create an AWS Site-to-Site VPN connection between Account A and Account B to transfer the model."
        ],
        "answer": "B",
        "explanation": "Amazon Comprehend provides the ImportModel API operation, which allows you to copy a custom model between AWS accounts. By creating a resource-based IAM policy on the model in Account A, you can grant Account B the necessary permissions to access and import the model. This approach requires minimal development effort and is the AWS-recommended method for sharing custom models across accounts."
    },
    {
        "question": "A company is building a web-based Al application by using Amazon SageMaker. The application will provide the following capabilities and features: ML experimentation, training, a central model registry, model deployment, and model monitoring. The application must ensure secure and isolated use of training data during the ML lifecycle. The training data is stored in Amazon S3. The company must run an on-demand workflow to monitor bias drift for models that are deployed to real- time endpoints from the application.\nWhich action will meet this requirement?",
        "options": [
            "A. Configure the application to invoke an AWS Lambda function that runs a SageMaker Clarify job.",
            "B. Invoke an AWS Lambda function to pull the sagemaker-model-monitor-analyzer built-in SageMaker image.",
            "C. Use AWS Glue Data Quality to monitor bias.",
            "D. Use SageMaker notebooks to compare the bias."
        ],
        "answer": "A",
        "explanation": "Monitoring bias drift in machine learning models is crucial to ensure fairness and accuracy over time. Amazon SageMaker Clarify provides tools to detect bias in ML models, both during training and after deployment. To monitor bias drift for models deployed to real-time endpoints, an effective approach involves orchestrating SageMaker Clarify jobs using AWS Lambda functions."
    },
    {
        "question": "A company needs to develop an ML model by using tabular data from its customers. The data contains meaningful ordered features with sensitive information that should not be discarded. An ML engineer must ensure that the sensitive data is masked before another team starts to build the model.\nWhich solution will meet these requirements?",
        "options": [
            "A. Use Amazon Macie to categorize the sensitive data.",
            "B. Prepare the data by using AWS Glue DataBrew.",
            "C. Run an AWS Batch job to change the sensitive data to random values.",
            "D. Run an Amazon EMR job to change the sensitive data to random values."
        ],
        "answer": "B",
        "explanation": "AWS Glue DataBrew provides an easy-to-use interface for preparing and transforming data, including masking or obfuscating sensitive information. It offers built-in data masking features, allowing the ML engineer to handle sensitive data securely while retaining its structure and meaning. This solution is efficient and requires minimal coding, making it ideal for ensuring sensitive data is masked before model building begins."
    },
    {
        "question": "An ML engineer needs to use AWS CloudFormation to create an ML model that an Amazon SageMaker endpoint will host.\nWhich resource should the ML engineer declare in the CloudFormation template to meet this requirement?",
        "options": [
            "A. AWS::SageMaker::Model",
            "B. AWS::SageMaker::Endpoint",
            "C. AWS::SageMaker::NotebookInstance",
            "D. AWS::SageMaker::Pipeline"
        ],
        "answer": "A",
        "explanation": "The AWS::SageMaker::Model resource in AWS CloudFormation is used to create an ML model in Amazon SageMaker. This model can then be hosted on an endpoint by using the AWS::SageMaker::Endpoint resource. The model resource defines the container or algorithm to use for hosting and the S3 location of the model artifacts."
    },
    {
        "question": "A company has trained an ML model in Amazon SageMaker. The company needs to host the model to provide inferences in a production environment. The model must be highly available and must respond with minimum latency. The size of each request will be between 1 KB and 3 MB. The model will receive unpredictable bursts of requests during the day. The inferences must adapt proportionally to the changes in demand.\nHow should the company deploy the model into production to meet these requirements?",
        "options": [
            "A. Create a SageMaker real-time inference endpoint. Configure auto scaling. Configure the endpoint to present the existing model.",
            "B. Deploy the model on an Amazon Elastic Container Service (Amazon ECS) cluster. Use ECS scheduled scaling that is based on the CPU of the ECS cluster.",
            "C. Install SageMaker Operator on an Amazon Elastic Kubernetes Service (Amazon EKS) cluster. Deploy the model in Amazon EKS. Set horizontal pod auto scaling to scale replicas based on the memory metric.",
            "D. Use Spot Instances with a Spot Fleet behind an Application Load Balancer (ALB) for inferences. Use the ALBRequestCountPerTarget metric as the metric for auto scaling."
        ],
        "answer": "A"
    },
    {
        "question": "A company has AWS Glue data processing jobs that are orchestrated by an AWS Glue workflow. The AWS Glue jobs can run on a schedule or can be launched manually. The company is developing pipelines in Amazon SageMaker Pipelines for ML model development. The pipelines will use the output of the AWS Glue jobs during the data processing phase of model development. An ML engineer needs to implement a solution that integrates the AWS Glue jobs with the pipelines.\nWhich solution will meet these requirements with the LEAST operational overhead?",
        "options": [
            "A. Use AWS Step Functions for orchestration of the pipelines and the AWS Glue jobs.",
            "B. Use processing steps in SageMaker Pipelines. Configure inputs that point to the Amazon Resource Names (ARNs) of the AWS Glue jobs.",
            "C. Use Callback steps in SageMaker Pipelines to start the AWS Glue workflow and to stop the pipelines until the AWS Glue jobs finish running.",
            "D. Use Amazon EventBridge to invoke the pipelines and the AWS Glue jobs in the desired order."
        ],
        "answer": "C",
        "explanation": "Callback steps in Amazon SageMaker Pipelines allow you to integrate external processes, such as AWS Glue jobs, into the pipeline workflow. By using a Callback step, the SageMaker pipeline can trigger the AWS Glue workflow and pause execution until the Glue jobs complete. This approach provides seamless integration with minimal operational overhead, as it directly ties the pipeline's execution flow to the completion of the AWS Glue jobs without requiring additional orchestration tools or complex setups."
    },
    {
        "question": "A company is running ML models on premises by using custom Python scripts and proprietary datasets. The company is using PyTorch. The model building requires unique domain knowledge. The company needs to move the models to AWS.\nWhich solution will meet these requirements with the LEAST effort?",
        "options": [
            "A. Use SageMaker built-in algorithms to train the proprietary datasets.",
            "B. Use SageMaker script mode and premade images for ML frameworks.",
            "C. Build a container on AWS that includes custom packages and a choice of ML frameworks.",
            "D. Purchase similar production models through AWS Marketplace."
        ],
        "answer": "B",
        "explanation": "SageMaker script mode allows you to bring existing custom Python scripts and run them on AWS with minimal changes. SageMaker provides prebuilt containers for ML frameworks like PyTorch, simplifying the migration process. This approach enables the company to leverage their existing Python scripts and domain knowledge while benefiting from the scalability and managed environment of SageMaker. It requires the least effort compared to building custom containers or retraining models from scratch."
    },
    {
        "question": "A company is gathering audio, video, and text data in various languages. The company needs to use a large language model (LLM) to summarize the gathered data that is in Spanish.\nWhich solution will meet these requirements in the LEAST amount of time?",
        "options": [
            "A. Train and deploy a model in Amazon SageMaker to convert the data into English text. Train and deploy an LLM in SageMaker to summarize the text.",
            "B. Use Amazon Transcribe and Amazon Translate to convert the data into English text. Use Amazon Bedrock with the Jurassic model to summarize the text.",
            "C. Use Amazon Rekognition and Amazon Translate to convert the data into English text. Use Amazon Bedrock with the Anthropic Claude model to summarize the text.",
            "D. Use Amazon Comprehend and Amazon Translate to convert the data into English text. Use Amazon Bedrock with the Stable Diffusion model to summarize the text."
        ],
        "answer": "B",
        "explanation": "Amazon Transcribeis well-suited for converting audio data into text, including Spanish. Amazon Translatecan efficiently translate Spanish text into English if needed. Amazon Bedrock, with theJurassic model, is designed for tasks like text summarization and can handle large language models (LLMs) seamlessly. This combination provides a low-code, managed solution to process audio, video, and text data with minimal time and effort."
    },
    {
        "question": "A company has trained and deployed an ML model by using Amazon SageMaker. The company needs to implement a solution to record and monitor all the API call events for the SageMaker endpoint. The solution also must provide a notification when the number of API call events breaches a threshold.\nWhich solution will meet these requirements?",
        "options": [
            "A. Use SageMaker Debugger to track the inferences and to report metrics. Create a custom rule to provide a notification when the threshold is breached.",
            "B. Use SageMaker Debugger to track the inferences and to report metrics. Use the tensor_variance built-in rule to provide a notification when the threshold is breached.",
            "C. Log all the endpoint invocation API events by using AWS CloudTrail. Use an Amazon CloudWatch dashboard for monitoring. Set up a CloudWatch alarm to provide notification when the threshold is breached.",
            "D. Add the Invocations metric to an Amazon CloudWatch dashboard for monitoring. Set up a CloudWatch alarm to provide notification when the threshold is breached."
        ],
        "answer": "D",
        "explanation": "Amazon SageMaker automatically tracks theInvocationsmetric, which represents the number of API calls made to the endpoint, inAmazon CloudWatch. By adding this metric to a CloudWatch dashboard, you can monitor the endpoint's activity in real-time. Setting up aCloudWatch alarmallows the system to send notifications whenever the API call events exceed the defined threshold, meeting both the monitoring and notification requirements efficiently."
    },
    {
        "question": "A company needs to create a central catalog for all the company's ML models. The models are in AWS accounts where the company developed the models initially. The models are hosted in Amazon Elastic Container Registry (Amazon ECR) repositories.\nWhich solution will meet these requirements?",
        "options": [
            "A. Configure ECR cross-account replication for each existing ECR repository. Ensure that each model is visible in each AWS account.",
            "B. Create a new AWS account with a new ECR repository as the central catalog. Configure ECR cross-account replication between the initial ECR repositories and the central catalog.",
            "C. Use the Amazon SageMaker Model Registry to create a model group for models hosted in Amazon ECR. Create a new AWS account. In the new account, use the SageMaker Model Registry as the central catalog. Attach a cross-account resource policy to each model group in the initial AWS accounts.",
            "D. Use an AWS Glue Data Catalog to store the models. Run an AWS Glue crawler to migrate the models from the ECR repositories to the Data Catalog. Configure cross-account access to the Data Catalog."
        ],
        "answer": "C",
        "explanation": "The Amazon SageMaker Model Registry is designed to manage and catalog ML models, including those hosted in Amazon ECR. By creating a model group for each model in the SageMaker Model Registry and setting up cross-account resource policies, the company can establish a central catalog in a new AWS account. This allows all models from the initial accounts to be accessible in a unified, centralized manner for better organization, management, and governance."
    },
    {
        "question": "A company is planning to create several ML prediction models. The training data is stored in Amazon S3. The entire dataset is more than 5 TB in size and consists of CSV, JSON, Apache Parquet, and simple text files. The data must be processed in several consecutive steps. The steps include complex manipulations that can take hours to finish running. Some of the processing involves natural language processing (NLP) transformations. The entire process must be automated.\nWhich solution will meet these requirements?",
        "options": [
            "A. Process data at each step by using Amazon SageMaker Data Wrangler. Automate the process by using Data Wrangler jobs.",
            "B. Use Amazon SageMaker notebooks for each data processing step. Automate the process by using Amazon EventBridge.",
            "C. Process data at each step by using AWS Lambda functions. Automate the process by using AWS Step Functions and Amazon EventBridge.",
            "D. Use Amazon SageMaker Pipelines to create a pipeline of data processing steps. Automate the pipeline by using Amazon EventBridge."
        ],
        "answer": "D",
        "explanation": "Amazon SageMaker Pipelines is designed for creating, automating, and managing end-to-end ML workflows, including complex data preprocessing tasks. It supports handling large datasets and can integrate with custom steps, such as NLP transformations. By combining SageMaker Pipelines with Amazon EventBridge, the entire workflow can be triggered and automated efficiently, meeting the requirements for scalability, automation, and processing complexity."
    },
    {
        "question": "An ML engineer needs to implement a solution to host a trained ML model. The rate of requests to the model will be inconsistent throughout the day. The ML engineer needs a scalable solution that minimizes costs when the model is not in use. The solution also must maintain the model's capacity to respond to requests during times of peak usage.\nWhich solution will meet these requirements?",
        "options": [
            "A. Create AWS Lambda functions that have fixed concurrency to host the model. Configure the Lambda functions to automatically scale based on the number of requests to the model.",
            "B. Deploy the model on an Amazon Elastic Container Service (Amazon ECS) cluster that uses AWS Fargate. Set a static number of tasks to handle requests during times of peak usage.",
            "C. Deploy the model to an Amazon SageMaker endpoint. Deploy multiple copies of the model to the endpoint. Create an Application Load Balancer to route traffic between the different copies of the model at the endpoint.",
            "D. Deploy the model to an Amazon SageMaker endpoint. Create SageMaker endpoint auto scaling policies that are based on Amazon CloudWatch metrics to adjust the number of instances dynamically."
        ],
        "answer": "D"
    },
    {
        "question": "A company wants to improve the sustainability of its ML operations. Which actions will reduce the energy usage and computational resources that are associated with the company's training jobs? (Choose two.)",
        "options": [
            "A. Use Amazon SageMaker Debugger to stop training jobs when non-converging conditions are detected.",
            "B. Use Amazon SageMaker Ground Truth for data labeling.",
            "C. Deploy models by using AWS Lambda functions.",
            "D. Use AWS Trainium instances for training.",
            "E. Use PyTorch or TensorFlow with the distributed training option."
        ],
        "answer": "AD",
        "explanation": "SageMaker Debuggercan identify when a training job is not converging or is stuck in a non-productive state. By stopping these jobs early, unnecessary energy and computational resources are conserved, improving sustainability. AWS Trainiuminstances are purpose-built for ML training and are optimized for energy efficiency and cost- effectiveness. They use less energy per training task compared to general-purpose instances, making them a sustainable choice."
    },
    {
        "question": "A company has a hybrid cloud environment. A model that is deployed on premises uses data in Amazon S3 to provide customers with a live conversational engine. The model is using sensitive data. An ML engineer needs to implement a solution to identify and remove the sensitive data.\nWhich solution will meet these requirements with the LEAST operational overhead?",
        "options": [
            "A. Deploy the model on Amazon SageMaker. Create a set of AWS Lambda functions to identify and remove the sensitive data.",
            "B. Deploy the model on an Amazon Elastic Container Service (Amazon ECS) cluster that uses AWS Fargate. Create an AWS Batch job to identify and remove the sensitive data.",
            "C. Use Amazon Macie to identify the sensitive data. Create a set of AWS Lambda functions to remove the sensitive data.",
            "D. Use Amazon Comprehend to identify the sensitive data. Launch Amazon EC2 instances to remove the sensitive data."
        ],
        "answer": "C",
        "explanation": "Amazon Macie is a fully managed data security and privacy service that uses machine learning to discover and classify sensitive data in Amazon S3. It is purpose-built to identify sensitive data with minimal operational overhead. After identifying the sensitive data, you can use AWS Lambda functions to automate the process of removing or redacting the sensitive data, ensuring efficiency and integration with the hybrid cloud environment."
    },
    {
        "question": "A company is using Amazon SageMaker Studio to develop an ML model. The company has a single SageMaker Studio domain. An ML engineer needs to implement a solution that provides an automated alert when SageMaker compute costs reach a specific threshold.\nWhich solution will meet these requirements?",
        "options": [
            "A. Add resource tagging by editing the SageMaker user profile in the SageMaker domain. Configure AWS Cost Explorer to send an alert when the threshold is reached.",
            "B. Add resource tagging by editing the SageMaker user profile in the SageMaker domain. Configure AWS Budgets to send an alert when the threshold is reached.",
            "C. Add resource tagging by editing each user's IAM profile. Configure AWS Cost Explorer to send an alert when the threshold is reached.",
            "D. Add resource tagging by editing each user's IAM profile. Configure AWS Budgets to send an alert when the threshold is reached."
        ],
        "answer": "B",
        "explanation": "Adding resource tagging to the SageMaker user profile enables tracking and monitoring of costs associated with specific SageMaker resources. AWS Budgets allows setting thresholds and automated alerts for costs and usage, making it the ideal service to notify the ML engineer when compute costs reach a specified limit."
    },
    {
        "question": "A financial company receives a high volume of real-time market data streams from an external provider. The streams consist of thousands of JSON records every second. The company needs to implement a scalable solution on AWS to identify anomalous data points.\nWhich solution will meet these requirements with the LEAST operational overhead?",
        "options": [
            "A. Ingest real-time data into Amazon Kinesis data streams. Use the built-in RANDOM_CUT_FOREST function in Amazon Managed Service for Apache Flink to process the data streams and to detect data anomalies.",
            "B. Ingest real-time data into Amazon Kinesis data streams. Deploy an Amazon SageMaker endpoint for real-time outlier detection. Create an AWS Lambda function to detect anomalies. Use the data streams to invoke the Lambda function.",
            "C. Ingest real-time data into Apache Kafka on Amazon EC2 instances. Deploy an Amazon SageMaker endpoint for real-time outlier detection. Create an AWS Lambda function to detect anomalies. Use the data streams to invoke the Lambda function.",
            "D. Send real-time data to an Amazon Simple Queue Service (Amazon SQS) FIFO queue. Create an AWS Lambda function to consume the queue messages. Program the Lambda function to start an AWS Glue extract, transform, and load (ETL) job for batch processing and anomaly detection."
        ],
        "answer": "A",
        "explanation": "This solution is the most efficient and involves the least operational overhead: Amazon Kinesis data streams efficiently handle real-time ingestion of high-volume streaming data. Amazon Managed Service for Apache Flink provides a fully managed environment for stream processing with built-in support for RANDOM_CUT_FOREST, an algorithm designed for anomaly detection in real- time streaming data. This approach eliminates the need for deploying and managing additional infrastructure like SageMaker endpoints, Lambda functions, or external tools, making it the most scalable and operationally simple solution."
    },
    {
        "question": "A company has deployed an ML model that detects fraudulent credit card transactions in real time in a banking application. The model uses Amazon SageMaker Asynchronous Inference. Consumers are reporting delays in receiving the inference results. An ML engineer needs to implement a solution to improve the inference performance. The solution also must provide a notification when a deviation in model quality occurs.\nWhich solution will meet these requirements?",
        "options": [
            "A. Use SageMaker real-time inference for inference. Use SageMaker Model Monitor for notifications about model quality.",
            "B. Use SageMaker batch transform for inference. Use SageMaker Model Monitor for notifications about model quality.",
            "C. Use SageMaker Serverless Inference for inference. Use SageMaker Inference Recommender for notifications about model quality.",
            "D. Keep using SageMaker Asynchronous Inference for inference. Use SageMaker Inference Recommender for notifications about model quality."
        ],
        "answer": "A",
        "explanation": "SageMaker real-time inference is designed for low-latency, real-time use cases, such as detecting fraudulent transactions in banking applications. It eliminates the delays associated with SageMaker Asynchronous Inference, improving inference performance. SageMaker Model Monitor provides tools to monitor deployed models for deviations in data quality, model performance, and other metrics. It can be configured to send notifications when a deviation in model quality is detected, ensuring the system remains reliable."
    },
    {
        "question": "An ML engineer is developing a fraud detection model on AWS. The training dataset includes transaction logs, customer profiles, and tables from an on-premises MySQL database. The transaction logs and customer profiles are stored in Amazon S3. The dataset has a class imbalance that affects the learning of the model's algorithm. Additionally, many of the features have interdependencies. The algorithm is not capturing all the desired underlying patterns in the data. The training dataset includes categorical data and numerical data. The ML engineer must prepare the training dataset to maximize the accuracy of the model.\nWhich action will meet this requirement with the LEAST operational overhead?",
        "options": [
            "A. Use AWS Glue to transform the categorical data into numerical data.",
            "B. Use AWS Glue to transform the numerical data into categorical data.",
            "C. Use Amazon SageMaker Data Wrangler to transform the categorical data into numerical data.",
            "D. Use Amazon SageMaker Data Wrangler to transform the numerical data into categorical data."
        ],
        "answer": "C",
        "explanation": "Preparing a training dataset that includes both categorical and numerical data is essential for maximizing the accuracy of a machine learning model. Transforming categorical data into numerical format is a critical step, as most ML algorithms require numerical input. Why Transform Categorical Data into Numerical Data? * Model Compatibility: Many ML algorithms cannot process categorical data directly and require numerical representations. * Improved Performance: Proper encoding of categorical variables can enhance model accuracy and convergence speed."
    },
    {
        "question": "An ML engineer needs to use data with Amazon SageMaker Canvas to train an ML model. The data is stored in Amazon S3 and is complex in structure. The ML engineer must use a file format that minimizes processing time for the data.\nWhich file format will meet these requirements?",
        "options": [
            "A. CSV files compressed with Snappy",
            "B. JSON objects in JSONL format",
            "C. JSON files compressed with gzip",
            "D. Apache Parquet files"
        ],
        "answer": "D",
        "explanation": "Apache Parquet is a columnar storage file format optimized for complex and large datasets. It provides efficient reading and processing by accessing only the required columns, which reduces I/O and speeds up data handling. This makes it ideal for use with Amazon SageMaker Canvas, where minimizing processing time is important for training ML models. Parquet is also compatible with S3 and widely supported in data analytics and ML workflows."
    },
    {
        "question": "A company is building a web-based Al application by using Amazon SageMaker. The application will provide the following capabilities and features: ML experimentation, training, a central model registry, model deployment, and model monitoring. The application must ensure secure and isolated use of training data during the ML lifecycle. The training data is stored in Amazon S3. The company needs to use the central model registry to manage different versions of models in the application.\nWhich action will meet this requirement with the LEAST operational overhead?",
        "options": [
            "A. Create a separate Amazon Elastic Container Registry (Amazon ECR) repository for each model.",
            "B. Use Amazon Elastic Container Registry (Amazon ECR) and unique tags for each model version.",
            "C. Use the SageMaker Model Registry and model groups to catalogthe models.",
            "D. Use the SageMaker Model Registry and unique tags for each model version."
        ],
        "answer": "C",
        "explanation": "Amazon SageMaker Model Registry is a feature designed to manage machine learning (ML) models throughout their lifecycle. It allows users to catalog, version, and deploy models systematically, ensuring efficient model governance and management."
    },
    {
        "question": "A company is using Amazon SageMaker and millions of files to train an ML model. Each file is several megabytes in size. The files are stored in an Amazon S3 bucket. The company needs to improve training performance.\nWhich solution will meet these requirements in the LEAST amount of time?",
        "options": [
            "A. Transfer the data to a new S3 bucket that provides S3 Express One Zone storage. Adjust the training job to use the new S3 bucket.",
            "B. Create an Amazon FSx for Lustre file system. Link the file system to the existing S3 bucket. Adjust the training job to read from the file system.",
            "C. Create an Amazon Elastic File System (Amazon EFS) file system. Transfer the existing data to the file system. Adjust the training job to read from the file system.",
            "D. Create an Amazon ElastiCache (Redis OSS) cluster. Link the Redis OSS cluster to the existing S3 bucket. Stream the data from the Redis OSS cluster directly to the training job."
        ],
        "answer": "B",
        "explanation": "Amazon FSx for Lustre is designed for high-performance workloads like ML training. It provides fast, low- latency access to data by linking directly to the existing S3 bucket and caching frequently accessed files locally. This significantly improves training performance compared to directly accessing millions of files from S3. It requires minimal changes to the training job and avoids the overhead of transferring or restructuring data, making it the fastest and most efficient solution."
    },
    {
        "question": "An ML engineer needs to use an AWS service to identify and extract meaningful unique keywords from documents.\nWhich solution will meet these requirements with the LEAST operational overhead?",
        "options": [
            "A. Use the Natural Language Toolkit (NLTK) library on Amazon EC2 instances for text pre-processing. Use the Latent Dirichlet Allocation (LDA) algorithm to identify and extract relevant keywords.",
            "B. Use Amazon SageMaker and the BlazingText algorithm. Apply custom pre-processing steps for stemming and removal of stop words. Calculate term frequency-inverse document frequency (TF-IDF) scores to identify and extract relevant keywords.",
            "C. Store the documents in an Amazon S3 bucket. Create AWS Lambda functions to process the documents and to run Python scripts for stemming and removal of stop words. Use bigram and trigram techniques to identify and extract relevant keywords.",
            "D. Use Amazon Comprehend custom entity recognition and key phrase extraction to identify and extract relevant keywords."
        ],
        "answer": "D",
        "explanation": "Amazon Comprehend provides pre-built functionality for key phrase extraction and can identify meaningful keywords from documents with minimal setup or operational overhead. It eliminates the need for manual preprocessing, stemming, or stop-word removal and does not require custom model development or infrastructure management. This makes it the most efficient and low-maintenance solution for the task."
    }
];
