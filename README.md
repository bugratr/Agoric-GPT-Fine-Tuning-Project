# Agoric GPT Fine-Tuning Project

This project demonstrates how to fine-tune the GPT-3.5-turbo model using training and validation datasets tailored for Agoric-related smart contract development tasks.

## Prerequisites

Before you begin, ensure you have the following:

- An OpenAI API key
- Python installed on your machine
- Jupyter Notebook installed (optional but recommended)
- Required Python libraries: `openai`, `json`

## Installation

1. **Install the required libraries:**

```bash
pip install openai

2. Clone the repository

git clone https://github.com/bugratr/agoric-gpt-fine-tuning.git
cd agoric-gpt-fine-tuning

## Dataset
The project includes two datasets:

training_set.jsonl: The training dataset containing examples for fine-tuning.
validation_set.jsonl: The validation dataset used to validate the fine-tuning process.
The datasets are expected to be in JSONL format.

## Jupyter Notebook
The following Jupyter Notebook script demonstrates the process of fine-tuning the GPT model using the provided datasets:

# Import necessary libraries
import openai
import json

# Set your OpenAI API key
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY_HERE"
openai.api_key = OPENAI_API_KEY

# Example of training and validation dataset
training_json_script = '''
# Training dataset content goes here
'''
validation_json_script = '''
# Validation dataset content goes here
'''

# Save training dataset to a JSONL file
with open('training_set.jsonl', 'w') as file:
    file.write(training_json_script)

# Save validation dataset to a JSONL file
with open('validation_set.jsonl', 'w') as file:
    file.write(validation_json_script)

# Load and display the training dataset
with open('training_set.jsonl', 'r', encoding='utf-8') as f:
    training_dataset = [json.loads(line) for line in f]

print("Training set contains examples:", len(training_dataset))
print("First example in the training set:")
for message in training_dataset[0]["messages"]:
    print(message)

# Load and display the validation dataset
with open('validation_set.jsonl', 'r', encoding='utf-8') as f:
    validation_dataset = [json.loads(line) for line in f]

print("\\nValidation set contains examples:", len(validation_dataset))
print("First example in the validation set:")
for message in validation_dataset[0]["messages"]:
    print(message)

# Upload the training dataset to OpenAI
training_file_response = openai.File.create(
    file=open("training_set.jsonl", "rb"),
    purpose="fine-tune"
)
print("Training File ID:", training_file_response.id)

# Upload the validation dataset to OpenAI
validation_file_response = openai.File.create(
    file=open("validation_set.jsonl", "rb"),
    purpose="fine-tune"
)
print("Validation File ID:", validation_file_response.id)

# Create a fine-tuning job
fine_tuning_response = openai.FineTuningJob.create(
    training_file=training_file_response.id,
    validation_file=validation_file_response.id,
    model="gpt-3.5-turbo"
)
print("Fine Tuning Job ID:", fine_tuning_response.id)

# Check the status of the fine-tuning job
fine_tuning_status = openai.FineTuningJob.retrieve(fine_tuning_response.id)
print("Fine Tuning Job Status:", fine_tuning_status.status)

# Example to generate a response using the fine-tuned model
response = openai.ChatCompletion.create(
  model=fine_tuning_response.fine_tuned_model,
  messages=[
    {"role": "system", "content": "Agoric GPT is a smart contract development assistant."},
    {"role": "user", "content": "Who is Linus Torvalds?"}
  ]
)
print(response.choices[0].message.content)

Steps to Run the Notebook
Set your OpenAI API Key:
Replace YOUR_OPENAI_API_KEY_HERE with your actual OpenAI API key.

Populate the Datasets:
Fill in the training_json_script and validation_json_script with your actual training and validation data.

Run the Notebook:
Open the Jupyter Notebook and run the cells sequentially to perform the fine-tuning process.

Monitoring the Fine-Tuning Job
To monitor the status of the fine-tuning job, use the following methods:

Retrieve the latest fine-tuning jobs:

openai.FineTuningJob.list(limit=10)

Check the status of a specific fine-tuning job:

openai.FineTuningJob.retrieve("fine_tuning_job_id")


Cancel a fine-tuning job:
openai.FineTuningJob.cancel("fine_tuning_job_id")

Delete a fine-tuned model:
openai.Model.delete("fine_tuned_model_id")

Usage
Once the model is fine-tuned, you can generate responses using the fine-tuned model by providing it with appropriate prompts.

License
This project is licensed under the MIT License. See the LICENSE file for more details.

