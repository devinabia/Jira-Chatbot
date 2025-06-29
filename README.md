# Jira Chatbot Backend

## Description

A FastAPI-powered Slack bot that provides instant access to your organization's Jira knowledge base through natural conversation. Users can ask questions using multiple interaction methods and get AI-powered responses directly in Slack.

🛠️ Tech Stack:

Backend: Python FastAPI
Integration: Slack Bolt SDK
VectorDB: Qdrant
LLM: Gemini


## Prerequisites

- Python 3.13

## Installation

1. Clone the repository:

```
git clone https://github.com/devinabia/Jira-Chatbot.git
```

2. Navigate to the project directory:


3. Create a virtual environment and activate it:

```
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

4. Install the required dependencies:

```
pip install -r requirements.txt
```

5. Set up the environment variables:

Create a `.env` file in the project root directory and add the variables given in env_sample:

## Running the Application

1. Start the FastAPI server:

```
uvicorn main:app --reload
```

The server will start running at `http://localhost:8000`.

2. Access the application in your web browser:

```
http://localhost:8000
```

You should see the "server is up and running" message.

## Deployment

To deploy the application, you can use a hosting platform like Heroku, AWS, or DigitalOcean. Make sure to set the environment variables on the deployment platform as well.
