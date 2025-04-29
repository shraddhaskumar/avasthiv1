import requests
import time
import csv

# API URLs
BASE_URL = "http://127.0.0.1:8000"
QUERY_URL = f"{BASE_URL}/query"
FEEDBACK_URL = f"{BASE_URL}/feedback"
EVALUATE_URL = f"{BASE_URL}/evaluate-sentiment"

# Authentication (replace with your actual token)
TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhY3F1IiwiZXhwIjoxNzQzMjY5MjI5fQ.Qs27l_gD5Vb9xIISEtcxi_gXY8cCELL0Uk1n1wVz1QI"
HEADERS = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}

# Load dataset
DATASET_FILE = "testdata.csv"
queries = []

with open(DATASET_FILE, newline="", encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        queries.append({"id": row["id"], "query": row["query"], "actual_sentiment": row["actual_sentiment"]})

# Step 1: Send Queries to /query endpoint
feedback_data = []
for q in queries:
    response = requests.post(QUERY_URL, json={"query": q["query"]}, headers=HEADERS)
    if response.status_code == 200:
        result = response.json()
        feedback_data.append({"query_id": q["id"], "actual_sentiment": q["actual_sentiment"]})
    else:
        print(f"Error with query ID {q['id']}: {response.json()}")
    time.sleep(1)  # Avoid API rate limits

# Step 2: Submit actual sentiment using /feedback
feedback_payload = {"feedback": feedback_data}
feedback_response = requests.post(FEEDBACK_URL, json=feedback_payload, headers=HEADERS)
print("Feedback Response:", feedback_response.json())

# Step 3: Get Evaluation Metrics from /evaluate-sentiment
eval_response = requests.get(EVALUATE_URL, headers=HEADERS)
if eval_response.status_code == 200:
    eval_data = eval_response.json()
    print("Evaluation Metrics:", eval_data)
else:
    print("Error during evaluation:", eval_response.json())
