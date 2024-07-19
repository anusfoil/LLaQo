import pandas as pd
import pickle
import os.path
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request

# Load the datasets
df1 = pd.read_csv('checkpoint_1191000.csv')
df2 = pd.read_csv('ltu_results.csv')
df3 = pd.read_csv('mullama_results.csv')

# Authenticate and create the service
def create_service():
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', scopes=['https://www.googleapis.com/auth/forms'])
            creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    service = build('forms', 'v1', credentials=creds)
    return service

service = create_service()

# Function to create a form
def create_form(index, questions):
    form_title = f"Audio Model Evaluation Form {index + 1}"
    form = service.forms().create(body={"info": {"title": form_title}}).execute()
    form_id = form['formId']

    # Add questions
    for idx, (entry1, entry2, entry3) in enumerate(zip(questions[0], questions[1], questions[2])):
        question_text = f"Q{idx+1}: {entry1['question']}\nListen to the audio: {entry1['audio_path']}"
        question = {
            'createItem': {
                'item': {
                    'title': question_text,
                    'questionItem': {
                        'question': {
                            'choiceQuestion': {
                                'type': 'RADIO',
                                'options': [
                                    {'value': f"Model 1: {entry1['verbal_output']}"},
                                    {'value': f"Model 2: {entry2['verbal_output']}"},
                                    {'value': f"Model 3: {entry3['verbal_output']}"}
                                ],
                                'shuffle': True
                            }
                        }
                    },
                    'description': "Select the best response based on your judgment."
                },
                'location': {
                    'index': idx
                }
            }
        }
        service.forms().items().create(formId=form_id, body=question).execute()
    
    print(f"Created form {index+1} with ID: {form_id}")

# Divide the dataframe into chunks and create forms
chunk_size = 50
num_forms = 20
total_questions = chunk_size * num_forms

# Ensure each dataset has the same number of rows and slice according to total questions needed
assert len(df1) == len(df2) == len(df3), "Dataframe lengths are not equal."
df1, df2, df3 = df1[:total_questions], df2[:total_questions], df3[:total_questions]

for i in range(num_forms):
    start_idx = i * chunk_size
    end_idx = start_idx + chunk_size
    chunks = (df1.iloc[start_idx:end_idx], df2.iloc[start_idx:end_idx], df3.iloc[start_idx:end_idx])
    create_form(i, chunks)
