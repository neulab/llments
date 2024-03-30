import os
from litellm import completion

class APIBasedLM:
    def __init__(self, api_key, model_name):
        # Set the API key
        self.api_key = api_key
        self.model_name = model_name
    
    def generate_response(self, prompt):
        # Generate response
        response = completion(
            model=self.model_name,
            messages=[{"content": prompt, "role": "user"}]
        )
        return response['choices'][0]['message']['content']

api_key = "your_api_key_here"
model_name = "model-name"
api = APIBasedLM(api_key=api_key, model_name=model_name)

# Generate response for a prompt
prompt = "sample-prompt"
response = api.generate_response(prompt)
print(response)