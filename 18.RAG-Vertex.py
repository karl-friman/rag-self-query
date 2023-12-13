import vertexai
from vertexai.language_models import TextGenerationModel

vertexai.init(project="karl-gke-workshop", location="us-central1")
parameters = {
    "candidate_count": 1,
    "max_output_tokens": 1024,
    "temperature": 0.2,
    "top_p": 0.8,
    "top_k": 40,
}
model = TextGenerationModel.from_pretrained("text-bison")
response = model.predict(
    """Respond only using json.

Give me a list of 3 african mammals and this data
average age (int)
color (string)
average height in cm (int)
average length in cm (int)""",
    **parameters,
)
print(f"Response from Model: {response.text}")
