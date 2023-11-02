from vertexai.preview.language_models import TextGenerationModel

print("Query: What are the olympics?")
prompt = "What are the olympics?"
model = TextGenerationModel.from_pretrained("text-bison@latest")
response = model.predict(prompt, temperature=0.1, top_k=40, top_p=0.8, max_output_tokens=1024)
print("Response from Vertex AI:")
print(response.text)
