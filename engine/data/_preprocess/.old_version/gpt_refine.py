import openai

openai.api_key = "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"

prompt = "I will give you a list of video descriptions. Process each individually. Extract the type of the sound and generate an audio caption describing possoble audio inside the video. The audio caption should be less than 20 words. Replace key workds such as 'photo with 'sound' when applicable' Delete the author of the sound. Delete locations, city names, country names. Delete the time. Delete device names. Delete the proper noun modifiers, number modifiers, and unit modifiers. Summarize each output into one sentence. Replace all named entities with their hypernyms. Replace people names with 'someone'. Do not write introductions or explanations. Only describe the sound events and do not use 'heard', 'recorded'. Start each output sentence with its index. Make sure you are using grammatical subject-verb-object sentences. Output 'Failure.' if the description is not related to sound. The descriptions are {}"
candidates = [
    "a cup of coffee with a spoon in it", 
    "a young girl with long hair and a smile on her face"
    ]

response = openai.Completion.create(
    engine="text-davinci-003",  # Specify the ChatGPT model variant
    prompt=prompt.format(candidates),
    max_tokens=40,
    # n=1,
    stop=None,
    temperature=0.7
)
generated_response = response.choices[0].text.strip()
print("Generated Response:", generated_response)