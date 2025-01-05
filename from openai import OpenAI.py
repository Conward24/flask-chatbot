from openai import OpenAI

client = OpenAI(
  api_key="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"
)

completion = client.chat.completions.create(
  model="gpt-4o-mini",
  store=True,
  messages=[
    {"role": "user", "content": "write a haiku about ai"}
  ]
)

print(completion.choices[0].message);