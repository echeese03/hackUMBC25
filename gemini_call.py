import google.generativeai as genai

# Paste API key directly here
genai.configure(api_key="AIzaSyDx4iMiM4EcWoKnMO_1OiCFI9HTgn3Zzug")

model = genai.GenerativeModel("gemini-pro-latest")

response = model.generate_content("Write a short poem about recycling.")

print(response.text)
