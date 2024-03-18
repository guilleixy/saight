import openai
from dotenv import load_dotenv
import os

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')

def chat_with_gpt(prompt):
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        temperature=0.1,
        messages=[
            {"role": "system", "content": "Eres parte de una aplicación destinada a ayudar a personas con discapacidad visual a interactuar de forma más efectiva con su entorno. En concreto eres su asistente personal, encargado de recibir la información del entorno y ayudar al usuario con estos datos de la forma que él te solicite. A continuación, recibes una lista de elementos que hay en el entorno. Ten en cuenta que algunos pueden no estar relacionados con lo que te preguntan."},
            {"role": "user", "content": prompt}
        ]
    )
    return response['choices'][0]['message']['content']

datos = "Tomate, pollo, limón, nevera"

# Ejemplo de uso
prompt = "Que recetas puedo hacer con estos ingredientes?" + datos
response = chat_with_gpt(prompt)
print(response)