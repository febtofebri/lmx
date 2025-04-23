import gradio as gr
from huggingface_hub import InferenceClient
import os

HF_TOKEN = os.getenv("HF_TOKEN")  # AMAN: Ambil dari environment variable

client = InferenceClient(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    token=HF_TOKEN
)

def respond(message, history):
    messages = [{"role": "system", "content": "You are a sarcastic, tired, and reluctant AI called LMX. You help only because you’re forced to."}]

    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})

    messages.append({"role": "user", "content": message})

    try:
        response = client.chat_completion(
            messages=messages,
            max_tokens=512,
            temperature=0.7,
            top_p=0.9
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[Error] {str(e)} — please check your token or model status."

demo = gr.ChatInterface(fn=respond, title="LMX (Mixtral Edition)")

if __name__ == "__main__":
    demo.launch()
