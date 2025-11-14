# import os
# import torch
# from dotenv import load_dotenv

# # 1. Load environment variables
# load_dotenv()
# hf_token = os.getenv("HF_TOKEN")
# if not hf_token:
#     print("HF_TOKEN not found. Please ensure it's set in your .env file.")
#     exit()
# print("HF_TOKEN loaded successfully!")

# # 2. Import and install dependencies if needed
# try:
#     from transformers import pipeline
# except ImportError:
#     print("Transformers not found. Run: pip install transformers")
#     exit()
# try:
#     import accelerate
# except ImportError:
#     print("Accelerate not found. Run: pip install accelerate")
#     exit()
# try:
#     import torch
# except ImportError:
#     print("PyTorch not found. Run: pip install torch")
#     exit()

# # 3. Set up the model pipeline
# model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# print(f"Loading model: {model_id}...")
# try:
#     pipe = pipeline(
#         "text-generation",
#         model=model_id,
#         device_map="auto",
#         # Use 'dtype' if your transformers version allows, otherwise use 'torch_dtype'
#         dtype=torch.bfloat16,   # or dtype=torch.bfloat16
#         token=hf_token,
#     )
# except Exception as e:
#     print(f"Error loading model: {e}")
#     print("Please ensure you have accepted the model's license on Hugging Face and have a GPU available.")
#     exit()
# print("Model loaded. You can now chat with your travel agent.")
# print("Type 'quit' or 'exit' to end the conversation.")

# # 4. Chat loop with proper prompt formatting
# messages = [
#     {"role": "system", "content": "You are a friendly and expert travel agent. You help users plan their dream vacations by providing helpful and concise information."},
# ]
# while True:
#     user_input = input("\nYou: ")
#     if user_input.lower() in ["quit", "exit"]:
#         break
#     messages.append({"role": "user", "content": user_input})

#     # Format messages history into a prompt
#     prompt = ""
#     for m in messages:
#         if m["role"] == "system":
#             prompt += f"[SYSTEM] {m['content']}\n"
#         elif m["role"] == "user":
#             prompt += f"[USER] {m['content']}\n"
#         elif m["role"] == "assistant":
#             prompt += f"[ASSISTANT] {m['content']}\n"

#     response = pipe(prompt, max_new_tokens=512, temperature=0.7, top_p=0.95)
#     generated_text = response[0]["generated_text"] if "generated_text" in response[0] else response[0]["text"]
#     assistant_response = generated_text[len(prompt):].strip()
#     print(f"\nTravel Agent: {assistant_response}")
#     messages.append({"role": "assistant", "content": assistant_response})


# import torch
# print(torch.cuda.is_available())  # Should print True
# print(torch.cuda.get_device_name(0))  # Should print "NVIDIA GeForce RTX 3060"

import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")


