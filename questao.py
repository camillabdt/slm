import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Configuração CPU
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu", torch_dtype=torch.float32)

# 2. O Prompt com EXEMPLO (One-Shot)
# Aqui nós mostramos para ele EXATAMENTE o que queremos antes de pedir
prompt_text = """### Instruction:
You are a Software Engineering professor. Create a multiple-choice question following the example format.

---
Example Question:
Stem: Which of the following is a Functional Requirement?
Options:
A) The system shall be able to handle 1000 users.
B) The system shall allow users to log in with email and password.
C) The system code must be written in Python.
D) The background color must be blue.
E) The system must respond in 2 seconds.
Correct Answer: B
Explanation: Logging in is a specific behavior/function the system provides to the user.
---

Now, generate a NEW question about 'Non-Functional Requirements'.

### Response:
Stem:"""

# 3. Gerar
print("Gerando com exemplo (One-Shot)...")
inputs = tokenizer(prompt_text, return_tensors="pt").to("cpu")

with torch.no_grad():
    outputs = model.generate(
        **inputs, 
        max_new_tokens=200, 
        do_sample=True, 
        temperature=0.6, # Baixei um pouco a temperatura para ele não "viajar" muito
        top_p=0.9,
        repetition_penalty=1.2
    )

# 4. Mostrar resultado
# O split remove o prompt e mostra só o que ele gerou novo
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
final_answer = "Stem:" + generated_text.split("Stem:")[-1] 

print("-" * 40)
print(final_answer)
print("-" * 40)