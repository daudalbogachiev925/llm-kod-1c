import ollama

def generate_code(task: str) -> str:
    prompt = open('prompts/query_gen.txt').read() + "\n\nЗадача: " + task
    r = ollama.chat(model='qwen2.5-coder:7b', messages=[
        {'role': 'system', 'content': open('prompts/system.txt').read()},
        {'role': 'user', 'content': prompt}])
    return r['message']['content']
