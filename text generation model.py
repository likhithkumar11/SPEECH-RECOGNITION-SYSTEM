# task4_text_generation.py

from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_text(prompt, max_length=100):
    # Load pre-trained model and tokenizer
    model_name = "gpt2"  # You can use "distilgpt2" for a smaller version
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Encode the prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Generate text
    output = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.7,
        top_k=50,
        top_p=0.95
    )

    # Decode and return the result
    return tokenizer.decode(output[0], skip_special_tokens=True)

if __name__ == "__main__":
    user_prompt = "Artificial Intelligence in healthcare"
    result = generate_text(user_prompt, max_length=120)
    print("Prompt:", user_prompt)
    print("\nGenerated Text:\n", result)
