from llama_kernel import LlamaInference

def main():
    # Initialize the model
    model = LlamaInference(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # Using a smaller model for CPU
        device="cpu",  # Using CPU for now since we're on macOS
        use_flash_attention=False  # Disabled since we didn't install flash-attn
    )

    # Generate text with a creative writing prompt
    prompt = """<|system|>
You are a creative and imaginative AI assistant that writes engaging stories and poems.</s>
<|user|>
Write a short poem about artificial intelligence and human creativity, exploring the relationship between technology and human imagination. The poem should be 4-6 lines long and use metaphors to convey its message.</s>
<|assistant|>"""

    output = model.generate(
        prompt,
        max_length=1024,
        temperature=0.8,  # Slightly increased for more creativity
        top_p=0.9,
        repetition_penalty=1.1,
        do_sample=True
    )

    print(output)

if __name__ == "__main__":
    main() 