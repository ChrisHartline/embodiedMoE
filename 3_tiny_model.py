"""
Working with tiny language models
Demonstrates loading and using small models
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class TinyLanguageModel:
    """Wrapper for tiny language models"""
    
    def __init__(self, model_name: str = "roneneldan/TinyStories-33M"):
        """
        Initialize tiny model
        
        Popular tiny models:
        - roneneldan/TinyStories-33M (33M params)
        - roneneldan/TinyStories-8M (8M params)
        - distilgpt2 (82M params)
        """
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
        print(f"Model loaded on {self.device}")
        print(f"Parameters: {self.count_parameters():,}")
    
    def count_parameters(self) -> int:
        """Count model parameters"""
        return sum(p.numel() for p in self.model.parameters())
    
    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """Generate text from prompt"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the prompt from output
        response = generated[len(prompt):].strip()
        return response


def demo_tiny_model():
    """Demonstrate tiny model usage"""
    
    print("=" * 60)
    print("Tiny Language Model Demo")
    print("=" * 60)
    
    # Initialize model (this will download ~130MB)
    model = TinyLanguageModel("roneneldan/TinyStories-33M")
    
    # Test prompts
    prompts = [
        "Once upon a time",
        "The robot learned to",
        "Clara is a helpful AI who"
    ]
    
    print("\nGenerating responses...")
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        response = model.generate(prompt, max_length=50)
        print(f"Response: {response}")
    
    print("\n" + "=" * 60)
    print("Note: This model is trained on children's stories,")
    print("so responses are simple. For Clara, you'd fine-tune")
    print("on conversational data.")
    print("=" * 60)


if __name__ == "__main__":
    demo_tiny_model()