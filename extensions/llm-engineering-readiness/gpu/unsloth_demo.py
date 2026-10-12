"""Optional GPU PEFT setup; not a claim of completed Unsloth SFT."""
from .preflight import require_gpu
def configure():
    require_gpu()
    from unsloth import FastLanguageModel
    model,tokenizer=FastLanguageModel.from_pretrained(
        model_name="unsloth/SmolLM2-135M-Instruct",max_seq_length=512,load_in_4bit=True)
    model=FastLanguageModel.get_peft_model(model,r=8,lora_alpha=16,lora_dropout=0,
        bias="none",target_modules=["q_proj","k_proj","v_proj","o_proj"])
    return model,tokenizer
if __name__=="__main__":
    model,tokenizer=configure()
    print("GPU model/LoRA loaded; training and evaluation not run")
