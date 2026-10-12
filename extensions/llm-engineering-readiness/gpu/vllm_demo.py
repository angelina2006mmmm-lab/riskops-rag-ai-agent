"""Optional GPU vLLM inference demo; requires model weights, CUDA and VRAM."""
from .preflight import require_gpu
def run(model="HuggingFaceTB/SmolLM2-135M-Instruct"):
    require_gpu()
    from vllm import LLM,SamplingParams
    lm=LLM(model=model,dtype="float16",gpu_memory_utilization=0.5,enforce_eager=True)
    output=lm.generate(["Summarize a synthetic credit risk incident."],SamplingParams(temperature=0,max_tokens=48))
    return output[0].outputs[0].text
if __name__=="__main__":print(run())
