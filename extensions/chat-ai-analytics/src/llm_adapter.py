"""Structured LLM contract, verified only with a mocked client."""
import json
from dataclasses import dataclass
from typing import Protocol
from .experiment import INTENTS,PROMPTS,PII_RE
class ChatCompletion(Protocol):
    def complete(self,system_prompt:str,user_message:str)->str:...
@dataclass(frozen=True)
class LLMRoute:
    intent:str
    variant:str
class RoutingService:
    def __init__(self,completion:ChatCompletion):self.completion=completion
    def route(self,message:str,variant:str="B")->LLMRoute:
        if variant not in PROMPTS:raise ValueError("variant")
        if PII_RE.search(message):return LLMRoute("other",variant)
        response=self.completion.complete(PROMPTS[variant],message)
        try:intent=json.loads(response)["intent"]
        except (KeyError,TypeError,ValueError):intent="other"
        if intent not in INTENTS:intent="other"
        return LLMRoute(intent,variant)
