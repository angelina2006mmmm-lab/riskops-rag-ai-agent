"""Optional Langfuse SDK v4 adapter; input data never sent to tracing.

The real SDK is imported in CI but no SaaS ingestion is claimed. An injected
client allows dry-run testing without Langfuse credentials or network access.
"""
import os
from .domain import NullTracer

class LangfuseTracer:
    def __init__(self,client):
        self._client=client
    def span(self,name:str):
        # SDK v4 follows OTel observation contexts.
        return self._client.start_as_current_observation(as_type="span",name=name)

def make_tracer(enabled:bool|None=None,client=None):
    if enabled is None:enabled=os.getenv("LANGFUSE_ENABLED")=="1"
    if not enabled:return NullTracer()
    if client is None:
        if not all(os.getenv(x) for x in ("LANGFUSE_PUBLIC_KEY","LANGFUSE_SECRET_KEY")):
            raise RuntimeError("Langfuse requested but credentials are not configured")
        from langfuse import get_client
        client=get_client()
    return LangfuseTracer(client)
