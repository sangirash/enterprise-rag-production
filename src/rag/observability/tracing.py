"""
OpenTelemetry distributed tracing setup.

Exports spans to an OTLP collector (e.g. OpenTelemetry Collector, Jaeger, Tempo).
Call setup_tracing() once at application startup; then use get_tracer() anywhere.
"""
from __future__ import annotations

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from ..config import settings

_tracer: trace.Tracer | None = None


def setup_tracing(service_name: str = "enterprise-rag") -> trace.Tracer:
    """Initialise the global TracerProvider and return a Tracer instance."""
    global _tracer
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=settings.otlp_endpoint, insecure=True)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer(service_name)
    return _tracer


def get_tracer() -> trace.Tracer:
    """Return the configured Tracer, initialising with defaults if needed."""
    global _tracer
    if _tracer is None:
        _tracer = setup_tracing()
    return _tracer
