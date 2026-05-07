"""rtmdk/production/telemetry.py — OpenTelemetry tracing integration.

Provides distributed tracing for RTMDK API requests.
Activate by setting OTEL_EXPORTER_OTLP_ENDPOINT env var.
"""

import os
from typing import Optional

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource


class TelemetryManager:
    """Manages OpenTelemetry tracer lifecycle."""

    def __init__(self, service_name: str = "rtmdk"):
        self.tracer: Optional[trace.Tracer] = None
        self._provider: Optional[TracerProvider] = None
        endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        if endpoint:
            resource = Resource(attributes={SERVICE_NAME: service_name})
            self._provider = TracerProvider(resource=resource)
            exporter = OTLPSpanExporter(endpoint=endpoint)
            self._provider.add_span_processor(BatchSpanProcessor(exporter))
            trace.set_tracer_provider(self._provider)
            self.tracer = trace.get_tracer(service_name)

    @property
    def enabled(self) -> bool:
        return self.tracer is not None

    def shutdown(self):
        if self._provider is not None:
            self._provider.shutdown()
