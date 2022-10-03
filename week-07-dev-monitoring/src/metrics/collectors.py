from prometheus_client import Counter, CollectorRegistry, Histogram

APP_NAME_IN_PROMETHEUS = 'genres'

api_metrics_registry = CollectorRegistry()

LATENCY_BUCKETS = (
    20, 50, 100, 150, 300, 500, 1000, 3000, 10000, float('inf'),
)


requests_counter = Counter(
    name=f'{APP_NAME_IN_PROMETHEUS}_requests_total',
    documentation='count number of requests',
    labelnames=['method', 'endpoint'],
    registry=api_metrics_registry,
)

requests_latency_hist = Histogram(
    name=f'{APP_NAME_IN_PROMETHEUS}_requests_latency_hist',
    documentation='count number of requests',
    labelnames=['method', 'endpoint'],
    registry=api_metrics_registry,
    buckets=LATENCY_BUCKETS,
)
