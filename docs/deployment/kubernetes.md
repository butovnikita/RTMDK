# Kubernetes Deployment

The `k8s/` directory contains ready-made manifests: namespace, ConfigMap,
PVC (memory persistence), Deployment, Service, and an HPA (3–10 replicas
based on 70% CPU / 80% memory).

## Quick Start

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml
```

## Verify & Access

```bash
kubectl get pods -n rtmdk
kubectl get svc -n rtmdk
kubectl get hpa -n rtmdk

kubectl port-forward svc/rtmdk-service -n rtmdk 8080:80
```

## Manifests

| File | Purpose |
|------|---------|
| `namespace.yaml` | `rtmdk` namespace |
| `configmap.yaml` | `RTMDK_*` env configuration |
| `pvc.yaml` | Persistent volume for memory data |
| `deployment.yaml` | RTMDK server pods |
| `service.yaml` | Cluster service (port 80 → 8080) |
| `hpa.yaml` | Horizontal Pod Autoscaler |

## Full Documentation

- [k8s/README.md — canonical guide](../../k8s/README.md)
- [Production Guide](../02_PRODUCTION_GUIDE.md)
- [Docker Deployment](docker.md)
