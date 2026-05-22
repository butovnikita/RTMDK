# Kubernetes Deployment

## Quick Start

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml
```

## Verify

```bash
kubectl get pods -n rtmdk
kubectl get svc -n rtmdk
kubectl get hpa -n rtmdk
```

## Access

```bash
kubectl port-forward svc/rtmdk-service -n rtmdk 8080:80
```

## Scaling

HPA is configured for 3-10 replicas based on CPU (70%) and memory (80%).
