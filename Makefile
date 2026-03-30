# Load local overrides (gitignored). Create .makerc.local to set PROJECT_ID etc.
# Example: echo "PROJECT_ID := reachymini" > .makerc.local
-include .makerc.local

PROJECT_ID  ?= $(shell gcloud config get-value project 2>/dev/null)
IMAGE       := gcr.io/$(PROJECT_ID)/emotion-cloud
TAG         ?= latest
REGION      := us-central1
CLUSTER     := emotion-cloud-cluster
NAMESPACE   := default

.PHONY: help proto build push deploy logs port-forward \
        download-weights package-model create-cluster delete-cluster \
        k8s-apply k8s-delete k8s-status secrets docker-run

help:
	@echo ""
	@echo "  emotion-cloud Makefile"
	@echo ""
	@echo "  Local dev"
	@echo "    make proto              Regenerate gRPC stubs from proto/emotion.proto"
	@echo "    make docker-run         Run locally with docker compose (GPU required)"
	@echo "    make download-weights   Pull weights from GCS to /opt/ml/model-store/weights"
	@echo "    make package-model      Create emotion-detector.mar for TorchServe"
	@echo ""
	@echo "  CI / CD"
	@echo "    make build              Build Docker image"
	@echo "    make push               Push image to GCR"
	@echo "    make deploy             build + push + k8s-apply"
	@echo ""
	@echo "  GKE"
	@echo "    make create-cluster     Create GKE Autopilot cluster in $(REGION)"
	@echo "    make k8s-apply          Apply all manifests in deploy/k8s/"
	@echo "    make k8s-delete         Delete all k8s resources"
	@echo "    make k8s-status         Show pod / service status"
	@echo "    make secrets            Create k8s Secret from .env"
	@echo "    make port-forward       Forward gRPC port 50051 to localhost"
	@echo "    make logs               Tail pod logs"
	@echo ""

# ── gRPC stubs ────────────────────────────────────────────────────────────────

proto:
	python -m grpc_tools.protoc \
		-I proto \
		--python_out=. \
		--grpc_python_out=. \
		proto/emotion.proto
	@echo "Generated: emotion_pb2.py  emotion_pb2_grpc.py"

# ── Local dev ─────────────────────────────────────────────────────────────────

docker-run:
	docker compose -f deploy/docker/docker-compose.yml up --build

download-weights:
	python scripts/download_weights.py

package-model:
	bash scripts/package_model.sh

# ── Docker / GCR ─────────────────────────────────────────────────────────────

build:
	docker build \
		-f deploy/docker/Dockerfile \
		-t $(IMAGE):$(TAG) \
		.

push: build
	docker push $(IMAGE):$(TAG)

# ── GKE cluster ───────────────────────────────────────────────────────────────

create-cluster:
	gcloud container clusters create-auto $(CLUSTER) \
		--region $(REGION) \
		--project $(PROJECT_ID)
	gcloud container clusters get-credentials $(CLUSTER) \
		--region $(REGION) \
		--project $(PROJECT_ID)

delete-cluster:
	gcloud container clusters delete $(CLUSTER) \
		--region $(REGION) \
		--project $(PROJECT_ID) \
		--quiet

# ── Kubernetes ────────────────────────────────────────────────────────────────

secrets:
	kubectl create secret generic emotion-cloud-secrets \
		--from-env-file=.env \
		--namespace $(NAMESPACE) \
		--dry-run=client -o yaml | kubectl apply -f -

k8s-apply: secrets
	# PROJECT_ID is substituted at deploy time from gcloud config (never committed).
	sed "s/PROJECT_ID/$(PROJECT_ID)/g" deploy/k8s/serviceaccount.yaml | kubectl apply -f -
	sed "s/PROJECT_ID/$(PROJECT_ID)/g" deploy/k8s/deployment.yaml     | kubectl apply -f -
	kubectl apply -f deploy/k8s/service.yaml
	kubectl apply -f deploy/k8s/hpa.yaml

k8s-delete:
	kubectl delete -f deploy/k8s/hpa.yaml        --ignore-not-found
	kubectl delete -f deploy/k8s/service.yaml     --ignore-not-found
	kubectl delete -f deploy/k8s/deployment.yaml  --ignore-not-found
	kubectl delete -f deploy/k8s/serviceaccount.yaml --ignore-not-found
	kubectl delete secret emotion-cloud-secrets   --ignore-not-found

k8s-status:
	@echo "\n── Pods ──────────────────────────────────────────────────────────────"
	kubectl get pods -l app=emotion-cloud -o wide
	@echo "\n── Service ───────────────────────────────────────────────────────────"
	kubectl get svc emotion-cloud-grpc
	@echo "\n── HPA ───────────────────────────────────────────────────────────────"
	kubectl get hpa emotion-cloud-hpa

port-forward:
	kubectl port-forward svc/emotion-cloud-grpc 50051:50051

logs:
	kubectl logs -l app=emotion-cloud -f --tail=100

# ── Full deploy ───────────────────────────────────────────────────────────────

deploy: push k8s-apply
	@echo ""
	@echo "Deployed $(IMAGE):$(TAG) to GKE cluster $(CLUSTER)"
	@echo "Run 'make k8s-status' to check readiness."
