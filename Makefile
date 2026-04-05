# Load local overrides (gitignored). Create .makerc.local to set PROJECT_ID etc.
# Example: echo "PROJECT_ID := reachymini" > .makerc.local
-include .makerc.local

PROJECT_ID  ?= $(shell gcloud config get-value project 2>/dev/null)
IMAGE       := gcr.io/$(PROJECT_ID)/emotion-cloud
TAG         ?= latest

# ── VM config ─────────────────────────────────────────────────────────────────
REGION       := us-east4
ZONE         := us-east4-a          # on-demand T4 available here
VM_NAME      := emotion-cloud-vm
MACHINE_TYPE := n1-standard-4       # 4 vCPU, 15 GB RAM
VM_SA        := emotion-cloud-vm-sa@$(PROJECT_ID).iam.gserviceaccount.com
STATIC_IP    := emotion-cloud-ip
VM_TAG       := emotion-cloud       # network tag used by the firewall rule
DLVM_FAMILY  := common-cu128-ubuntu-2204-nvidia-570
DLVM_PROJECT := deeplearning-platform-release

.PHONY: help proto build push \
        vm-setup vm-setup-sa vm-setup-fw vm-setup-ip \
        vm-create vm-delete vm-start vm-stop \
        vm-ssh vm-logs vm-status vm-ip vm-deploy \
        download-weights package-model docker-run \
        test test-install

help:
	@echo ""
	@echo "  emotion-cloud Makefile"
	@echo ""
	@echo "  VM deployment"
	@echo "    make vm-setup        One-time: create service account, firewall, static IP"
	@echo "    make vm-create       Provision VM, copy .env, start container (~3 min)"
	@echo "    make vm-deploy       Build + push image, pull on VM, restart container"
	@echo "    make vm-start        Start a stopped VM"
	@echo "    make vm-stop         Stop VM (halts compute + GPU billing; disk + IP kept)"
	@echo "    make vm-delete       Delete the VM (keeps static IP and GCR image)"
	@echo "    make vm-status       Show VM state, external IP, container status"
	@echo "    make vm-ip           Print the external IP"
	@echo "    make vm-logs         Tail container logs on the VM"
	@echo "    make vm-ssh          SSH into the VM"
	@echo ""
	@echo "  Docker / GCR"
	@echo "    make build           Build Docker image (linux/amd64)"
	@echo "    make push            Build + push to GCR"
	@echo ""
	@echo "  Local dev"
	@echo "    make proto           Regenerate gRPC stubs from proto/emotion.proto"
	@echo "    make docker-run      Run locally with docker compose (GPU required)"
	@echo "    make download-weights Pull weights from GCS to local model-store"
	@echo "    make package-model   Create emotion-detector.mar for TorchServe"
	@echo ""
	@echo "  Testing"
	@echo "    make test-install    Install dev dependencies (requirements-dev.txt)"
	@echo "    make test            Run unit + integration tests with pytest"
	@echo ""

# ── Tests ─────────────────────────────────────────────────────────────────────

test-install:
	pip3 install -r requirements-dev.txt

test:
	pytest -v --tb=short

# ── gRPC stubs ────────────────────────────────────────────────────────────────

proto:
	python3 -m grpc_tools.protoc \
		-I proto \
		--python_out=. \
		--grpc_python_out=. \
		proto/emotion.proto
	@echo "Generated: emotion_pb2.py  emotion_pb2_grpc.py"

# ── Local dev ─────────────────────────────────────────────────────────────────

docker-run:
	docker compose -f deploy/docker/docker-compose.yml up --build

download-weights:
	python3 scripts/download_weights.py

package-model:
	bash scripts/package_model.sh

# ── Docker / GCR ─────────────────────────────────────────────────────────────

build:
	docker build \
		--platform linux/amd64 \
		-f deploy/docker/Dockerfile \
		-t $(IMAGE):$(TAG) \
		.

push: build
	gcloud auth configure-docker --project=$(PROJECT_ID) --quiet
	docker push $(IMAGE):$(TAG)

# ── VM: one-time setup ────────────────────────────────────────────────────────

vm-setup: vm-setup-sa vm-setup-fw vm-setup-ip
	@echo ""
	@echo "One-time setup complete. Run 'make vm-create' to provision the VM."

vm-setup-sa:
	gcloud iam service-accounts create emotion-cloud-vm-sa \
		--display-name="emotion-cloud VM" \
		--project=$(PROJECT_ID)
	gcloud projects add-iam-policy-binding $(PROJECT_ID) \
		--member="serviceAccount:$(VM_SA)" \
		--role="roles/storage.objectViewer"
	gcloud projects add-iam-policy-binding $(PROJECT_ID) \
		--member="serviceAccount:$(VM_SA)" \
		--role="roles/logging.logWriter"
	gcloud projects add-iam-policy-binding $(PROJECT_ID) \
		--member="serviceAccount:$(VM_SA)" \
		--role="roles/monitoring.metricWriter"
	@echo "Service account $(VM_SA) created with Storage/Logging/Monitoring roles."

vm-setup-fw:
	gcloud compute firewall-rules create emotion-cloud-grpc \
		--project=$(PROJECT_ID) \
		--direction=INGRESS \
		--priority=1000 \
		--network=default \
		--action=ALLOW \
		--rules=tcp:50051 \
		--source-ranges=0.0.0.0/0 \
		--target-tags=$(VM_TAG)
	@echo "Firewall rule created: TCP 50051 open on tag '$(VM_TAG)'."

vm-setup-ip:
	gcloud compute addresses create $(STATIC_IP) \
		--project=$(PROJECT_ID) \
		--region=$(REGION)
	@echo "Static IP reserved:"
	@gcloud compute addresses describe $(STATIC_IP) \
		--project=$(PROJECT_ID) \
		--region=$(REGION) \
		--format="value(address)"

# ── VM: lifecycle ─────────────────────────────────────────────────────────────

vm-create:
	@echo "Creating VM $(VM_NAME) in $(ZONE) with T4 GPU..."
	gcloud compute instances create $(VM_NAME) \
		--project=$(PROJECT_ID) \
		--zone=$(ZONE) \
		--machine-type=$(MACHINE_TYPE) \
		--accelerator=type=nvidia-tesla-t4,count=1 \
		--image-family=$(DLVM_FAMILY) \
		--image-project=$(DLVM_PROJECT) \
		--maintenance-policy=TERMINATE \
		--restart-on-failure \
		--boot-disk-size=100GB \
		--boot-disk-type=pd-ssd \
		--metadata-from-file=startup-script=deploy/vm/startup.sh \
		--service-account=$(VM_SA) \
		--scopes=cloud-platform \
		--tags=$(VM_TAG) \
		--address=$(STATIC_IP)
	@echo "Waiting 90s for VM to initialize..."
	@sleep 90
	@echo "Copying .env to VM..."
	gcloud compute scp .env $(VM_NAME):/tmp/emotion-cloud.env \
		--zone=$(ZONE) --project=$(PROJECT_ID)
	gcloud compute ssh $(VM_NAME) --zone=$(ZONE) --project=$(PROJECT_ID) -- \
		"sudo mv /tmp/emotion-cloud.env /etc/emotion-cloud.env && \
		 sudo chmod 600 /etc/emotion-cloud.env"
	@echo "Pulling image and starting container..."
	gcloud compute ssh $(VM_NAME) --zone=$(ZONE) --project=$(PROJECT_ID) -- \
		"gcloud auth configure-docker gcr.io --quiet && \
		 docker pull $(IMAGE):$(TAG) && \
		 docker run -d --name emotion-cloud --restart=unless-stopped \
		   --gpus all -p 50051:50051 \
		   --env-file /etc/emotion-cloud.env \
		   $(IMAGE):$(TAG)"
	@echo ""
	@echo "emotion-cloud is starting up. External IP:"
	@make --no-print-directory vm-ip
	@echo ""
	@echo "Allow ~3 min for TorchServe + model load, then test with:"
	@echo "  python3 examples/grpc_client.py --host \$$(make vm-ip) --health-only"

vm-delete:
	gcloud compute instances delete $(VM_NAME) \
		--zone=$(ZONE) --project=$(PROJECT_ID) --quiet
	@echo "VM deleted. Static IP and GCR image preserved."

vm-start:
	gcloud compute instances start $(VM_NAME) \
		--zone=$(ZONE) --project=$(PROJECT_ID)
	@echo "VM starting. Run 'make vm-status' to watch."

vm-stop:
	gcloud compute instances stop $(VM_NAME) \
		--zone=$(ZONE) --project=$(PROJECT_ID)
	@echo "VM stopped. Compute + GPU billing halted. Static IP preserved."

vm-ssh:
	gcloud compute ssh $(VM_NAME) \
		--zone=$(ZONE) --project=$(PROJECT_ID)

vm-logs:
	gcloud compute ssh $(VM_NAME) --zone=$(ZONE) --project=$(PROJECT_ID) -- \
		"docker logs emotion-cloud -f --tail=100"

vm-status:
	@echo "\n── VM ──────────────────────────────────────────────────────────────"
	@gcloud compute instances describe $(VM_NAME) \
		--zone=$(ZONE) --project=$(PROJECT_ID) \
		--format="table(name,status,networkInterfaces[0].accessConfigs[0].natIP:label=EXTERNAL_IP)"
	@echo "\n── Container ───────────────────────────────────────────────────────"
	@gcloud compute ssh $(VM_NAME) --zone=$(ZONE) --project=$(PROJECT_ID) -- \
		"docker ps --filter name=emotion-cloud \
		 --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'" \
		2>/dev/null || echo "  (SSH unavailable — VM may still be starting)"

vm-ip:
	@gcloud compute instances describe $(VM_NAME) \
		--zone=$(ZONE) --project=$(PROJECT_ID) \
		--format="value(networkInterfaces[0].accessConfigs[0].natIP)"

# ── VM: deploy update ─────────────────────────────────────────────────────────

vm-deploy: push
	@echo "Deploying $(IMAGE):$(TAG) to $(VM_NAME)..."
	gcloud compute ssh $(VM_NAME) --zone=$(ZONE) --project=$(PROJECT_ID) -- \
		"gcloud auth configure-docker gcr.io --quiet && \
		 docker pull $(IMAGE):$(TAG) && \
		 docker stop emotion-cloud 2>/dev/null || true && \
		 docker rm   emotion-cloud 2>/dev/null || true && \
		 docker run -d --name emotion-cloud --restart=unless-stopped \
		   --gpus all -p 50051:50051 \
		   --env-file /etc/emotion-cloud.env \
		   $(IMAGE):$(TAG) && \
		 echo 'Container restarted successfully'"
	@echo ""
	@echo "Deployed $(IMAGE):$(TAG). Allow ~3 min for model load."
