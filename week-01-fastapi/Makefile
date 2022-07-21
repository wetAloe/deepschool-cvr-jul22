APP_PORT := 5000
DOCKER_IMAGE := genres
DOCKER_TAG := latest
DEPLOY_HOST := demo_host
KEY_FILE := ~/.ssh/id_rsa

.PHONY: run_app
run_app:
	python3 -m uvicorn app:app --host='0.0.0.0' --port=$(APP_PORT)

.PHONY: install
install:
	pip install -r requirements.txt

.PHONY: download_weights
download_weights:
	wget -O weights/genre_classifier.pt https://www.dropbox.com/s/xlax7mfwmzh4rjl/genre_classifier.pt?dl=0
	wget -O weights/rotate_classifier.pt https://www.dropbox.com/s/976wwhi1531hmcg/rotate_classifier.pt?dl=0

.PHONY: build
build:
	docker build -f Dockerfile . --force-rm=true -t $(DOCKER_IMAGE):$(DOCKER_TAG)


.PHONY: deploy
deploy:
	ansible-playbook -i deploy/ansible/inventory.ini deploy/ansible/deploy.yml \
		-e host=$(DEPLOY_HOST) \
		-e docker_image=$(DOCKER_IMAGE) \
		-e docker_tag=$(DOCKER_TAG) \
		-e docker_registry_user=$(CI_REGISTRY_USER) \
		-e docker_registry_password=$(CI_REGISTRY_PASSWORD) \
		-e docker_registry=$(CI_REGISTRY)

.PHONY: destroy
destroy:
	ansible-playbook -i deploy/ansible/inventory.ini deploy/ansible/destroy.yml \
		-e host=$(DEPLOY_HOST)
