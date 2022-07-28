APP_PORT := 5000
DOCKER_TAG := latest
DOCKER_IMAGE := genres

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

.PHONY: run_unit_tests
run_unit_tests:
	PYTHONPATH=. pytest tests/unit/

.PHONY: run_integration_tests
run_integration_tests:
	PYTHONPATH=. pytest tests/integration/

.PHONY: run_all_tests
run_all_tests:
	make run_unit_tests
	make run_integration_tests

.PHONY: generate_coverage_report
generate_coverage_report:
	PYTHONPATH=. pytest --cov=src --cov-report html  tests/

.PHONY: lint
lint:
	flake8 src/


.PHONY: build
build:
	docker build -f Dockerfile . -t $(DOCKER_IMAGE):$(DOCKER_TAG)
