TORCHSERVE_IMAGE := my-torchserve
MAR_FILES_DIR := $(shell pwd)/model_mars
APP_PORT := 5003
FILEBROWSER_PORT := 5002

MACHINE_IP := 192.168.0.105


.PHONY: download_weights
download_weights:
	wget -O weights/genre_classifier.pt https://www.dropbox.com/s/xlax7mfwmzh4rjl/genre_classifier.pt?dl=0

.PHONY: generate_mar
generate_mar:
	PYTHONPATH=. python ts_serialize/model2torchserve.py

.PHONY: upload_model
upload_model:
	PYTHONPATH=. python upload_model.py

.PHONY: build_ts_cuda
build_ts_cuda:
	cd ts/build && ./build_image.sh -g -cv ${CUDA_VERSION} -t ${TORCHSERVE_IMAGE}:${CUDA_VERSION}

.PHONY: build_ts_cpu
build_ts_cpu:
	cd ts/build && ./build_image.sh -t ${TORCHSERVE_IMAGE}:cpu

.PHONY: run_ts_cuda
run_ts_cuda:
	docker run --rm -it -p 8080:8080 -p 8081:8081 -p 8082:8082 -p 7070:7070 -p 7071:7071  my-torchserve:${CUDA_VERSION}

.PHONY: run_ts_cpu
run_ts_cpu:
	docker run --rm -it -p 8080:8080 -p 8081:8081 -p 8082:8082 -p 7070:7070 -p 7071:7071  my-torchserve:cpu

.PHONY: run_nginx_filebrowser
run_nginx_filebrowser:
	docker run -p ${FILEBROWSER_PORT}:80 -v ${MAR_FILES_DIR}:/opt/www/files/ mohamnag/nginx-file-browser

.PHONY: upload_mar_file
upload_mar_file:
	curl -X POST  "http://localhost:8081/models?url=http://${MACHINE_IP}:${FILEBROWSER_PORT}/files/${MAR_FILE}"

.PHONY: run_app
run_app:
	python3 -m uvicorn app:app --host='0.0.0.0' --port=$(APP_PORT)

.PHONY: run_stress
run_stress:
	PYTHONPATH=. python3 stress.py