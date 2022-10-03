#!/bin/bash

set -ue

cd {{ playbook_dest }}

export PROMETHEUS_PORT={{ prometheus_port }}
export GRAFANA_PORT={{ grafana_port }}
export NODE_EXPORTER_PORT={{ node_exporter_port }}
export APP_PORT={{ service_port }}


docker-compose down