#!/bin/bash
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script launches the serving framework, run as the entrypoint.

# Exit if expanding an undefined variable.
set -u

export MODEL_REST_PORT=8600

if [[ -v "AIP_STORAGE_URI" && -n "$AIP_STORAGE_URI" ]]; then
  export MODEL_FILES="/model_files"
  mkdir "$MODEL_FILES"
  gcloud storage cp "$AIP_STORAGE_URI/*" "$MODEL_FILES" --recursive
fi

echo "Serving framework start, launching model server"

(/opt/tritonserver/bin/tritonserver \
    --model-repository="/serving/model_repository" \
    --allow-grpc=true \
    --grpc-address=127.0.0.1 \
    --grpc-port=8500 \
    --http-address=127.0.0.1 \
    --http-port="${MODEL_REST_PORT}" \
    --allow-vertex-ai=false \
    --strict-readiness=true || exit) &

echo "Launching front end"

(/server-env/bin/python3.12 -m serving.server_gunicorn --alsologtostderr \
    --verbosity=1 || exit)&

# Wait for any process to exit
wait -n

# Exit with status of process that exited first
exit $?
