# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This is used to build a Docker image that includes the necessary dependencies
# for running MedSigLIP as a microservice.

from nvcr.io/nvidia/tritonserver:25.03-py3

WORKDIR /
COPY ./python/data_accessors /data_accessors
COPY ./python/data_processing /data_processing
COPY ./python/serving /serving
COPY ./python/pre_processor_configs /pre_processor_configs
COPY ./LICENSE /LICENSE
RUN chmod a+x /serving/entrypoint.sh

# Install python3.12 and git
RUN apt-get update && \
    apt-get -y upgrade && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get install -y git python3.12 python3.12-venv

# Install model dependencies for the triton backend.
RUN python3 -m pip install --require-hashes -r /serving/model_requirements.txt

# Get pypi requirements
RUN python3.12 -m venv /server-env && \
    /server-env/bin/python3.12 -m pip install --require-hashes \
        -r /serving/serving_framework/pip-install.txt && \
    /server-env/bin/python3.12 -m pip install --require-hashes \
        -r /serving/requirements.txt

# Install gcloud CLI
RUN apt-get install -y apt-transport-https curl gnupg
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
    | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    apt-get update -y && apt-get install google-cloud-cli -y

# begin code mirror collection
WORKDIR /code-mirror

# Clone python-certifi to meet MPL 2.0 License terms for source code mirroring
RUN git clone https://github.com/certifi/python-certifi.git

# Clone tdqm to meet MPL 2.0 License terms for source code mirroring
RUN git clone https://github.com/tqdm/tqdm.git

# Clone GPL and LGPL source code for mirroring.
RUN git clone https://git.launchpad.net/launchpadlib
RUN git clone https://git.launchpad.net/lazr.restfulclient
RUN git clone https://git.launchpad.net/lazr.uri
RUN git clone https://git.launchpad.net/wadllib
RUN git clone https://github.com/openslide/openslide-bin.git
RUN git clone https://github.com/openslide/openslide-python.git
RUN git clone https://gitlab.gnome.org/GNOME/pygobject.git
RUN git clone https://git.launchpad.net/python-apt

WORKDIR /
# end code mirror collection

ENTRYPOINT ["/serving/entrypoint.sh"]
