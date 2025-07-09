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
"""Data types for the medsiglip predictor."""

from typing import Union

from data_accessors.dicom_generic import data_accessor_definition as dicom_generic_data_accessor_definition
from data_accessors.dicom_wsi import data_accessor_definition as dicom_wsi_data_accessor_definition
from data_accessors.gcs_generic import data_accessor_definition as gcs_generic_data_accessor_definition
from data_accessors.http_image import data_accessor_definition as http_image_data_accessor_definition
from data_accessors.inline_bytes import data_accessor_definition as inline_bytes_data_accessor_definition
from data_accessors.inline_text import data_accessor_definition as inline_text_data_accessor_definition


class InlineTextInstance(inline_text_data_accessor_definition.InlineText):

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    # text does not have patch coordinates but adding to enable text accessor
    # to be used identically with image accessors.
    self.patch_coordinates = []


EmbeddingInstance = Union[
    dicom_wsi_data_accessor_definition.DicomWSIImage,
    dicom_generic_data_accessor_definition.DicomGenericImage,
    gcs_generic_data_accessor_definition.GcsGenericBlob,
    inline_bytes_data_accessor_definition.InlineBytes,
    InlineTextInstance,
    http_image_data_accessor_definition.HttpImage,
]
