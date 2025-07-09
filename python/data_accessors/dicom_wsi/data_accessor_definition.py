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

"""Request dataclasses for DICOM WSI data accessor."""

import dataclasses
import json
from typing import Any, List, Mapping

from ez_wsi_dicomweb import credential_factory as credential_factory_module
from ez_wsi_dicomweb import dicom_web_interface
from ez_wsi_dicomweb.ml_toolkit import dicom_path

from data_accessors import data_accessor_const
from data_accessors import data_accessor_errors
from data_accessors.dicom_wsi import configuration
from data_accessors.utils import json_validation_utils
from data_accessors.utils import patch_coordinate as patch_coordinate_module

_InstanceJsonKeys = data_accessor_const.InstanceJsonKeys


class _InstanceUIDMetadataError(Exception):
  pass


@dataclasses.dataclass(frozen=True)
class DicomWSIImage:
  """An instance in a DICOM Embedding Request as described in the schema file."""

  credential_factory: credential_factory_module.AbstractCredentialFactory
  series_path: str
  extensions: Mapping[str, Any]
  instance_uid: str
  patch_coordinates: List[patch_coordinate_module.PatchCoordinate]
  dicom_instances_metadata: List[dicom_web_interface.DicomObject]


def _generate_instance_metadata_error_string(
    metadata: Mapping[str, Any], *keys: str
) -> str:
  """returns instance metadata as a error string."""
  result = {}
  for key in keys:
    if key not in metadata:
      continue
    if key == _InstanceJsonKeys.EXTENSIONS:
      value = metadata[key]
      if isinstance(value, Mapping):
        value = dict(value)
        # Strip ez_wsi_state from output.
        # Not contributing to validation errors here and may be very large.
        if _InstanceJsonKeys.EZ_WSI_STATE in value:
          del value[_InstanceJsonKeys.EZ_WSI_STATE]
          result[key] = value
          continue
    elif key == _InstanceJsonKeys.BEARER_TOKEN:
      value = metadata[key]
      # If bearer token is present, and defined strip
      if isinstance(value, str) and value:
        result[key] = 'PRESENT'
        continue
    # otherwise just associate key and value.
    result[key] = metadata[key]
  return json.dumps(result, sort_keys=True)


def json_to_dicom_wsi_image(
    credential_factory: credential_factory_module.AbstractCredentialFactory,
    instance: Mapping[str, Any],
    settings: configuration.ConfigurationSettings,
    dicom_instances_metadata: List[dicom_web_interface.DicomObject],
) -> DicomWSIImage:
  """Converts json to DicomWSIImage."""
  try:
    patch_coordinates = patch_coordinate_module.parse_patch_coordinates(
        instance.get(_InstanceJsonKeys.PATCH_COORDINATES, []),
        settings.endpoint_input_width,
        settings.endpoint_input_height,
        settings.require_patch_dim_match_default_dim,
    )
  except patch_coordinate_module.InvalidCoordinateError as exp:
    instance_error_msg = _generate_instance_metadata_error_string(
        instance,
        _InstanceJsonKeys.PATCH_COORDINATES,
    )
    raise data_accessor_errors.InvalidRequestFieldError(
        f'Invalid patch coordinate; {exp}; {instance_error_msg}'
    ) from exp
  try:
    dcm_path = json_validation_utils.validate_not_empty_str(
        instance.get(_InstanceJsonKeys.DICOM_WEB_URI)
    )
  except json_validation_utils.ValidationError as exp:
    raise data_accessor_errors.InvalidRequestFieldError(
        'Invalid DICOM path.'
    ) from exp
  dcm_path = dicom_path.FromString(dcm_path)
  try:
    return DicomWSIImage(
        credential_factory=credential_factory,
        series_path=dcm_path.GetSeriesPath().complete_url,
        extensions=json_validation_utils.validate_str_key_dict(
            instance.get(
                _InstanceJsonKeys.EXTENSIONS,
                {},
            )
        ),
        instance_uid=dcm_path.instance_uid,
        patch_coordinates=patch_coordinates,
        dicom_instances_metadata=dicom_instances_metadata,
    )
  except _InstanceUIDMetadataError as exp:
    error_msg = _generate_instance_metadata_error_string(
        instance,
        _InstanceJsonKeys.DICOM_WEB_URI,
        _InstanceJsonKeys.BEARER_TOKEN,
        _InstanceJsonKeys.EXTENSIONS,
    )
    raise data_accessor_errors.InvalidRequestFieldError(
        f'Invalid DICOM SOP Instance UID metadata; {error_msg}'
    ) from exp
  except json_validation_utils.ValidationError as exp:
    error_msg = _generate_instance_metadata_error_string(
        instance,
        _InstanceJsonKeys.DICOM_WEB_URI,
        _InstanceJsonKeys.BEARER_TOKEN,
        _InstanceJsonKeys.EXTENSIONS,
    )
    raise data_accessor_errors.InvalidRequestFieldError(
        f'DICOM instance JSON formatting is invalid; {error_msg}'
    ) from exp
