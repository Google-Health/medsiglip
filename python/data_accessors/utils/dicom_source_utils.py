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
# ==============================================================================
"""Determines the SOPClassUIDs of a DICOM data for modality specfic processing."""

import dataclasses
import enum
from typing import Any, List, Mapping

from ez_wsi_dicomweb import credential_factory
from ez_wsi_dicomweb import dicom_web_interface
from ez_wsi_dicomweb.ml_toolkit import dicom_path
from ez_wsi_dicomweb.ml_toolkit import tags

from data_accessors import data_accessor_const
from data_accessors import data_accessor_errors
from data_accessors.utils import json_validation_utils

_InstanceJsonKeys = data_accessor_const.InstanceJsonKeys
_EZ_WSI_STATE = 'ez_wsi_state'

# DICOM VL Microscopy SOPClassUIDs
# https://dicom.nema.org/dicom/2013/output/chtml/part04/sect_i.4.html
VL_WHOLE_SLIDE_MICROSCOPY_IMAGE_SOP_CLASS_UID = '1.2.840.10008.5.1.4.1.1.77.1.6'
_VL_MICROSCOPIC_IMAGE_SOP_CLASS_UID = '1.2.840.10008.5.1.4.1.1.77.1.2'
_VL_SLIDE_COORDINATES_MICROSCOPIC_IMAGE_SOP_CLASS_UID = (
    '1.2.840.10008.5.1.4.1.1.77.1.3'
)

DICOM_MICROSCOPIC_IMAGE_IODS = frozenset([
    _VL_MICROSCOPIC_IMAGE_SOP_CLASS_UID,
    _VL_SLIDE_COORDINATES_MICROSCOPIC_IMAGE_SOP_CLASS_UID,
])

DICOM_MICROSCOPY_IODS = frozenset([
    VL_WHOLE_SLIDE_MICROSCOPY_IMAGE_SOP_CLASS_UID,
    _VL_MICROSCOPIC_IMAGE_SOP_CLASS_UID,
    _VL_SLIDE_COORDINATES_MICROSCOPIC_IMAGE_SOP_CLASS_UID,
])


class DicomDataSourceEnum(enum.Enum):
  """Enum for DICOM data source type."""

  SLIDE_MICROSCOPY_IMAGE = 'slide_microscope_image'
  GENERIC_DICOM = 'generic_dicom'


@dataclasses.dataclass(frozen=True)
class _DicomSourceType:
  dicom_source_type: DicomDataSourceEnum
  dicom_instances_metadata: List[dicom_web_interface.DicomObject]


def _get_instance_dicom_path(
    instance: Mapping[str, Any],
) -> dicom_path.Path:
  """Returns normalized instance DICOM path."""
  dicomweb_uri = instance[_InstanceJsonKeys.DICOM_WEB_URI]
  if not isinstance(dicomweb_uri, str) or not dicomweb_uri:
    raise data_accessor_errors.InvalidRequestFieldError('Invalid DICOMweb uri.')
  json_validation_utils.validate_not_empty_str(dicomweb_uri)
  try:
    instance_path = dicom_path.FromString(dicomweb_uri)
  except ValueError as exp:
    raise data_accessor_errors.InvalidRequestFieldError(
        'Invalid DICOMweb uri.'
    ) from exp
  if instance_path.type != dicom_path.Type.INSTANCE:
    raise data_accessor_errors.InvalidRequestFieldError(
        'DICOM path is not a instance instance path.'
    )
  return instance_path


def _get_vl_whole_slide_microscopy_image_instances(
    selected_instance: dicom_web_interface.DicomObject,
    instances: List[dicom_web_interface.DicomObject],
) -> List[dicom_web_interface.DicomObject]:
  """Returns DICOM instances for VL whole slide microscopy image pyramid layer."""
  concatination_uid = selected_instance.get_value(tags.CONCATENATION_UID)
  if concatination_uid is None:
    return [selected_instance]
  found_instances = []
  for i in instances:
    if i.sop_instance_uid != selected_instance.sop_instance_uid:
      continue
    found_concatination_uid = i.get_value(tags.CONCATENATION_UID)
    if (
        found_concatination_uid is not None
        and found_concatination_uid == concatination_uid
    ):
      found_instances.append(i)
  return found_instances


def get_dicom_source_type(
    auth: credential_factory.AbstractCredentialFactory,
    instance: Mapping[str, Any],
) -> _DicomSourceType:
  """Returns dicom source type."""
  dcm_path = _get_instance_dicom_path(instance)
  extensions = instance.get(_InstanceJsonKeys.EXTENSIONS, {})
  ez_wsi_state = json_validation_utils.validate_str_key_dict(
      extensions.get(_EZ_WSI_STATE, {})
  )
  if ez_wsi_state:
    return _DicomSourceType(DicomDataSourceEnum.SLIDE_MICROSCOPY_IMAGE, [])
  dwi = dicom_web_interface.DicomWebInterface(auth)
  series_path = dcm_path.GetSeriesPath()
  instances = dwi.get_instances(series_path)
  if not instances:
    raise data_accessor_errors.InvalidRequestFieldError(
        f'No instances found for DICOM path: {series_path}.'
    )
  i_uid = dcm_path.instance_uid
  selected_instance = [i for i in instances if i.sop_instance_uid == i_uid]
  if len(selected_instance) != 1:
    raise data_accessor_errors.InvalidRequestFieldError(
        f'No instance found for DICOM path: {series_path}.'
    )
  selected_instance = selected_instance[0]
  if selected_instance.sop_class_uid in DICOM_MICROSCOPIC_IMAGE_IODS:
    return _DicomSourceType(
        DicomDataSourceEnum.SLIDE_MICROSCOPY_IMAGE,
        [selected_instance],
    )
  elif (
      selected_instance.sop_class_uid
      == VL_WHOLE_SLIDE_MICROSCOPY_IMAGE_SOP_CLASS_UID
  ):
    return _DicomSourceType(
        DicomDataSourceEnum.SLIDE_MICROSCOPY_IMAGE,
        _get_vl_whole_slide_microscopy_image_instances(
            selected_instance, instances
        ),
    )
  return _DicomSourceType(
      DicomDataSourceEnum.GENERIC_DICOM, [selected_instance]
  )
