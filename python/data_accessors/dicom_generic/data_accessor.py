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
"""Data accessor for generic DICOM images stored in a DICOM store."""
import contextlib
import os
import tempfile
from typing import Iterator

from ez_wsi_dicomweb import dicom_frame_decoder
from ez_wsi_dicomweb import dicom_web_interface
from ez_wsi_dicomweb import ez_wsi_errors
from ez_wsi_dicomweb.ml_toolkit import dicom_path
import numpy as np
import pydicom

from data_accessors import abstract_data_accessor
from data_accessors import data_accessor_errors
from data_accessors.dicom_generic import data_accessor_definition
from data_accessors.local_file_handlers import generic_dicom_handler
from data_accessors.utils import icc_profile_utils
from data_accessors.utils import image_dimension_utils
from data_accessors.utils import patch_coordinate as patch_coordinate_module


# Transfer Syntax UID for uncompressed little endian.
_UNCOMPRESSED_LITTLE_ENDIAN_TRANSFER_SYNTAX = '1.2.840.10008.1.2.1'


def _can_decode_transfer_syntax(
    instance: data_accessor_definition.DicomGenericImage,
):
  transfer_syntax_uid = instance.dicom_instances_metadata[0].transfer_syntax_uid
  if transfer_syntax_uid == _UNCOMPRESSED_LITTLE_ENDIAN_TRANSFER_SYNTAX:
    return True
  return dicom_frame_decoder.can_decompress_dicom_transfer_syntax(
      transfer_syntax_uid
  )


def _download_dicom_instance(
    stack: contextlib.ExitStack,
    instance: data_accessor_definition.DicomGenericImage,
) -> str:
  """Downloads DICOM instance to a local file."""
  dwi = dicom_web_interface.DicomWebInterface(instance.credential_factory)
  instance_path = dicom_path.FromString(instance.instance_path)
  temp_dir = stack.enter_context(tempfile.TemporaryDirectory())
  temp_file = os.path.join(temp_dir, 'temp.dcm')
  with open(temp_file, 'wb') as output_file:
    if not instance.dicom_instances_metadata:
      raise ValueError('No DICOM instances metadata found.')
    locally_decode_dicom = _can_decode_transfer_syntax(instance)
    try:
      if locally_decode_dicom:
        dwi.download_instance_untranscoded(instance_path, output_file)
      else:
        # transcode to uncompressed little endian.
        dwi.download_instance(
            instance_path,
            _UNCOMPRESSED_LITTLE_ENDIAN_TRANSFER_SYNTAX,
            output_file,
        )
    except ez_wsi_errors.HttpError as exp:
      raise data_accessor_errors.HttpError(str(exp)) from exp
  return temp_file


def _get_dicom_image(
    instance: data_accessor_definition.DicomGenericImage,
    local_file_path: str,
) -> Iterator[np.ndarray]:
  """Returns image patch bytes from DICOM series."""
  with contextlib.ExitStack() as stack:
    if not local_file_path:
      local_file_path = _download_dicom_instance(stack, instance)
    try:
      with pydicom.dcmread(local_file_path) as dcm:
        target_icc_profile = icc_profile_utils.get_target_icc_profile(
            instance.extensions
        )
        patch_required_to_be_fully_in_source_image = (
            patch_coordinate_module.patch_required_to_be_fully_in_source_image(
                instance.extensions
            )
        )
        resize_image_dimensions = (
            image_dimension_utils.get_resize_image_dimensions(
                instance.extensions
            )
        )
        yield from generic_dicom_handler.decode_dicom_image(
            dcm,
            target_icc_profile,
            instance.patch_coordinates,
            resize_image_dimensions,
            patch_required_to_be_fully_in_source_image,
        )
    except pydicom.errors.InvalidDicomError as exp:
      raise data_accessor_errors.DicomError(
          'DICOM cannot decode pixel data.'
      ) from exp


class DicomGenericData(
    abstract_data_accessor.AbstractDataAccessor[
        data_accessor_definition.DicomGenericImage, np.ndarray
    ]
):
  """Data accessor for generic DICOM images stored in a DICOM store."""

  def __init__(
      self,
      instance_class: data_accessor_definition.DicomGenericImage,
  ):
    super().__init__(instance_class)
    self._local_file_path = ''

  @contextlib.contextmanager
  def _reset_local_file_path(self, *args, **kwds):
    del args, kwds
    try:
      yield
    finally:
      self._local_file_path = ''

  def load_data(self, stack: contextlib.ExitStack) -> None:
    """Method pre-loads data prior to data_iterator.

    Required that context manger must exist for life time of data accesor
    iterator after data is loaded.

    Args:
     stack: contextlib.ExitStack to manage resources.

    Returns:
      None
    """
    if self._local_file_path:
      return
    self._local_file_path = _download_dicom_instance(stack, self.instance)
    stack.enter_context(self._reset_local_file_path())

  def data_iterator(self) -> Iterator[np.ndarray]:
    return _get_dicom_image(self.instance, self._local_file_path)

  def is_accessor_data_embedded_in_request(self) -> bool:
    """Returns true if data is inline with request."""
    return False

  def __len__(self) -> int:
    """Returns number of data sets returned by iterator."""
    if self.instance.patch_coordinates:
      return len(self.instance.patch_coordinates)
    return 1
