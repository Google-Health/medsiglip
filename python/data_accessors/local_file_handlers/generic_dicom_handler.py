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
"""local handler for handling generic DICOM files."""
import enum
import io
from typing import Any, Iterator, Mapping, Optional, Sequence, Union

from ez_wsi_dicomweb import dicom_frame_decoder
import numpy as np
from PIL import ImageCms
import pydicom
import pydicom.errors

from data_accessors import data_accessor_errors
from data_accessors.local_file_handlers import abstract_handler
from data_accessors.utils import dicom_source_utils
from data_accessors.utils import icc_profile_utils
from data_accessors.utils import image_dimension_utils
from data_accessors.utils import patch_coordinate as patch_coordinate_module
from data_processing import image_utils


_PYDICOM_MAJOR_VERSION = int((pydicom.__version__).split('.')[0])

# DICOM Tag Keywords
_PIXEL_DATA = 'PixelData'
_WINDOW_CENTER = 'WindowCenter'
_WINDOW_WIDTH = 'WindowWidth'

# DICOM Transfer Syntax UIDs
_IMPLICIT_VR_ENDIAN_TRANSFER_SYNTAX = '1.2.840.10008.1.2'
_EXPLICIT_VR_ENDIAN_TRANSFER_SYNTAX = '1.2.840.10008.1.2.1'
_DEFLATED_EXPLICIT_VR_LITTLE_ENDIAN_TRANSFER_SYNTAX = '1.2.840.10008.1.2.1.99'

_VALID_UNENCAPSULATED_DICOM_TRANSFER_SYNTAXES = frozenset([
    _IMPLICIT_VR_ENDIAN_TRANSFER_SYNTAX,
    _EXPLICIT_VR_ENDIAN_TRANSFER_SYNTAX,
    _DEFLATED_EXPLICIT_VR_LITTLE_ENDIAN_TRANSFER_SYNTAX,
])

# PhotometricInterpretation Coded Values
MONOCHROME1 = 'MONOCHROME1'
_MONOCHROME2 = 'MONOCHROME2'
_RGB = 'RGB'
_SINGLE_CHANNEL_PHOTOMETRIC_INTERPRETATION = frozenset(
    [MONOCHROME1, _MONOCHROME2]
)
_SUPPORTED_UNENCAPSULATED_PHOTOMETRIC_INTERPRETATIONS = frozenset(
    [MONOCHROME1, _MONOCHROME2, _RGB]
)

_SUPPORTED_SAMPLES_PER_PIXEL = frozenset([1, 3])


class _MODALITY(enum.Enum):
  """Modality Coded Values."""

  CR = 'CR'  # Computed Radiography
  DX = 'DX'  # Digital X-Ray
  GM = 'GM'  # General Microscopy
  SM = 'SM'  # Slide Microscopy
  XC = 'XC'  # External Camera


_CXR_MODALITIES = {_MODALITY.CR.value, _MODALITY.DX.value}
_MICROSCOPY_MODALITIES = {_MODALITY.SM.value, _MODALITY.GM.value}


def _validate_modality_supported(dcm: pydicom.FileDataset) -> None:
  """Validates DICOM modality is supported."""
  try:
    modality = dcm.Modality
  except (AttributeError, ValueError, TypeError) as _:
    raise data_accessor_errors.DicomError(
        'DICOM missing modality tag metadata.'
    )
  if modality in _CXR_MODALITIES or modality in _MICROSCOPY_MODALITIES:
    return
  if modality == _MODALITY.XC.value:
    return
  raise data_accessor_errors.DicomError(
      f'DICOM encodes a unsupported Modality; Modality: {modality}.'
  )


def validate_transfer_syntax(dcm: pydicom.FileDataset) -> None:
  transfer_syntax_uid = dcm.file_meta.TransferSyntaxUID
  if transfer_syntax_uid in _VALID_UNENCAPSULATED_DICOM_TRANSFER_SYNTAXES:
    return
  if dicom_frame_decoder.can_decompress_dicom_transfer_syntax(
      transfer_syntax_uid
  ):
    return
  raise data_accessor_errors.DicomError(
      'DICOM instance encoded using unsupported transfer syntax.'
      f' {transfer_syntax_uid}.'
  )


def _transform_image_to_target_icc_profile(
    decoded_image_bytes: np.ndarray,
    compressed_image_bytes: bytes,
    dcm: pydicom.FileDataset,
    target_icc_profile: Optional[ImageCms.core.CmsProfile],
) -> np.ndarray:
  """Transforms image to target ICC profile."""
  if target_icc_profile is None:
    return decoded_image_bytes
  if dcm.SamplesPerPixel != 3:
    return decoded_image_bytes
  if dcm.BitsAllocated != 8:
    return decoded_image_bytes
  if decoded_image_bytes.ndim != 3 or decoded_image_bytes.shape[2] != 3:
    return decoded_image_bytes
  icc_profile_bytes = icc_profile_utils.get_dicom_icc_profile_bytes(dcm)
  if compressed_image_bytes and not icc_profile_bytes:
    icc_profile_bytes = (
        icc_profile_utils.get_icc_profile_bytes_from_compressed_image(
            compressed_image_bytes
        )
    )
  if not icc_profile_bytes:
    return decoded_image_bytes
  icc_profile_transformation = (
      icc_profile_utils.create_icc_profile_transformation(
          icc_profile_bytes, target_icc_profile
      )
  )
  return icc_profile_utils.transform_image_bytes_to_target_icc_profile(
      decoded_image_bytes, icc_profile_transformation
  )


def _get_encapsulated_dicom_frame_bytes(ds: pydicom.FileDataset) -> bytes:
  """Returns DICOM bytes from encapsulated PixelData."""
  if _PIXEL_DATA not in ds or not ds.PixelData:
    return b''
  try:
    number_of_frames = int(ds.NumberOfFrames)
  except (TypeError, ValueError, AttributeError) as _:
    # DICOM IOD that do not define multi-frame do not contain NumberOfFrames
    # tag. For these IOD, we assume that the image has only one frame.
    number_of_frames = 1
  if number_of_frames < 1:
    return b''
  if _PYDICOM_MAJOR_VERSION <= 2:
    # pytype: disable=module-attr
    frame_bytes_generator = pydicom.encaps.generate_pixel_data_frame(
        ds.PixelData, number_of_frames
    )
    # pytype: enable=module-attr
  else:
    # pytype: disable=module-attr
    frame_bytes_generator = pydicom.encaps.generate_frames(
        ds.PixelData, number_of_frames=number_of_frames
    )
    # pytype: enable=module-attr
  for frame_bytes in frame_bytes_generator:
    return frame_bytes
  return b''


def _rescale_cxr_dynamic_range(image_bytes: np.ndarray) -> np.ndarray:
  """Rescales dynamic range of image bytes to make across image range."""
  try:
    # For uint8 images, rescaling is not needed
    if image_bytes.dtype == np.uint8:
      return image_bytes
    if np.dtype(image_bytes.dtype).kind != 'u':
      image_bytes = image_utils.shift_to_unsigned(image_bytes)
    # Rescaling dynamic range enables 12 bit imaging to be scaled to uint16.
    # Also will scale signed imaging across full range.
    # Side effect is that will make imaging relative to self.
    return image_utils.rescale_dynamic_range(image_bytes)
  except ValueError as exp:
    raise data_accessor_errors.DicomError(
        'DICOM PixelData contains has incompatible encoding.'
    ) from exp


def _norm_cxr_imaging(arr: np.ndarray, ds: pydicom.FileDataset) -> np.ndarray:
  """Applies data handling from pydicom."""
  pixel_array = pydicom.pixels.processing.apply_modality_lut(arr, ds)
  if _WINDOW_WIDTH in ds and _WINDOW_CENTER in ds:
    # windowing will normalize imaging to uint16.
    # with dynamic range scaled across the windowed range.
    pixel_array = image_utils.window(
        pixel_array, ds.WindowCenter, ds.WindowWidth, np.uint16
    )
  if pixel_array.dtype == np.float64:
    # if pixel array is altered by the LUT will be transformed to float64.
    # https://pydicom.github.io/pydicom/dev/reference/generated/pydicom.pixels.apply_modality_lut.html
    # cast back to the original integer dtype for windowing.
    pixel_array = pixel_array.astype(arr.dtype)
  # Scale imaging
  pixel_array = _rescale_cxr_dynamic_range(pixel_array)
  if ds.PhotometricInterpretation == MONOCHROME1:
    return np.iinfo(pixel_array.dtype).max - pixel_array
  return pixel_array


def validate_samples_per_pixel(dcm: pydicom.FileDataset) -> None:
  """Validates samples per pixel metadata."""
  try:
    if dcm.SamplesPerPixel not in _SUPPORTED_SAMPLES_PER_PIXEL:
      raise data_accessor_errors.DicomError(
          'DICOM instance contains unsupported number of samples per pixel;'
          f' expected: {_SUPPORTED_SAMPLES_PER_PIXEL} found:'
          f' {dcm.SamplesPerPixel}.'
      )
  except (ValueError, AttributeError) as _:
    raise data_accessor_errors.DicomError(
        'DICOM instance missing SamplesPerPixel metadata.'
    )


def validate_samples_per_pixel_and_photometric_interpretation_match(
    dcm: pydicom.FileDataset,
) -> None:
  """Validates samples per pixel and photometric interpretation."""
  if (
      dcm.SamplesPerPixel == 1
      and dcm.PhotometricInterpretation
      not in _SINGLE_CHANNEL_PHOTOMETRIC_INTERPRETATION
  ):
    raise data_accessor_errors.DicomError(
        'DICOM instance has 1 sample per pixel but contains multichannel'
        ' PhotometricInterpretation.'
    )
  if (
      dcm.SamplesPerPixel == 3
      and dcm.PhotometricInterpretation
      in _SINGLE_CHANNEL_PHOTOMETRIC_INTERPRETATION
  ):
    raise data_accessor_errors.DicomError(
        'DICOM instance has 3 sample per pixel but contains single channel'
        ' PhotometricInterpretation.'
    )


def _validate_number_of_frames(dcm: pydicom.FileDataset) -> None:
  try:
    if int(dcm.NumberOfFrames) != 1:
      raise data_accessor_errors.DicomError(
          'DICOM contains more than one frame; number of frames:'
          f' {dcm.NumberOfFrames}.'
      )
  except (TypeError, ValueError, AttributeError) as _:
    return


def validate_unencapsulated_photometric_interpretation(
    dcm: pydicom.FileDataset,
) -> None:
  """Validates pixel unencapsulated pixel encoding pixel encoding."""
  try:
    photometric_interpretation = dcm.PhotometricInterpretation
  except (ValueError, AttributeError) as exp:
    raise data_accessor_errors.DicomError(
        'PhotometricInterpretation is required for DICOM images.'
    ) from exp
  if (
      photometric_interpretation
      not in _SUPPORTED_UNENCAPSULATED_PHOTOMETRIC_INTERPRETATIONS
  ):
    raise data_accessor_errors.DicomError(
        'DICOM image encoded using unsupported PhotometricInterpretation:'
        f' {photometric_interpretation}.'
    )


def decode_dicom_image(
    dcm: pydicom.FileDataset,
    target_icc_profile: Optional[ImageCms.core.CmsProfile],
    patch_coordinates: Sequence[patch_coordinate_module.PatchCoordinate],
    resize_image_dimensions: Optional[image_dimension_utils.ImageDimensions],
    patch_required_to_be_fully_in_source_image: bool,
) -> Iterator[np.ndarray]:
  """Decode DICOM image and return decoded image bytes."""
  _validate_modality_supported(dcm)
  validate_transfer_syntax(dcm)
  validate_samples_per_pixel(dcm)
  _validate_number_of_frames(dcm)
  try:
    encapsulated_dicom = (
        dicom_frame_decoder.can_decompress_dicom_transfer_syntax(
            dcm.file_meta.TransferSyntaxUID
        )
    )
  except (AttributeError, ValueError) as exp:
    raise data_accessor_errors.DicomError(
        'DICOM missing TransferSyntaxUID.'
    ) from exp
  if encapsulated_dicom:
    compressed_image_bytes = _get_encapsulated_dicom_frame_bytes(dcm)
    if not compressed_image_bytes:
      raise data_accessor_errors.DicomError('DICOM missing pixel data.')
    try:
      transfer_syntax_uid = dcm.file_meta.TransferSyntaxUID
    except (AttributeError, ValueError) as exp:
      raise data_accessor_errors.DicomError(
          'DICOM missing TransferSyntaxUID.'
      ) from exp
    decoded_image_bytes = (
        dicom_frame_decoder.decode_dicom_compressed_frame_bytes(
            compressed_image_bytes, transfer_syntax_uid
        )
    )
    if decoded_image_bytes is None:
      raise data_accessor_errors.DicomError('DICOM cannot decode pixel data.')
    if dcm.SamplesPerPixel == 1 and decoded_image_bytes.shape[2] == 3:
      decoded_image_bytes = decoded_image_bytes[..., 0]
  else:
    compressed_image_bytes = b''
    validate_unencapsulated_photometric_interpretation(dcm)
    try:
      decoded_image_bytes = dcm.pixel_array
    except (AttributeError, ValueError) as exp:
      raise data_accessor_errors.DicomError(
          f'Cannot decode pixel data: {exp}.'
      ) from exp
  validate_samples_per_pixel_and_photometric_interpretation_match(dcm)
  if dcm.SamplesPerPixel == 1 and decoded_image_bytes.ndim == 2:
    decoded_image_bytes = np.expand_dims(decoded_image_bytes, 2)
  decoded_image_bytes = _transform_image_to_target_icc_profile(
      decoded_image_bytes, compressed_image_bytes, dcm, target_icc_profile
  )
  if (
      dcm.Modality in _CXR_MODALITIES
      and decoded_image_bytes.ndim == 3
      and decoded_image_bytes.shape[2] == 1
  ):
    decoded_image_bytes = _norm_cxr_imaging(decoded_image_bytes, dcm)
  if resize_image_dimensions is not None:
    decoded_image_bytes = image_dimension_utils.resize_image_dimensions(
        decoded_image_bytes, resize_image_dimensions
    )
  if not patch_coordinates:
    yield decoded_image_bytes
  else:
    image_shape = image_dimension_utils.ImageDimensions(
        width=decoded_image_bytes.shape[1],
        height=decoded_image_bytes.shape[0],
    )
    for pc in patch_coordinates:
      if patch_required_to_be_fully_in_source_image:
        pc.validate_patch_in_dim(image_shape)
      yield patch_coordinate_module.get_patch_from_memory(
          pc, decoded_image_bytes
      )


class GenericDicomHandler(abstract_handler.AbstractHandler):
  """Reads a generic DICOM image from file system. Returns None on failure."""

  def process_file(
      self,
      instance_patch_coordinates: Sequence[
          patch_coordinate_module.PatchCoordinate
      ],
      base_request: Mapping[str, Any],
      file_path: Union[str, io.BytesIO],
  ) -> Iterator[np.ndarray]:
    instance_extensions = abstract_handler.get_base_request_extensions(
        base_request
    )
    try:
      with pydicom.dcmread(file_path, specific_tags=['SOPClassUID']) as dcm:
        if (
            dcm.SOPClassUID
            == dicom_source_utils.VL_WHOLE_SLIDE_MICROSCOPY_IMAGE_SOP_CLASS_UID
        ):
          return
      if isinstance(file_path, io.BytesIO):
        file_path.seek(0)
      with pydicom.dcmread(file_path) as dcm:
        target_icc_profile = icc_profile_utils.get_target_icc_profile(
            instance_extensions
        )
        patch_required_to_be_fully_in_source_image = (
            patch_coordinate_module.patch_required_to_be_fully_in_source_image(
                instance_extensions
            )
        )
        resize_image_dimensions = (
            image_dimension_utils.get_resize_image_dimensions(
                instance_extensions
            )
        )
        yield from decode_dicom_image(
            dcm,
            target_icc_profile,
            instance_patch_coordinates,
            resize_image_dimensions,
            patch_required_to_be_fully_in_source_image,
        )
    except pydicom.errors.InvalidDicomError:
      # The handler is purposefully eating the message here.
      # if a handler fails to process the image it returns an empty iterator.
      return
