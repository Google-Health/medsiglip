# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the 'License");
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
"""Unit tests for generic dicom handler."""

import io
import os
import tempfile
from typing import Any, Mapping

from absl.testing import absltest
from absl.testing import parameterized
import cv2
from ez_wsi_dicomweb import dicom_slide
import numpy as np
import PIL.Image
import pydicom

from data_accessors import data_accessor_const
from data_accessors import data_accessor_errors
from data_accessors.local_file_handlers import generic_dicom_handler
from data_accessors.utils import patch_coordinate
from data_accessors.utils import test_utils


_generic_dicom_handler = generic_dicom_handler.GenericDicomHandler()
_InstanceJsonKeys = data_accessor_const.InstanceJsonKeys


def _mock_instance_extension_metadata(
    extensions: Mapping[str, Any],
) -> Mapping[str, Any]:
  return {_InstanceJsonKeys.EXTENSIONS: extensions}


_MOCK_INSTANCE_METADATA_REQUEST_ICCPROFILE_NORM = (
    _mock_instance_extension_metadata(
        {_InstanceJsonKeys.TRANSFORM_IMAGING_TO_ICC_PROFILE: 'ADOBERGB'}
    )
)


def _encapsulated_dicom_path() -> str:
  return test_utils.testdata_path('cxr', 'encapsulated_cxr.dcm')


class GenericDicomHandlerTest(parameterized.TestCase):

  def test_load_encapsulated_dicom_file_path(self):
    images = list(
        _generic_dicom_handler.process_file([], {}, _encapsulated_dicom_path())
    )
    self.assertLen(images, 1)
    self.assertEqual(images[0].shape, (1024, 1024, 1))

  def test_does_not_process_wsi_dicom(self):
    self.assertEmpty(
        list(
            _generic_dicom_handler.process_file(
                [],
                {},
                test_utils.testdata_path(
                    'wsi', 'multiframe_camelyon_challenge_image.dcm'
                ),
            )
        )
    )

  def test_load_encapsulated_dicom_from_bytes_io(self):
    with open(_encapsulated_dicom_path(), 'rb') as f:
      with io.BytesIO(f.read()) as binary_file:
        images = list(_generic_dicom_handler.process_file([], {}, binary_file))
        self.assertLen(images, 1)
        self.assertEqual(images[0].shape, (1024, 1024, 1))

  def test_loadpatches_coordinates(self):
    images = list(
        _generic_dicom_handler.process_file(
            [
                patch_coordinate.PatchCoordinate(0, 0, 10, 10),
                patch_coordinate.PatchCoordinate(10, 10, 10, 10),
            ],
            {},
            _encapsulated_dicom_path(),
        )
    )
    self.assertLen(images, 2)
    for img in images:
      self.assertEqual(img.shape, (10, 10, 1))

  def test_load_dicom_file_missing_photometric_interpretation_raises_error_path(
      self,
  ):
    with tempfile.TemporaryDirectory() as temp_dir:
      pixeldata = np.zeros((10, 10, 3), dtype=np.uint8)
      dcm = test_utils.create_test_dicom_instance(
          '1.2.840.10008.5.1.4.1.1.1.1', '1.1', '1.1.1', '1.1.1.1', pixeldata
      )
      dcm.Modality = 'CR'
      dcm_path = os.path.join(temp_dir, 'test.dcm')
      dcm.save_as(dcm_path)
      with self.assertRaisesRegex(
          data_accessor_errors.DicomError,
          '.*PhotometricInterpretation is required for DICOM.*',
      ):
        list(_generic_dicom_handler.process_file([], {}, dcm_path))

  def test_load_dicom_file_unsupported_samples_per_pixel_raises_error(
      self,
  ):
    with tempfile.TemporaryDirectory() as temp_dir:
      pixeldata = np.zeros((10, 10, 2), dtype=np.uint8)
      dcm = test_utils.create_test_dicom_instance(
          '1.2.840.10008.5.1.4.1.1.1.1', '1.1', '1.1.1', '1.1.1.1', pixeldata
      )
      dcm.Modality = 'DX'
      dcm_path = os.path.join(temp_dir, 'test.dcm')
      dcm.save_as(dcm_path)
      with self.assertRaisesRegex(
          data_accessor_errors.DicomError,
          '.*unsupported number of samples per pixel.*',
      ):
        list(_generic_dicom_handler.process_file([], {}, dcm_path))

  @parameterized.named_parameters(
      dict(
          testcase_name='MONOCHROME1',
          photometric_interpretation='MONOCHROME1',
      ),
      dict(
          testcase_name='MONOCHROME2',
          photometric_interpretation='MONOCHROME2',
      ),
  )
  def test_load_dicom_photometric_interpretation_raises_error_path(
      self, photometric_interpretation
  ):
    with tempfile.TemporaryDirectory() as temp_dir:
      pixeldata = np.zeros((10, 10, 3), dtype=np.uint8)
      dcm = test_utils.create_test_dicom_instance(
          '1.2.840.10008.5.1.4.1.1.1.1', '1.1', '1.1.1', '1.1.1.1', pixeldata
      )
      dcm.Modality = 'SM'
      dcm.PhotometricInterpretation = photometric_interpretation
      dcm_path = os.path.join(temp_dir, 'test.dcm')
      dcm.save_as(dcm_path)
      with self.assertRaisesRegex(
          data_accessor_errors.DicomError,
          'DICOM instance has 3 sample per pixel but contains single channel'
          ' PhotometricInterpretation.',
      ):
        list(_generic_dicom_handler.process_file([], {}, dcm_path))

  @parameterized.parameters(
      ['BitsStored', 'PlanarConfiguration', 'PixelRepresentation']
  )
  def test_load_dicom_fails_if_missing_required_tag_raises_error_path(
      self, required_elements
  ):
    with tempfile.TemporaryDirectory() as temp_dir:
      pixeldata = np.zeros((10, 10, 3), dtype=np.uint8)
      dcm = test_utils.create_test_dicom_instance(
          '1.2.840.10008.5.1.4.1.1.1.1', '1.1', '1.1.1', '1.1.1.1', pixeldata
      )
      dcm.Modality = 'GM'
      dcm.PhotometricInterpretation = 'RGB'
      dcm_path = os.path.join(temp_dir, 'test.dcm')
      del dcm[required_elements]
      dcm.save_as(dcm_path)
      with self.assertRaises(data_accessor_errors.DicomError):
        list(_generic_dicom_handler.process_file([], {}, dcm_path))

  @parameterized.named_parameters(
      dict(
          testcase_name='MONOCHROME1',
          photometric_interpretation='MONOCHROME1',
          channels=1,
      ),
      dict(
          testcase_name='MONOCHROME2',
          photometric_interpretation='MONOCHROME2',
          channels=1,
      ),
      dict(testcase_name='RGB', photometric_interpretation='RGB', channels=3),
  )
  def test_load_dicom_path(self, photometric_interpretation, channels):
    with tempfile.TemporaryDirectory() as temp_dir:
      pixeldata = np.zeros((10, 10, channels), dtype=np.uint8)
      dcm = test_utils.create_test_dicom_instance(
          '1.2.840.10008.5.1.4.1.1.1.1', '1.1', '1.1.1', '1.1.1.1', pixeldata
      )
      dcm.Modality = 'XC'
      dcm.PhotometricInterpretation = photometric_interpretation
      dcm_path = os.path.join(temp_dir, 'test.dcm')
      dcm.save_as(dcm_path)
      img = list(_generic_dicom_handler.process_file([], {}, dcm_path))
    self.assertLen(img, 1)
    self.assertEqual(img[0].shape, (10, 10, channels))

  def test_load_dicom_does_not_transform_if_missing_embedded_icc_profile(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      with PIL.Image.open(test_utils.testdata_path('image.jpeg')) as source_img:
        pixel_data = np.asarray(source_img)
      dcm = test_utils.create_test_dicom_instance(
          '1.2.840.10008.5.1.4.1.1.1.1', '1.1', '1.1.1', '1.1.1.1', pixel_data
      )
      dcm.Modality = 'SM'
      dcm.PhotometricInterpretation = 'RGB'
      dcm_path = os.path.join(temp_dir, 'test.dcm')
      dcm.save_as(dcm_path)
      img = list(
          _generic_dicom_handler.process_file(
              [],
              _MOCK_INSTANCE_METADATA_REQUEST_ICCPROFILE_NORM,
              dcm_path,
          )
      )
    self.assertLen(img, 1)
    np.testing.assert_array_equal(img[0], pixel_data)

  def test_load_dicom_does_not_icc_transform_if_monochrome(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      with PIL.Image.open(
          test_utils.testdata_path('image_bw.jpeg')
      ) as source_img:
        pixel_data = np.asarray(source_img)
      dcm = test_utils.create_test_dicom_instance(
          '1.2.840.10008.5.1.4.1.1.1.1', '1.1', '1.1.1', '1.1.1.1', pixel_data
      )
      dcm.Modality = 'CR'
      dcm.PhotometricInterpretation = 'MONOCHROME2'
      dcm.ICCProfile = dicom_slide.get_rommrgb_icc_profile_bytes()
      dcm_path = os.path.join(temp_dir, 'test.dcm')
      dcm.save_as(dcm_path)
      img = list(
          _generic_dicom_handler.process_file(
              [],
              _MOCK_INSTANCE_METADATA_REQUEST_ICCPROFILE_NORM,
              dcm_path,
          )
      )
    self.assertLen(img, 1)
    np.testing.assert_array_equal(img[0], np.expand_dims(pixel_data, axis=2))

  def test_load_dicom_transform_icc_if_profile_embedded_in_dicom(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      with PIL.Image.open(test_utils.testdata_path('image.jpeg')) as source_img:
        pixel_data = np.asarray(source_img)
      dcm = test_utils.create_test_dicom_instance(
          '1.2.840.10008.5.1.4.1.1.1.1', '1.1', '1.1.1', '1.1.1.1', pixel_data
      )
      dcm.Modality = 'DX'
      dcm.ICCProfile = dicom_slide.get_rommrgb_icc_profile_bytes()
      dcm.PhotometricInterpretation = 'RGB'
      dcm_path = os.path.join(temp_dir, 'test.dcm')
      dcm.save_as(dcm_path)
      img = list(
          _generic_dicom_handler.process_file(
              [],
              _MOCK_INSTANCE_METADATA_REQUEST_ICCPROFILE_NORM,
              dcm_path,
          )
      )
    self.assertLen(img, 1)
    self.assertFalse(np.array_equal(img[0], pixel_data))

  def test_load_dicom_transform_icc_if_profile_embedded_in_image(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      with io.BytesIO() as img_data:
        with PIL.Image.open(
            test_utils.testdata_path('image.jpeg')
        ) as source_img:
          pixel_data = np.asarray(source_img)
          source_img.save(
              img_data,
              icc_profile=dicom_slide.get_rommrgb_icc_profile_bytes(),
              format='JPEG',
          )
          width, height = source_img.width, source_img.height
        dcm_path = os.path.join(temp_dir, 'test.dcm')
        with pydicom.dcmread(_encapsulated_dicom_path()) as dcm:
          dcm.PixelData = pydicom.encaps.encapsulate([img_data.getvalue()])
          dcm.Width = width
          dcm.Height = height
          dcm.PhotometricInterpretation = 'RGB'
          dcm.SamplesPerPixel = 3
          dcm.save_as(dcm_path)
      img = list(
          _generic_dicom_handler.process_file(
              [],
              _MOCK_INSTANCE_METADATA_REQUEST_ICCPROFILE_NORM,
              dcm_path,
          )
      )
    self.assertLen(img, 1)
    self.assertFalse(np.array_equal(img[0], pixel_data))

  def test_load_dicom_does_not_transform_icc_if_profile_not_embedded(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      with io.BytesIO() as img_data:
        with PIL.Image.open(
            test_utils.testdata_path('image.jpeg')
        ) as source_img:
          pixel_data = np.asarray(source_img)
          source_img.save(img_data, format='JPEG')
          width, height = source_img.width, source_img.height
        dcm_path = os.path.join(temp_dir, 'test.dcm')
        with pydicom.dcmread(_encapsulated_dicom_path()) as dcm:
          dcm.PixelData = pydicom.encaps.encapsulate([img_data.getvalue()])
          dcm.Width = width
          dcm.Height = height
          dcm.PhotometricInterpretation = 'RGB'
          dcm.SamplesPerPixel = 3
          dcm.save_as(dcm_path)
      img = list(
          _generic_dicom_handler.process_file(
              [],
              _MOCK_INSTANCE_METADATA_REQUEST_ICCPROFILE_NORM,
              dcm_path,
          )
      )
    self.assertLen(img, 1)
    self.assertFalse(np.array_equal(img[0], pixel_data))

  @parameterized.named_parameters(
      dict(
          testcase_name='downsample_2x',
          scale_factor=1 / 2,
          interpolation=cv2.INTER_AREA,
      ),
      dict(
          testcase_name='upsample_2x',
          scale_factor=2,
          interpolation=cv2.INTER_CUBIC,
      ),
  )
  def test_load_whole_dicom_resize(self, scale_factor, interpolation):
    with tempfile.TemporaryDirectory() as temp_dir:
      with io.BytesIO() as img_data:
        with open(
            test_utils.testdata_path('image.jpeg'),
            'rb',
        ) as source_img:
          img_data.write(source_img.read())
        img_data.seek(0)
        with PIL.Image.open(img_data) as source_img:
          pixel_data = np.asarray(source_img)
          width, height = source_img.width, source_img.height
        img_data.seek(0)
        dcm_path = os.path.join(temp_dir, 'test.dcm')
        with pydicom.dcmread(_encapsulated_dicom_path()) as dcm:
          dcm.PixelData = pydicom.encaps.encapsulate([img_data.getvalue()])
          dcm.Width = width
          dcm.Height = height
          dcm.PhotometricInterpretation = 'RGB'
          dcm.SamplesPerPixel = 3
          dcm.save_as(dcm_path)
      img = list(
          _generic_dicom_handler.process_file(
              [],
              _mock_instance_extension_metadata({
                  _InstanceJsonKeys.IMAGE_DIMENSIONS: {
                      'width': int(width * scale_factor),
                      'height': int(height * scale_factor),
                  }
              }),
              dcm_path,
          )
      )
    self.assertLen(img, 1)
    np.testing.assert_array_equal(
        img[0],
        cv2.resize(
            pixel_data,
            (int(width * scale_factor), int(height * scale_factor)),
            interpolation=interpolation,
        ),
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='downsample_2x',
          scale_factor=1 / 2,
          interpolation=cv2.INTER_AREA,
      ),
      dict(
          testcase_name='upsample_2x',
          scale_factor=2,
          interpolation=cv2.INTER_CUBIC,
      ),
  )
  def test_load_whole_dicom_resize_patchs(self, scale_factor, interpolation):
    patch_coordinates = [
        patch_coordinate.PatchCoordinate(0, 0, 10, 10),
        patch_coordinate.PatchCoordinate(10, 10, 10, 10),
    ]
    with tempfile.TemporaryDirectory() as temp_dir:
      with io.BytesIO() as img_data:
        with open(
            test_utils.testdata_path('image.jpeg'),
            'rb',
        ) as source_img:
          img_data.write(source_img.read())
        img_data.seek(0)
        with PIL.Image.open(img_data) as source_img:
          pixel_data = np.asarray(source_img)
          width, height = source_img.width, source_img.height
        img_data.seek(0)
        dcm_path = os.path.join(temp_dir, 'test.dcm')
        with pydicom.dcmread(_encapsulated_dicom_path()) as dcm:
          dcm.PixelData = pydicom.encaps.encapsulate([img_data.getvalue()])
          dcm.Width = width
          dcm.Height = height
          dcm.PhotometricInterpretation = 'RGB'
          dcm.SamplesPerPixel = 3
          dcm.save_as(dcm_path)
      images = list(
          _generic_dicom_handler.process_file(
              patch_coordinates,
              _mock_instance_extension_metadata({
                  _InstanceJsonKeys.IMAGE_DIMENSIONS: {
                      'width': int(width * scale_factor),
                      'height': int(height * scale_factor),
                  }
              }),
              dcm_path,
          )
      )
    expected_img = cv2.resize(
        pixel_data,
        (int(width * scale_factor), int(height * scale_factor)),
        interpolation=interpolation,
    )
    self.assertLen(images, 2)
    for pc, img in zip(patch_coordinates, images):
      np.testing.assert_array_equal(
          img,
          expected_img[
              pc.y_origin : pc.y_origin + pc.height,
              pc.x_origin : pc.x_origin + pc.width,
              ...,
          ],
      )

  def test_validate_transfer_syntax(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      dcm_path = os.path.join(temp_dir, 'test.dcm')
      with pydicom.dcmread(_encapsulated_dicom_path()) as dcm:
        dcm.file_meta.TransferSyntaxUID = '1.2.840.10008.1.2.4.100'
        dcm.save_as(dcm_path)
      with self.assertRaisesRegex(
          data_accessor_errors.DicomError,
          '.*DICOM instance encoded using unsupported transfer syntax.*'
      ):
        list(_generic_dicom_handler.process_file([], {}, dcm_path))


if __name__ == '__main__':
  absltest.main()
