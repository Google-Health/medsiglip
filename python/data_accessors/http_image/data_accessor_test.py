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
"""Tests for http_generic data accessor."""

from typing import Any, Mapping, Sequence
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from ez_wsi_dicomweb import credential_factory as credential_factory_module
import google.auth.credentials
import numpy as np
import PIL.Image
import requests_mock

from data_accessors import data_accessor_const
from data_accessors import data_accessor_errors
from data_accessors.http_image import data_accessor
from data_accessors.http_image import data_accessor_definition
from data_accessors.local_file_handlers import generic_dicom_handler
from data_accessors.local_file_handlers import traditional_image_handler
from data_accessors.utils import test_utils


_InstanceJsonKeys = data_accessor_const.InstanceJsonKeys


def _test_load_from_http(
    file_handlers,
    blob_file_name,
    credential_factory: credential_factory_module.AbstractCredentialFactory,
    json_instance: Mapping[str, Any],
    source_image_path: str,
) -> Sequence[np.ndarray]:
  with open(source_image_path, 'rb') as f:
    data = f.read()
  with requests_mock.Mocker() as m:
    m.get(f'http://earth.com/{blob_file_name}', content=data)
    instance = data_accessor_definition.json_to_http_image(
        credential_factory,
        json_instance,
        default_patch_width=256,
        default_patch_height=256,
        require_patch_dim_match_default_dim=False,
    )
    http_data_accessor = data_accessor.HttpImageData(
        instance,
        file_handlers=file_handlers,
    )
    return list(http_data_accessor.data_iterator())


def _test_load_image_from_http(
    credential_factory: credential_factory_module.AbstractCredentialFactory,
    json_instance: Mapping[str, Any],
    source_image_path: str,
) -> Sequence[np.ndarray]:
  return _test_load_from_http(
      [
          generic_dicom_handler.GenericDicomHandler(),
          traditional_image_handler.TraditionalImageHandler(),
      ],
      'image.jpeg',
      credential_factory,
      json_instance,
      source_image_path,
  )


def _test_load_generic_dicom_from_http(
    credential_factory: credential_factory_module.AbstractCredentialFactory,
    json_instance: Mapping[str, Any],
    source_image_path: str,
) -> Sequence[np.ndarray]:
  return _test_load_from_http(
      [
          traditional_image_handler.TraditionalImageHandler(),
          generic_dicom_handler.GenericDicomHandler(),
      ],
      'image.dcm',
      credential_factory,
      json_instance,
      source_image_path,
  )


class DataAccessorTest(parameterized.TestCase):

  def test_http_traditional_image_handler_color_image(self):
    json_instance = {
        _InstanceJsonKeys.IMAGE_URL: 'http://earth.com/image.jpeg',
    }
    source_image_path = test_utils.testdata_path('image.jpeg')
    img = _test_load_image_from_http(
        credential_factory_module.NoAuthCredentialsFactory(),
        json_instance,
        source_image_path,
    )
    self.assertLen(img, 1)
    self.assertEqual(img[0].shape, (67, 100, 3))
    with PIL.Image.open(source_image_path) as source_img:
      expected_img = np.asarray(source_img)
    np.testing.assert_array_equal(img[0], expected_img)

  def test_http_traditional_image_handler_bw_image(self):
    json_instance = {
        _InstanceJsonKeys.IMAGE_URL: 'http://earth.com/image.jpeg',
    }
    source_image_path = test_utils.testdata_path('image_bw.jpeg')
    img = _test_load_image_from_http(
        credential_factory_module.NoAuthCredentialsFactory(),
        json_instance,
        source_image_path,
    )
    self.assertLen(img, 1)
    self.assertEqual(img[0].shape, (67, 100, 1))
    with PIL.Image.open(source_image_path) as source_img:
      expected_img = np.asarray(source_img)
    np.testing.assert_array_equal(img[0][..., 0], expected_img)

  def test_http_traditional_image_credential_pass_through(self):
    json_instance = {
        _InstanceJsonKeys.IMAGE_URL: 'http://earth.com/image.jpeg',
    }
    source_image_path = test_utils.testdata_path('image.jpeg')
    img = _test_load_image_from_http(
        credential_factory_module.TokenPassthroughCredentialFactory('token'),
        json_instance,
        source_image_path,
    )
    self.assertLen(img, 1)
    self.assertEqual(img[0].shape, (67, 100, 3))
    with PIL.Image.open(source_image_path) as source_img:
      expected_img = np.asarray(source_img)
    np.testing.assert_array_equal(img[0], expected_img)

  def test_http_traditional_image_default_credential(self):
    json_instance = {_InstanceJsonKeys.IMAGE_URL: 'http://earth.com/image.jpeg'}
    source_image_path = test_utils.testdata_path('image.jpeg')
    credentials_mock = mock.create_autospec(
        google.auth.credentials.Credentials, instance=True
    )
    type(credentials_mock).token = mock.PropertyMock(return_value='TOKEN')
    type(credentials_mock).valid = mock.PropertyMock(return_value='True')
    type(credentials_mock).expired = mock.PropertyMock(return_value='False')
    with mock.patch(
        'google.auth.default', return_value=(credentials_mock, 'project')
    ):
      img = _test_load_image_from_http(
          credential_factory_module.DefaultCredentialFactory(),
          json_instance,
          source_image_path,
      )
    self.assertLen(img, 1)
    self.assertEqual(img[0].shape, (67, 100, 3))
    with PIL.Image.open(source_image_path) as source_img:
      expected_img = np.asarray(source_img)
    np.testing.assert_array_equal(img[0], expected_img)

  @parameterized.named_parameters(
      dict(
          testcase_name='patch_list',
          patch_coordinates=[
              {'x_origin': 0, 'y_origin': 0, 'width': 256, 'height': 256}
          ],
      ),
  )
  def test_http_traditional_image_patch_coordinates_outside_of_image_raises(
      self, patch_coordinates
  ):
    json_instance = {
        _InstanceJsonKeys.IMAGE_URL: 'http://earth.com/image.jpeg',
        _InstanceJsonKeys.PATCH_COORDINATES: patch_coordinates,
    }
    with self.assertRaises(
        data_accessor_errors.PatchOutsideOfImageDimensionsError
    ):
      _test_load_image_from_http(
          credential_factory_module.NoAuthCredentialsFactory(),
          json_instance,
          test_utils.testdata_path('image.jpeg'),
      )

  @parameterized.named_parameters(
      dict(
          testcase_name='patch_list',
          patch_coordinates=[
              {'x_origin': 0, 'y_origin': 0, 'width': 10, 'height': 10}
          ],
          expected_shape=(10, 10, 3),
      ),
      dict(
          testcase_name='empty_patch_list',
          patch_coordinates=[],
          expected_shape=(67, 100, 3),
      ),
  )
  def test_http_traditional_image_patch_coordinates(
      self, patch_coordinates, expected_shape
  ):
    json_instance = {
        _InstanceJsonKeys.IMAGE_URL: 'http://earth.com/image.jpeg',
        _InstanceJsonKeys.PATCH_COORDINATES: patch_coordinates,
    }
    img = _test_load_image_from_http(
        credential_factory_module.NoAuthCredentialsFactory(),
        json_instance,
        test_utils.testdata_path('image.jpeg'),
    )
    self.assertLen(img, 1)
    self.assertEqual(img[0].shape, expected_shape)

  @parameterized.named_parameters(
      dict(
          testcase_name='patch_list',
          patch_coordinates=[
              {'x_origin': 0, 'y_origin': 0, 'width': 10, 'height': 10}
          ],
          expected_shape=(10, 10, 1),
      ),
      dict(
          testcase_name='empty_patch_list',
          patch_coordinates=[],
          expected_shape=(1024, 1024, 1),
      ),
  )
  def test_http_dicom_image_patch_coordinates(
      self, patch_coordinates, expected_shape
  ):
    json_instance = {
        _InstanceJsonKeys.IMAGE_URL: 'http://earth.com/image.dcm',
        _InstanceJsonKeys.PATCH_COORDINATES: patch_coordinates,
    }
    img = _test_load_generic_dicom_from_http(
        credential_factory_module.NoAuthCredentialsFactory(),
        json_instance,
        test_utils.testdata_path('cxr', 'encapsulated_cxr.dcm'),
    )
    self.assertLen(img, 1)
    self.assertEqual(img[0].shape, expected_shape)

  def test_http_dicom_image_patch_coordinates_outside_of_image_raises(self):
    json_instance = {
        _InstanceJsonKeys.IMAGE_URL: 'http://earth.com/image.dcm',
        _InstanceJsonKeys.PATCH_COORDINATES: [
            {'x_origin': 0, 'y_origin': 0, 'width': 5000, 'height': 10}
        ],
    }
    with self.assertRaises(
        data_accessor_errors.PatchOutsideOfImageDimensionsError
    ):
      _test_load_generic_dicom_from_http(
          credential_factory_module.NoAuthCredentialsFactory(),
          json_instance,
          test_utils.testdata_path('cxr', 'encapsulated_cxr.dcm'),
      )

  def test_is_accessor_data_embedded_in_request(self):
    json_instance = {
        _InstanceJsonKeys.IMAGE_URL: 'http://earth.com/image.dcm',
        _InstanceJsonKeys.PATCH_COORDINATES: [
            {'x_origin': 0, 'y_origin': 0, 'width': 5000, 'height': 10}
        ],
    }
    instance = data_accessor_definition.json_to_http_image(
        credential_factory_module.NoAuthCredentialsFactory(),
        json_instance,
        default_patch_width=256,
        default_patch_height=256,
        require_patch_dim_match_default_dim=False,
    )
    http_data_accessor = data_accessor.HttpImageData(
        instance,
        file_handlers=[
            traditional_image_handler.TraditionalImageHandler(),
            generic_dicom_handler.GenericDicomHandler(),
        ],
    )
    self.assertFalse(http_data_accessor.is_accessor_data_embedded_in_request())

  @parameterized.named_parameters(
      dict(
          testcase_name='no_patch_coordinates',
          metadata={},
          expected=1,
      ),
      dict(
          testcase_name='one_patch',
          metadata={
              _InstanceJsonKeys.PATCH_COORDINATES: [
                  {'x_origin': 0, 'y_origin': 0, 'width': 10, 'height': 10},
              ]
          },
          expected=1,
      ),
      dict(
          testcase_name='two_patches',
          metadata={
              _InstanceJsonKeys.PATCH_COORDINATES: [
                  {'x_origin': 0, 'y_origin': 0, 'width': 10, 'height': 10},
                  {'x_origin': 0, 'y_origin': 0, 'width': 10, 'height': 10},
              ]
          },
          expected=2,
      ),
  )
  def test_accessor_length(self, metadata, expected):
    json_instance = {
        _InstanceJsonKeys.IMAGE_URL: 'http://earth.com/image.dcm',
    }
    json_instance.update(metadata)
    instance = data_accessor_definition.json_to_http_image(
        credential_factory_module.NoAuthCredentialsFactory(),
        json_instance,
        default_patch_width=256,
        default_patch_height=256,
        require_patch_dim_match_default_dim=False,
    )
    http_data_accessor = data_accessor.HttpImageData(
        instance,
        file_handlers=[
            traditional_image_handler.TraditionalImageHandler(),
            generic_dicom_handler.GenericDicomHandler(),
        ],
    )
    self.assertLen(http_data_accessor, expected)


if __name__ == '__main__':
  absltest.main()
