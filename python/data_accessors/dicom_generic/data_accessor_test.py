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

"""Unit tests for dicom_generic data accessor."""

from typing import Any, Mapping, Sequence

from absl.testing import absltest
from absl.testing import parameterized
from ez_wsi_dicomweb import credential_factory as credential_factory_module
from ez_wsi_dicomweb import dicom_web_interface
from ez_wsi_dicomweb.ml_toolkit import dicom_path
import numpy as np
import pydicom

from data_accessors import data_accessor_const
from data_accessors import data_accessor_errors
from data_accessors.dicom_generic import data_accessor
from data_accessors.dicom_generic import data_accessor_definition
from data_accessors.utils import test_utils
from ez_wsi_dicomweb.test_utils.dicom_store_mock import dicom_store_mock

_InstanceJsonKeys = data_accessor_const.InstanceJsonKeys

_MOCK_DICOM_STORE_PATH = 'https://www.mock_dicom_store.com'


def _create_dicom_web_uri(dcm: pydicom.FileDataset) -> str:
  return f'{_MOCK_DICOM_STORE_PATH}/studies/{dcm.StudyInstanceUID}/series/{dcm.SeriesInstanceUID}/instances/{dcm.SOPInstanceUID}'


def _test_load_generic_dicom(
    credential_factory: credential_factory_module.AbstractCredentialFactory,
    json_instance: Mapping[str, Any],
    dcm: pydicom.FileDataset,
) -> Sequence[np.ndarray]:
  with dicom_store_mock.MockDicomStores(_MOCK_DICOM_STORE_PATH) as dicom_store:
    dicom_store[_MOCK_DICOM_STORE_PATH].add_instance(dcm)
    dwi = dicom_web_interface.DicomWebInterface(
        credential_factory_module.NoAuthCredentialsFactory()
    )
    instance_path = dicom_path.FromString(
        json_instance[_InstanceJsonKeys.DICOM_WEB_URI]
    )
    instance = data_accessor_definition.json_to_generic_dicom_image(
        credential_factory,
        json_instance,
        default_patch_width=256,
        default_patch_height=256,
        require_patch_dim_match_default_dim=False,
        dicom_instances_metadata=dwi.get_instances(instance_path),
    )
    return list(data_accessor.DicomGenericData(instance).data_iterator())


class DataAccessorTest(parameterized.TestCase):

  def test_dicom_missing_metadata_raises(self):
    with pydicom.dcmread(
        test_utils.testdata_path('cxr', 'encapsulated_cxr.dcm')
    ) as dcm:
      json_instance = {
          _InstanceJsonKeys.DICOM_WEB_URI: _create_dicom_web_uri(dcm),
          _InstanceJsonKeys.PATCH_COORDINATES: [],
      }
      with dicom_store_mock.MockDicomStores(
          _MOCK_DICOM_STORE_PATH
      ) as dicom_store:
        dicom_store[_MOCK_DICOM_STORE_PATH].add_instance(dcm)
        instance = data_accessor_definition.json_to_generic_dicom_image(
            credential_factory_module.NoAuthCredentialsFactory(),
            json_instance,
            default_patch_width=256,
            default_patch_height=256,
            require_patch_dim_match_default_dim=False,
            dicom_instances_metadata=[],
        )
        with self.assertRaises(ValueError):
          list(data_accessor.DicomGenericData(instance).data_iterator())

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
  def test_dicom_image_patch_coordinates(
      self, patch_coordinates, expected_shape
  ):
    with pydicom.dcmread(
        test_utils.testdata_path('cxr', 'encapsulated_cxr.dcm')
    ) as dcm:
      json_instance = {
          _InstanceJsonKeys.DICOM_WEB_URI: _create_dicom_web_uri(dcm),
          _InstanceJsonKeys.PATCH_COORDINATES: patch_coordinates,
      }
      img = _test_load_generic_dicom(
          credential_factory_module.NoAuthCredentialsFactory(),
          json_instance,
          dcm,
      )
      self.assertLen(img, 1)
      self.assertEqual(img[0].shape, expected_shape)

  def test_dicom_image_patch_coordinates_outside_of_image_raises(self):
    with pydicom.dcmread(
        test_utils.testdata_path('cxr', 'encapsulated_cxr.dcm')
    ) as dcm:
      json_instance = {
          _InstanceJsonKeys.DICOM_WEB_URI: _create_dicom_web_uri(dcm),
          _InstanceJsonKeys.PATCH_COORDINATES: [
              {'x_origin': 0, 'y_origin': 0, 'width': 5000, 'height': 10}
          ],
      }
      with self.assertRaises(
          data_accessor_errors.PatchOutsideOfImageDimensionsError
      ):
        _test_load_generic_dicom(
            credential_factory_module.NoAuthCredentialsFactory(),
            json_instance,
            dcm,
        )

  def test_is_accessor_data_embedded_in_request(self):
    with pydicom.dcmread(
        test_utils.testdata_path('cxr', 'encapsulated_cxr.dcm')
    ) as dcm:
      json_instance = {
          _InstanceJsonKeys.DICOM_WEB_URI: _create_dicom_web_uri(dcm),
          _InstanceJsonKeys.PATCH_COORDINATES: [],
      }
      instance = data_accessor_definition.json_to_generic_dicom_image(
          credential_factory_module.NoAuthCredentialsFactory(),
          json_instance,
          default_patch_width=256,
          default_patch_height=256,
          require_patch_dim_match_default_dim=False,
          dicom_instances_metadata=[],
      )
      self.assertFalse(
          data_accessor.DicomGenericData(
              instance
          ).is_accessor_data_embedded_in_request()
      )

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
    with pydicom.dcmread(
        test_utils.testdata_path('cxr', 'encapsulated_cxr.dcm')
    ) as dcm:
      json_instance = {
          _InstanceJsonKeys.DICOM_WEB_URI: _create_dicom_web_uri(dcm),
      }
      json_instance.update(metadata)
      instance = data_accessor_definition.json_to_generic_dicom_image(
          credential_factory_module.NoAuthCredentialsFactory(),
          json_instance,
          default_patch_width=256,
          default_patch_height=256,
          require_patch_dim_match_default_dim=False,
          dicom_instances_metadata=[],
      )
      self.assertLen(data_accessor.DicomGenericData(instance), expected)


if __name__ == '__main__':
  absltest.main()
