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

"""Tests for dicom source utils."""

from absl.testing import absltest
from absl.testing import parameterized
from ez_wsi_dicomweb import credential_factory
import pydicom

from data_accessors import data_accessor_const
from data_accessors.utils import dicom_source_utils
from data_accessors.utils import test_utils
from ez_wsi_dicomweb.test_utils.dicom_store_mock import dicom_store_mock


_InstanceJsonKeys = data_accessor_const.InstanceJsonKeys
_MOCK_DICOM_STORE_PATH = 'https://www.mock_dicom_store.com'


class DicomSourceUtilsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='dicom_source_type_generic',
          dcm_file=test_utils.testdata_path('cxr', 'encapsulated_cxr.dcm'),
          expected_source_type=dicom_source_utils.DicomDataSourceEnum.GENERIC_DICOM,
      ),
      dict(
          testcase_name='dicom_source_type_slide_microscopy_image',
          dcm_file=test_utils.testdata_path(
              'wsi', 'multiframe_camelyon_challenge_image.dcm'
          ),
          expected_source_type=dicom_source_utils.DicomDataSourceEnum.SLIDE_MICROSCOPY_IMAGE,
      ),
  )
  def test_identify_dicom_source_type(self, dcm_file, expected_source_type):
    with pydicom.dcmread(dcm_file) as dcm:
      dcm_path = f'{_MOCK_DICOM_STORE_PATH}/studies/{dcm.StudyInstanceUID}/series/{dcm.SeriesInstanceUID}/instances/{dcm.SOPInstanceUID}'
      with dicom_store_mock.MockDicomStores(
          _MOCK_DICOM_STORE_PATH
      ) as dicom_store:
        dicom_store[_MOCK_DICOM_STORE_PATH].add_instance(dcm)
        source_type = dicom_source_utils.get_dicom_source_type(
            credential_factory.NoAuthCredentialsFactory(),
            {
                _InstanceJsonKeys.DICOM_WEB_URI: dcm_path,
            },
        )
        self.assertEqual(source_type.dicom_source_type, expected_source_type)
        self.assertLen(source_type.dicom_instances_metadata, 1)
        self.assertEqual(
            source_type.dicom_instances_metadata[0].sop_instance_uid,
            dcm.SOPInstanceUID,
        )


if __name__ == '__main__':
  absltest.main()
