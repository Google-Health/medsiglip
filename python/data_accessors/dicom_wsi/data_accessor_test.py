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

"""Unit tests for pathology 2.0 endpoint predictor."""

import dataclasses
import json
import typing
from typing import Sequence
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from ez_wsi_dicomweb import credential_factory
from ez_wsi_dicomweb import dicom_slide
from ez_wsi_dicomweb import dicom_web_interface
from ez_wsi_dicomweb.ml_toolkit import dicom_path
import numpy as np
import pydicom

from data_accessors import data_accessor_const
from data_accessors import data_accessor_errors
from data_accessors.dicom_wsi import configuration
from data_accessors.dicom_wsi import data_accessor
from data_accessors.dicom_wsi import data_accessor_definition
from data_accessors.dicom_wsi import icc_profile_cache
from data_accessors.dicom_wsi.test_utils import test_files
from data_accessors.utils import image_dimension_utils
from data_accessors.utils import patch_coordinate
from serving.serving_framework import model_runner
from ez_wsi_dicomweb.test_utils.dicom_store_mock import dicom_store_mock


_InstanceJsonKeys = data_accessor_const.InstanceJsonKeys

_DEBUG_SETTINGS = configuration.ConfigurationSettings(
    224,
    224,
    None,
    configuration.IccProfileCacheConfiguration(testing=True),
)


def _dicom_series_path(dcm: pydicom.Dataset) -> dicom_path.Path:
  return dicom_path.FromString(
      f'/projects/project/locations/location/datasets/dataset/dicomStores/dicomstore/dicomWeb/studies/{dcm.StudyInstanceUID}/series/{dcm.SeriesInstanceUID}'
  )


class MockModelRunner:
  """Mock embedding, return mean for each channel in patch."""

  def batch_model(self, data: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
    """Compute and return mock embeddings."""
    return [np.mean(d, axis=(1, 2)) for d in data]


_mock_model_runner = typing.cast(model_runner.ModelRunner, MockModelRunner())


class DicomDigitalPathologyDataTest(parameterized.TestCase):

  def test_get_dicom_patches_instance_not_found(self):
    dcm = pydicom.dcmread(
        test_files.testdata_path('multiframe_camelyon_challenge_image.dcm')
    )
    path = _dicom_series_path(dcm)
    coordinates = [
        patch_coordinate.create_patch_coordinate(
            {'x_origin': 0, 'y_origin': 0},
            default_width=_DEBUG_SETTINGS.endpoint_input_width,
            default_height=_DEBUG_SETTINGS.endpoint_input_height,
        )
    ]
    instance = data_accessor_definition.DicomWSIImage(
        credential_factory.TokenPassthroughCredentialFactory('mock_token'),
        str(path),
        {},
        '1.42',
        coordinates,
        [],
    )
    store_path = str(path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mk_dicom_stores:
      mk_dicom_stores[store_path].add_instance(dcm)
      with self.assertRaises(data_accessor_errors.DicomPathError):
        list(
            data_accessor.DicomDigitalPathologyData(
                instance, _DEBUG_SETTINGS
            ).data_iterator()
        )

  def test_get_dicom_patches_dicom_slide_not_found(self):
    dcm = pydicom.dcmread(
        test_files.testdata_path('multiframe_camelyon_challenge_image.dcm')
    )
    path = dicom_path.FromString(
        '/projects/project/locations/location/datasets/dataset/dicomStores/dicomstore/dicomWeb/studies/1.42/series/1.42'
    )
    coordinates = [
        patch_coordinate.create_patch_coordinate(
            {'x_origin': 0, 'y_origin': 0},
            default_width=_DEBUG_SETTINGS.endpoint_input_width,
            default_height=_DEBUG_SETTINGS.endpoint_input_height,
        )
    ]
    instance = data_accessor_definition.DicomWSIImage(
        credential_factory.TokenPassthroughCredentialFactory('mock_token'),
        str(path),
        {},
        '1.42',
        coordinates,
        [],
    )
    store_path = str(path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mk_dicom_stores:
      mk_dicom_stores[store_path].add_instance(dcm)
      with self.assertRaises(data_accessor_errors.DicomPathError):
        list(
            data_accessor.DicomDigitalPathologyData(
                instance, _DEBUG_SETTINGS
            ).data_iterator()
        )

  @parameterized.parameters(['https://bad_path', '', 'bad_path'])
  def test_get_dicom_patches_bad_path(self, bad_path):
    coordinates = [
        patch_coordinate.create_patch_coordinate(
            {'x_origin': 0, 'y_origin': 0},
            default_width=_DEBUG_SETTINGS.endpoint_input_width,
            default_height=_DEBUG_SETTINGS.endpoint_input_height,
        )
    ]
    instance = data_accessor_definition.DicomWSIImage(
        credential_factory.TokenPassthroughCredentialFactory('mock_token'),
        bad_path,
        {},
        '1.42',
        coordinates,
        [],
    )
    with self.assertRaises(data_accessor_errors.DicomPathError):
      list(
          data_accessor.DicomDigitalPathologyData(
              instance, _DEBUG_SETTINGS
          ).data_iterator()
      )

  @parameterized.parameters(['', 'mock_bearer_token'])
  def test_get_dicom_patches(self, bearer_token):
    dcm = pydicom.dcmread(
        test_files.testdata_path('multiframe_camelyon_challenge_image.dcm')
    )
    path = _dicom_series_path(dcm)
    coordinates = [
        patch_coordinate.create_patch_coordinate(
            {'x_origin': 0, 'y_origin': 0},
            default_width=_DEBUG_SETTINGS.endpoint_input_width,
            default_height=_DEBUG_SETTINGS.endpoint_input_height,
        )
    ]
    instance = data_accessor_definition.DicomWSIImage(
        credential_factory.TokenPassthroughCredentialFactory(bearer_token),
        str(path),
        {},
        dcm.SOPInstanceUID,
        coordinates,
        [],
    )
    store_path = str(path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mk_dicom_stores:
      mk_dicom_stores[store_path].add_instance(dcm)
      images = list(
          data_accessor.DicomDigitalPathologyData(
              instance, _DEBUG_SETTINGS
          ).data_iterator()
      )
    self.assertLen(images, 1)
    self.assertEqual(images[0].shape, (224, 224, 3))
    self.assertEqual(np.min(images[0]), 27)
    self.assertEqual(np.max(images[0]), 255)

  @parameterized.parameters(['', 'mock_bearer_token'])
  def test_get_dicom_whole_slide(self, bearer_token):
    dcm = pydicom.dcmread(
        test_files.testdata_path('multiframe_camelyon_challenge_image.dcm')
    )
    path = _dicom_series_path(dcm)
    coordinates = []
    instance = data_accessor_definition.DicomWSIImage(
        credential_factory.TokenPassthroughCredentialFactory(bearer_token),
        str(path),
        {},
        dcm.SOPInstanceUID,
        coordinates,
        [],
    )
    store_path = str(path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mk_dicom_stores:
      mk_dicom_stores[store_path].add_instance(dcm)
      images = list(
          data_accessor.DicomDigitalPathologyData(
              instance, _DEBUG_SETTINGS
          ).data_iterator()
      )
    self.assertLen(images, 1)
    self.assertEmpty(instance.patch_coordinates)
    self.assertEqual(images[0].shape, (700, 1152, 3))
    self.assertEqual(np.min(images[0]), 19)
    self.assertEqual(np.max(images[0]), 255)

  @parameterized.named_parameters([
      dict(testcase_name='no_extension', extension={}),
      dict(
          testcase_name='defines_icc_profile_transform',
          extension={
              _InstanceJsonKeys.TRANSFORM_IMAGING_TO_ICC_PROFILE: 'SRGB',
          },
      ),
  ])
  @mock.patch.object(
      dicom_slide, 'create_icc_profile_transformation', autospec=True
  )
  def test_get_patches_from_dicom_with_out_icc_profile_not_create_transform(
      self, create_transform, extension
  ):
    dcm = pydicom.dcmread(
        test_files.testdata_path('multiframe_camelyon_challenge_image.dcm')
    )
    path = _dicom_series_path(dcm)
    coordinates = [
        patch_coordinate.create_patch_coordinate(
            {'x_origin': 0, 'y_origin': 0},
            default_width=_DEBUG_SETTINGS.endpoint_input_width,
            default_height=_DEBUG_SETTINGS.endpoint_input_height,
        )
    ]
    instance = data_accessor_definition.DicomWSIImage(
        credential_factory.TokenPassthroughCredentialFactory(
            'mock_bearer_token'
        ),
        str(path),
        extension,
        dcm.SOPInstanceUID,
        coordinates,
        [],
    )
    store_path = str(path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mk_dicom_stores:
      mk_dicom_stores[store_path].add_instance(dcm)
      images = list(
          data_accessor.DicomDigitalPathologyData(
              instance, _DEBUG_SETTINGS
          ).data_iterator()
      )
    self.assertLen(images, 1)
    self.assertEqual(images[0].shape, (224, 224, 3))
    self.assertEqual(np.min(images[0]), 27)
    self.assertEqual(np.max(images[0]), 255)
    create_transform.assert_not_called()

  def test_get_dicom_patches_no_pixel_spacing(self):
    dcm = pydicom.dcmread(test_files.testdata_path('test.dcm'))
    # remove pixel spacing
    del dcm['SharedFunctionalGroupsSequence']  # pylint: disable=invalid-delete
    path = _dicom_series_path(dcm)
    coordinates = [
        patch_coordinate.create_patch_coordinate(
            {'x_origin': 0, 'y_origin': 0},
            default_width=_DEBUG_SETTINGS.endpoint_input_width,
            default_height=_DEBUG_SETTINGS.endpoint_input_height,
        )
    ]
    instance = data_accessor_definition.DicomWSIImage(
        credential_factory.TokenPassthroughCredentialFactory(
            'mock_bearer_token'
        ),
        str(path),
        {_InstanceJsonKeys.REQUIRE_PATCHES_FULLY_IN_SOURCE_IMAGE: False},
        dcm.SOPInstanceUID,
        coordinates,
        [],
    )
    store_path = str(path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mk_dicom_stores:
      mk_dicom_stores[store_path].add_instance(dcm)
      images = list(
          data_accessor.DicomDigitalPathologyData(
              instance, _DEBUG_SETTINGS
          ).data_iterator()
      )
    self.assertLen(images, 1)
    self.assertEqual(images[0].shape, (224, 224, 3))
    self.assertEqual(round(float(np.min(images[0])), 4), 0.0)
    self.assertEqual(np.max(images[0]), 255)

  @parameterized.named_parameters([
      dict(
          testcase_name='VL_SLIDE_COORDINATES_MIROSCOPIC_IMAGE_SOP_CLASS_UID',
          sop_class_uid='1.2.840.10008.5.1.4.1.1.77.1.3',
      ),
      dict(
          testcase_name='VL_MIROSCOPIC_IMAGE_SOP_CLASS_UID ',
          sop_class_uid='1.2.840.10008.5.1.4.1.1.77.1.2',
      ),
  ])
  def test_get_dicom_patches_from_non_tiled_dicom(self, sop_class_uid):
    dcm = pydicom.dcmread(test_files.testdata_path('test.dcm'))
    # remove pixel spacing
    del dcm['SharedFunctionalGroupsSequence']  # pylint: disable=invalid-delete
    dcm.file_meta.MediaStorageSOPClassUID = sop_class_uid
    dcm.SOPClassUID = sop_class_uid
    path = _dicom_series_path(dcm)
    coordinates = [
        patch_coordinate.create_patch_coordinate(
            {'x_origin': 0, 'y_origin': 0},
            default_width=_DEBUG_SETTINGS.endpoint_input_width,
            default_height=_DEBUG_SETTINGS.endpoint_input_height,
        )
    ]
    instance = data_accessor_definition.DicomWSIImage(
        credential_factory.TokenPassthroughCredentialFactory(
            'mock_bearer_token'
        ),
        str(path),
        {_InstanceJsonKeys.REQUIRE_PATCHES_FULLY_IN_SOURCE_IMAGE: False},
        dcm.SOPInstanceUID,
        coordinates,
        [],
    )
    store_path = str(path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mk_dicom_stores:
      mk_dicom_stores[store_path].add_instance(dcm)
      images = list(
          data_accessor.DicomDigitalPathologyData(
              instance, _DEBUG_SETTINGS
          ).data_iterator()
      )
    self.assertLen(images, 1)
    self.assertEqual(images[0].shape, (224, 224, 3))
    self.assertEqual(round(float(np.min(images[0])), 4), 0.0)
    self.assertEqual(np.max(images[0]), 255)

  def test_get_dicom_patches_from_sparse_dicom_raises(self):
    dcm = pydicom.dcmread(
        test_files.testdata_path('multiframe_camelyon_challenge_image.dcm')
    )
    del dcm['00209311']  # pylint: disable=invalid-delete
    path = _dicom_series_path(dcm)
    coordinates = [
        patch_coordinate.create_patch_coordinate(
            {'x_origin': 0, 'y_origin': 0},
            default_width=_DEBUG_SETTINGS.endpoint_input_width,
            default_height=_DEBUG_SETTINGS.endpoint_input_height,
        )
    ]
    instance = data_accessor_definition.DicomWSIImage(
        credential_factory.TokenPassthroughCredentialFactory('bearer_token'),
        str(path),
        {},
        dcm.SOPInstanceUID,
        coordinates,
        [],
    )
    store_path = str(path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mk_dicom_stores:
      mk_dicom_stores[store_path].add_instance(dcm)
      with self.assertRaises(data_accessor_errors.DicomTiledFullError):
        list(
            data_accessor.DicomDigitalPathologyData(
                instance, _DEBUG_SETTINGS
            ).data_iterator()
        )

  def test_get_dicom_patches_from_missing_instance_raises(
      self,
  ):
    dcm = pydicom.dcmread(
        test_files.testdata_path('multiframe_camelyon_challenge_image.dcm')
    )
    path = _dicom_series_path(dcm)
    coordinates = [
        patch_coordinate.create_patch_coordinate(
            {'x_origin': 0, 'y_origin': 0},
            default_width=_DEBUG_SETTINGS.endpoint_input_width,
            default_height=_DEBUG_SETTINGS.endpoint_input_height,
        )
    ]
    instance = data_accessor_definition.DicomWSIImage(
        credential_factory.TokenPassthroughCredentialFactory('bearer_token'),
        str(path),
        {},
        '1.42',
        coordinates,
        [],
    )
    store_path = str(path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mk_dicom_stores:
      mk_dicom_stores[store_path].add_instance(dcm)
      with self.assertRaises(data_accessor_errors.DicomPathError):
        list(
            data_accessor.DicomDigitalPathologyData(
                instance, _DEBUG_SETTINGS
            ).data_iterator()
        )

  @parameterized.parameters('mock_bearer_token', '')
  def test_repeated_get_dicom_patches_does_not_re_int(self, bearer_token):
    dcm = pydicom.dcmread(
        test_files.testdata_path('multiframe_camelyon_challenge_image.dcm')
    )
    path = _dicom_series_path(dcm)
    coordinates = [
        patch_coordinate.create_patch_coordinate(
            {'x_origin': 0, 'y_origin': 0},
            default_width=_DEBUG_SETTINGS.endpoint_input_width,
            default_height=_DEBUG_SETTINGS.endpoint_input_height,
        )
    ]
    instance = data_accessor_definition.DicomWSIImage(
        credential_factory.TokenPassthroughCredentialFactory(bearer_token),
        str(path),
        {},
        dcm.SOPInstanceUID,
        coordinates,
        [],
    )
    store_path = str(path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mk_dicom_stores:
      mk_dicom_stores[store_path].add_instance(dcm)
      list(
          data_accessor.DicomDigitalPathologyData(
              instance, _DEBUG_SETTINGS
          ).data_iterator()
      )
      # repeate prior call
      images = list(
          data_accessor.DicomDigitalPathologyData(
              instance, _DEBUG_SETTINGS
          ).data_iterator()
      )
      # check patch results are as expected.
      self.assertLen(images, 1)
      self.assertEqual(images[0].shape, (224, 224, 3))
      self.assertEqual(np.min(images[0]), 27)
      self.assertEqual(np.max(images[0]), 255)

  @parameterized.parameters([1, 1.2, 'abc', ([],)])
  def test_get_ez_wsi_state_invalid_value_raises(self, val):
    with self.assertRaises(data_accessor_errors.EzWsiStateError):
      data_accessor._get_ez_wsi_state({_InstanceJsonKeys.EZ_WSI_STATE: val})

  def test_get_ez_wsi_state_default(self):
    self.assertEqual(data_accessor._get_ez_wsi_state({}), {})

  def test_get_ez_wsi_state_expected(self):
    expected = {'abc': 123}
    self.assertEqual(
        data_accessor._get_ez_wsi_state(
            {_InstanceJsonKeys.EZ_WSI_STATE: expected}
        ),
        expected,
    )

  def test_get_dicom_instances_with_different_transfer_syntax_raise(self):
    dcm = pydicom.dcmread(
        test_files.testdata_path('multiframe_camelyon_challenge_image.dcm')
    )
    # define two concatenation instances with different transfer syntaxs.
    dcm.InConcatenationNumber = 1
    dcm.ConcatenationUID = '1.43'
    dcm.ConcatenationFrameOffsetNumber = 0
    dcm2 = pydicom.dcmread(
        test_files.testdata_path('multiframe_camelyon_challenge_image.dcm')
    )
    dcm2.file_meta.MediaStorageSOPInstanceUID = '1.42'
    dcm2.ConcatenationFrameOffsetNumber = dcm.NumberOfFrames
    dcm2.SOPInstanceUID = '1.42'
    dcm2.InConcatenationNumber = 2
    dcm2.ConcatenationUID = '1.43'
    dcm2.file_meta.TransferSyntaxUID = '1.2.840.10008.1.​2.​1'
    path = _dicom_series_path(dcm)
    coordinates = [
        patch_coordinate.create_patch_coordinate(
            {'x_origin': 0, 'y_origin': 0},
            default_width=_DEBUG_SETTINGS.endpoint_input_width,
            default_height=_DEBUG_SETTINGS.endpoint_input_height,
        )
    ]
    instance = data_accessor_definition.DicomWSIImage(
        credential_factory.TokenPassthroughCredentialFactory('mock_token'),
        str(path),
        {},
        dcm.SOPInstanceUID,
        coordinates,
        [],
    )
    store_path = str(path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mk_dicom_stores:
      mk_dicom_stores[store_path].add_instance(dcm)
      mk_dicom_stores[store_path].add_instance(dcm2)
      with self.assertRaisesRegex(
          data_accessor_errors.DicomError,
          'All DICOM instances in a pyramid level are required to have same'
          ' TransferSyntaxUID.',
      ):
        list(
            data_accessor.DicomDigitalPathologyData(
                instance, _DEBUG_SETTINGS
            ).data_iterator()
        )

  def test_get_dicom_instances_invalid_tags_raises(self):
    dcm = pydicom.dcmread(
        test_files.testdata_path('multiframe_camelyon_challenge_image.dcm')
    )
    del dcm['00080008']  # pylint: disable=invalid-delete
    path = _dicom_series_path(dcm)
    coordinates = [
        patch_coordinate.create_patch_coordinate(
            {'x_origin': 0, 'y_origin': 0},
            default_width=_DEBUG_SETTINGS.endpoint_input_width,
            default_height=_DEBUG_SETTINGS.endpoint_input_height,
        )
    ]
    instance = data_accessor_definition.DicomWSIImage(
        credential_factory.TokenPassthroughCredentialFactory('mock_token'),
        str(path),
        {},
        dcm.SOPInstanceUID,
        coordinates,
        [],
    )
    store_path = str(path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mk_dicom_stores:
      mk_dicom_stores[store_path].add_instance(dcm)
      with self.assertRaisesRegex(
          data_accessor_errors.DicomError,
          'DICOM instance missing required tags.',
      ):
        list(
            data_accessor.DicomDigitalPathologyData(
                instance, _DEBUG_SETTINGS
            ).data_iterator()
        )

  def test_can_not_find_dicom_level_raises(self):
    dcm = pydicom.dcmread(
        test_files.testdata_path('multiframe_camelyon_challenge_image.dcm')
    )
    path = _dicom_series_path(dcm)
    coordinates = [
        patch_coordinate.create_patch_coordinate(
            {'x_origin': 0, 'y_origin': 0},
            default_width=_DEBUG_SETTINGS.endpoint_input_width,
            default_height=_DEBUG_SETTINGS.endpoint_input_height,
        )
    ]
    store_path = str(path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mk_dicom_stores:
      mk_dicom_stores[store_path].add_instance(dcm)
      ds = dicom_slide.DicomSlide(
          dicom_web_interface.DicomWebInterface(
              credential_factory.NoAuthCredentialsFactory()
          ),
          path,
      )
      # error occures due to instance requesting predictions for an instance
      # which is not defined in the metadata.
      metadata = ds.json_metadata()
      metadata = metadata.replace(dcm.SOPInstanceUID, '1.42')
      metadata = json.loads(metadata)
      # modifying metadata to remove instance.
      instance = data_accessor_definition.DicomWSIImage(
          credential_factory.TokenPassthroughCredentialFactory('mock_token'),
          str(path),
          {_InstanceJsonKeys.EZ_WSI_STATE: metadata},
          dcm.SOPInstanceUID,
          coordinates,
          [],
      )
      with self.assertRaises(data_accessor_errors.LevelNotFoundError):
        list(
            data_accessor.DicomDigitalPathologyData(
                instance, _DEBUG_SETTINGS
            ).data_iterator()
        )

  def test_dicom_level_resize_greater_than_8x_raises(self):
    dcm = pydicom.dcmread(
        test_files.testdata_path('multiframe_camelyon_challenge_image.dcm')
    )
    path = _dicom_series_path(dcm)
    coordinates = [
        patch_coordinate.create_patch_coordinate(
            {_InstanceJsonKeys.X_ORIGIN: 0, _InstanceJsonKeys.Y_ORIGIN: 0},
            default_width=_DEBUG_SETTINGS.endpoint_input_width,
            default_height=_DEBUG_SETTINGS.endpoint_input_height,
        )
    ]
    store_path = str(path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mk_dicom_stores:
      mk_dicom_stores[store_path].add_instance(dcm)
      instance = data_accessor_definition.DicomWSIImage(
          credential_factory.TokenPassthroughCredentialFactory('mock_token'),
          str(path),
          {
              _InstanceJsonKeys.IMAGE_DIMENSIONS: dataclasses.asdict(
                  image_dimension_utils.ImageDimensions(
                      int(dcm.TotalPixelMatrixColumns // 9),
                      int(dcm.TotalPixelMatrixRows // 9),
                  )
              )
          },
          dcm.SOPInstanceUID,
          coordinates,
          [],
      )
      with self.assertRaisesRegex(
          data_accessor_errors.DicomImageDownsamplingTooLargeError,
          'Image downsampling, 9.09091X exceeds 8X',
      ):
        list(
            data_accessor.DicomDigitalPathologyData(
                instance, _DEBUG_SETTINGS
            ).data_iterator()
        )

  def test_dicom_level_resize(self):
    dcm = pydicom.dcmread(
        test_files.testdata_path('multiframe_camelyon_challenge_image.dcm')
    )
    path = _dicom_series_path(dcm)
    coordinates = [
        patch_coordinate.create_patch_coordinate(
            {'x_origin': 0, 'y_origin': 0},
            default_width=_DEBUG_SETTINGS.endpoint_input_width,
            default_height=_DEBUG_SETTINGS.endpoint_input_height,
        )
    ]
    store_path = str(path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mk_dicom_stores:
      mk_dicom_stores[store_path].add_instance(dcm)
      instance = data_accessor_definition.DicomWSIImage(
          credential_factory.TokenPassthroughCredentialFactory('mock_token'),
          str(path),
          {
              _InstanceJsonKeys.IMAGE_DIMENSIONS: dataclasses.asdict(
                  image_dimension_utils.ImageDimensions(
                      int(dcm.TotalPixelMatrixColumns // 3),
                      int(dcm.TotalPixelMatrixRows // 3),
                  )
              )
          },
          dcm.SOPInstanceUID,
          coordinates,
          [],
      )
      images = list(
          data_accessor.DicomDigitalPathologyData(
              instance, _DEBUG_SETTINGS
          ).data_iterator()
      )

      # check patch results are as expected.
      self.assertLen(images, 1)
      self.assertEqual(images[0].shape, (224, 224, 3))
      self.assertEqual(np.min(images[0]), 51)
      self.assertEqual(np.max(images[0]), 239)

  def test_dicom_patch_outside_level_dim(self):
    dcm = pydicom.dcmread(
        test_files.testdata_path('multiframe_camelyon_challenge_image.dcm')
    )
    path = _dicom_series_path(dcm)
    coordinates = [
        patch_coordinate.create_patch_coordinate(
            {'x_origin': 0, 'y_origin': 0},
            default_width=_DEBUG_SETTINGS.endpoint_input_width,
            default_height=_DEBUG_SETTINGS.endpoint_input_height,
        )
    ]
    store_path = str(path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mk_dicom_stores:
      mk_dicom_stores[store_path].add_instance(dcm)
      instance = data_accessor_definition.DicomWSIImage(
          credential_factory.TokenPassthroughCredentialFactory('mock_token'),
          str(path),
          {
              _InstanceJsonKeys.IMAGE_DIMENSIONS: dataclasses.asdict(
                  image_dimension_utils.ImageDimensions(
                      int(dcm.TotalPixelMatrixColumns // 7),
                      int(dcm.TotalPixelMatrixRows // 7),
                  )
              )
          },
          dcm.SOPInstanceUID,
          coordinates,
          [],
      )
      with self.assertRaisesRegex(
          data_accessor_errors.PatchOutsideOfImageDimensionsError,
          'Patch dimensions.*fall outside of DICOM level pyramid imaging'
          ' dimensions.*',
      ):
        list(
            data_accessor.DicomDigitalPathologyData(
                instance, _DEBUG_SETTINGS
            ).data_iterator()
        )

  def test_dicom_bits_allocated_not_8_raises(self):
    dcm = pydicom.dcmread(
        test_files.testdata_path('multiframe_camelyon_challenge_image.dcm')
    )
    dcm.BitsAllocated = 12
    path = _dicom_series_path(dcm)
    coordinates = [
        patch_coordinate.create_patch_coordinate(
            {'x_origin': 0, 'y_origin': 0},
            default_width=_DEBUG_SETTINGS.endpoint_input_width,
            default_height=_DEBUG_SETTINGS.endpoint_input_height,
        )
    ]
    store_path = str(path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mk_dicom_stores:
      mk_dicom_stores[store_path].add_instance(dcm)
      instance = data_accessor_definition.DicomWSIImage(
          credential_factory.TokenPassthroughCredentialFactory('mock_token'),
          str(path),
          {},
          dcm.SOPInstanceUID,
          coordinates,
          [],
      )
      with self.assertRaisesRegex(
          data_accessor_errors.DicomError,
          'DICOM contains instances with imaging bits allocated != 8',
      ):
        list(
            data_accessor.DicomDigitalPathologyData(
                instance, _DEBUG_SETTINGS
            ).data_iterator()
        )

  def test_dicom_icc_profile_correction_changes_pixel_values(self):
    dcm = pydicom.dcmread(
        test_files.testdata_path('multiframe_camelyon_challenge_image.dcm')
    )
    dcm.ICCProfile = dicom_slide.get_rommrgb_icc_profile_bytes()
    path = _dicom_series_path(dcm)
    coordinates = [
        patch_coordinate.create_patch_coordinate(
            {'x_origin': 0, 'y_origin': 0},
            default_width=_DEBUG_SETTINGS.endpoint_input_width,
            default_height=_DEBUG_SETTINGS.endpoint_input_height,
        )
    ]
    store_path = str(path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mk_dicom_stores:
      mk_dicom_stores[store_path].add_instance(dcm)
      instance = data_accessor_definition.DicomWSIImage(
          credential_factory.TokenPassthroughCredentialFactory('mock_token'),
          str(path),
          {_InstanceJsonKeys.TRANSFORM_IMAGING_TO_ICC_PROFILE: 'SRGB'},
          dcm.SOPInstanceUID,
          coordinates,
          [],
      )
      srgb_result = list(
          data_accessor.DicomDigitalPathologyData(
              instance, _DEBUG_SETTINGS
          ).data_iterator()
      )
      instance = data_accessor_definition.DicomWSIImage(
          credential_factory.TokenPassthroughCredentialFactory('mock_token'),
          str(path),
          {_InstanceJsonKeys.TRANSFORM_IMAGING_TO_ICC_PROFILE: 'ROMMRGB'},
          dcm.SOPInstanceUID,
          coordinates,
          [],
      )
      rommrgb_result = list(
          data_accessor.DicomDigitalPathologyData(
              instance, _DEBUG_SETTINGS
          ).data_iterator()
      )
      self.assertLen(srgb_result, 1)
      self.assertLen(rommrgb_result, 1)
      # color normalization changes pixel values
      self.assertGreater(np.max(np.abs(rommrgb_result[0] - srgb_result[0])), 0)

  def test_dicom_icc_profile_no_effect_of_correction_for_same_profile(self):
    dcm = pydicom.dcmread(
        test_files.testdata_path('multiframe_camelyon_challenge_image.dcm')
    )
    dcm.ICCProfile = dicom_slide.get_rommrgb_icc_profile_bytes()
    path = _dicom_series_path(dcm)
    coordinates = [
        patch_coordinate.create_patch_coordinate(
            {'x_origin': 0, 'y_origin': 0},
            default_width=_DEBUG_SETTINGS.endpoint_input_width,
            default_height=_DEBUG_SETTINGS.endpoint_input_height,
        )
    ]
    store_path = str(path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mk_dicom_stores:
      mk_dicom_stores[store_path].add_instance(dcm)
      instance = data_accessor_definition.DicomWSIImage(
          credential_factory.TokenPassthroughCredentialFactory('mock_token'),
          str(path),
          {},
          dcm.SOPInstanceUID,
          coordinates,
          [],
      )
      none_result = list(
          data_accessor.DicomDigitalPathologyData(
              instance, _DEBUG_SETTINGS
          ).data_iterator()
      )
      instance = data_accessor_definition.DicomWSIImage(
          credential_factory.TokenPassthroughCredentialFactory('mock_token'),
          str(path),
          {_InstanceJsonKeys.TRANSFORM_IMAGING_TO_ICC_PROFILE: 'ROMMRGB'},
          dcm.SOPInstanceUID,
          coordinates,
          [],
      )
      rommrgb_result = list(
          data_accessor.DicomDigitalPathologyData(
              instance, _DEBUG_SETTINGS
          ).data_iterator()
      )
      self.assertLen(none_result, 1)
      self.assertLen(rommrgb_result, 1)
      # no change in changes pixel values
      self.assertEqual(np.max(np.abs(rommrgb_result[0] - none_result[0])), 0)

  @parameterized.named_parameters([
      dict(testcase_name='no_profile_transform_defined', exensions={}),
      dict(
          testcase_name='none_transform',
          exensions={
              _InstanceJsonKeys.TRANSFORM_IMAGING_TO_ICC_PROFILE: 'NONE'
          },
      ),
  ])
  @mock.patch.object(icc_profile_cache, 'get_dicom_icc_profile', autospec=True)
  def test_dicom_icc_profile_not_called(self, mock_get_profile, exensions):
    dcm = pydicom.dcmread(
        test_files.testdata_path('multiframe_camelyon_challenge_image.dcm')
    )
    dcm.ICCProfile = dicom_slide.get_rommrgb_icc_profile_bytes()
    path = _dicom_series_path(dcm)
    coordinates = [
        patch_coordinate.create_patch_coordinate(
            {'x_origin': 0, 'y_origin': 0},
            default_width=_DEBUG_SETTINGS.endpoint_input_width,
            default_height=_DEBUG_SETTINGS.endpoint_input_height,
        )
    ]
    store_path = str(path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mk_dicom_stores:
      mk_dicom_stores[store_path].add_instance(dcm)
      instance = data_accessor_definition.DicomWSIImage(
          credential_factory.TokenPassthroughCredentialFactory('mock_token'),
          str(path),
          exensions,
          dcm.SOPInstanceUID,
          coordinates,
          [],
      )
      _ = list(
          data_accessor.DicomDigitalPathologyData(
              instance, _DEBUG_SETTINGS
          ).data_iterator()
      )
      mock_get_profile.assert_not_called()

  @parameterized.named_parameters([
      dict(testcase_name='monochrome_1', shape=(224, 224)),
      dict(testcase_name='monochrome_2', shape=(224, 224, 1)),
      dict(testcase_name='rgb', shape=(224, 224, 3)),
      dict(testcase_name='rgba', shape=(224, 224, 4)),
  ])
  def test_fetch_patch_bytes_norms_monochrome_images_to_three_channels(
      self, shape
  ):
    mem = np.zeros(shape=shape, dtype=np.uint8)
    mock_patch = mock.create_autospec(dicom_slide.DicomPatch, instance=True)
    mock_patch.image_bytes.return_value = mem
    self.assertEqual(
        data_accessor._fetch_image_bytes(mock_patch, None).shape,
        (224, 224, 3),
    )

  def test_validate_dicom_image_accessor_raises(self):
    with self.assertRaises(data_accessor_errors.UnapprovedDicomStoreError):
      data_accessor._validate_dicom_image_accessor(
          'http://test_bucket/google.png',
          dataclasses.replace(
              _DEBUG_SETTINGS,
              approved_dicom_stores=['http://abc', 'http://123'],
          ),
      )

  @parameterized.parameters(['http://abc/studies', 'http://123/studies'])
  def test_validate_dicom_image_accessor_valid(self, source):
    self.assertIsNone(
        data_accessor._validate_dicom_image_accessor(
            source,
            dataclasses.replace(
                _DEBUG_SETTINGS,
                approved_dicom_stores=['http://abc', 'http://123'],
            ),
        )
    )

  def test_validate_default_dicom_image_accessor_valid(self):
    self.assertIsNone(
        data_accessor._validate_dicom_image_accessor(
            'http://test_bucket/studies', _DEBUG_SETTINGS
        )
    )

  def test_is_accessor_data_embedded_in_request(self):
    with pydicom.dcmread(
        test_files.testdata_path('multiframe_camelyon_challenge_image.dcm')
    ) as dcm:
      path = _dicom_series_path(dcm)
      coordinates = [
          patch_coordinate.create_patch_coordinate(
              {'x_origin': 0, 'y_origin': 0},
              default_width=_DEBUG_SETTINGS.endpoint_input_width,
              default_height=_DEBUG_SETTINGS.endpoint_input_height,
          )
      ]
      instance = data_accessor_definition.DicomWSIImage(
          credential_factory.TokenPassthroughCredentialFactory('mock_token'),
          str(path),
          {},
          dcm.SOPInstanceUID,
          coordinates,
          [],
      )
      self.assertFalse(
          data_accessor.DicomDigitalPathologyData(
              instance, _DEBUG_SETTINGS
          ).is_accessor_data_embedded_in_request()
      )

  @parameterized.named_parameters(
      dict(
          testcase_name='no_patch_coordinates',
          coordinates=[],
          expected=1,
      ),
      dict(
          testcase_name='one_patch',
          coordinates=[
              patch_coordinate.create_patch_coordinate(
                  {'x_origin': 0, 'y_origin': 0},
                  default_width=_DEBUG_SETTINGS.endpoint_input_width,
                  default_height=_DEBUG_SETTINGS.endpoint_input_height,
              )
          ],
          expected=1,
      ),
      dict(
          testcase_name='two_patches',
          coordinates=[
              patch_coordinate.create_patch_coordinate(
                  {'x_origin': 0, 'y_origin': 0},
                  default_width=_DEBUG_SETTINGS.endpoint_input_width,
                  default_height=_DEBUG_SETTINGS.endpoint_input_height,
              ),
              patch_coordinate.create_patch_coordinate(
                  {'x_origin': 0, 'y_origin': 0},
                  default_width=_DEBUG_SETTINGS.endpoint_input_width,
                  default_height=_DEBUG_SETTINGS.endpoint_input_height,
              ),
          ],
          expected=2,
      ),
  )
  def test_accessor_length(self, coordinates, expected):
    with pydicom.dcmread(
        test_files.testdata_path('multiframe_camelyon_challenge_image.dcm')
    ) as dcm:
      path = _dicom_series_path(dcm)
      instance = data_accessor_definition.DicomWSIImage(
          credential_factory.TokenPassthroughCredentialFactory('mock_token'),
          str(path),
          {},
          dcm.SOPInstanceUID,
          coordinates,
          [],
      )
      self.assertLen(
          data_accessor.DicomDigitalPathologyData(instance, _DEBUG_SETTINGS),
          expected,
      )


if __name__ == '__main__':
  absltest.main()
