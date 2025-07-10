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

"""Tests for MedSigLIP predictor."""
import base64
import io
import os
import typing
from typing import Any, Mapping, Set, Union

from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
import numpy as np
import PIL.Image
import pydicom
import requests_mock

from serving.serving_framework import model_runner
from serving import predictor
from ez_wsi_dicomweb.test_utils.dicom_store_mock import dicom_store_mock
from ez_wsi_dicomweb.test_utils.gcs_mock import gcs_mock


def _round_embeddings(
    data: Mapping[str, Any], ndigits: int
) -> Mapping[str, Any]:
  """Round embeddings to ndigits."""
  result = {}
  for key, value in data.items():
    if key == 'embedding':
      result[key] = [round(x, ndigits) for x in value]
    elif isinstance(value, dict):
      result[key] = _round_embeddings(value, ndigits)
    elif isinstance(value, list):
      result[key] = [
          _round_embeddings(v, ndigits) if isinstance(v, dict) else v
          for v in value
      ]
    else:
      result[key] = value
  return result


def _pc(
    x_origin: int, y_origin: int, width: int = 224, height: int = 224
) -> Mapping[str, int]:
  """Return dictionary defining of patch coordinate."""
  return {
      'x_origin': x_origin,
      'y_origin': y_origin,
      'width': width,
      'height': height,
  }


class MockModelRunner:
  """Mock embedding, return mean for each channel in patch."""

  def run_model_multiple_output(
      self, mode_input: Mapping[str, np.ndarray], model_output_keys: Set[str]
  ) -> Mapping[str, Union[np.ndarray, float]]:
    """Compute and return mock embeddings."""
    result = {}
    pixels = mode_input[predictor._PIXEL_INPUT_KEY]
    if pixels.ndim != 4:
      raise ValueError(
          f'MockModelRunner: Unexpected pixel data shape: {pixels.ndim}.'
      )
    result[predictor._IMAGE_EMBEDS_KEY] = np.mean(pixels, axis=(2, 3))
    text_embeddings = mode_input[predictor._TOKENIZED_TEXT_INPUT_KEY]
    if text_embeddings.ndim != 2:
      raise ValueError(
          f'MockModelRunner: Unexpected textdata shape: {text_embeddings.ndim}.'
      )
    result[predictor._TEXT_EMBEDS_KEY] = np.asarray(
        [np.mean(text_embeddings, axis=1)]
    )
    # mock temperature and bias.
    result[predictor._SCALE_KEY] = np.asarray([1.0], dtype=np.float32)
    result[predictor._BIAS_KEY] = np.asarray([0.0], dtype=np.float32)
    if not result:
      raise ValueError('MockModelRunner called with no data.')
    return {k: v for k, v in result.items() if k in model_output_keys}


_mock_model_runner = typing.cast(model_runner.ModelRunner, MockModelRunner())

_MOCK_STORE_PATH = 'https://test_store'


def _read_test_path_dcm() -> pydicom.FileDataset:
  path = os.path.join(
      os.path.dirname(__file__),
      'testdata',
      'multiframe_camelyon_challenge_image.dcm',
  )
  return pydicom.dcmread(path)


def _read_test_cxr_dcm() -> pydicom.FileDataset:
  path = os.path.join(
      os.path.dirname(__file__), 'testdata', 'encapsulated_cxr.dcm'
  )
  return pydicom.dcmread(path)


def _read_test_jpeg() -> bytes:
  path = os.path.join(os.path.dirname(__file__), 'testdata', 'image.jpeg')
  with open(path, 'rb') as infile:
    return infile.read()


class DicomDigitalPathologyDataTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='batch_prediction',
          batch_prediction=True,
      ),
      dict(
          testcase_name='parallel_prediction',
          batch_prediction=False,
      ),
  )
  def test_path_dicom_prediction_embeddings(self, batch_prediction):
    dcm = _read_test_path_dcm()
    instance_path = f'{_MOCK_STORE_PATH}/studies/{dcm.StudyInstanceUID}/series/{dcm.SeriesInstanceUID}/instances/{dcm.SOPInstanceUID}'
    mock_prediction_input = {
        'instances': [
            {
                'image': {
                    'dicomweb_uri': instance_path,
                    'patch_coordinates': [_pc(0, 0), _pc(1, 1)],
                }
            },
            {
                'image': {
                    'dicomweb_uri': instance_path,
                    'patch_coordinates': [_pc(2, 2), _pc(3, 3)],
                }
            },
        ]
    }
    with dicom_store_mock.MockDicomStores(_MOCK_STORE_PATH) as dicom_store:
      dicom_store[_MOCK_STORE_PATH].add_instance(dcm)
      with flagsaver.flagsaver(batch_prediction=batch_prediction):
        pred = predictor.MedSiglipPredictor()
        result = pred.predict(mock_prediction_input, _mock_model_runner)
      self.assertEqual(pred.last_request_model_prediction_count, 4)
      self.assertEqual(
          _round_embeddings(result, 4),
          {
              'predictions': [
                  {
                      'model_temperature': 1.0,
                      'model_bias': 0.0,
                      'input_type': 'image',
                      'patch_embeddings': [
                          {
                              'embedding': [0.5502, 0.4308, 0.6527],
                              'patch_coordinate': _pc(0, 0),
                          },
                          {
                              'embedding': [0.5481, 0.4274, 0.6513],
                              'patch_coordinate': _pc(1, 1),
                          },
                      ],
                  },
                  {
                      'model_temperature': 1.0,
                      'model_bias': 0.0,
                      'input_type': 'image',
                      'patch_embeddings': [
                          {
                              'embedding': [0.5459, 0.424, 0.6498],
                              'patch_coordinate': _pc(2, 2),
                          },
                          {
                              'embedding': [0.5437, 0.4206, 0.6483],
                              'patch_coordinate': _pc(3, 3),
                          },
                      ],
                  },
              ]
          },
      )

  @parameterized.named_parameters(
      dict(
          testcase_name='batch_prediction',
          batch_prediction=True,
      ),
      dict(
          testcase_name='parallel_prediction',
          batch_prediction=False,
      ),
  )
  def test_jpeg_image_embeddings(self, batch_prediction):
    base64_jpeg = base64.b64encode(_read_test_jpeg()).decode('utf-8')
    mock_prediction_input = {
        'instances': [
            {'image': {'input_bytes': base64_jpeg}},
            {
                'image': {
                    'input_bytes': base64_jpeg,
                    'patch_coordinates': [_pc(2, 2, 10, 10), _pc(3, 3, 10, 10)],
                }
            },
        ]
    }
    with flagsaver.flagsaver(batch_prediction=batch_prediction):
      pred = predictor.MedSiglipPredictor()
      result = pred.predict(mock_prediction_input, _mock_model_runner)
    self.assertEqual(pred.last_request_model_prediction_count, 3)
    self.assertEqual(
        _round_embeddings(result, 4),
        {
            'predictions': [
                {
                    'model_temperature': 1.0,
                    'model_bias': 0.0,
                    'input_type': 'image',
                    'embedding': [0.2801, 0.2639, 0.2785],
                },
                {
                    'model_temperature': 1.0,
                    'model_bias': 0.0,
                    'input_type': 'image',
                    'patch_embeddings': [
                        {
                            'embedding': [1.0, 1.0, 1.0],
                            'patch_coordinate': _pc(2, 2, 10, 10),
                        },
                        {
                            'embedding': [1.0, 0.9984, 0.9992],
                            'patch_coordinate': _pc(3, 3, 10, 10),
                        },
                    ],
                },
            ]
        },
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='batch_prediction',
          batch_prediction=True,
      ),
      dict(
          testcase_name='parallel_prediction',
          batch_prediction=False,
      ),
  )
  def test_cxr_dicom_embeddings(self, batch_prediction):
    dcm = _read_test_cxr_dcm()
    instance_path = f'{_MOCK_STORE_PATH}/studies/{dcm.StudyInstanceUID}/series/{dcm.SeriesInstanceUID}/instances/{dcm.SOPInstanceUID}'
    mock_prediction_input = {
        'instances': [
            {
                'image': {
                    'dicomweb_uri': instance_path,
                    'patch_coordinates': [_pc(0, 0), _pc(1, 1)],
                }
            },
            {
                'image': {
                    'dicomweb_uri': instance_path,
                    'patch_coordinates': [_pc(2, 2), _pc(3, 3)],
                }
            },
        ]
    }
    with dicom_store_mock.MockDicomStores(_MOCK_STORE_PATH) as dicom_store:
      dicom_store[_MOCK_STORE_PATH].add_instance(dcm)
      with flagsaver.flagsaver(batch_prediction=batch_prediction):
        pred = predictor.MedSiglipPredictor()
        result = pred.predict(mock_prediction_input, _mock_model_runner)
      self.assertEqual(pred.last_request_model_prediction_count, 4)
      self.assertEqual(
          _round_embeddings(result, 4),
          {
              'predictions': [
                  {
                      'model_temperature': 1.0,
                      'model_bias': 0.0,
                      'input_type': 'image',
                      'patch_embeddings': [
                          {
                              'embedding': [-0.999, -0.999, -0.999],
                              'patch_coordinate': _pc(0, 0),
                          },
                          {
                              'embedding': [-0.9985, -0.9985, -0.9985],
                              'patch_coordinate': _pc(1, 1),
                          },
                      ],
                  },
                  {
                      'model_temperature': 1.0,
                      'model_bias': 0.0,
                      'input_type': 'image',
                      'patch_embeddings': [
                          {
                              'embedding': [-0.998, -0.998, -0.998],
                              'patch_coordinate': _pc(2, 2),
                          },
                          {
                              'embedding': [-0.9974, -0.9974, -0.9974],
                              'patch_coordinate': _pc(3, 3),
                          },
                      ],
                  },
              ]
          },
      )

  @parameterized.named_parameters(
      dict(
          testcase_name='batch_prediction',
          batch_prediction=True,
      ),
      dict(
          testcase_name='parallel_prediction',
          batch_prediction=False,
      ),
  )
  def test_text_prediction_embeddings(self, batch_prediction):
    mock_prediction_input = {
        'instances': [{'text': 'test_text_1'}, {'text': 'test_text_2'}]
    }
    with flagsaver.flagsaver(batch_prediction=batch_prediction):
      pred = predictor.MedSiglipPredictor()
      result = pred.predict(mock_prediction_input, _mock_model_runner)
    self.assertEqual(pred.last_request_model_prediction_count, 2)
    self.assertEqual(
        _round_embeddings(result, 4),
        {
            'predictions': [
                {
                    'model_temperature': 1.0,
                    'model_bias': 0.0,
                    'input_type': 'text',
                    'embedding': [155.0156],
                },
                {
                    'model_temperature': 1.0,
                    'model_bias': 0.0,
                    'input_type': 'text',
                    'embedding': [153.0469],
                },
            ]
        },
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='batch_prediction',
          batch_prediction=True,
      ),
      dict(
          testcase_name='parallel_prediction',
          batch_prediction=False,
      ),
  )
  def test_embedd_very_long_text_clips_input_text(self, batch_prediction):
    mock_prediction_input = {
        'instances': [{'text': f'{' '.join([str(i) for i in range(1000)])}'}]
    }
    with flagsaver.flagsaver(batch_prediction=batch_prediction):
      pred = predictor.MedSiglipPredictor()
      result = pred.predict(mock_prediction_input, _mock_model_runner)
    self.assertEqual(pred.last_request_model_prediction_count, 1)
    self.assertEqual(
        _round_embeddings(result, 4),
        {
            'predictions': [
                {
                    'model_temperature': 1.0,
                    'model_bias': 0.0,
                    'input_type': 'text',
                    'embedding': [3283.0312],
                },
            ]
        },
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='batch_prediction',
          batch_prediction=True,
          expected_count=3,
      ),
      dict(
          testcase_name='parallel_prediction',
          batch_prediction=False,
          expected_count=5,
      ),
  )
  def test_image_and_text_prediction_embeddings(
      self, batch_prediction, expected_count
  ):
    dcm = _read_test_path_dcm()
    instance_path = f'{_MOCK_STORE_PATH}/studies/{dcm.StudyInstanceUID}/series/{dcm.SeriesInstanceUID}/instances/{dcm.SOPInstanceUID}'
    with dicom_store_mock.MockDicomStores(_MOCK_STORE_PATH) as dicom_store:
      dicom_store[_MOCK_STORE_PATH].add_instance(dcm)
      mock_prediction_input = {
          'instances': [
              {'text': 'test_text_1'},
              {
                  'image': {
                      'dicomweb_uri': instance_path,
                      'patch_coordinates': [_pc(0, 0), _pc(1, 1)],
                  }
              },
              {'text': 'test_text_2'},
              {
                  'image': {
                      'dicomweb_uri': instance_path,
                  }
              },
          ]
      }
      with flagsaver.flagsaver(batch_prediction=batch_prediction):
        pred = predictor.MedSiglipPredictor()
        result = pred.predict(mock_prediction_input, _mock_model_runner)
      self.assertEqual(pred.last_request_model_prediction_count, expected_count)
      self.assertEqual(
          _round_embeddings(result, 4),
          {
              'predictions': [
                  {
                      'model_temperature': 1.0,
                      'model_bias': 0.0,
                      'input_type': 'text',
                      'embedding': [155.0156],
                  },
                  {
                      'model_temperature': 1.0,
                      'model_bias': 0.0,
                      'input_type': 'image',
                      'patch_embeddings': [
                          {
                              'embedding': [0.5502, 0.4308, 0.6527],
                              'patch_coordinate': _pc(0, 0),
                          },
                          {
                              'embedding': [0.5481, 0.4274, 0.6513],
                              'patch_coordinate': _pc(1, 1),
                          },
                      ],
                  },
                  {
                      'model_temperature': 1.0,
                      'model_bias': 0.0,
                      'input_type': 'text',
                      'embedding': [153.0469],
                  },
                  {
                      'model_temperature': 1.0,
                      'model_bias': 0.0,
                      'input_type': 'image',
                      'embedding': [-0.0118, -0.082, 0.0254],
                  },
              ]
          },
      )

  @parameterized.named_parameters(
      dict(
          testcase_name='buffered_images',
          mock_prediction_input={
              'instances': [
                  {'text': 'test_text_1'},
                  {'text': 'test_text_2'},
                  {
                      'image': {
                          'gcs_uri': 'gs://earth/test.dcm',
                          'patch_coordinates': [_pc(0, 0), _pc(1, 1)],
                      }
                  },
                  {
                      'image': {
                          'gcs_uri': 'gs://earth/test.dcm',
                      }
                  },
              ]
          },
          expected_result={
              'predictions': [
                  {
                      'model_temperature': 1.0,
                      'model_bias': 0.0,
                      'input_type': 'text',
                      'embedding': [155.0156],
                  },
                  {
                      'model_temperature': 1.0,
                      'model_bias': 0.0,
                      'input_type': 'text',
                      'embedding': [153.0469],
                  },
                  {
                      'model_temperature': 1.0,
                      'model_bias': 0.0,
                      'input_type': 'image',
                      'patch_embeddings': [
                          {
                              'embedding': [0.5502, 0.4308, 0.6527],
                              'patch_coordinate': _pc(0, 0),
                          },
                          {
                              'embedding': [0.5481, 0.4274, 0.6513],
                              'patch_coordinate': _pc(1, 1),
                          },
                      ],
                  },
                  {
                      'model_temperature': 1.0,
                      'model_bias': 0.0,
                      'input_type': 'image',
                      'embedding': [-0.0118, -0.082, 0.0254],
                  },
              ]
          },
      ),
      dict(
          testcase_name='buffered_text',
          mock_prediction_input={
              'instances': [
                  {
                      'image': {
                          'gcs_uri': 'gs://earth/test.dcm',
                          'patch_coordinates': [_pc(0, 0), _pc(1, 1)],
                      }
                  },
                  {
                      'image': {
                          'gcs_uri': 'gs://earth/test.dcm',
                      }
                  },
                  {
                      'text': 'test_text_1',
                  },
                  {
                      'text': 'test_text_2',
                  },
              ]
          },
          expected_result={
              'predictions': [
                  {
                      'model_temperature': 1.0,
                      'model_bias': 0.0,
                      'input_type': 'image',
                      'patch_embeddings': [
                          {
                              'embedding': [0.5502, 0.4308, 0.6527],
                              'patch_coordinate': _pc(0, 0),
                          },
                          {
                              'embedding': [0.5481, 0.4274, 0.6513],
                              'patch_coordinate': _pc(1, 1),
                          },
                      ],
                  },
                  {
                      'model_temperature': 1.0,
                      'model_bias': 0.0,
                      'input_type': 'image',
                      'embedding': [-0.0118, -0.082, 0.0254],
                  },
                  {
                      'model_temperature': 1.0,
                      'model_bias': 0.0,
                      'input_type': 'text',
                      'embedding': [155.0156],
                  },
                  {
                      'model_temperature': 1.0,
                      'model_bias': 0.0,
                      'input_type': 'text',
                      'embedding': [153.0469],
                  },
              ]
          },
      ),
  )
  @flagsaver.flagsaver(batch_prediction=True)
  def test_buffered_prediction_embeddings_results_are_ordered_correctly(
      self, mock_prediction_input, expected_result
  ):
    dcm = _read_test_path_dcm()
    temp_dir = self.create_tempdir()
    dcm.save_as(os.path.join(temp_dir.full_path, 'test.dcm'))
    with gcs_mock.GcsMock({'earth': temp_dir}):
      pred = predictor.MedSiglipPredictor()
      results = pred.predict(mock_prediction_input, _mock_model_runner)
      self.assertEqual(_round_embeddings(results, 4), expected_result)
      self.assertEqual(pred.last_request_model_prediction_count, 3)

  @parameterized.named_parameters(
      dict(
          testcase_name='batch_prediction',
          batch_prediction=True,
      ),
      dict(
          testcase_name='parallel_prediction',
          batch_prediction=False,
      ),
  )
  def test_data_accessor_error(self, batch_prediction):
    mock_prediction_input = {
        'instances': [
            {
                'image': {
                    'gcs_uri': 'gs://earth/test.dcm',
                }
            }
        ]
    }
    with gcs_mock.GcsMock():
      with flagsaver.flagsaver(batch_prediction=batch_prediction):
        pred = predictor.MedSiglipPredictor()
        result = pred.predict(mock_prediction_input, _mock_model_runner)
      result['predictions'][0]['error']['description'] = ''
      self.assertEqual(
          result,
          {
              'predictions': [
                  {'error': {'code': 'HTTP_ERROR', 'description': ''}}
              ],
          },
      )
      self.assertEqual(pred.last_request_model_prediction_count, 0)

  @parameterized.named_parameters(
      dict(
          testcase_name='request_is_not_a_dict',
          mock_input=[],
      ),
      dict(
          testcase_name='request_is_missing_instances',
          mock_input={},
      ),
      dict(
          testcase_name='instancees_not_a_list',
          mock_input={'instances': 'foo'},
      ),
      dict(
          testcase_name='instance_not_contain_dict',
          mock_input={'instances': ['foo']},
      ),
      dict(
          testcase_name='unrecognized_request_instance',
          mock_input={'instances': [{'foo': {}}]},
      ),
      dict(
          testcase_name='image_instance_is_not_a_dict',
          mock_input={'instances': [{'image': []}]},
      ),
      dict(
          testcase_name='unrecognized_image_instance',
          mock_input={'instances': [{'image': {'foo': {}}}]},
      ),
  )
  @flagsaver.flagsaver(batch_prediction=True)
  def test_invalid_instance_input_format(self, mock_input):
    pred = predictor.MedSiglipPredictor()
    result = pred.predict(mock_input, _mock_model_runner)
    self.assertEqual(
        result,
        {
            'error': 'INVALID_REQUEST_FIELD_ERROR',
        },
    )
    self.assertEqual(pred.last_request_model_prediction_count, 0)

  @flagsaver.flagsaver(batch_prediction=True, max_embeddings_per_request=3)
  def test_requested_embeddings_exceeds_max_per_request_returns_error(self):
    dcm = _read_test_path_dcm()
    instance_path = f'{_MOCK_STORE_PATH}/studies/{dcm.StudyInstanceUID}/series/{dcm.SeriesInstanceUID}/instances/{dcm.SOPInstanceUID}'
    mock_prediction_input = {
        'instances': [
            {
                'image': {
                    'dicomweb_uri': instance_path,
                    'patch_coordinates': [_pc(0, 0), _pc(1, 1)],
                }
            },
            {
                'image': {
                    'dicomweb_uri': instance_path,
                    'patch_coordinates': [_pc(2, 2), _pc(3, 3)],
                }
            },
        ]
    }
    with dicom_store_mock.MockDicomStores(_MOCK_STORE_PATH) as dicom_store:
      dicom_store[_MOCK_STORE_PATH].add_instance(dcm)
      pred = predictor.MedSiglipPredictor()
      result = pred.predict(mock_prediction_input, _mock_model_runner)
      self.assertEqual(pred.last_request_model_prediction_count, 0)
      self.assertEqual(result, {'error': 'TOO_MANY_PATCHES_ERROR'})

  @parameterized.named_parameters(
      dict(
          testcase_name='batch_prediction',
          batch_prediction=True,
      ),
      dict(
          testcase_name='parallel_prediction',
          batch_prediction=False,
      ),
  )
  def test_read_32bit_png_from_gcs(self, batch_prediction):
    with io.BytesIO(_read_test_jpeg()) as f:
      with PIL.Image.open(f) as img:
        img_bytes = np.asarray(img)
      img_shape = list(img_bytes.shape)
      img_shape[-1] = 4
      png_bytes = np.zeros(img_shape, dtype=np.uint8)
      png_bytes[:, :, :3] = img_bytes[...]
      png_bytes[:, :, 3] = 125
    temp_dir = self.create_tempdir()
    temp_png_path = os.path.join(temp_dir.full_path, 'test.png')
    mock_prediction_input = {
        'instances': [
            {'image': {'gcs_uri': 'gs://earth/test.png'}},
        ]
    }
    with PIL.Image.fromarray(png_bytes) as img:
      img.save(temp_png_path)
    with gcs_mock.GcsMock({'earth': temp_dir}):
      with flagsaver.flagsaver(batch_prediction=batch_prediction):
        pred = predictor.MedSiglipPredictor()
        result = pred.predict(mock_prediction_input, _mock_model_runner)
      self.assertEqual(
          _round_embeddings(result, 4),
          {
              'predictions': [{
                  'embedding': [0.2801, 0.2639, 0.2785],
                  'input_type': 'image',
                  'model_temperature': 1.0,
                  'model_bias': 0.0,
              }]
          },
      )
      self.assertEqual(pred.last_request_model_prediction_count, 1)

  @parameterized.named_parameters(
      dict(
          testcase_name='batch_prediction',
          batch_prediction=True,
      ),
      dict(
          testcase_name='parallel_prediction',
          batch_prediction=False,
      ),
  )
  def test_read_32bit_png_inline(self, batch_prediction):
    with io.BytesIO(_read_test_jpeg()) as f:
      with PIL.Image.open(f) as img:
        img_bytes = np.asarray(img)
      img_shape = list(img_bytes.shape)
      img_shape[-1] = 4
      png_bytes = np.zeros(img_shape, dtype=np.uint8)
      png_bytes[:, :, :3] = img_bytes[...]
      png_bytes[:, :, 3] = 125
    with io.BytesIO() as outfile:
      with PIL.Image.fromarray(png_bytes) as img:
        img.save(outfile, format='PNG')
      png_bytes = outfile.getvalue()
    mock_prediction_input = {
        'instances': [
            {
                'image': {
                    'input_bytes': base64.b64encode(png_bytes).decode('utf-8')
                }
            },
        ]
    }
    with flagsaver.flagsaver(batch_prediction=batch_prediction):
      pred = predictor.MedSiglipPredictor()
      result = pred.predict(mock_prediction_input, _mock_model_runner)
    self.assertEqual(
        _round_embeddings(result, 4),
        {
            'predictions': [{
                'embedding': [0.2801, 0.2639, 0.2785],
                'input_type': 'image',
                'model_temperature': 1.0,
                'model_bias': 0.0,
            }]
        },
    )
    self.assertEqual(pred.last_request_model_prediction_count, 1)

  @parameterized.named_parameters(
      dict(
          testcase_name='batch_prediction_from_url_string',
          batch_prediction=True,
          image_url='http://earth.com/image.jpeg',
      ),
      dict(
          testcase_name='parallel_prediction_from_url_string',
          batch_prediction=False,
          image_url='http://earth.com/image.jpeg',
      ),
      dict(
          testcase_name='batch_prediction_from_url_dict',
          batch_prediction=True,
          image_url={'url': 'http://earth.com/image.jpeg'},
      ),
      dict(
          testcase_name='parallel_prediction_from_url_dict',
          batch_prediction=False,
          image_url={'url': 'http://earth.com/image.jpeg'},
      ),
  )
  def test_http_image_prediction(self, batch_prediction, image_url):
    mock_prediction_input = {
        'instances': [
            {'image': {'image_url': image_url}},
        ]
    }
    with requests_mock.Mocker() as m:
      m.get('http://earth.com/image.jpeg', content=_read_test_jpeg())
      with flagsaver.flagsaver(batch_prediction=batch_prediction):
        pred = predictor.MedSiglipPredictor()
        result = pred.predict(mock_prediction_input, _mock_model_runner)
      self.assertEqual(
          _round_embeddings(result, 4),
          {
              'predictions': [{
                  'embedding': [0.2801, 0.2639, 0.2785],
                  'input_type': 'image',
                  'model_temperature': 1.0,
                  'model_bias': 0.0,
              }]
          },
      )

  @parameterized.named_parameters([
      dict(
          testcase_name='square_image',
          img=np.ones((1, 1, 3), dtype=np.uint8),
          expected=np.ones((1, 1, 3), dtype=np.uint8),
      ),
      dict(
          testcase_name='pad_height_one',
          img=np.ones((1, 2, 1), dtype=np.uint8),
          expected=np.asarray([[[1], [1]], [[0], [0]]], dtype=np.uint8),
      ),
      dict(
          testcase_name='pad_width_one',
          img=np.ones((2, 1, 1), dtype=np.uint8),
          expected=np.asarray([[[1], [0]], [[1], [0]]], dtype=np.uint8),
      ),
      dict(
          testcase_name='pad_height_two',
          img=np.ones((1, 3, 1), dtype=np.uint8),
          expected=np.asarray(
              [[[0], [0], [0]], [[1], [1], [1]], [[0], [0], [0]]],
              dtype=np.uint8,
          ),
      ),
      dict(
          testcase_name='pad_width_two',
          img=np.ones((3, 1, 1), dtype=np.uint8),
          expected=np.asarray(
              [[[0], [1], [0]], [[0], [1], [0]], [[0], [1], [0]]],
              dtype=np.uint8,
          ),
      ),
  ])
  def test_zero_pad_image_to_square(self, img, expected):
    np.testing.assert_array_equal(
        predictor._zero_pad_image_to_square(img), expected
    )


if __name__ == '__main__':
  absltest.main()
