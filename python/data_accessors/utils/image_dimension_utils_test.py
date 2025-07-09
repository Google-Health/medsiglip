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
"""Tests for image dimensions utils for data accessors."""

from absl.testing import absltest
from absl.testing import parameterized

from data_accessors import data_accessor_const
from data_accessors import data_accessor_errors
from data_accessors.utils import image_dimension_utils

_InstanceJsonKeys = data_accessor_const.InstanceJsonKeys


class ImageDimensionUtilsTest(parameterized.TestCase):

  def test_get_resize_image_dimensions_succeeds(self):
    val = {
        _InstanceJsonKeys.IMAGE_DIMENSIONS: {
            _InstanceJsonKeys.WIDTH: 10,
            _InstanceJsonKeys.HEIGHT: 20,
        }
    }
    test = image_dimension_utils.get_resize_image_dimensions(val)
    self.assertEqual((test.width, test.height), (10, 20))  # pytype: disable=attribute-error

  def test_get_resize_image_dimensions_returns_none(self):
    self.assertIsNone(image_dimension_utils.get_resize_image_dimensions({}))

  @parameterized.named_parameters([
      dict(
          testcase_name='dict_ref_int',
          val={_InstanceJsonKeys.IMAGE_DIMENSIONS: 1},
      ),
      dict(
          testcase_name='dict_ref_list',
          val={_InstanceJsonKeys.IMAGE_DIMENSIONS: [1, 2, 3]},
      ),
      dict(
          testcase_name='dict_ref_string_invalid_json',
          val={_InstanceJsonKeys.IMAGE_DIMENSIONS: '1,2,3'},
      ),
      dict(
          testcase_name='dict_ref_string_missing_value',
          val={_InstanceJsonKeys.IMAGE_DIMENSIONS: '{}'},
      ),
      dict(
          testcase_name='dict_ref_string_json_list',
          val={_InstanceJsonKeys.IMAGE_DIMENSIONS: '[1,2,3]'},
      ),
      dict(
          testcase_name='dict_ref_width_only',
          val={
              _InstanceJsonKeys.IMAGE_DIMENSIONS: {_InstanceJsonKeys.WIDTH: 1}
          },
      ),
      dict(
          testcase_name='dict_ref_height_only',
          val={
              _InstanceJsonKeys.IMAGE_DIMENSIONS: {_InstanceJsonKeys.HEIGHT: 1}
          },
      ),
      dict(
          testcase_name='dict_ref_width_zero',
          val={
              _InstanceJsonKeys.IMAGE_DIMENSIONS: {
                  _InstanceJsonKeys.WIDTH: 0,
                  _InstanceJsonKeys.HEIGHT: 1,
              }
          },
      ),
      dict(
          testcase_name='dict_ref_height_zero',
          val={
              _InstanceJsonKeys.IMAGE_DIMENSIONS: {
                  _InstanceJsonKeys.WIDTH: 1,
                  _InstanceJsonKeys.HEIGHT: 0,
              }
          },
      ),
      dict(
          testcase_name='dict_ref_width_minus_one',
          val={
              _InstanceJsonKeys.IMAGE_DIMENSIONS: {
                  _InstanceJsonKeys.WIDTH: -1,
                  _InstanceJsonKeys.HEIGHT: 1,
              }
          },
      ),
      dict(
          testcase_name='dict_ref_height_minus_one',
          val={
              _InstanceJsonKeys.IMAGE_DIMENSIONS: {
                  _InstanceJsonKeys.WIDTH: 1,
                  _InstanceJsonKeys.HEIGHT: -1,
              }
          },
      ),
  ])
  def test_get_resize_image_dimensions_raises(self, val):
    with self.assertRaises(data_accessor_errors.InvalidRequestFieldError):
      image_dimension_utils.get_resize_image_dimensions(val)

  @parameterized.parameters([(10, 20), (20, 10), (20, 20)])
  def test_get_resize_image_dimensions_raises_if_exceeds_max(
      self, width, height
  ):
    val = {
        _InstanceJsonKeys.IMAGE_DIMENSIONS: (
            '{"width": %d, "height": %d}' % (width, height)
        )
    }
    with self.assertRaises(data_accessor_errors.InvalidRequestFieldError):
      image_dimension_utils.get_resize_image_dimensions(val, 15)

  def test_get_resize_image_dimensions_does_not_raises_if_at_max(self):
    val = {
        _InstanceJsonKeys.IMAGE_DIMENSIONS: {
            _InstanceJsonKeys.WIDTH: 15,
            _InstanceJsonKeys.HEIGHT: 10,
        }
    }
    dim = image_dimension_utils.get_resize_image_dimensions(val, 15)
    self.assertEqual((dim.width, dim.height), (15, 10))  # pytype: disable=attribute-error


if __name__ == '__main__':
  absltest.main()
