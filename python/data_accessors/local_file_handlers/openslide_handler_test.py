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
"""Unit tests for openslide handler."""

from absl.testing import absltest
from absl.testing import parameterized
import cv2
import numpy as np
import openslide

from data_accessors import data_accessor_const
from data_accessors import data_accessor_errors
from data_accessors.local_file_handlers import openslide_handler
from data_accessors.utils import patch_coordinate
from data_accessors.utils import test_utils


_InstanceJsonKeys = data_accessor_const.InstanceJsonKeys
_openslide_handler = openslide_handler.OpenSlideHandler()


def openslide_data_path() -> str:
  return test_utils.testdata_path('openslide', 'ndpi_test.ndpi')


class OpenslideHandlerTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='level_-2',
          index=-2,
          expected_shape=(298, 400, 3),
      ),
      dict(
          testcase_name='level_-1',
          index=-1,
          expected_shape=(149, 200, 3),
      ),
      dict(
          testcase_name='level_6',
          index=6,
          expected_shape=(596, 800, 3),
      ),
      dict(
          testcase_name='level_7',
          index=7,
          expected_shape=(298, 400, 3),
      ),
  )
  def test_index_openslide_level_by_index(self, index, expected_shape):
    instance_json = {
        _InstanceJsonKeys.OPENSLIDE_PYRAMID_LEVEL: {
            _InstanceJsonKeys.OPENSLIDE_LEVEL_INDEX: index
        }
    }
    images = list(
        _openslide_handler.process_file(
            [], instance_json, openslide_data_path()
        )
    )
    self.assertLen(images, 1)
    self.assertEqual(images[0].shape, expected_shape)

  @parameterized.named_parameters(
      dict(
          testcase_name='smaller_than_smallest_level',
          level_shape=(100, 100),
          expected_shape=(149, 200, 3),
      ),
      dict(
          testcase_name='equal_to_smallest_level',
          level_shape=(298, 400),
          expected_shape=(298, 400, 3),
      ),
      dict(
          testcase_name='bigger_than_smallest_level',
          level_shape=(298, 401),
          expected_shape=(596, 800, 3),
      ),
  )
  def test_index_openslide_by_dim(self, level_shape, expected_shape):
    instance_json = {
        _InstanceJsonKeys.OPENSLIDE_PYRAMID_LEVEL: {
            _InstanceJsonKeys.OPENSLIDE_LEVEL_WIDTH_PX: level_shape[1],
            _InstanceJsonKeys.OPENSLIDE_LEVEL_HEIGHT_PX: level_shape[0],
        }
    }
    images = list(
        _openslide_handler.process_file(
            [], instance_json, openslide_data_path()
        )
    )
    self.assertLen(images, 1)
    self.assertEqual(images[0].shape, expected_shape)

  @parameterized.named_parameters(
      dict(
          testcase_name='larger_than_smallest_level',
          level_pixel_spacing=(0.13, 0.15),
          expected_shape=(149, 200, 3),
      ),
      dict(
          testcase_name='smaller_than_smallest_level',
          level_pixel_spacing=(0.10, 0.10),
          expected_shape=(298, 400, 3),
      ),
  )
  def test_index_openslide_by_pixel_spacing(
      self, level_pixel_spacing, expected_shape
  ):
    instance_json = {
        _InstanceJsonKeys.OPENSLIDE_PYRAMID_LEVEL: {
            _InstanceJsonKeys.OPENSLIDE_LEVEL_WIDTH_PIXEL_SPACING_MMP: (
                level_pixel_spacing[1]
            ),
            _InstanceJsonKeys.OPENSLIDE_LEVEL_HEIGHT_PIXEL_SPACING_MMP: (
                level_pixel_spacing[0]
            ),
        }
    }
    images = list(
        _openslide_handler.process_file(
            [], instance_json, openslide_data_path()
        )
    )
    self.assertLen(images, 1)
    self.assertEqual(images[0].shape, expected_shape)

  def test_get_patches(self):
    patch_coordinates = [
        patch_coordinate.PatchCoordinate(
            x_origin=0,
            y_origin=0,
            width=10,
            height=11,
        ),
        patch_coordinate.PatchCoordinate(
            x_origin=1,
            y_origin=2,
            width=10,
            height=11,
        ),
    ]
    instance_json = {
        _InstanceJsonKeys.OPENSLIDE_PYRAMID_LEVEL: {
            _InstanceJsonKeys.OPENSLIDE_LEVEL_INDEX: 8
        }
    }
    images = list(
        _openslide_handler.process_file(
            patch_coordinates, instance_json, openslide_data_path()
        )
    )
    self.assertLen(images, 2)
    with openslide.OpenSlide(openslide_data_path()) as slide:
      width, height = slide.level_dimensions[8]
      whole_level = np.asarray(slide.read_region((0, 0), 8, (width, height)))
      for i, img in enumerate(images):
        pc = patch_coordinates[i]
        expected = whole_level[
            pc.y_origin : pc.y_origin + pc.height,
            pc.x_origin : pc.x_origin + pc.width,
            0:3,
        ]
        np.testing.assert_array_equal(img, expected)

  def test_patch_outside_of_image_dimensions_raises_error(self):
    patch_coordinates = [
        patch_coordinate.PatchCoordinate(
            x_origin=0,
            y_origin=0,
            width=500,
            height=500,
        ),
    ]
    instance_json = {
        _InstanceJsonKeys.OPENSLIDE_PYRAMID_LEVEL: {
            _InstanceJsonKeys.OPENSLIDE_LEVEL_INDEX: 8
        },
    }
    with self.assertRaises(
        data_accessor_errors.PatchOutsideOfImageDimensionsError
    ):
      list(
          _openslide_handler.process_file(
              patch_coordinates, instance_json, openslide_data_path()
          )
      )

  def test_patch_outside_of_resized_image_dimensions_raises_error(self):
    patch_coordinates = [
        patch_coordinate.PatchCoordinate(
            x_origin=0,
            y_origin=0,
            width=11,
            height=11,
        ),
    ]
    instance_json = {
        _InstanceJsonKeys.OPENSLIDE_PYRAMID_LEVEL: {
            _InstanceJsonKeys.OPENSLIDE_LEVEL_INDEX: 8
        },
        _InstanceJsonKeys.EXTENSIONS: {
            _InstanceJsonKeys.IMAGE_DIMENSIONS: {
                _InstanceJsonKeys.WIDTH: 10,
                _InstanceJsonKeys.HEIGHT: 10,
            }
        },
    }
    with self.assertRaises(
        data_accessor_errors.PatchOutsideOfImageDimensionsError
    ):
      list(
          _openslide_handler.process_file(
              patch_coordinates, instance_json, openslide_data_path()
          )
      )

  def test_patch_outside_of_dim_disable_raise_error(self):
    patch_coordinates = [
        patch_coordinate.PatchCoordinate(
            x_origin=0,
            y_origin=0,
            width=11,
            height=11,
        ),
    ]
    instance_json = {
        _InstanceJsonKeys.OPENSLIDE_PYRAMID_LEVEL: {
            _InstanceJsonKeys.OPENSLIDE_LEVEL_INDEX: 8
        },
        _InstanceJsonKeys.EXTENSIONS: {
            _InstanceJsonKeys.IMAGE_DIMENSIONS: {
                _InstanceJsonKeys.WIDTH: 10,
                _InstanceJsonKeys.HEIGHT: 10,
            },
            _InstanceJsonKeys.REQUIRE_PATCHES_FULLY_IN_SOURCE_IMAGE: False,
        },
    }
    val = list(
        _openslide_handler.process_file(
            patch_coordinates, instance_json, openslide_data_path()
        )
    )
    self.assertLen(val, 1)
    self.assertEqual(val[0].shape, (11, 11, 3))

  @parameterized.named_parameters([
      dict(
          testcase_name='full_image',
          patch_coordinates=[
              patch_coordinate.PatchCoordinate(
                  x_origin=0,
                  y_origin=0,
                  width=100,
                  height=100,
              )
          ],
          max_mean_diff=0.0,
      ),
      dict(
          testcase_name='upper_left_corner',
          patch_coordinates=[
              patch_coordinate.PatchCoordinate(
                  x_origin=0,
                  y_origin=0,
                  width=10,
                  height=11,
              )
          ],
          max_mean_diff=0.09,
      ),
      dict(
          testcase_name='offset_patch',
          patch_coordinates=[
              patch_coordinate.PatchCoordinate(
                  x_origin=1,
                  y_origin=2,
                  width=10,
                  height=11,
              )
          ],
          max_mean_diff=0.09,
      ),
  ])
  def test_downsample_image(self, patch_coordinates, max_mean_diff):
    instance_json = {
        _InstanceJsonKeys.OPENSLIDE_PYRAMID_LEVEL: {
            _InstanceJsonKeys.OPENSLIDE_LEVEL_INDEX: 8
        },
        _InstanceJsonKeys.EXTENSIONS: {
            _InstanceJsonKeys.IMAGE_DIMENSIONS: {
                _InstanceJsonKeys.WIDTH: 100,
                _InstanceJsonKeys.HEIGHT: 100,
            }
        },
    }
    images = list(
        _openslide_handler.process_file(
            patch_coordinates, instance_json, openslide_data_path()
        )
    )
    self.assertLen(images, 1)
    with openslide.OpenSlide(openslide_data_path()) as slide:
      width, height = slide.level_dimensions[8]
      whole_level = np.asarray(slide.read_region((0, 0), 8, (width, height)))
      whole_level = cv2.resize(
          whole_level, (100, 100), interpolation=cv2.INTER_AREA
      )
      for i, img in enumerate(images):
        pc = patch_coordinates[i]
        expected = whole_level[
            pc.y_origin : pc.y_origin + pc.height,
            pc.x_origin : pc.x_origin + pc.width,
            0:3,
        ]
        self.assertLessEqual(
            np.mean(
                np.abs(img.astype(np.int32) - expected.astype(np.int32)).astype(
                    np.uint8
                )
            ),
            max_mean_diff,
        )

  @parameterized.named_parameters(
      dict(
          testcase_name='full_image',
          patch_coordinates=[
              patch_coordinate.PatchCoordinate(
                  x_origin=0,
                  y_origin=0,
                  width=300,
                  height=300,
              )
          ],
          max_mean_diff=0.001,
      ),
      dict(
          testcase_name='upper_left_corner',
          patch_coordinates=[
              patch_coordinate.PatchCoordinate(
                  x_origin=0,
                  y_origin=0,
                  width=10,
                  height=11,
              )
          ],
          max_mean_diff=0.2,
      ),
      dict(
          testcase_name='offset_patch',
          patch_coordinates=[
              patch_coordinate.PatchCoordinate(
                  x_origin=1,
                  y_origin=2,
                  width=10,
                  height=11,
              )
          ],
          max_mean_diff=0.1,
      ),
  )
  def test_upsample_image(self, patch_coordinates, max_mean_diff: float):
    instance_json = {
        _InstanceJsonKeys.OPENSLIDE_PYRAMID_LEVEL: {
            _InstanceJsonKeys.OPENSLIDE_LEVEL_INDEX: 8
        },
        _InstanceJsonKeys.EXTENSIONS: {
            _InstanceJsonKeys.IMAGE_DIMENSIONS: {
                _InstanceJsonKeys.WIDTH: 300,
                _InstanceJsonKeys.HEIGHT: 300,
            }
        },
    }
    images = list(
        _openslide_handler.process_file(
            patch_coordinates, instance_json, openslide_data_path()
        )
    )
    self.assertLen(images, 1)
    with openslide.OpenSlide(openslide_data_path()) as slide:
      width, height = slide.level_dimensions[8]
      whole_level = np.asarray(slide.read_region((0, 0), 8, (width, height)))
      whole_level = cv2.resize(
          whole_level, (300, 300), interpolation=cv2.INTER_CUBIC
      )
      for i, img in enumerate(images):
        pc = patch_coordinates[i]
        expected = whole_level[
            pc.y_origin : pc.y_origin + pc.height,
            pc.x_origin : pc.x_origin + pc.width,
            0:3,
        ]
        self.assertLess(
            np.mean(
                np.abs(img.astype(np.int32) - expected.astype(np.int32)).astype(
                    np.uint8
                )
            ),
            max_mean_diff,
        )

  def test_returned_patch_upper_left_corner_outside_of_dim_is_as_expected(self):
    patch_coordinates = [
        patch_coordinate.PatchCoordinate(
            x_origin=-10,
            y_origin=-10,
            width=20,
            height=20,
        ),
    ]
    instance_json = {
        _InstanceJsonKeys.OPENSLIDE_PYRAMID_LEVEL: {
            _InstanceJsonKeys.OPENSLIDE_LEVEL_INDEX: 8
        },
        _InstanceJsonKeys.EXTENSIONS: {
            _InstanceJsonKeys.REQUIRE_PATCHES_FULLY_IN_SOURCE_IMAGE: False,
        },
    }
    val = list(
        _openslide_handler.process_file(
            patch_coordinates, instance_json, openslide_data_path()
        )
    )
    self.assertLen(val, 1)
    self.assertEqual(val[0].shape, (20, 20, 3))
    expected = np.zeros((20, 20, 3), dtype=np.uint8)
    with openslide.OpenSlide(openslide_data_path()) as slide:
      whole_level = np.asarray(slide.read_region((0, 0), 8, (10, 10)))[
          :, :, 0:3
      ]
      expected[10:, 10:, :] = whole_level[...]
    np.testing.assert_array_equal(val[0], expected)

  def test_returned_patch_lower_right_corner_outside_of_dim_is_as_expected(
      self,
  ):
    patch_coordinates = [
        patch_coordinate.PatchCoordinate(
            x_origin=0,
            y_origin=0,
            width=300,
            height=200,
        ),
    ]
    instance_json = {
        _InstanceJsonKeys.OPENSLIDE_PYRAMID_LEVEL: {
            _InstanceJsonKeys.OPENSLIDE_LEVEL_INDEX: 8
        },
        _InstanceJsonKeys.EXTENSIONS: {
            _InstanceJsonKeys.REQUIRE_PATCHES_FULLY_IN_SOURCE_IMAGE: False,
        },
    }
    val = list(
        _openslide_handler.process_file(
            patch_coordinates, instance_json, openslide_data_path()
        )
    )
    self.assertLen(val, 1)
    self.assertEqual(val[0].shape, (200, 300, 3))
    expected = np.zeros((200, 300, 3), dtype=np.uint8)
    with openslide.OpenSlide(openslide_data_path()) as slide:
      whole_level = np.asarray(slide.read_region((0, 0), 8, (200, 149)))[
          :, :, 0:3
      ]
      expected[0:149, 0:200, :] = whole_level[...]
    np.testing.assert_array_equal(val[0], expected)

  def test_returned_patch_both_edges_outside_of_dim_is_as_expected(
      self,
  ):
    patch_coordinates = [
        patch_coordinate.PatchCoordinate(
            x_origin=-10,
            y_origin=-10,
            width=300,
            height=200,
        ),
    ]
    instance_json = {
        _InstanceJsonKeys.OPENSLIDE_PYRAMID_LEVEL: {
            _InstanceJsonKeys.OPENSLIDE_LEVEL_INDEX: 8
        },
        _InstanceJsonKeys.EXTENSIONS: {
            _InstanceJsonKeys.REQUIRE_PATCHES_FULLY_IN_SOURCE_IMAGE: False,
        },
    }
    val = list(
        _openslide_handler.process_file(
            patch_coordinates, instance_json, openslide_data_path()
        )
    )
    self.assertLen(val, 1)
    self.assertEqual(val[0].shape, (200, 300, 3))
    expected = np.zeros((200, 300, 3), dtype=np.uint8)
    with openslide.OpenSlide(openslide_data_path()) as slide:
      whole_level = np.asarray(slide.read_region((0, 0), 8, (200, 149)))[
          :, :, 0:3
      ]
      expected[10:159, 10:210, :] = whole_level[...]
    np.testing.assert_array_equal(val[0], expected)


if __name__ == '__main__':
  absltest.main()
