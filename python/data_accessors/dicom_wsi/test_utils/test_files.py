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

"""test_util for mi_siglip."""
from data_accessors.utils import test_utils


def testdata_path(*args: str) -> str:
  base_path = ['wsi']
  base_path.extend(args)
  return test_utils.testdata_path(*base_path)


def test_multi_frame_dicom_instance_path() -> str:
  return testdata_path('multiframe_camelyon_challenge_image.dcm')
