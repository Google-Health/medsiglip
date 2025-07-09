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
"""abstract handler for processing local data files."""

import abc
import io
from typing import Any, Iterator, Mapping, Sequence, Union

import numpy as np

from data_accessors import data_accessor_const
from data_accessors.utils import json_validation_utils
from data_accessors.utils import patch_coordinate

_InstanceJsonKeys = data_accessor_const.InstanceJsonKeys


class AbstractHandler(metaclass=abc.ABCMeta):
  """Abstract class for handling files read from GCS."""

  @abc.abstractmethod
  def process_file(
      self,
      instance_patch_coordinates: Sequence[patch_coordinate.PatchCoordinate],
      base_request: Mapping[str, Any],
      file_path: Union[str, io.BytesIO],
  ) -> Iterator[np.ndarray]:
    """Return processed data from files or None if not processed."""


def get_base_request_extensions(
    base_request: Mapping[str, Any],
) -> Mapping[str, Any]:
  return json_validation_utils.validate_str_key_dict(
      base_request.get(
          _InstanceJsonKeys.EXTENSIONS,
          {},
      )
  )
