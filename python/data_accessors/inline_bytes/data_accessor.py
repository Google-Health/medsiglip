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
"""Data accessor for bytes passed inline in a request."""

import contextlib
import io
from typing import Iterator, Sequence

import numpy as np

from data_accessors import abstract_data_accessor
from data_accessors import data_accessor_errors
from data_accessors.inline_bytes import data_accessor_definition
from data_accessors.local_file_handlers import abstract_handler


def _get_generic_bytes_images(
    file_handlers: Sequence[abstract_handler.AbstractHandler],
    instance: data_accessor_definition.InlineBytes,
) -> Iterator[np.ndarray]:
  """Returns image patch bytes from DICOM series."""
  with io.BytesIO(instance.input_bytes) as input_bytes:
    for file_handler in file_handlers:
      input_bytes.seek(0)
      processed = file_handler.process_file(
          instance.patch_coordinates, instance.base_request, input_bytes
      )
      yield_result = False
      for data in processed:
        yield data
        yield_result = True
      if yield_result:
        return
  raise data_accessor_errors.UnhandledGenericBytesError(
      'No file handler processed the bytes.'
  )


class InlineBytesData(
    abstract_data_accessor.AbstractDataAccessor[
        data_accessor_definition.InlineBytes, np.ndarray
    ]
):
  """Data accessor for bytes passed inline in a request."""

  def __init__(
      self,
      instance_class: data_accessor_definition.InlineBytes,
      file_handlers: Sequence[abstract_handler.AbstractHandler],
  ):
    super().__init__(instance_class)
    self._file_handlers = file_handlers

  def data_iterator(self) -> Iterator[np.ndarray]:
    return _get_generic_bytes_images(self._file_handlers, self.instance)

  def is_accessor_data_embedded_in_request(self) -> bool:
    return True

  def load_data(self, stack: contextlib.ExitStack) -> None:
    """Method pre-loads data prior to data_iterator."""
    return

  def __len__(self) -> int:
    """Returns number of data sets returned by iterator."""
    if self.instance.patch_coordinates:
      return len(self.instance.patch_coordinates)
    return 1
