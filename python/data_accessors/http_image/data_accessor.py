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
"""Data accessor for generic DICOM images stored in a DICOM store."""

import contextlib
import os
import tempfile
from typing import Iterator, Sequence
import urllib.parse

import numpy as np
import requests

from data_accessors import abstract_data_accessor
from data_accessors import data_accessor_errors
from data_accessors.http_image import data_accessor_definition
from data_accessors.local_file_handlers import abstract_handler


def _download_http_image(
    stack: contextlib.ExitStack,
    instance: data_accessor_definition.HttpImage,
) -> str:
  """Downloads DICOM instance to a local file."""
  parsed_url = urllib.parse.urlparse(instance.url)
  temp_dir = stack.enter_context(tempfile.TemporaryDirectory())
  url_filename = os.path.basename(parsed_url.path)
  local_filename = os.path.join(temp_dir, url_filename)
  with open(local_filename, 'wb') as output_file:
    with requests.get(instance.url, stream=True) as r:
      r.raise_for_status()
      for chunk in r.iter_content(chunk_size=102400):
        output_file.write(chunk)
  return local_filename


def _get_http_image(
    instance: data_accessor_definition.HttpImage,
    file_handlers: Sequence[abstract_handler.AbstractHandler],
    local_file_path: str,
) -> Iterator[np.ndarray]:
  """Returns image patch bytes from DICOM series."""
  with contextlib.ExitStack() as stack:
    if not local_file_path:
      local_file_path = _download_http_image(stack, instance)
    for file_handler in file_handlers:
      processed = file_handler.process_file(
          instance.patch_coordinates,
          instance.base_request,
          local_file_path,
      )
      yield_result = False
      for data in processed:
        yield data
        yield_result = True
      if yield_result:
        return

  raise data_accessor_errors.UnhandledHttpFileError(
      'No file handler processed the files.'
  )


class HttpImageData(
    abstract_data_accessor.AbstractDataAccessor[
        data_accessor_definition.HttpImage, np.ndarray
    ]
):
  """Data accessor for generic DICOM images stored in a DICOM store."""

  def __init__(
      self,
      instance_class: data_accessor_definition.HttpImage,
      file_handlers: Sequence[abstract_handler.AbstractHandler],
  ):
    super().__init__(instance_class)
    self._file_handlers = file_handlers
    self._local_file_path = ''

  @contextlib.contextmanager
  def _reset_local_file_path(self, *args, **kwds):
    del args, kwds
    try:
      yield
    finally:
      self._local_file_path = ''

  def load_data(self, stack: contextlib.ExitStack) -> None:
    """Method pre-loads data prior to data_iterator.

    Required that context manger must exist for life time of data accesor
    iterator after data is loaded.

    Args:
     stack: contextlib.ExitStack to manage resources.

    Returns:
      None
    """
    if self._local_file_path:
      return
    self._local_file_path = _download_http_image(stack, self.instance)
    stack.enter_context(self._reset_local_file_path())

  def data_iterator(self) -> Iterator[np.ndarray]:
    return _get_http_image(
        self.instance, self._file_handlers, self._local_file_path
    )

  def is_accessor_data_embedded_in_request(self) -> bool:
    """Returns true if data is inline with request."""
    return False

  def __len__(self) -> int:
    """Returns number of data sets returned by iterator."""
    if self.instance.patch_coordinates:
      return len(self.instance.patch_coordinates)
    return 1
