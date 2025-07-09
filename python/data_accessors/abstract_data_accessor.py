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
"""Defines abstract representation for accessing data referenced in a request."""

import abc
import contextlib
from typing import Generic, Iterator, TypeVar

InstanceDataClass = TypeVar('InstanceDataClass')
InstanceDataType = TypeVar('InstanceDataType')


class AbstractDataAccessor(
    Generic[InstanceDataClass, InstanceDataType], metaclass=abc.ABCMeta
):
  """Defines abstract representation of data accessor, imaging and embeddings."""

  def __init__(
      self,
      instance: InstanceDataClass,
  ):
    self._instance = instance

  @property
  def instance(self) -> InstanceDataClass:
    return self._instance

  @abc.abstractmethod
  def load_data(self, stack: contextlib.ExitStack) -> None:
    """Method pre-loads data prior to data_iterator.

    Required that context manger must exist for life time of data accesor
    iterator after data is loaded.

    Args:
     stack: contextlib.ExitStack to manage resources.

    Returns:
      None
    """

  @abc.abstractmethod
  def is_accessor_data_embedded_in_request(self) -> bool:
    """Returns true if data is embedded inline within the request."""

  @abc.abstractmethod
  def __len__(self) -> int:
    """Returns number of data sets returned by iterator."""

  @abc.abstractmethod
  def data_iterator(self) -> Iterator[InstanceDataType]:
    """Returns iterator of data."""
