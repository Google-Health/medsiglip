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
"""Async batch predictor for running inference on provided patches."""

from __future__ import annotations

import abc
import concurrent.futures
import contextlib
import dataclasses
import functools
from typing import Callable, Generic, Iterator, Sequence, TypeVar, Union

import numpy as np

from data_accessors import abstract_data_accessor
from data_accessors import data_accessor_errors
from serving.serving_framework import model_runner
from serving.logging_lib import cloud_logging_client

_InstanceInput = TypeVar('_InstanceInput')
_GetInstanceDataAsyncReturnType = TypeVar('_GetInstanceDataAsyncReturnType')


class InternalBugError(Exception):
  """Internal error capture exceptions which should never happen."""


@dataclasses.dataclass(frozen=True)
class DataAccessorEmbeddings(
    Generic[_InstanceInput, _GetInstanceDataAsyncReturnType]
):
  source: abstract_data_accessor.AbstractDataAccessor[
      _InstanceInput, _GetInstanceDataAsyncReturnType
  ]
  embeddings: Sequence[np.ndarray]


def _get_inst_data_map_func(
    get_inst_data: Callable[
        [_InstanceInput],
        _GetInstanceDataAsyncReturnType,
    ],
    requests: abstract_data_accessor.AbstractDataAccessor[
        _InstanceInput, _GetInstanceDataAsyncReturnType
    ],
) -> Union[
    data_accessor_errors.DataAccessorError,
    _GetInstanceDataAsyncReturnType,
]:
  """Returns retrieved data or exception."""
  try:
    return get_inst_data(requests)
  except data_accessor_errors.DataAccessorError as exp:
    return exp


class AsyncBatchModelPredictor(
    contextlib.ExitStack,
    Generic[_InstanceInput, _GetInstanceDataAsyncReturnType],
    metaclass=abc.ABCMeta,
):
  """Retrieves data and runs data across model predictor."""

  def __init__(self, threadpool_max_workers: int, thread_pool_timeout: int):
    super().__init__()
    self._thread_pool = None
    self._threadpool_max_workers = max(threadpool_max_workers, 1)
    self._thread_pool_timeout = thread_pool_timeout
    self._model_prediction_count = 0
    self._model_temperature = None
    self._model_bias = None

  @property
  def model_prediction_count(self) -> int:
    return self._model_prediction_count

  @property
  def model_temperature(self) -> Union[float, None]:
    return self._model_temperature

  @model_temperature.setter
  def model_temperature(self, temperature: float) -> None:
    self._model_temperature = temperature

  @property
  def model_bias(self) -> Union[float, None]:
    return self._model_bias

  @model_bias.setter
  def model_bias(self, bias: float) -> None:
    self._model_bias = bias

  def __enter__(self) -> AsyncBatchModelPredictor:
    self._thread_pool = self.enter_context(
        concurrent.futures.ThreadPoolExecutor(
            max_workers=self._threadpool_max_workers,
        )
    )
    return self

  @abc.abstractmethod
  def _run_model_predictor(
      self,
      model: model_runner.ModelRunner,
      images: Sequence[
          abstract_data_accessor.AbstractDataAccessor[
              _InstanceInput, _GetInstanceDataAsyncReturnType
          ]
      ],
  ) -> Iterator[np.ndarray]:
    """Runs model on seqeunce of images in size batches in and returns results."""

  def _get_single_instance_prediction(
      self,
      model: model_runner.ModelRunner,
      instance_data: abstract_data_accessor.AbstractDataAccessor[
          _InstanceInput, _GetInstanceDataAsyncReturnType
      ],
  ) -> Union[
      data_accessor_errors.DataAccessorError,
      DataAccessorEmbeddings[_InstanceInput, _GetInstanceDataAsyncReturnType],
  ]:
    """Return embeddings for single instance."""
    try:
      prediction = list(self._run_model_predictor(model, [instance_data]))
    except data_accessor_errors.DataAccessorError as exp:
      return exp
    return DataAccessorEmbeddings[
        _InstanceInput, _GetInstanceDataAsyncReturnType
    ](instance_data, prediction)

  def _get_instance_data_async(
      self,
      get_inst_data: Callable[
          [_InstanceInput],
          _GetInstanceDataAsyncReturnType,
      ],
      requests: Sequence[
          abstract_data_accessor.AbstractDataAccessor[
              _InstanceInput, _GetInstanceDataAsyncReturnType
          ]
      ],
  ) -> Iterator[
      Union[
          data_accessor_errors.DataAccessorError,
          _GetInstanceDataAsyncReturnType,
      ]
  ]:
    """Calls function in parallel to init each instance."""
    if not requests:
      return iter([])
    if self._thread_pool is None:
      msg = 'Must be run in context mangaged block.'
      cloud_logging_client.error(msg)
      raise InternalBugError(msg)
    if len(requests) == 1:
      return iter([_get_inst_data_map_func(get_inst_data, requests[0])])
    return self._thread_pool.map(
        functools.partial(_get_inst_data_map_func, get_inst_data),
        requests,
        timeout=self._thread_pool_timeout,
    )

  def _get_and_load_instance_data(
      self,
      context: contextlib.ExitStack,
      request: abstract_data_accessor.AbstractDataAccessor[
          _InstanceInput, _GetInstanceDataAsyncReturnType
      ],
  ) -> Union[
      data_accessor_errors.DataAccessorError,
      abstract_data_accessor.AbstractDataAccessor[
          _InstanceInput, _GetInstanceDataAsyncReturnType
      ],
  ]:
    """Get and load instance data in parallel."""
    try:
      request.load_data(context)
    except data_accessor_errors.DataAccessorError as exp:
      return exp
    return request

  def batch_predict_embeddings(
      self,
      model: model_runner.ModelRunner,
      requests: Sequence[
          abstract_data_accessor.AbstractDataAccessor[
              _InstanceInput, _GetInstanceDataAsyncReturnType
          ]
      ],
  ) -> Iterator[
      Union[
          data_accessor_errors.DataAccessorError,
          DataAccessorEmbeddings[
              _InstanceInput, _GetInstanceDataAsyncReturnType
          ],
      ]
  ]:
    """Get imaging in parallel and then run ML model in batch."""
    if not requests:
      return
    with contextlib.ExitStack() as stack:
      ds = list(
          self._get_instance_data_async(
              functools.partial(self._get_and_load_instance_data, stack),
              requests,
          )
      )
      all_embeddings = self._run_model_predictor(
          model,
          [
              img
              for img in ds
              if not isinstance(img, data_accessor_errors.DataAccessorError)
          ],
      )
      for request, img in zip(requests, ds):
        if isinstance(img, data_accessor_errors.DataAccessorError):
          yield img
          continue
        yield (
            DataAccessorEmbeddings[
                _InstanceInput, _GetInstanceDataAsyncReturnType
            ](img, [next(all_embeddings) for _ in range(len(request))])
        )

  def parallel_predict_embeddings(
      self, model: model_runner.ModelRunner, requests: Sequence[_InstanceInput]
  ) -> Iterator[
      Union[
          data_accessor_errors.DataAccessorError,
          DataAccessorEmbeddings[
              _InstanceInput, _GetInstanceDataAsyncReturnType
          ],
      ]
  ]:
    """Get imaging and execute model in parallel."""
    return self._get_instance_data_async(
        functools.partial(self._get_single_instance_prediction, model),
        requests,
    )
