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

"""Response dataclasses for Pete."""

import dataclasses
from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np

from data_accessors import data_accessor_errors
from serving import async_batch_predictor
from serving import predictor_const
from serving import predictor_data_types

_MAX_ERROR_DESCRIPTION_LENGTH = 1024


def _patch_embeddings(
    instance: async_batch_predictor.DataAccessorEmbeddings[
        predictor_data_types.EmbeddingInstance, np.ndarray
    ],
) -> Sequence[Mapping[str, Any]]:
  return [
      {
          predictor_const.EMBEDDING: embedding.tolist(),
          predictor_const.PATCH_COORDINATE: dataclasses.asdict(
              instance.source.instance.patch_coordinates[i]
          ),
      }
      for i, embedding in enumerate(instance.embeddings)
  ]


def _get_instance_type(
    instance: async_batch_predictor.DataAccessorEmbeddings[
        predictor_data_types.EmbeddingInstance, np.ndarray
    ],
) -> str:
  if isinstance(
      instance.source.instance, predictor_data_types.InlineTextInstance
  ):
    return predictor_const.TEXT_INPUT_TYPE
  else:
    return predictor_const.IMAGE_INPUT_TYPE


def _get_embedding(
    instance: async_batch_predictor.DataAccessorEmbeddings[
        predictor_data_types.EmbeddingInstance, np.ndarray
    ],
) -> MutableMapping[str, Any]:
  """Returns embeddings generated for a instance."""
  if not instance.embeddings:
    raise data_accessor_errors.InternalError(
        'Request did not generate an embedding.'
    )
  embedding_count = len(instance.embeddings)
  ds = instance.source.instance
  if ds.patch_coordinates:
    if embedding_count != len(ds.patch_coordinates):
      raise data_accessor_errors.InternalError(
          f'Number of embeddings generated, {embedding_count}, does not match'
          f' number of patches, {len(ds.patch_coordinates)}, in the request.'
      )
    return {
        predictor_const.INPUT_TYPE: _get_instance_type(instance),
        predictor_const.PATCH_EMBEDDINGS: _patch_embeddings(instance),
    }
  if embedding_count != 1:
    raise data_accessor_errors.InternalError(
        f'Number of embeddings generated, {embedding_count}, != 1'
    )
  return {
      predictor_const.EMBEDDING: instance.embeddings[0].tolist(),
      predictor_const.INPUT_TYPE: _get_instance_type(instance),
  }


def embedding_instance_response(
    predictor: async_batch_predictor.AsyncBatchModelPredictor,
    instance: async_batch_predictor.DataAccessorEmbeddings[
        predictor_data_types.EmbeddingInstance, np.ndarray
    ],
) -> Mapping[str, Any]:
  """Returns a JSON-serializable embedding instance responses."""
  result = _get_embedding(instance)
  result[predictor_const.MODEL_TEMPERATURE] = predictor.model_temperature
  result[predictor_const.MODEL_BIAS] = predictor.model_bias
  return result


def instance_error_response(
    ds_error: data_accessor_errors.DataAccessorError,
) -> Mapping[str, Any]:
  error = {predictor_const.ERROR_CODE: ds_error.error_code.value}
  if ds_error.api_description:
    error[predictor_const.ERROR_CODE_DESCRIPTION] = ds_error.api_description[
        :_MAX_ERROR_DESCRIPTION_LENGTH
    ]
  return {predictor_const.ERROR: error}


def prediction_error_response(
    ds_error: data_accessor_errors.DataAccessorError,
) -> Mapping[str, Any]:
  return {predictor_const.VERTEXAI_ERROR: ds_error.error_code.value}
