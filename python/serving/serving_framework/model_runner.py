# Copyright 2024 Google LLC
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

"""Abstract base class for dependency injection of model handling.

Wraps execution of models on input tensors in an implementation-agnostic
interface. Provides a mixin method for batching model execution.
"""

import abc
from collections.abc import Mapping, Sequence, Set

import numpy as np


class ModelRunner(abc.ABC):
  """Runs a model with tensor inputs and outputs."""

  @abc.abstractmethod
  def run_model_multiple_output(
      self,
      model_input: Mapping[str, np.ndarray] | np.ndarray,
      *,
      model_name: str = "default",
      model_version: int | None = None,
      model_output_keys: Set[str],
  ) -> Mapping[str, np.ndarray]:
    """Runs a model on the given input and returns multiple outputs.

    Args:
      model_input: An array or map of arrays comprising the input tensors for
        the model. A bare array is given a default input key.
      model_name: The name of the model to run.
      model_version: The version of the model to run. Uses default if None.
      model_output_keys: The desired model output keys.

    Returns:
      A mapping of model output keys to tensors.
    """

  def run_model(
      self,
      model_input: Mapping[str, np.ndarray] | np.ndarray,
      *,
      model_name: str = "default",
      model_version: int | None = None,
      model_output_key: str = "output_0",
  ) -> np.ndarray:
    """Runs a model on the given input.

    Args:
      model_input: An array or map of arrays comprising the input tensors for
        the model.  A bare array is given a default input key.
      model_name: The name of the model to run.
      model_version: The version of the model to run. Uses default if None.
      model_output_key: The key to pull the output from. Defaults to "output_0".

    Returns:
      The single output tensor.
    """
    return self.run_model_multiple_output(
        model_input,
        model_name=model_name,
        model_version=model_version,
        model_output_keys={model_output_key},
    )[model_output_key]

  def batch_model(
      self,
      model_inputs: Sequence[Mapping[str, np.ndarray]] | Sequence[np.ndarray],
      *,
      model_name: str = "default",
      model_version: int | None = None,
      model_output_key: str = "output_0",
  ) -> list[np.ndarray]:
    """Runs a model on each of the given inputs.

    Args:
      model_inputs: A sequence of arrays or maps of arrays comprising the input
        tensors for the model.  Bare arrays are given a default input key.
      model_name: The name of the model to run.
      model_version: The version of the model to run. Uses default if None.
      model_output_key: The key to pull the output from. Defaults to "output_0".

    Returns:
      A list of the single output tensor from each input.
    """
    return [
        self.run_model(
            model_input,
            model_name=model_name,
            model_version=model_version,
            model_output_key=model_output_key,
        )
        for model_input in model_inputs
    ]

  def batch_model_multiple_output(
      self,
      model_inputs: Sequence[Mapping[str, np.ndarray]] | Sequence[np.ndarray],
      *,
      model_name: str = "default",
      model_version: int | None = None,
      model_output_keys: Set[str],
  ) -> list[Mapping[str, np.ndarray]]:
    """Runs a model on the given inputs and returns multiple outputs.

    Args:
      model_inputs: An array or map of arrays comprising the input tensors for
        the model. Bare arrays are given a default input key.
      model_name: The name of the model to run.
      model_version: The version of the model to run. Uses default if None.
      model_output_keys: The desired model output keys.

    Returns:
      A list containing the mapping of model output keys to tensors from each
      input.
    """
    return [
        self.run_model_multiple_output(
            model_input,
            model_name=model_name,
            model_version=model_version,
            model_output_keys=model_output_keys,
        )
        for model_input in model_inputs
    ]
