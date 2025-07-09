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

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from data_accessors import data_accessor_errors
from data_accessors.inline_bytes import data_accessor
from data_accessors.inline_bytes import data_accessor_definition
from data_accessors.utils import patch_coordinate as patch_coordinate_module
from serving import async_batch_predictor
from serving import embedding_response


class EmbeddingResponseTest(parameterized.TestCase):

  @parameterized.named_parameters([
      dict(
          testcase_name="no_embeddings",
          patch_coordiantes=[],
          embeddings=[],
          exp_msg=r"Request did not generate an embedding.",
      ),
      dict(
          testcase_name="multiple_embeddings_no_patch_coordinates",
          patch_coordiantes=[],
          embeddings=[np.asarray([1.0, 2.0]), np.asarray([13.0, 32.0])],
          exp_msg=r"Number of embeddings generated, 2, != 1",
      ),
      dict(
          testcase_name="returned_embeddings_does_not_match_patch_coord_count",
          patch_coordiantes=[
              patch_coordinate_module.PatchCoordinate(0, 0, 100, 100)
          ],
          embeddings=[np.asarray([1.0, 2.0]), np.asarray([13.0, 32.0])],
          exp_msg=(
              r"Number of embeddings generated, 2, does not match number of"
              r" patches, 1, in the request."
          ),
      ),
  ])
  def test_invalid_reponse_raises(self, patch_coordiantes, embeddings, exp_msg):
    instance = async_batch_predictor.DataAccessorEmbeddings[
        data_accessor_definition.InlineBytes, np.ndarray
    ](
        data_accessor.InlineBytesData(
            data_accessor_definition.InlineBytes(
                input_bytes=b"mock_bytes",
                base_request={},
                patch_coordinates=patch_coordiantes,
            ),
            [],
        ),
        embeddings,
    )
    with self.assertRaisesRegex(data_accessor_errors.InternalError, exp_msg):
      embedding_response.embedding_instance_response(
          mock.create_autospec(
              async_batch_predictor.AsyncBatchModelPredictor, instance=True
          ),
          instance,
      )


if __name__ == "__main__":
  absltest.main()
