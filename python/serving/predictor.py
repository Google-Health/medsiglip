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

"""Generates embeddings for text and imaging data."""
import abc
import dataclasses
import time
import typing
from typing import Any, Generic, Iterator, Mapping, MutableMapping, Optional, Sequence, TypeVar, Union

import numpy as np
from transformers.models import siglip

from data_accessors import abstract_data_accessor
from data_accessors import data_accessor_errors
from data_accessors.dicom_generic import data_accessor as dicom_generic_data_accessor
from data_accessors.dicom_generic import data_accessor_definition as dicom_generic_data_accessor_definition
from data_accessors.dicom_wsi import configuration as dicom_wsi_configuration
from data_accessors.dicom_wsi import data_accessor as dicom_wsi_data_accessor
from data_accessors.dicom_wsi import data_accessor_definition as dicom_wsi_data_accessor_definition
from data_accessors.gcs_generic import data_accessor as gcs_generic_data_accessor
from data_accessors.gcs_generic import data_accessor_definition as gcs_generic_data_accessor_definition
from data_accessors.http_image import data_accessor as http_image_data_accessor
from data_accessors.http_image import data_accessor_definition as http_image_data_accessor_definition
from data_accessors.inline_bytes import data_accessor as inline_bytes_data_accessor
from data_accessors.inline_bytes import data_accessor_definition as inline_bytes_data_accessor_definition
from data_accessors.inline_text import data_accessor as inline_text_data_accessor
from data_accessors.inline_text import data_accessor_definition as inline_text_data_accessor_definition
from data_accessors.local_file_handlers import generic_dicom_handler
from data_accessors.local_file_handlers import openslide_handler
from data_accessors.local_file_handlers import traditional_image_handler
from data_accessors.local_file_handlers import wsi_dicom_handler
from data_accessors.utils import authentication_utils
from data_accessors.utils import dicom_source_utils
from serving.serving_framework import model_runner
from pre_processor_configs import config_utils
from serving import async_batch_predictor
from serving import embedding_response
from serving import flags
from serving import predictor_const
from serving import predictor_data_types
from serving.logging_lib import cloud_logging_client

_MAX_TEXT_TOKEN_COUNT = 64

# Siglip image preprocessor model input dimensions.
_WIDTH = 'width'
_HEIGHT = 'height'

# Siglip image preprocessor data output keys.
_PIXEL_VALUES = 'pixel_values'
_INPUT_IDS = 'input_ids'

# Model input dictionary keys
_TOKENIZED_TEXT_INPUT_KEY = 'input_ids__0'
_PIXEL_INPUT_KEY = 'pixel_values__1'

# Model output dictionary keys
_TEXT_EMBEDS_KEY = 'text_embeds__0'
_IMAGE_EMBEDS_KEY = 'image_embeds__1'
_SCALE_KEY = 'logit_scale__2'
_BIAS_KEY = 'logit_bias__3'
_MODEL_OUTPUT_KEYS = frozenset([
    _TEXT_EMBEDS_KEY,
    _IMAGE_EMBEDS_KEY,
    _SCALE_KEY,
    _BIAS_KEY,
])
_IMAGE_EMBEDDING_INDEX = 'image_embedding_index'
_TEXT_EMBEDDING_INDEX = 'text_embedding_index'
_ZERO_DIM_ARRAY = np.zeros((0,), dtype=np.uint8)

_LOCAL_FILE_HANDLERS = [
    generic_dicom_handler.GenericDicomHandler(),
    traditional_image_handler.TraditionalImageHandler(),
    openslide_handler.OpenSlideHandler(),
    wsi_dicom_handler.WsiDicomHandler(),
]
_GCS_DOWNLOAD_THREAD_COUNT = int(2)


_siglip_image_processor: Optional[siglip.SiglipImageProcessor] = None


def _get_siglip_image_processor() -> siglip.SiglipImageProcessor:
  """Returns a SiglipImageProcessor for the endpoint."""
  # TODO: b/414473350 - Change from flags to config file init. Model parameters
  # likely very simiar to huggingface defaults with changes for modelinput size.
  # https://huggingface.co/google/med-siglip-448-staging/blob/main/preprocessor_config.json
  global _siglip_image_processor
  if _siglip_image_processor is None:
    # Serving stack rescales inputs 0..1 to support image inputs with both 8 &
    # 16 bits per channel.
    _siglip_image_processor = siglip.SiglipImageProcessor(
        do_rescale=False,
        size={
            _HEIGHT: flags.MODEL_INPUT_HEIGHT_FLAG.value,
            _WIDTH: flags.MODEL_INPUT_WIDTH_FLAG.value,
        },
    )
  return _siglip_image_processor


_siglip_tokenizer: Optional[siglip.SiglipTokenizer] = None


def _run_siglip_tokenizer(text: str, truncation: bool) -> Mapping[str, Any]:
  global _siglip_tokenizer
  if _siglip_tokenizer is None:
    _siglip_tokenizer = siglip.SiglipTokenizer.from_pretrained(
        config_utils.config_directory()
    )
  return _siglip_tokenizer(
      text,
      return_tensors='np',
      padding='max_length',
      max_length=_MAX_TEXT_TOKEN_COUNT,
      truncation=truncation,
  )


def _tokenize_text(text: str) -> Mapping[str, Any]:
  """Returns tokenize text."""
  tokens = _run_siglip_tokenizer(text, False)
  token_count = tokens[_INPUT_IDS].shape[-1]
  if token_count <= _MAX_TEXT_TOKEN_COUNT:
    return tokens
  msg = text if len(text) <= 256 else f'{text[:256]}...'
  cloud_logging_client.warning(
      f'Input text: {msg} generated {token_count} tokens. Tokens'
      f' trucated to {_MAX_TEXT_TOKEN_COUNT}.'
  )
  # Unclear how token are a fixed size and if coud truncate simply by clipping
  # the token vector.
  return _run_siglip_tokenizer(text, True)


_PAD_TEXT_TOKENS: Optional[np.ndarray] = None
_PAD_INPUT_IMAGE: Optional[np.ndarray] = None


def _get_pad_text_tokens() -> np.ndarray:
  """Returns a text tokens to use for padding text input."""
  global _PAD_TEXT_TOKENS
  if _PAD_TEXT_TOKENS is None:
    _PAD_TEXT_TOKENS = _tokenize_text('')[_INPUT_IDS]
  return _PAD_TEXT_TOKENS


def _get_pad_input_image() -> np.ndarray:
  """Returns a input image to use for padding image input."""
  global _PAD_INPUT_IMAGE
  if _PAD_INPUT_IMAGE is None:
    # shape: (batch, channel, height, width)
    _PAD_INPUT_IMAGE = np.zeros(
        (
            1,
            3,
            flags.MODEL_INPUT_HEIGHT_FLAG.value,
            flags.MODEL_INPUT_WIDTH_FLAG.value,
        ),
        dtype=np.float32,
    )
  return _PAD_INPUT_IMAGE


@dataclasses.dataclass(frozen=True)
class _ModelInput:
  """model input."""

  index: int  # index of embedding request
  data: Mapping[str, np.ndarray]  # data to be used as model input


def _call_model(
    model: model_runner.ModelRunner,
    img_list: Sequence[_ModelInput],
    token_list: Sequence[_ModelInput],
) -> Mapping[str, Union[np.ndarray, Sequence[int]]]:
  """Calls siglip image embedding model."""
  model_input = {}
  # shape: (batch, channel, height, width)
  images = [img.data[_PIXEL_VALUES] for img in img_list]
  pad_image_count = (
      flags.IMAGE_EMBEDDINGS_PER_BATCH_PREDICTION_FLAG.value - len(images)
  )
  if pad_image_count > 0:
    images.extend([_get_pad_input_image()] * pad_image_count)
  model_input[_PIXEL_INPUT_KEY] = np.concatenate(images, axis=0)

  text_token_list = [token.data[_INPUT_IDS] for token in token_list]
  pad_text_count = flags.TEXT_EMBEDDINGS_PER_BATCH_PREDICTION_FLAG.value - len(
      text_token_list
  )
  if pad_text_count > 0:
    text_token_list.extend([_get_pad_text_tokens()] * pad_text_count)
  model_input[_TOKENIZED_TEXT_INPUT_KEY] = np.concatenate(
      text_token_list, axis=0
  )
  model_output = model.run_model_multiple_output(
      model_input, model_output_keys=_MODEL_OUTPUT_KEYS
  )
  model_output = dict(model_output)
  # Add returned embedding request indexs to output.
  model_output[_IMAGE_EMBEDDING_INDEX] = [img.index for img in img_list]
  model_output[_TEXT_EMBEDDING_INDEX] = [token.index for token in token_list]
  if pad_image_count == 0 and pad_text_count == 0:
    return model_output
  if pad_image_count > 0:
    model_output[_IMAGE_EMBEDS_KEY] = model_output[_IMAGE_EMBEDS_KEY][
        :-pad_image_count, ...
    ]

  if pad_text_count > 0:
    model_output[_TEXT_EMBEDS_KEY] = model_output[_TEXT_EMBEDS_KEY][
        :-pad_text_count, ...
    ]
  return model_output


def _validate_instance_list(json_metadata: Mapping[str, Any]) -> Sequence[Any]:
  if not isinstance(json_metadata, dict):
    raise data_accessor_errors.InvalidRequestFieldError(
        'Request is not a dict.'
    )
  val = json_metadata.get(predictor_const.INSTANCES)
  if isinstance(val, list):
    return val
  raise data_accessor_errors.InvalidRequestFieldError(
      'Invalid input, missing expected'
      f' key: {predictor_const.INSTANCES} and associated list of values.'
  )


def _parse_image_instance(
    config: dicom_wsi_configuration.ConfigurationSettings,
    instance: Mapping[str, Any],
) -> abstract_data_accessor.AbstractDataAccessor[
    predictor_data_types.EmbeddingInstance, np.ndarray
]:
  """Parses image instance request from json."""
  if not isinstance(instance, dict):
    raise data_accessor_errors.InvalidRequestFieldError(
        'Request image instance is not a dict.'
    )
  if predictor_const.INPUT_BYTES in instance:
    parsed_instance = (
        inline_bytes_data_accessor_definition.json_to_generic_bytes(
            instance,
            config.endpoint_input_width,
            config.endpoint_input_height,
            False,  # Require patch dim match default dim.
        )
    )
    return inline_bytes_data_accessor.InlineBytesData(
        parsed_instance, _LOCAL_FILE_HANDLERS
    )
  auth = authentication_utils.create_auth_from_instance(
      instance.get(predictor_const.BEARER_TOKEN, '')
  )
  # support reading imaging from HTTP.
  if predictor_const.IMAGE_URL in instance:
    parsed_instance = http_image_data_accessor_definition.json_to_http_image(
        auth,
        instance,
        config.endpoint_input_width,
        config.endpoint_input_height,
        require_patch_dim_match_default_dim=False,
    )
    return http_image_data_accessor.HttpImageData(
        parsed_instance,
        _LOCAL_FILE_HANDLERS,
    )
  # support GCS and DICOM.
  if predictor_const.GCS_URI in instance:
    parsed_instance = (
        gcs_generic_data_accessor_definition.json_to_generic_gcs_image(
            auth,
            instance,
            config.endpoint_input_width,
            config.endpoint_input_height,
            False,  # Require patch dim match default dim.
        )
    )
    return gcs_generic_data_accessor.GcsGenericData(
        parsed_instance,
        _LOCAL_FILE_HANDLERS,
        _GCS_DOWNLOAD_THREAD_COUNT,
    )
  if predictor_const.DICOM_WEB_URI in instance:
    # decode dicom path
    # determine dicom source type may query dicom store for series
    # instance metadata.
    result = dicom_source_utils.get_dicom_source_type(auth, instance)
    # if slide microscope image
    if (
        result.dicom_source_type
        == dicom_source_utils.DicomDataSourceEnum.SLIDE_MICROSCOPY_IMAGE
    ):
      # Define pathology DICOM input.
      parsed_instance = (
          dicom_wsi_data_accessor_definition.json_to_dicom_wsi_image(
              auth,
              instance,
              config,
              result.dicom_instances_metadata,
          )
      )
      return dicom_wsi_data_accessor.DicomDigitalPathologyData(
          parsed_instance, config
      )
    parsed_instance = (
        dicom_generic_data_accessor_definition.json_to_generic_dicom_image(
            auth,
            instance,
            config.endpoint_input_width,
            config.endpoint_input_height,
            False,  # Require patch dim match default dim.
            result.dicom_instances_metadata,
        )
    )
    return dicom_generic_data_accessor.DicomGenericData(parsed_instance)
  raise data_accessor_errors.InvalidRequestFieldError(
      'Unidentified image instance.'
  )


class PredictionInputParser:
  """Class containing methods for transforming embedding request and responses."""

  def json_to_embedding_request(
      self,
      config: dicom_wsi_configuration.ConfigurationSettings,
      json_metadata: Mapping[str, Any],
  ) -> Sequence[
      abstract_data_accessor.AbstractDataAccessor[
          predictor_data_types.EmbeddingInstance, Union[np.ndarray, str]
      ]
  ]:
    """Converts json to embedding request.

    Args:
      config: The configuration settings for the data source.
      json_metadata: The value of the JSON payload provided to the API.

    Returns:
      Structured EmbeddingRequest object.

    Raises:
      InvalidRequestFieldError: If the provided fields are invalid.
    """
    parsed_instances = []
    for instance in _validate_instance_list(json_metadata):
      if not isinstance(instance, dict):
        raise data_accessor_errors.InvalidRequestFieldError(
            'Request instance is not a dict.'
        )
      image_instance = instance.get(predictor_const.IMAGE)
      if image_instance is not None:
        parsed_instances.append(_parse_image_instance(config, image_instance))
      elif predictor_const.TEXT in instance:
        inline_text = inline_text_data_accessor_definition.json_to_text(
            instance
        )
        parsed_instance = predictor_data_types.InlineTextInstance(
            text=inline_text.text, base_request=inline_text.base_request
        )
        parsed_instance = typing.cast(
            abstract_data_accessor.AbstractDataAccessor[
                predictor_data_types.InlineTextInstance, str
            ],
            inline_text_data_accessor.InlineText(parsed_instance),
        )
        parsed_instances.append(parsed_instance)
      else:
        raise data_accessor_errors.InvalidRequestFieldError(
            'Unidentified instance.'
        )
    return parsed_instances


@dataclasses.dataclass(frozen=True)
class _InputIterInput:
  index: int  # index of first embedding returned.
  # if data returns multiple embeddings, then the returned
  # embeddings indices will be index..index+n
  data: abstract_data_accessor.AbstractDataAccessor  # data source for embedding


_InputIterDataType = TypeVar('_InputIterDataType')


class _InputIter(Generic[_InputIterDataType], metaclass=abc.ABCMeta):
  """Returns batches of embedding input data from a list of data sources."""

  def __init__(self, input_list: Sequence[_InputIterInput]):
    """Constructor takes a list of data sources."""
    self._input_list = input_list
    self._index = 0
    if self._index < len(self._input_list):
      self._iter = self._input_list[self._index].data.data_iterator()
      self._embedding_index = self._input_list[self._index].index
    else:
      self._iter = None
      self._embedding_index = 0

  @abc.abstractmethod
  def _pre_process(self, input_data: _InputIterDataType) -> Mapping[str, Any]:
    """Pre-process data source input data."""

  def get_next(self, batch_size: int) -> Sequence[_ModelInput]:
    """Returns next batch of input data max size batch_size."""
    result = []
    if self._iter is None:
      # no data return nothing.
      return result
    # return up to batch size embedding input data
    for _ in range(batch_size):
      while True:
        try:
          # try to get next iterator input data
          val = next(self._iter)
          result.append(
              _ModelInput(self._embedding_index, self._pre_process(val))
          )
          self._embedding_index += 1
          break
        except StopIteration:
          # if current iterator has no more data, go to next iterator.
          self._index += 1
          if self._index < len(self._input_list):
            # get next data iterator
            # and set starting index for next embedding.
            self._iter = self._input_list[self._index].data.data_iterator()
            self._embedding_index = self._input_list[self._index].index
          else:
            # no more data set iter to none.
            self._iter = None
            self._embedding_index = 0
            return result
    return result


class _InputTextIter(_InputIter[str]):

  def _pre_process(self, input_data: str) -> Mapping[str, Any]:
    return _tokenize_text(input_data)


def _zero_pad_image_to_square(norm_img: np.ndarray) -> np.ndarray:
  """Pads image with zeros to be dimensionally square."""
  height, width = norm_img.shape[:2]
  if height < width:
    dh = width - height
    half_dh = dh // 2
    return np.pad(norm_img, ((half_dh, dh - half_dh), (0, 0), (0, 0)))
  elif width < height:
    dw = height - width
    half_dw = dw // 2
    return np.pad(norm_img, ((0, 0), (half_dw, dw - half_dw), (0, 0)))
  return norm_img


class _InputImageIter(_InputIter[np.ndarray]):
  """Pre-process image input data."""

  def _pre_process(self, input_data: np.ndarray) -> Mapping[str, Any]:
    # Normalize images to [0, 1]
    # Do externally to enable 8 and 16 bit images to be passed in base
    # transformer only supports 8 or 16 bit normalization
    norm_img = input_data.astype(np.float32) / float(
        np.iinfo(input_data.dtype).max
    )
    if norm_img.shape[-1] == 1 and norm_img.ndim == 3:
      # if single channel monochrome image, represent as 3 channel image.
      norm_img = np.concatenate([norm_img, norm_img, norm_img], axis=-1)
    elif norm_img.shape[-1] == 4 and norm_img.ndim == 3:
      # if RGBA image remove alpha channel.
      norm_img = norm_img[..., :3]

    # method performed in training
    # pad image with zeros to be dimensionally square
    norm_img = _zero_pad_image_to_square(norm_img)

    return _get_siglip_image_processor().preprocess(
        [norm_img],
        data_format='channels_first',
        input_data_format='channels_last' if norm_img.ndim == 3 else 'none',
    )


class _MlOutputDecoder:
  """Returns model embedding results matching embedding indexs."""

  def __init__(
      self,
      model_output: Mapping[str, Any],
      embedding_key: str,
      index_key: str,
  ):
    self._embeds = model_output.get(embedding_key, _ZERO_DIM_ARRAY)
    self._embed_indexs = model_output.get(index_key, [])
    self._index = 0

  def get_embedding(self, embedding_index: int) -> Optional[np.ndarray]:
    if (
        self._index < self._embeds.shape[0]
        and self._embed_indexs[self._index] == embedding_index
    ):
      result = self._embeds[self._index, ...]
      self._index += 1
      return result
    return None

  def buffer_unreturned_embeddings(
      self, buffered_results: MutableMapping[int, np.ndarray]
  ) -> None:
    for i in range(self._index, self._embeds.shape[0]):
      buffered_results[self._embed_indexs[i]] = self._embeds[i, ...]


def _unpack_model_output(
    model_output: Mapping[str, Any],
    buffered_results: MutableMapping[int, np.ndarray],
    embedding_index: int,
) -> Iterator[np.ndarray]:
  """Returns model output in the order data was requested, buffers as needed."""
  # Get model output
  embedding_result_decoders = (
      _MlOutputDecoder(model_output, _TEXT_EMBEDS_KEY, _TEXT_EMBEDDING_INDEX),
      _MlOutputDecoder(model_output, _IMAGE_EMBEDS_KEY, _IMAGE_EMBEDDING_INDEX),
  )
  while True:
    # if text embedding index matches return text embedding.
    for decoder in embedding_result_decoders:
      embedding = decoder.get_embedding(embedding_index)
      if embedding is not None:
        yield embedding
        embedding_index += 1
        break
    else:
      # check buffered results if embedding index was previously computed.
      embedding = buffered_results.get(embedding_index)
      if embedding is not None:
        del buffered_results[embedding_index]
        yield embedding
        embedding_index += 1
        continue
      # next sequential embedding index not found stop returning embedings.
      break
  # buffer any remaining embeddings that were computed out of order and not
  # returned.
  for decoder in embedding_result_decoders:
    decoder.buffer_unreturned_embeddings(buffered_results)


class ModelPredictor(
    async_batch_predictor.AsyncBatchModelPredictor[
        predictor_data_types.EmbeddingInstance, Union[np.ndarray, str]
    ]
):
  """Retrieves data and runs data across model predictor."""

  def __init__(self):
    super().__init__(
        flags.THREAD_POOL_MAX_WORKERS_FLAG.value,
        flags.THREAD_POOL_TIMEOUT_FLAG.value,
    )

  def _run_model_predictor(
      self,
      model: model_runner.ModelRunner,
      data_list: Sequence[
          abstract_data_accessor.AbstractDataAccessor[
              predictor_data_types.EmbeddingInstance, Union[np.ndarray, str]
          ]
      ],
  ) -> Iterator[np.ndarray]:
    """Runs model on sequence of images in size batches in and returns results."""
    images_per_batch = flags.IMAGE_EMBEDDINGS_PER_BATCH_PREDICTION_FLAG.value
    text_per_batch = flags.TEXT_EMBEDDINGS_PER_BATCH_PREDICTION_FLAG.value
    start_time = time.time()
    embedding_count = 0
    text_input = []
    image_input = []
    # create a lists of text input and image input sources
    # embedding_count is the index of the first embedding for the data source.
    # in the response.
    for data in data_list:
      if isinstance(data, inline_text_data_accessor.InlineText):
        text_input.append(_InputIterInput(embedding_count, data))
      else:
        image_input.append(_InputIterInput(embedding_count, data))
      embedding_count += len(data)
    text_input = _InputTextIter(text_input)
    image_input = _InputImageIter(image_input)
    buffered_results = {}
    returned_embedding_count = 0
    # while embedding values to return
    while returned_embedding_count < embedding_count:
      # get next chunk of image and text inputs.
      img_list = image_input.get_next(images_per_batch)
      token_list = text_input.get_next(text_per_batch)
      # call model
      model_output = _call_model(model, img_list, token_list)
      self._model_prediction_count += 1
      # immediatly yield ml results in the order they were requested.
      # buffer results that were processed out of order.

      # set model temperature and bias if not set.
      # will be same for all predictions.
      if self.model_temperature is None:
        self.model_temperature = model_output[_SCALE_KEY].item(0)
        self.model_bias = model_output[_BIAS_KEY].item(0)

      for embedding in _unpack_model_output(
          model_output, buffered_results, returned_embedding_count
      ):
        yield embedding
        returned_embedding_count += 1
    cloud_logging_client.info(
        f'Called embedding model; {time.time() - start_time} (sec).'
    )


def _validate_encoder_request(
    requests: Sequence[
        abstract_data_accessor.AbstractDataAccessor[
            predictor_data_types.EmbeddingInstance, np.ndarray
        ]
    ],
) -> None:
  """Validates the request does not exceed the maximum number of embeddings."""
  instance_count = 0
  for request in requests:
    if request.instance.patch_coordinates:
      instance_count += len(request.instance.patch_coordinates)
    else:
      instance_count += 1
    if instance_count > flags.MAX_EMBEDDINGS_PER_REQUEST_FLAG.value:
      raise data_accessor_errors.TooManyPatchesError(
          'Request exceeds maximum number of images per request:'
          f' {flags.MAX_EMBEDDINGS_PER_REQUEST_FLAG.value}'
      )


class MedSiglipPredictor:
  """Callable responsible for generating embeddings."""

  def __init__(self):
    self._last_request_model_prediction_count = 0

  @property
  def last_request_model_prediction_count(self) -> int:
    return self._last_request_model_prediction_count

  def predict(
      self,
      prediction_input: Mapping[str, Any],
      model: model_runner.ModelRunner,
  ) -> dict[str, Any]:
    """Runs inference on provided patches.

    Args:
      prediction_input: JSON formatted input for embedding prediction.
      model: ModelRunner to handle model step.

    Returns:
      JSON formatted output.

    Raises:
      ERROR_LOADING_DICOM: If the provided patches are not concated.
    """
    self._last_request_model_prediction_count = 0
    config = dicom_wsi_configuration.ConfigurationSettings(
        endpoint_input_width=flags.MODEL_INPUT_WIDTH_FLAG.value,
        endpoint_input_height=flags.MODEL_INPUT_HEIGHT_FLAG.value,
        approved_dicom_stores=flags.APPROVED_DICOM_STORE_SOURCE_LIST_FLAG.value,
        icc_profile_cache_configuration=dicom_wsi_configuration.IccProfileCacheConfiguration(
            gcs_bucket=flags.ICC_PROFILE_CACHE_GCS_BUCKET_FLAG.value,
            redis_ip=flags.ICC_PROFILE_CACHE_REDIS_IP_FLAG.value,
            redis_port=flags.ICC_PROFILE_CACHE_REDIS_PORT_FLAG.value,
            store_icc_profile_bytes_in_redis=flags.STORE_ICC_PROFILE_BYTES_IN_REDIS_FLAG.value,
            testing=flags.IS_DEBUGGING_FLAG.value,
        ),
    )
    try:
      request = PredictionInputParser().json_to_embedding_request(
          config, prediction_input
      )
      _validate_encoder_request(request)
    except data_accessor_errors.DataAccessorError as exp:
      return dict(embedding_response.prediction_error_response(exp))

    with ModelPredictor() as predictor:
      if flags.BATCH_PREDICTION_FLAG.value:
        # Get imaging in parallel and then run ML model in batch.
        instances = predictor.batch_predict_embeddings(model, request)
      else:
        # Get imaging and run ML model for each requested instance in parallel.
        instances = predictor.parallel_predict_embeddings(model, request)
      del request  # request metadata not needed further.

      # build response for each instance.
      embedding_results = []
      for instance in instances:
        if isinstance(instance, data_accessor_errors.DataAccessorError):
          embedding_results.append(
              embedding_response.instance_error_response(instance)
          )
          continue
        try:
          embedding_results.append(
              embedding_response.embedding_instance_response(
                  predictor, instance
              )
          )
        except data_accessor_errors.DataAccessorError as exp:
          embedding_results.append(
              embedding_response.instance_error_response(exp)
          )
      self._last_request_model_prediction_count = (
          predictor.model_prediction_count
      )
    cloud_logging_client.info('Returning embeddings.')
    return {predictor_const.PREDICTIONS: embedding_results}
