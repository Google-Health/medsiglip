"""Model file compatible with Triton PyTorch 2.0 backend."""

import os
from typing import Optional

import torch
from transformers.models import siglip


# Override dir to hide imports from the Triton backend's loading strategy.
# Without this, the backend can attempt to load from the imports
# instead of SiglipWrapper.
def __dir__():
  return ["SiglipWrapper", "__name__", "__spec__"]


class SiglipWrapper(torch.nn.Module):
  """Wraps SiglipModel with custom weight loading and return structure."""

  def __init__(self):
    super(SiglipWrapper, self).__init__()
    token = None
    if os.getenv("AIP_STORAGE_URI"):
      # Using model files copied from Vertex GCS bucket.
      model_origin = os.getenv("MODEL_FILES")
    else:
      # Using model files from HF repository.
      model_origin = os.getenv("MODEL_ID")
      if not model_origin:
        raise ValueError(
            "No model origin found. MODEL_ID or AIP_STORAGE_URI must be set."
        )
      token = os.getenv("HF_TOKEN")  # optional for access to non-public models.
    self.model = siglip.SiglipModel.from_pretrained(
        model_origin,
        token=token,
    )

  def forward(
      self,
      input_ids: Optional[torch.LongTensor] = None,
      pixel_values: Optional[torch.FloatTensor] = None,
  ):
    output = self.model(input_ids=input_ids, pixel_values=pixel_values)
    return (
        output.text_embeds,
        output.image_embeds,
        self.model.logit_scale,
        self.model.logit_bias,
    )
