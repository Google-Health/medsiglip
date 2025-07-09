# MedSigLIP

MedSigLIP is a variant of [SigLIP](https://arxiv.org/abs/2303.15343)(Sigmoid
Loss for Language Image Pre-training) that is trained to encode medical images
and text into a common embedding space. Developers can use MedSigLIP to
accelerate building healthcare-based AI applications. MedSigLIP contains a 400M
parameter vision encoder and 400M parameter text encoder, it supports 448x448
image resolution with up to 64 text tokens.

MedSigLIP is recommended for medical image interpretation applications without a
need for text generation, such as data-efficient classification, zero-shot
classification, and semantic image retrieval. For medical applications that
require text generation,
[MedGemma](https://developers.google.com/health-ai-developer-foundations/medgemma)
is recommended.

## Get started

*   Read our
    [developer documentation](https://developers.google.com/health-ai-developer-foundations/medsiglip/get-started)
    to see the full range of next steps available, including learning more about
    the model through its
    [model card](https://developers.google.com/health-ai-developer-foundations/medsiglip/model-card).

*   Explore this repository, which contains [notebooks](./notebooks) for using
    the model from Hugging Face and Vertex AI as well as the
    [implementation](./python/serving) of the container that you can deploy to
    Vertex AI.

*   Visit the model on [Hugging Face](https://huggingface.co/google/medsiglip)
    or
    [Model Garden](https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/medsiglip).

## Contributing

We are open to bug reports, pull requests (PR), and other contributions. See
[CONTRIBUTING](CONTRIBUTING.md) and
[community guidelines](https://developers.google.com/health-ai-developer-foundations/community-guidelines)
for details.

## License

While the model is licensed under the
[Health AI Developer Foundations License](https://developers.google.com/health-ai-developer-foundations/terms),
everything in this repository is licensed under the Apache 2.0 license, see
[LICENSE](LICENSE).
