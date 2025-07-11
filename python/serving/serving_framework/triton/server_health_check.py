"""REST-based health check implementation for Tensorflow model servers."""

import http

import requests
from typing_extensions import override

from serving.serving_framework import server_gunicorn


class TritonServerHealthCheck(server_gunicorn.ModelServerHealthCheck):
  """Checks the health of the local model server via REST request."""

  def __init__(self, health_check_port: int):
    self._health_check_url = (
        f"http://localhost:{health_check_port}/v2/health/ready"
    )

  @override
  def check_health(self) -> bool:
    try:
      r = requests.get(self._health_check_url)
      return r.status_code == http.HTTPStatus.OK.value
    except requests.exceptions.ConnectionError:
      return False
