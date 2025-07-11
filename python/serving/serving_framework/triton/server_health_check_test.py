import http
import os
from unittest import mock

import requests
import requests_mock

from absl.testing import absltest
from serving.serving_framework import server_gunicorn
from serving.serving_framework.triton import server_health_check


class ServerHealthCheckTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    os.environ["AIP_PREDICT_ROUTE"] = "/fake-predict-route"
    os.environ["AIP_HEALTH_ROUTE"] = "/fake-health-route"

  @requests_mock.Mocker()
  def test_health_route_pass_check(self, mock_requests):
    mock_requests.register_uri(
        "GET",
        "http://localhost:12345/v2/health/ready",
        text="assorted_metadata",
        status_code=http.HTTPStatus.OK,
    )

    executor = mock.create_autospec(
        server_gunicorn.PredictionExecutor,
        instance=True,
    )

    app = server_gunicorn.PredictionApplication(
        executor,
        health_check=server_health_check.TritonServerHealthCheck(
            12345
        ),
    ).load()
    service = app.test_client()

    response = service.get("/fake-health-route")

    self.assertEqual(response.status_code, http.HTTPStatus.OK)
    self.assertEqual(response.text, "ok")

  @requests_mock.Mocker()
  def test_health_route_fail_check(self, mock_requests):
    mock_requests.register_uri(
        "GET",
        "http://localhost:12345/v2/health/ready",
        exc=requests.exceptions.ConnectionError,
    )
    executor = mock.create_autospec(
        server_gunicorn.PredictionExecutor,
        instance=True,
    )

    app = server_gunicorn.PredictionApplication(
        executor,
        health_check=server_health_check.TritonServerHealthCheck(
            12345
        ),
    ).load()
    service = app.test_client()

    response = service.get("/fake-health-route")

    self.assertEqual(response.status_code, http.HTTPStatus.SERVICE_UNAVAILABLE)
    self.assertEqual(response.text, "not ok")


if __name__ == "__main__":
  absltest.main()
