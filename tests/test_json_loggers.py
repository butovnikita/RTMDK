"""Tests for rtmdk/utils/json_logger.py and rtmdk/production/json_logger.py."""

import io
import json
import logging

import pytest

from rtmdk.production.json_logger import JSONFormatter as ProdJSONFormatter
from rtmdk.production.json_logger import setup_json_logging as prod_setup
from rtmdk.utils.json_logger import JSONFormatter as UtilsJSONFormatter
from rtmdk.utils.json_logger import setup_json_logging as utils_setup


@pytest.fixture
def record():
    logger = logging.getLogger("rtmdk.test")
    return logger.makeRecord(
        "rtmdk.test", logging.WARNING, __file__, 42, "hello %s", ("world",), None, func="make_test_record"
    )


class TestUtilsFormatter:
    def test_basic_fields(self, record):
        out = json.loads(UtilsJSONFormatter().format(record))

        assert out["level"] == "WARNING"
        assert out["logger"] == "rtmdk.test"
        assert out["message"] == "hello world"
        assert "timestamp" in out
        assert "exception" not in out
        assert "context" not in out

    def test_static_fields_merged(self, record):
        out = json.loads(UtilsJSONFormatter(static_fields={"service": "rtmdk", "env": "test"}).format(record))
        assert out["service"] == "rtmdk"
        assert out["env"] == "test"

    def test_exception_included(self, record):
        try:
            raise ValueError("kaboom")
        except ValueError:
            import sys

            record.exc_info = sys.exc_info()
        out = json.loads(UtilsJSONFormatter().format(record))
        assert "ValueError: kaboom" in out["exception"]

    def test_context_included(self, record):
        record.context = {"request_id": "abc"}
        out = json.loads(UtilsJSONFormatter().format(record))
        assert out["context"] == {"request_id": "abc"}


class TestProdFormatter:
    def test_basic_fields(self, record):
        out = json.loads(ProdJSONFormatter().format(record))

        assert out["level"] == "WARNING"
        assert out["logger"] == "rtmdk.test"
        assert out["message"] == "hello world"
        assert out["timestamp"].endswith("Z")
        assert out["source"]["line"] == 42
        assert out["source"]["function"] == "make_test_record"
        assert "exception" not in out

    def test_static_fields_and_exception(self, record):
        import sys

        try:
            raise RuntimeError("prod boom")
        except RuntimeError:
            record.exc_info = sys.exc_info()

        out = json.loads(ProdJSONFormatter(static_fields={"service": "api"}).format(record))
        assert out["service"] == "api"
        assert "RuntimeError: prod boom" in out["exception"]


@pytest.fixture
def restore_root_logger():
    root = logging.getLogger()
    old_handlers, old_level = root.handlers[:], root.level
    yield
    root.handlers = old_handlers
    root.setLevel(old_level)


class TestSetup:
    def test_utils_setup_emits_json_to_stream(self, restore_root_logger):
        stream = io.StringIO()
        utils_setup(level=logging.INFO, static_fields={"app": "rtmdk"}, stream=stream)

        logging.getLogger("rtmdk.memory").info("ingested %d nodes", 5)

        out = json.loads(stream.getvalue().strip())
        assert out["message"] == "ingested 5 nodes"
        assert out["app"] == "rtmdk"
        assert out["level"] == "INFO"

    def test_utils_setup_respects_level(self, restore_root_logger):
        stream = io.StringIO()
        utils_setup(level=logging.ERROR, stream=stream)

        logging.getLogger("rtmdk").warning("suppressed")

        assert stream.getvalue() == ""

    def test_prod_setup_emits_json_with_service(self, restore_root_logger, capsys):
        prod_setup(level=logging.INFO, service="my-service")

        logging.getLogger("rtmdk").error("disk full")

        err = capsys.readouterr().err.strip()
        out = json.loads(err.splitlines()[-1])
        assert out["message"] == "disk full"
        assert out["service"] == "my-service"
