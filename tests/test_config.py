from nlp_platform.config import settings


def test_application_name():
    assert settings.app_name == "NLP Intelligence Platform"


def test_application_version():
    assert settings.version == "0.1.0"


def test_environment():
    assert settings.environment == "development"