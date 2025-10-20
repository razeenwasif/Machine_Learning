from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient

from autoML.api import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@patch('autoML.api.AutoMLPipeline')
def test_run_automl_success(MockAutoMLPipeline):
    # Arrange
    mock_result = {
        "task": "classification",
        "model_name": "test_model",
        "metrics": {"accuracy": 0.95},
        "best_config": {},
        "hpo_score": 0.95,
        "analysis_report": {},
        "cleaning_report": {},
    }
    mock_pipeline_instance = MagicMock()
    mock_pipeline_instance.run.return_value = mock_result
    MockAutoMLPipeline.return_value = mock_pipeline_instance

    request_payload = {
        "data_path": "/path/to/data.csv",
        "target": "my_target",
        "task": "classification",
        "max_trials": 10,
    }

    # Act
    response = client.post("/run", json=request_payload)

    # Assert
    assert response.status_code == 200
    assert response.json() == mock_result
    MockAutoMLPipeline.assert_called_once()
    mock_pipeline_instance.run.assert_called_once()


@patch('autoML.api.AutoMLPipeline')
def test_run_automl_pipeline_error(MockAutoMLPipeline):
    # Arrange
    mock_pipeline_instance = MagicMock()
    mock_pipeline_instance.run.side_effect = Exception("Something went wrong")
    MockAutoMLPipeline.return_value = mock_pipeline_instance

    request_payload = {
        "data_path": "/path/to/data.csv",
        "target": "my_target",
    }

    # Act
    response = client.post("/run", json=request_payload)

    # Assert
    assert response.status_code == 500
    assert "Pipeline execution failed" in response.json()["detail"]
