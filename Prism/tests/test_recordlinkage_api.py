from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient

# It's important that the app is imported after the mock setup for some scenarios,
# but for this structure, direct import is fine.
from recordLinkage.src.api import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@patch('recordLinkage.src.api.RecordLinkagePipeline')
def test_run_record_linkage_success(MockRecordLinkagePipeline):
    # Arrange
    mock_result = {
        "dataset_key": "test_data",
        "match_count": 100,
        # ... other fields
    }
    mock_pipeline_instance = MagicMock()
    # The run method returns a dataclass, which FastAPI will serialize.
    # Returning a dict is sufficient for mocking.
    mock_pipeline_instance.run.return_value = mock_result
    MockRecordLinkagePipeline.return_value = mock_pipeline_instance

    request_payload = {
        "dataset_key": "test_data",
    }

    # Act
    response = client.post("/run", json=request_payload)

    # Assert
    assert response.status_code == 200
    # The actual response will be a JSON string of the dict
    assert response.json() == mock_result
    MockRecordLinkagePipeline.assert_called_once()
    mock_pipeline_instance.run.assert_called_once()


@patch('recordLinkage.src.api.RecordLinkagePipeline')
def test_run_record_linkage_error(MockRecordLinkagePipeline):
    # Arrange
    mock_pipeline_instance = MagicMock()
    mock_pipeline_instance.run.side_effect = Exception("RL pipeline failed")
    MockRecordLinkagePipeline.return_value = mock_pipeline_instance

    request_payload = {
        "dataset_key": "test_data",
    }

    # Act
    response = client.post("/run", json=request_payload)

    # Assert
    assert response.status_code == 500
    assert "An unexpected error occurred" in response.json()["detail"]
