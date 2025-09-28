from fastapi.testclient import TestClient
from IntegratedRecyclableClassifier import app  # import your FastAPI app instance

client = TestClient(app)

def test_classify_image():
    # Assuming your /classify endpoint takes JSON with image path
    payload = {"image_path": "waterbottle.jpg"}
    with TestClient(app) as client:
        response = client.post("/classify/", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert 'plastic_water_bottles' in data
        print("Predicted class:", data)