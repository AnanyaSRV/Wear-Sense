
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.config import FRONTEND_ORIGINS
from backend.schemas.activity import PredictionRequest
from backend.services.model_service import model_service
from backend.services.dataset_service import dataset_service


app = FastAPI(
    title="WearSense API",
    description=(
        "Activity recognition research API powered by the "
        "UCI HAR dataset and a trained machine-learning model."
    ),
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "name": "WearSense API",
        "status": "running",
        "message": "Welcome to the WearSense activity recognition API.",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model_service.model is not None,
        "dataset_loaded": dataset_service.features is not None,
    }


@app.get("/model/info")
def model_info():
    try:
        return model_service.get_model_info()
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


@app.get("/dataset/info")
def dataset_info():
    try:
        return dataset_service.get_dataset_info()
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


@app.get("/dataset/samples")
def dataset_samples(
    start: int = 0,
    limit: int = 20,
):
    try:
        return dataset_service.get_samples(
            start=start,
            limit=limit,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )


@app.get("/dataset/sample/{index}")
def dataset_sample(index: int):
    try:
        sample = dataset_service.get_sample(index)

        # Do not return the full feature vector here.
        # The prediction endpoint uses it internally.
        sample.pop("features", None)

        return sample

    except IndexError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        )


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        return model_service.predict(request.features)

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


@app.post("/predict/sample/{index}")
def predict_dataset_sample(index: int):
    try:
        sample = dataset_service.get_sample(index)

        prediction = model_service.predict(
            sample["features"]
        )

        return {
            "sample_index": sample["sample_index"],
            "subject_id": sample["subject_id"],
            "actual_activity_id": sample["actual_activity_id"],
            "actual_activity": sample["actual_activity"],
            "prediction": prediction,
            "data_source": "UCI HAR test dataset",
            "is_live_sensor_data": False,
        }

    except IndexError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )
