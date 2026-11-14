
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    features: list[float] = Field(
        min_length=561,
        max_length=561,
        description="A single UCI HAR feature vector containing 561 values.",
    )
