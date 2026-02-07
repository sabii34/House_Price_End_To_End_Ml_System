from pydantic import BaseModel, Field


class HouseFeatures(BaseModel):
    MedInc: float = Field(..., description="Median income in block group")
    HouseAge: float = Field(..., ge=0, description="Median house age in block group")
    AveRooms: float = Field(..., gt=0, description="Average rooms per household")
    AveBedrms: float = Field(..., gt=0, description="Average bedrooms per household")
    Population: float = Field(..., ge=0, description="Block group population")
    AveOccup: float = Field(..., gt=0, description="Average occupants per household")
    Latitude: float = Field(..., ge=-90, le=90)
    Longitude: float = Field(..., ge=-180, le=180)


class PredictResponse(BaseModel):
    prediction: float
    units: str = "100k_dollars"
    model_artifact: str
