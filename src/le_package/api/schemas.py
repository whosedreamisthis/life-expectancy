from pydantic import BaseModel, Field, validator
from typing import Optional
import numpy as np

class PredictionInput(BaseModel):
    # Categorical Features
    Country: str = Field(..., example="Afghanistan")
    Year: int = Field(..., ge=2000, le=2015, example=2015)
    Status: str = Field(..., pattern="^(Developed|Developing)$", example="Developing")

    # Demographic & Mortality Data
    Adult_Mortality: float = Field(..., alias="Adult Mortality", ge=0, example=263.0)
    infant_deaths: int = Field(..., alias="infant deaths", ge=0, example=62)
    under_five_deaths: int = Field(..., alias="under-five deaths", ge=0, example=83)

    # Health & Lifestyle Factors
    Alcohol: Optional[float] = Field(None, ge=0, example=0.01)
    BMI: Optional[float] = Field(None, ge=0, example=19.1)
    Measles: int = Field(..., ge=0, example=1154)
    HIV_AIDS: float = Field(..., alias="HIV/AIDS", ge=0, example=0.1)

    # Immunization Rates (Used for Immunization Score)
    Hepatitis_B: Optional[float] = Field(None, alias="Hepatitis B", ge=0, le=100, example=65.0)
    Polio: Optional[float] = Field(None, ge=0, le=100, example=6.0)
    Diphtheria: Optional[float] = Field(None, alias="Diphtheria", ge=0, le=100, example=65.0)

    # Economic & Resource Indicators
    percentage_expenditure: float = Field(..., alias="percentage expenditure", ge=0, example=71.27)
    Total_expenditure: Optional[float] = Field(None, alias="Total expenditure", ge=0, example=8.16)
    GDP: Optional[float] = Field(None, ge=0, example=584.25)
    Population: Optional[float] = Field(None, ge=0, example=33736494.0)
    Income_composition_of_resources: Optional[float] = Field(
        None, alias="Income composition of resources", ge=0, le=1, example=0.479
    )
    Schooling: Optional[float] = Field(None, ge=0, le=25, example=10.1)

    # Clinical Physical Status
    thinness_1_19_years: Optional[float] = Field(None, alias="thinness  1-19 years", ge=0, example=17.2)
    thinness_5_9_years: Optional[float] = Field(None, alias="thinness 5-9 years", ge=0, example=17.3)

    class Config:
        # Allows the API to accept both "Adult_Mortality" and "Adult Mortality"
        populate_by_name = True
        
    @validator('Alcohol', 'BMI', 'GDP', 'Population', pre=True)
    def handle_empty_strings(cls, v):
        """Convert empty strings to None so they can be imputed by the pipeline."""
        if v == "":
            return None
        return v