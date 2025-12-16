import asyncio
import tempfile
from contextlib import asynccontextmanager
from http import HTTPStatus

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from deploy import __version__
from deploy.config import get_logger
from deploy.inference import InferencePipeline
from deploy.utils import download_file_from_s3
import os

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading inference artifacts...")

    global pipe
    pipe = InferencePipeline()
    logger.info("Inference artifacts loaded.")
    yield

    del pipe


app = FastAPI(title="Dimensions Calculation Inference API", version=__version__, lifespan=lifespan)


class GemstoneDimensionsCalculationResponse(BaseModel):
    gemstone_length_prediction: float
    gemstone_width_prediction: float
    gemstone_thickness_prediction: float


@app.post(
    "/calculate-gemstone-dimensions",
    response_model=GemstoneDimensionsCalculationResponse,
    tags=["GemstoneDimensionsCalculation"],
)
async def calculate_gemstone_dimensions(input_path: str = Query(...), shape: str = Query(...)) -> GemstoneDimensionsCalculationResponse:
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4") as temp_input_file:
            video_path = temp_input_file.name
            await asyncio.to_thread(download_file_from_s3, input_path, video_path)

            with tempfile.TemporaryDirectory() as temp_folder:
                results = await asyncio.to_thread(
                    pipe,
                    video_path=video_path,
                    input_folder=temp_folder,
                    output_folder=temp_folder, 
                    shape=shape
                )
                logger.info(f"Prediction results: {results}")

        return GemstoneDimensionsCalculationResponse(
            gemstone_length_prediction=results["GemstoneLengthPrediction"],
            gemstone_width_prediction=results["GemstoneWidthPrediction"],
            gemstone_thickness_prediction=results["GemstoneThicknessPrediction"],
        )

    except Exception as e:
        logger.exception(f"An exception occurred during gemstone dimensions prediction: {e}")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"An exception occurred during gemstone dimensions prediction: {e}",
        )
