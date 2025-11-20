from typing import Any
from deploy.preprocessing import process_dimensions

class InferencePipeline:
    def __call__(self, video_path: str, input_folder: str, output_folder: str) -> dict[str, Any]:  
        output = {}  

        length, width, thickness = process_dimensions(
            video_path, 
            input_folder, 
            output_folder
        )

        output["GemstoneLengthPrediction"] = length
        output["GemstoneWidthPrediction"] = width
        output["GemstoneThicknessPrediction"] = thickness

        return output