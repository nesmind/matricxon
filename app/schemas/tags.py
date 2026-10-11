from pydantic import BaseModel


class ModelDetails(BaseModel):
    family: str
    parameter_size: str
    context_length: int


class TagEntry(BaseModel):
    name: str
    capabilities: list[str]
    size: int
    details: ModelDetails
    # Matricxon's minimum RAM to load this tag; see app.models.load_dtype.estimate_ram_gb (the same
    # number `/api/show` reports, exposed here so a caller listing every installed model,
    # ../pAIring/app/services/model_catalog_service.py, gets all in one call instead of an
    # `/api/show` per tag).
    estimated_ram_gb: float


class TagsResponse(BaseModel):
    models: list[TagEntry]
