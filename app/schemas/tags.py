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
    # Matricxon's own real, current minimum RAM to actually load this exact tag - see
    # app.models.load_dtype.estimate_ram_gb's own docstring for the full reasoning (same real
    # number `/api/show` reports, exposed here too so a caller listing every installed model - see
    # ../pAIring/app/services/model_catalog_service.py - gets it for all of them in one call,
    # rather than needing a separate `/api/show` round trip per tag).
    estimated_ram_gb: float


class TagsResponse(BaseModel):
    models: list[TagEntry]
