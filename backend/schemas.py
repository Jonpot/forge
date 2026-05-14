from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_COMMENT_COLOR = "#64748b"


def _normalize_identifier(value: str | None, *, fallback: str) -> str:
    text = str(value or "").strip()
    return text or fallback


def _default_edge_id(
    source: str,
    target: str,
    source_output: int | None,
    source_handle: str | None,
    target_input: int | None,
    target_handle: str | None,
    index: int,
) -> str:
    source_part = (
        str(source_output)
        if isinstance(source_output, int)
        else _normalize_identifier(source_handle, fallback="output_0")
    )
    target_part = (
        str(target_input)
        if isinstance(target_input, int)
        else _normalize_identifier(target_handle, fallback="input_0")
    )
    return f"edge_{source}_{source_part}_{target}_{target_part}_{index}"


class NodePositionModel(BaseModel):
    x: float
    y: float


class PipelineGroupModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    name: str
    description: str = ""
    comment_id: str | None = None


class CommentItemModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    title: str = ""
    description: str = ""
    color: str = DEFAULT_COMMENT_COLOR
    position: NodePositionModel
    width: float = 300
    height: float = 150
    managed: bool = False
    group_id: str | None = None


class PipelineNodeModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    block: str
    params: dict[str, Any] = Field(default_factory=dict)
    notes: str | None = None
    group_ids: list[str] = Field(default_factory=list)
    position: NodePositionModel | None = None
    width: float | None = None
    height: float | None = None


class PipelineEdgeModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str | None = None
    source: str
    target: str
    source_output: int | None = None
    sourceHandle: str | None = None
    target_input: int | None = None
    targetHandle: str | None = None


class PipelineModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    nodes: list[PipelineNodeModel] = Field(default_factory=list)
    edges: list[PipelineEdgeModel] = Field(default_factory=list)
    comments: list[CommentItemModel] = Field(default_factory=list)
    groups: list[PipelineGroupModel] = Field(default_factory=list)

    @model_validator(mode="after")
    def normalize_metadata(self) -> "PipelineModel":
        seen_edge_ids: set[str] = set()
        normalized_edges: list[PipelineEdgeModel] = []
        for index, edge in enumerate(self.edges):
            candidate = _normalize_identifier(
                edge.id,
                fallback=_default_edge_id(
                    edge.source,
                    edge.target,
                    edge.source_output,
                    edge.sourceHandle,
                    edge.target_input,
                    edge.targetHandle,
                    index,
                ),
            )
            if candidate in seen_edge_ids:
                suffix = 1
                while f"{candidate}_{suffix}" in seen_edge_ids:
                    suffix += 1
                candidate = f"{candidate}_{suffix}"
            seen_edge_ids.add(candidate)
            normalized_edges.append(edge.model_copy(update={"id": candidate}))
        self.edges = normalized_edges

        seen_group_ids: set[str] = set()
        normalized_groups: list[PipelineGroupModel] = []
        for index, group in enumerate(self.groups):
            candidate = _normalize_identifier(group.id, fallback=f"group_{index}")
            if candidate in seen_group_ids:
                suffix = 1
                while f"{candidate}_{suffix}" in seen_group_ids:
                    suffix += 1
                candidate = f"{candidate}_{suffix}"
            seen_group_ids.add(candidate)
            normalized_groups.append(group.model_copy(update={"id": candidate}))
        self.groups = normalized_groups

        seen_comment_ids: set[str] = set()
        normalized_comments: list[CommentItemModel] = []
        for index, comment in enumerate(self.comments):
            candidate = _normalize_identifier(comment.id, fallback=f"comment_{index}")
            if candidate in seen_comment_ids:
                suffix = 1
                while f"{candidate}_{suffix}" in seen_comment_ids:
                    suffix += 1
                candidate = f"{candidate}_{suffix}"
            seen_comment_ids.add(candidate)
            normalized_comments.append(comment.model_copy(update={"id": candidate}))
        self.comments = normalized_comments
        return self


class PipelineSummaryModel(BaseModel):
    id: str
    name: str
    path: str
    updated_at: float


class PipelineEnvelopeModel(BaseModel):
    id: str
    pipeline: PipelineModel


class ExecuteResponseModel(BaseModel):
    pipeline_id: str
    topological_order: list[str]
    executed_nodes: list[str]
    reused_nodes: list[str]
    node_results: dict[str, dict[str, Any]]


class CancelExecutionResponseModel(BaseModel):
    pipeline_id: str
    status: str


class StalenessResponseModel(BaseModel):
    pipeline_id: str
    stale: dict[str, bool]
    history_hashes: dict[str, str]


class CheckpointPreviewModel(BaseModel):
    checkpoint_id: str
    rows: list[dict[str, Any]]
    columns: list[str]
    dtypes: dict[str, str]
    total_rows: int


def normalize_pipeline_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return PipelineModel.model_validate(payload).model_dump(mode="json")
