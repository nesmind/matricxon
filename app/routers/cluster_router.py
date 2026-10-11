from fastapi import APIRouter, Depends, HTTPException

from app.cluster.configure import ConfigureRefusedError, ConfigureRequest, NodeConfigurator
from app.cluster.freshness import FreshnessChecker
from app.cluster.provisioning import (
    InvalidProvisionError,
    NodeProvisioner,
    NotFreshError,
    ProvisionRequest,
)
from app.cluster.readiness import ClusterReadiness, ReadinessReport
from app.cluster.reset import NodeResetter, ResetRefusedError
from app.dependencies import (
    get_cluster_readiness,
    get_freshness_checker,
    get_node_configurator,
    get_node_provisioner,
    get_node_resetter,
)
from app.schemas.cluster import (
    AdminSecretBody,
    AdminSecretResponse,
    ConfigureBody,
    FreshnessResponse,
    ProvisionBody,
    ProvisionResponse,
    ReadinessCheckInfo,
    ReadinessResponse,
)
from app.server.admin_auth import require_admin

router = APIRouter()


class ClusterHandler:
    """Cluster-mode checklist for pAIring's admin UI; like /api/pull it has no auth of its own."""

    def __init__(self, readiness: ClusterReadiness) -> None:
        self._readiness = readiness

    def readiness(self) -> ReadinessResponse:
        return self._response(self._readiness.report())

    def activate(self) -> ReadinessResponse:
        return self._response(self._readiness.activate())

    @staticmethod
    def _response(report: ReadinessReport) -> ReadinessResponse:
        return ReadinessResponse(
            role=report.role,
            node_id=report.node_id,
            ready=report.ready,
            active=report.active,
            checks=[ReadinessCheckInfo(**c.__dict__) for c in report.checks],
            nodes=report.nodes,
            node_kind=report.node_kind,
            admin_auth=report.admin_auth,
        )


@router.get("/api/cluster/readiness", response_model=ReadinessResponse)
def get_readiness(
    readiness: ClusterReadiness = Depends(get_cluster_readiness),
) -> ReadinessResponse:
    return ClusterHandler(readiness).readiness()


@router.post(
    "/api/cluster/activate", dependencies=[Depends(require_admin)], response_model=ReadinessResponse
)
def activate_cluster(
    readiness: ClusterReadiness = Depends(get_cluster_readiness),
) -> ReadinessResponse:
    return ClusterHandler(readiness).activate()


@router.get("/api/cluster/fresh", response_model=FreshnessResponse)
def get_freshness(checker: FreshnessChecker = Depends(get_freshness_checker)) -> FreshnessResponse:
    report = checker.report()
    return FreshnessResponse(fresh=report.fresh, reasons=report.reasons)


@router.post(
    "/api/cluster/provision",
    dependencies=[Depends(require_admin)],
    response_model=ProvisionResponse,
)
def provision_node(
    body: ProvisionBody, provisioner: NodeProvisioner = Depends(get_node_provisioner)
) -> ProvisionResponse:
    """Makes a fresh install a cluster joiner (409 if not fresh, 400 for unusable folders). It works
    only while
    the node is fresh, and is called by pAIring's admin-only Cluster page. Signed when an admin
    secret is set.
    """
    try:
        restarting = provisioner.provision(ProvisionRequest(**body.model_dump()))
    except NotFreshError as error:
        raise HTTPException(status_code=409, detail=f"not a fresh install: {error}") from error
    except InvalidProvisionError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return ProvisionResponse(restarting=restarting)


@router.post(
    "/api/cluster/reset", dependencies=[Depends(require_admin)], response_model=ProvisionResponse
)
def reset_node(
    force: bool = False, resetter: NodeResetter = Depends(get_node_resetter)
) -> ProvisionResponse:
    """Hard reset: THIS node returns to a fresh install and restarts, whatever its state (409 only
    for an active voter,
    unless `force=true`). Models and shared folders are untouched. Signed when an admin secret is
    set; called by
    pAIring's admin-only Cluster page.
    """
    try:
        return ProvisionResponse(restarting=resetter.reset(force))
    except ResetRefusedError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@router.post(
    "/api/cluster/configure",
    dependencies=[Depends(require_admin)],
    response_model=ProvisionResponse,
)
def configure_node(
    body: ConfigureBody, configurator: NodeConfigurator = Depends(get_node_configurator)
) -> ProvisionResponse:
    """Turns cluster mode on/off for THIS node (the seed) and restarts it gracefully. 409 for a
    joiner, 400 for a
    missing folder or key. Signed when an admin secret is set; called by pAIring's admin-only
    Cluster page.
    """
    try:
        restarting = configurator.configure(ConfigureRequest(**body.model_dump()))
    except ConfigureRefusedError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except InvalidProvisionError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return ProvisionResponse(restarting=restarting)


@router.put(
    "/api/cluster/admin-secret",
    dependencies=[Depends(require_admin)],
    response_model=AdminSecretResponse,
)
def set_admin_secret(
    body: AdminSecretBody, provisioner: NodeProvisioner = Depends(get_node_provisioner)
) -> AdminSecretResponse:
    """Sets (or with "" removes) the admin secret on THIS node, at once and in its `.env`. Open
    while no secret is set; after that it must be signed with the current one (rotation).
    """
    provisioner.set_admin_secret(body.secret)
    return AdminSecretResponse(admin_auth=bool(body.secret))
