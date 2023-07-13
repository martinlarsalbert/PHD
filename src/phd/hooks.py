import logging

from kedro.framework.cli.hooks import cli_hook_impl
from kedro.io import DataCatalog
from kedro.framework.startup import ProjectMetadata

logger = logging.getLogger(__name__)


class ProjectHooks:
    @cli_hook_impl
    def after_command_run(
        self, project_metadata: ProjectMetadata, command_args: list[str], exit_code: int
    ) -> None:
        logger.info("Kedro is done!")
