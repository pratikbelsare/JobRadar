"""Connector registry for selecting source implementations by connector type."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from ..http import HttpClient
from ..models import Company, ConnectorType
from .base import JobSourceConnector, UnsupportedConnectorError
from .greenhouse import GreenhouseConnector


ConnectorFactory = Callable[[HttpClient], JobSourceConnector]


class ConnectorRegistry:
    """Map configured connector types to constructor functions."""

    def __init__(
        self,
        http_client: HttpClient,
        factories: Mapping[ConnectorType, ConnectorFactory] | None = None,
    ) -> None:
        self._http_client = http_client
        self._factories: dict[ConnectorType, ConnectorFactory] = dict(
            factories
            or {ConnectorType.GREENHOUSE: lambda client: GreenhouseConnector(client)}
        )

    def register(self, connector_type: ConnectorType, factory: ConnectorFactory) -> None:
        self._factories[connector_type] = factory

    def for_company(self, company: Company) -> JobSourceConnector:
        factory = self._factories.get(company.connector_type)
        if factory is None:
            raise UnsupportedConnectorError(company.connector_type)
        return factory(self._http_client)


__all__ = ["ConnectorFactory", "ConnectorRegistry"]
