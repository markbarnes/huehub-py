"""Unit tests for bridge health diagnostics."""

import pytest
from pytest_httpx import HTTPXMock

from huehub.client import HueBridgeClient
from huehub.config import HueConfig
from huehub.diagnostics import build_diagnostics
from huehub.simulator import FakeBridge

pytestmark = pytest.mark.httpx_mock(assert_all_responses_were_requested=False)


def _device(did: str, name: str) -> dict:
    return {
        "id": did,
        "type": "device",
        "metadata": {"name": name},
        "product_data": {
            "model_id": "LWU001",
            "product_name": "Hue white lamp",
            "software_version": "1.122.8",
        },
    }


def _owned(rtype: str, did: str, **fields: object) -> dict:
    return {
        "id": f"{rtype}-{did}",
        "type": rtype,
        "owner": {"rid": did, "rtype": "device"},
        **fields,
    }


@pytest.fixture
def raw() -> list[dict]:
    return [
        _device("bridge", "Bridge"),
        _owned(
            "zigbee_connectivity",
            "bridge",
            status="connected",
            channel={"status": "set", "value": "channel_15"},
        ),
        _device("d1", "Lamp A"),
        _owned("zigbee_connectivity", "d1", status="connected"),
        _owned(
            "light",
            "d1",
            metadata={"name": "Lamp A"},
            dimming={"brightness": 0.0, "min_dim_level": 2.0},
        ),
        _device("d2", "Lamp B"),
        _owned("zigbee_connectivity", "d2", status="connectivity_issue"),
        _owned("device_software_update", "d2", state="update_pending"),
        _owned(
            "light", "d2", metadata={"name": "Lamp B"}, dimming={"brightness": 80.0}
        ),
        _device("d3", "Switch"),
        _owned("zigbee_connectivity", "d3", status="connected"),
        _owned(
            "device_power",
            "d3",
            power_state={"battery_state": "critical", "battery_level": 1},
        ),
        _device("d4", "Sensor"),
        _owned(
            "device_power",
            "d4",
            power_state={"battery_state": "normal", "battery_level": 90},
        ),
        {
            "id": "room-1",
            "type": "room",
            "metadata": {"name": "Lounge"},
            "children": [{"rid": "d2", "rtype": "device"}],
        },
    ]


class TestBuildDiagnostics:
    def test_zigbee_channel(self, raw: list[dict]) -> None:
        assert build_diagnostics(raw)["zigbee_channel"] == "channel_15"

    def test_unreachable_devices(self, raw: list[dict]) -> None:
        result = build_diagnostics(raw)["unreachable_devices"]
        assert result == [
            {
                "name": "Lamp B",
                "status": "connectivity_issue",
                "room": "Lounge",
                "model_id": "LWU001",
                "product_name": "Hue white lamp",
                "software_version": "1.122.8",
            }
        ]

    def test_low_batteries(self, raw: list[dict]) -> None:
        result = build_diagnostics(raw)["low_batteries"]
        assert result == [
            {"name": "Switch", "battery_level": 1, "battery_state": "critical"}
        ]

    def test_low_battery_by_level(self, raw: list[dict]) -> None:
        raw[-2]["power_state"]["battery_level"] = 15
        names = [b["name"] for b in build_diagnostics(raw)["low_batteries"]]
        assert names == ["Switch", "Sensor"]

    def test_pending_updates(self, raw: list[dict]) -> None:
        result = build_diagnostics(raw)["pending_updates"]
        assert result == [{"name": "Lamp B", "state": "update_pending"}]

    def test_zero_brightness_lights(self, raw: list[dict]) -> None:
        assert build_diagnostics(raw)["zero_brightness_lights"] == ["Lamp A"]

    def test_summary(self, raw: list[dict]) -> None:
        assert build_diagnostics(raw)["summary"] == {
            "devices": 5,
            "unreachable": 1,
            "low_batteries": 1,
            "pending_updates": 1,
            "zero_brightness_lights": 1,
            "unknown_connectivity": 1,
        }

    def test_zgp_connectivity_issue_is_unreachable(self, raw: list[dict]) -> None:
        raw.append(_owned("zgp_connectivity", "d4", status="disconnected"))
        result = build_diagnostics(raw)
        names = [d["name"] for d in result["unreachable_devices"]]
        assert names == ["Lamp B", "Sensor"]
        assert result["unknown_connectivity_devices"] == []

    def test_unknown_connectivity_devices(self, raw: list[dict]) -> None:
        assert build_diagnostics(raw)["unknown_connectivity_devices"] == [
            {
                "name": "Sensor",
                "room": None,
                "model_id": "LWU001",
                "product_name": "Hue white lamp",
            }
        ]

    def test_missing_status_counts_as_reachable(self, raw: list[dict]) -> None:
        raw.append(
            {"id": "zb-x", "type": "zigbee_connectivity", "owner": {"rid": "d4"}}
        )
        names = [d["name"] for d in build_diagnostics(raw)["unreachable_devices"]]
        assert names == ["Lamp B"]

    def test_empty(self) -> None:
        result = build_diagnostics([])
        assert result["zigbee_channel"] is None
        assert result["unreachable_devices"] == []


class TestClientDiagnose:
    async def test_diagnose_uses_fresh_data(
        self, hue_config: HueConfig, httpx_mock: HTTPXMock, fake_bridge: FakeBridge
    ) -> None:
        fake_bridge.zigbee_connectivity[0]["status"] = "disconnected"
        fake_bridge.setup_mocks(httpx_mock)
        async with HueBridgeClient(hue_config) as hue:
            result = await hue.diagnose()
        names = [d["name"] for d in result["unreachable_devices"]]
        assert names == ["Desk Lamp"]
