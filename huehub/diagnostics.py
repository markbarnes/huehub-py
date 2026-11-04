"""Bridge health diagnostics built from the raw CLIP v2 resource list.

Surfaces the usual causes of lights not responding: Zigbee connectivity
problems, low switch/sensor batteries, pending firmware updates, and lights
whose stored brightness is 0.
"""

import logging
from typing import Any

log = logging.getLogger(__name__)

LOW_BATTERY_LEVEL = 20


def build_diagnostics(raw: list[dict]) -> dict[str, Any]:
    """Build a health report from ``GET /clip/v2/resource`` items.

    Args:
        raw: Flat list of raw bridge resource dicts.

    Returns:
        Dict with ``zigbee_channel``, ``unreachable_devices``,
        ``low_batteries``, ``pending_updates``, ``zero_brightness_lights``
        and ``summary``.
    """
    by_type: dict[str, list[dict]] = {}
    for item in raw:
        by_type.setdefault(item.get("type", ""), []).append(item)

    devices = {d["id"]: d for d in by_type.get("device", []) if "id" in d}
    rooms = _room_by_device(by_type.get("room", []))

    report: dict[str, Any] = {
        "zigbee_channel": _zigbee_channel(by_type.get("zigbee_connectivity", [])),
        "unreachable_devices": _unreachable(
            by_type.get("zigbee_connectivity", []), devices, rooms
        ),
        "low_batteries": _low_batteries(by_type.get("device_power", []), devices),
        "pending_updates": _pending_updates(
            by_type.get("device_software_update", []), devices
        ),
        "zero_brightness_lights": _zero_brightness(by_type.get("light", [])),
    }
    report["summary"] = {
        "devices": len(devices),
        "unreachable": len(report["unreachable_devices"]),
        "low_batteries": len(report["low_batteries"]),
        "pending_updates": len(report["pending_updates"]),
        "zero_brightness_lights": len(report["zero_brightness_lights"]),
    }
    log.debug("Diagnostics summary: %s", report["summary"])
    return report


def _owner(item: dict) -> str:
    return (item.get("owner") or {}).get("rid", "")


def _name(device_id: str, devices: dict[str, dict]) -> str:
    device = devices.get(device_id) or {}
    return (device.get("metadata") or {}).get("name") or device_id


def _room_by_device(raw_rooms: list[dict]) -> dict[str, str]:
    result: dict[str, str] = {}
    for room in raw_rooms:
        name = (room.get("metadata") or {}).get("name", "")
        for child in room.get("children") or []:
            if child.get("rtype") == "device":
                result[child.get("rid", "")] = name
    return result


def _zigbee_channel(raw_zigbee: list[dict]) -> str | None:
    for item in raw_zigbee:
        channel = item.get("channel") or {}
        if channel.get("value"):
            return channel["value"]
    return None


def _unreachable(
    raw_zigbee: list[dict], devices: dict[str, dict], rooms: dict[str, str]
) -> list[dict[str, Any]]:
    result = []
    for item in raw_zigbee:
        if item.get("status") == "connected":
            continue
        device_id = _owner(item)
        product = (devices.get(device_id) or {}).get("product_data") or {}
        result.append(
            {
                "name": _name(device_id, devices),
                "status": item.get("status"),
                "room": rooms.get(device_id),
                "model_id": product.get("model_id"),
                "product_name": product.get("product_name"),
                "software_version": product.get("software_version"),
            }
        )
    return sorted(result, key=lambda d: d["name"])


def _low_batteries(
    raw_power: list[dict], devices: dict[str, dict]
) -> list[dict[str, Any]]:
    result = []
    for item in raw_power:
        state = item.get("power_state") or {}
        level = state.get("battery_level")
        battery_state = state.get("battery_state")
        low_level = level is not None and level <= LOW_BATTERY_LEVEL
        if battery_state not in (None, "normal") or low_level:
            result.append(
                {
                    "name": _name(_owner(item), devices),
                    "battery_level": level,
                    "battery_state": battery_state,
                }
            )
    return sorted(result, key=lambda b: b["battery_level"] or 0)


def _pending_updates(
    raw_updates: list[dict], devices: dict[str, dict]
) -> list[dict[str, Any]]:
    return sorted(
        (
            {"name": _name(_owner(item), devices), "state": item.get("state")}
            for item in raw_updates
            if item.get("state") not in (None, "no_update")
        ),
        key=lambda u: u["name"],
    )


def _zero_brightness(raw_lights: list[dict]) -> list[str]:
    return sorted(
        (light.get("metadata") or {}).get("name", light.get("id", ""))
        for light in raw_lights
        if (light.get("dimming") or {}).get("brightness") == 0
    )
