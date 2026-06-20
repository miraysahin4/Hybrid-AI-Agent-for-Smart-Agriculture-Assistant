import json
import math
from docplex.mp.model import Model


def load_sensor_catalog(path="sensor_catalog.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)

    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def solve_sensor_placement(
    targets=None,
    target_count=None,
    recommended_sensors=None,
    box_cost=5000,
    box_capacity=6,
    volume_capacity=12,
    sensing_range_m=80
):
    catalog = load_sensor_catalog()

    if recommended_sensors is None:
        recommended_sensors = ["pH", "Moisture", "EC", "SoilTemp", "NDVI"]

    S = [s for s in recommended_sensors if s in catalog]

    if targets and len(targets) > 0:
        N = [f"T{i+1}" for i in range(len(targets))]
        coords = {
            f"T{i+1}": (float(t["lat"]), float(t["lng"]))
            for i, t in enumerate(targets)
        }
    else:
        count = target_count if target_count else 5
        N = [f"T{i+1}" for i in range(count)]
        coords = None

    if not N or not S:
        return {
            "status": "No data",
            "minimum_cost": None,
            "selected_boxes": [],
            "sensor_assignments": []
        }

    cs = {s: catalog[s]["cost"] for s in S}
    v = {s: catalog[s]["volume"] for s in S}

    coverage = {}

    for i in N:
        for j in N:
            if coords:
                lat_i, lng_i = coords[i]
                lat_j, lng_j = coords[j]
                distance = haversine_m(lat_i, lng_i, lat_j, lng_j)
                coverage[i, j] = 1 if distance <= sensing_range_m else 0
            else:
                i_num = int(i[1:])
                j_num = int(j[1:])
                coverage[i, j] = 1 if abs(i_num - j_num) <= 1 else 0

    mdl = Model("SmartAgricultureSensorPlacement")

    x = mdl.binary_var_dict(N, name="box")
    y = mdl.binary_var_matrix(N, S, name="sensor")

    mdl.minimize(
        mdl.sum(box_cost * x[j] for j in N)
        + mdl.sum(cs[s] * y[j, s] for j in N for s in S)
    )

    for j in N:
        for s in S:
            mdl.add_constraint(y[j, s] <= x[j])

    for j in N:
        mdl.add_constraint(x[j] <= mdl.sum(y[j, s] for s in S))

    for j in N:
        mdl.add_constraint(mdl.sum(y[j, s] for s in S) <= box_capacity)

    for j in N:
        mdl.add_constraint(
            mdl.sum(v[s] * y[j, s] for s in S) <= volume_capacity * x[j]
        )

    for i in N:
        for s in S:
            mdl.add_constraint(
                mdl.sum(coverage[i, j] * y[j, s] for j in N) >= 1
            )

    solution = mdl.solve(log_output=False)

    if not solution:
        return {
            "status": "No feasible solution",
            "minimum_cost": None,
            "selected_boxes": [],
            "sensor_assignments": []
        }

    sensor_cost = 0

    for j in N:
        for s in S:
            if solution[y[j, s]] > 0.5:
                sensor_cost += cs[s]

    selected_boxes = [
        j for j in N
        if solution[x[j]] > 0.5
    ]

    box_cost_total = len(selected_boxes) * box_cost

    gateway_cost = 3500
    installation_cost = 2000

    total_cost = (
        sensor_cost
        + box_cost_total
        + gateway_cost
        + installation_cost
    )

    return {
        "status": "Optimal",
        "minimum_cost": total_cost,

        "sensor_cost": sensor_cost,
        "box_cost": box_cost_total,
        "gateway_cost": gateway_cost,
        "installation_cost": installation_cost,

        "selected_boxes": selected_boxes,

        "sensor_assignments": [
            {"location": j, "sensor": s}
            for j in N
            for s in S
            if solution[y[j, s]] > 0.5
        ],

        "sensing_range_m": sensing_range_m
    }


if __name__ == "__main__":
    sample_targets = [
        {"lat": 41.06827, "lng": 28.94506},
        {"lat": 41.06865, "lng": 28.94521},
        {"lat": 41.06910, "lng": 28.94580}
    ]

    result = solve_sensor_placement(
        targets=sample_targets,
        recommended_sensors=["pH", "Moisture", "EC", "SoilTemp", "NDVI"]
    )

    print(result)