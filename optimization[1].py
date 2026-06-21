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
    box_capacity=2,
    volume_capacity=6,
    sensing_range_m=45,
    max_boxes_per_location=2
):
    catalog = load_sensor_catalog()

    if recommended_sensors is None:
        recommended_sensors = ["pH", "Moisture", "SoilTemp", "NDVI"]

    S = [s for s in recommended_sensors if s in catalog]

    if targets and len(targets) > 0:
        N = [f"T{i+1}" for i in range(len(targets))]
        coords = {
            f"T{i+1}": (float(t["lat"]), float(t["lng"]))
            for i, t in enumerate(targets)
        }
    else:
        count = target_count if target_count else 8
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

    # Candidate box slots:
    # Her target noktasına en fazla 2 küçük sensör kutusu kurulabilir.
    # Böylece tek kutuya bütün sensörleri doldurmak engellenir.
    C = []
    candidate_location = {}

    for j in N:
        for b in range(1, max_boxes_per_location + 1):
            c = f"{j}_B{b}"
            C.append(c)
            candidate_location[c] = j

    coverage = {}

    for i in N:
        for c in C:
            j = candidate_location[c]

            if coords:
                lat_i, lng_i = coords[i]
                lat_j, lng_j = coords[j]
                distance = haversine_m(lat_i, lng_i, lat_j, lng_j)
                coverage[i, c] = 1 if distance <= sensing_range_m else 0
            else:
                i_num = int(i[1:])
                j_num = int(j[1:])
                coverage[i, c] = 1 if abs(i_num - j_num) <= 1 else 0

    mdl = Model("SmartAgricultureSensorPlacement")

    x = mdl.binary_var_dict(C, name="box")
    y = mdl.binary_var_matrix(C, S, name="sensor")

    mdl.minimize(
        mdl.sum(box_cost * x[c] for c in C)
        + mdl.sum(cs[s] * y[c, s] for c in C for s in S)
    )

    # Sensor can be assigned only if box is installed
    for c in C:
        for s in S:
            mdl.add_constraint(y[c, s] <= x[c])

    # If a box is installed, it must contain at least one sensor
    for c in C:
        mdl.add_constraint(x[c] <= mdl.sum(y[c, s] for s in S))

    # Each box can contain limited number of sensors
    for c in C:
        mdl.add_constraint(mdl.sum(y[c, s] for s in S) <= box_capacity)

    # Volume capacity
    for c in C:
        mdl.add_constraint(
            mdl.sum(v[s] * y[c, s] for s in S) <= volume_capacity * x[c]
        )

    # Coverage constraint
    # Each target must be covered by every selected sensor type.
    for i in N:
        for s in S:
            mdl.add_constraint(
                mdl.sum(coverage[i, c] * y[c, s] for c in C) >= 1
            )

    solution = mdl.solve(log_output=False)

    if not solution:
        return {
            "status": "No feasible solution",
            "minimum_cost": None,
            "selected_boxes": [],
            "sensor_assignments": [],
            "message": "No feasible solution found. Try increasing sensing range or target density."
        }

    selected_boxes = [c for c in C if solution[x[c]] > 0.5]

    sensor_assignments = []
    box_plan = []

    for c in selected_boxes:
        loc = candidate_location[c]
        sensors_in_box = []

        for s in S:
            if solution[y[c, s]] > 0.5:
                sensors_in_box.append(s)
                sensor_assignments.append({
                    "box": c,
                    "location": loc,
                    "sensor": s
                })

        item = {
            "box": c,
            "location": loc,
            "sensors": sensors_in_box
        }

        if coords:
            item["lat"] = coords[loc][0]
            item["lng"] = coords[loc][1]

        box_plan.append(item)

    sensor_cost = sum(
        cs[item["sensor"]]
        for item in sensor_assignments
    )

    box_cost_total = len(selected_boxes) * box_cost
    gateway_cost = 3500
    installation_cost = 2000

    total_cost = sensor_cost + box_cost_total + gateway_cost + installation_cost

    return {
        "status": "Optimal",
        "minimum_cost": total_cost,

        "sensor_cost": sensor_cost,
        "box_cost": box_cost_total,
        "gateway_cost": gateway_cost,
        "installation_cost": installation_cost,

        "selected_boxes": selected_boxes,
        "box_plan": box_plan,
        "sensor_assignments": sensor_assignments,

        "used_sensors": S,
        "target_count": len(N),
        "box_count": len(selected_boxes),
        "sensing_range_m": sensing_range_m,
        "box_capacity": box_capacity
    }


if __name__ == "__main__":
    sample_targets = [
        {"lat": 41.06902, "lng": 28.94537},
        {"lat": 41.06851, "lng": 28.94459},
        {"lat": 41.06837, "lng": 28.94540},
        {"lat": 41.06872, "lng": 28.94569},
        {"lat": 41.06882, "lng": 28.94505},
        {"lat": 41.06867, "lng": 28.94520},
    ]

    result = solve_sensor_placement(
        targets=sample_targets,
        recommended_sensors=["pH", "Moisture", "SoilTemp", "NDVI"]
    )

    print(result)
