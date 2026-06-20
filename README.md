# Hybrid-AI-Agent-for-Smart-Agriculture-Assistant

Smart Agriculture Assistant is an AI-powered decision support system developed as a senior design (capstone) project.

The system helps farmers determine suitable crops, select appropriate sensors, optimize sensor deployment, and estimate system costs using geospatial information, large language models (LLMs), and mathematical optimization techniques.

---

## Project Overview

The application allows users to:

- Select an agricultural field on an interactive map
- Define target monitoring points
- Receive crop recommendations based on location and field conditions
- Obtain sensor recommendations for selected crops
- Generate coverage matrices automatically
- Calculate sensor deployment costs
- Optimize sensor placement using IBM CPLEX

The project combines Artificial Intelligence, Geographic Information Systems (GIS), and Operations Research techniques into a single decision-support platform.

---

## Features

### Crop Recommendation
Users can ask:

> What grows best in this field?

The system analyzes the selected location and provides suitable crop recommendations.

### Sensor Recommendation

Users can ask:

> I want to grow apples in this field. Which sensors do you recommend?

The system recommends appropriate sensors such as:

- pH
- Moisture
- EC
- Soil Temperature
- NDVI
- NPK

### Technical Sensor Analysis

After users select sensors, the system automatically provides:

- Measurement intervals
- Critical value ranges
- Detection ranges
- Coverage matrix information

### Coverage Matrix Generation

Coverage matrices are generated automatically using:

- User-defined target points
- Sensor sensing ranges
- Geographic coordinates

The matrices are calculated in Python and used by the optimization model.

### Cost Analysis

The system estimates:

- Sensor costs
- Sensor box costs
- Gateway costs
- Installation costs

and calculates the total deployment cost.

### Sensor Placement Optimization

IBM CPLEX is used to determine:

- Optimal sensor box locations
- Sensor assignments
- Minimum deployment cost

subject to:

- Coverage constraints
- Box capacity constraints
- Sensor volume constraints

---

## Technologies Used

### Frontend

- HTML
- CSS
- JavaScript
- Bootstrap
- Leaflet.js
- Leaflet Draw

### Backend

- Python
- Flask

### Artificial Intelligence

- Claude API (Anthropic)

### Optimization

- IBM CPLEX
- DOcplex

### GIS

- OpenStreetMap
- Geospatial Target Point Analysis

---

## Mathematical Model

The optimization model minimizes the total deployment cost:

Minimize:

- Sensor box costs
- Sensor costs

Subject to:

- Coverage constraints
- Sensor capacity constraints
- Volume capacity constraints

Binary decision variables are used to determine:

- Sensor box installation locations
- Sensor assignments

The model is solved using IBM CPLEX.

---

## Example Workflow

1. Draw field boundary
2. Add target monitoring points
3. Ask for crop recommendations
4. Request sensor recommendations
5. Select desired sensors
6. Generate coverage matrix
7. Run cost analysis
8. Obtain optimal sensor deployment plan

---

## Authors

Gamze Kılıç

Miray Şahin

Senior Design Project

Industrial Engineering 

Istanbul Bilgi University

2025–2026
