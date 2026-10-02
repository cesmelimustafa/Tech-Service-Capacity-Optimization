# Discrete Event Simulation & Capacity Optimization for Tech Service Center

## Project Overview
This project develops a Discrete Event Simulation model to analyze and optimize customer acceptance and fault detection processes in a high-traffic smartphone repair center. 

Using Arena, the model identifies critical bottlenecks in resource allocation and evaluates pilot scenarios to increase system throughput, reduce flow time, and integrate idle staff into customer retention.

## Key Objectives
*   Bottleneck Analysis: Measure resource utilization, queue lengths, and flow times to identify workflow constraints.
*   Input Modeling: Perform statistical goodness-of-fit tests on historical data to model interarrival and service time distributions.
*   Scenario Testing: Simulate capacity expansion and process reallocations to improve financial sustainability and customer satisfaction.

## Methodology & Modeling Logic
*   Data Analysis: Analyzed raw repair data using Python to extract distributions. Interarrival times follow an Exponential distribution, and service times follow a Lognormal distribution.
*   Routing Logic: Modeled decision nodes including warranty validation, user-induced fault detection, and quote approval rates.
*   Queueing Discipline: Implemented strict FIFO policies across all processing modules.

## Current State vs. Proposed Optimization
### Baseline Model Bottleneck
*   Resource Saturation: Technician utilization reached 99.58 percent, establishing the repair phase as the primary system constraint.
*   Excessive Flow Time: Average total flow time for a device reached 143.76 hours due to extreme queueing.

### Strategic Intervention
*   Capacity Expansion: Increased technical repair staff from 2 to 3 operators.
*   Process Reallocation: Reassigned idle time from the cashier node to handle informational tasks for out-of-warranty customers.

## Business Impact & Results
*   Flow Time Reduction: Reduced average flow time from 143.76 hours down to 83.02 hours.
*   Queue Elimination: Eliminated the critical queue metric of customers waiting over 35 minutes down to zero.
*   Financial Efficiency: Utilizing the cashier for customer information mitigated the loss of out-of-warranty clients and increased paid repair acceptance rates.

## Repository Structure
*   `src/input_data_analysis.py`: Python script for statistical distribution fitting.
*   `docs/Tech_Service_Capacity_Optimization.pdf`: Complete methodology and simulation report.

## Author
**Mustafa Cesmeli**
Industrial Engineer
(LinkedIn: https://www.linkedin.com/in/mustafacesmeli/) | (Mail: cesmelimustafa0@gmail.com)
