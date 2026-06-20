<p align="center">
  <img
    src="GitHubbanner.png"
    width="100%"
  />
</p>

# A Third Way in Neoclassical Realist Theory of International Politics

The project develops a historically anchored agent-based model (ABM) to examine how three Type III Neoclassical Realist variables shape alliance-related decision-making and system-level alliance-network outcomes:

Material capacity
Strategic environment
Systemic clarity

The model covers 38 countries between 1995 and 2012. It combines cross-national historical data with a rule-based simulation of state decision-making under external pressure.

## Core Argument

The model represents a Type III Neoclassical Realist mechanism in which systemic pressures are filtered through domestic and perceptual conditions before shaping state behaviour.

The three focal systemic variables operate through distinct channels:

Variable	Role in the model
Strategic environment	Captures the intensity of external pressure facing a state.
Systemic clarity	Determines how clearly a state can interpret external pressure and whether pressure translates into action.
Material capacity	Shapes whether a state can resist its primary threat independently or with support from allies.

Domestic and unit-level variables modify the systemic inputs before they enter the ABM. These filters include:

leader images and beliefs;
strategic culture;
state-society relations;
domestic institutional constraints.

## Model Overview

Each annual simulation round follows a historically anchored sequence:

The model begins from the observed Correlates of War alliance network in year t.
Country agents receive country-year inputs for strategic environment, systemic clarity, and material capacity.
Each country identifies its primary threat from directed dyadic pressure.
Agents calculate action pressure, relative resistance, vulnerability, and policy utilities.
Agents select one of three policy orientations:
Status Quo
Balance
Bandwagon
Eligible agents stage potential alliance additions or removals.
The model applies formalisation rules, receiver acceptance, and an empirical annual alliance-change budget.
The resulting simulated network is compared with the observed alliance network in year t + 1.
The next annual round begins again from the observed historical network for that year.

The model therefore performs a sequence of historically anchored annual transition experiments rather than an unrestricted recursive simulation.

## Historical Anchoring

A central design feature of the model is its historical anchor.

At the beginning of each year, the alliance graph is reset to the observed Correlates of War alliance network for that year. The model then simulates a limited set of theoretically motivated alliance changes.

This means that:

the simulated network from year t is not carried forward as the starting network for year t + 1;
every annual round begins from historically observed alliance relations;
the model evaluates one-year-ahead structural plausibility rather than unrestricted long-run prediction;
the number of committed changes is constrained by an empirical historical formalisation budget.

This design allows the model to isolate the effect of the focal variables while avoiding the accumulation of unrealistic errors across many years.

## Ablation Design

The model evaluates the relative importance of the three focal variables using mean-neutralised ablations.

Four conditions are estimated:

Condition	Description
baseline	Retains observed variation in all focal variables.
neutralized_clarity	Replaces systemic clarity with its global sample mean.
neutralized_strategic_env	Replaces strategic environment with its global sample mean.
neutralized_capacity	Replaces the relevant capacity input with its global sample mean in the resistance-vulnerability mechanism.

For each ablation, the model compares deviations from the baseline across three dimensions:

Policy deviation
Changes in the distribution of Status Quo, Balance, and Bandwagon orientations.
Mechanism deviation
Changes in action pressure, resistance, vulnerability, utilities, and formalisation eligibility.
System deviation
Changes in alliance-network structure, including edge count, density, modularity, and community structure.

A larger score indicates that neutralising the relevant variable produces a larger deviation from the baseline model within the specified mechanism.

## Interpretation of Results

The ablation analysis should be interpreted as evidence of relative importance within the specified model mechanism.

The model does not claim that a single variable universally determines real-world alliance politics. Instead, it evaluates how strongly each focal input affects:

policy orientations;
internal decision mechanisms;
constrained local deviations from the historically observed alliance network.

The historical anchor is an intentional design choice. Strong similarity between baseline and observed alliance networks should therefore be interpreted as structural plausibility within a historically constrained simulation, not as evidence that the model reconstructs alliance history from first principles.
