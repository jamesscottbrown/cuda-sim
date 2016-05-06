
@model:2.3.0=SIR "SIR model with delay"
@compartments
 cell=1

@species
 cell:Susceptible=1000
 cell:Infected=100
 cell:Recovered=0

@parameters
 tau1 = 1
 tau2 = 10

@reactions



@r=Infection
 Susceptible -> Infected
 Susceptible*delay(Infected, tau1)

@r=Recovery
 Infected -> Recovered
 Infected

@r=Relaps
 Recovered -> Susceptible
 delay(Infected, tau2)