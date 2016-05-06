@model:2.2.1=Hasty "Model from a *A synchronized quorum of genetic clocks* "
@compartments
 cell=1
 environment=1

@species
 cell:A=1000
 cell:I=100
 cell:Hi=0
 environment:He=0


@parameters
 tau = 10
 k1 = 0.1
 C_A = 1
 C_I = 4
 d_0 = 0.88
 gamma_A = 15
 gamma_I = 24
 gamma_H = 0.01
 f = 0.3
 b = 0.06
 k = 1
 g = 0.01
 d = 0.5
 delta = 0.001
 alpha = 2500
 D = 2.5
 mu = 0.5


@reactions
@r=AiiA_synthesis
  -> A
  cell * C_A * (1 - (d/d_0)^4) * (delta + alpha * delay(Hi, tau)^2) / (1 + k1 * delay(Hi, tau)^2)

@reactions
@r=LuxI_synthesis
  -> I
  cell * C_I * (1 - (d/d_0)^4) * (delta + alpha * delay(Hi, tau)^2) / (1 + k1 * delay(Hi, tau)^2)

@reactions
@r=AiiA_degradation
  A -> 
  cell * gamma_A * A / (1 + f * (A + I))

@reactions
@r=LuxI_degradation
  I ->
  cell * gamma_I * I / (1 + f * (A + I))


@reactions
@r=AHL_synthesis
  -> Hi 
  cell * b * I / (1 + k * I)

@reactions
@r=AHL_enzymatic_degradation
  Hi ->  
  cell * gamma_H * A * Hi / (1 + g * A)

@reactions
@r=AHL_degradation
  He ->  
  environment * mu * He

@reactions
@r=AHL_diffusion1
  Hi -> He  
  cell * D * Hi

@reactions
@r=AHL_diffusion2
  He -> Hi
  cell * D * He