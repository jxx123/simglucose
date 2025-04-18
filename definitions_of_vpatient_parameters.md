## Definitions of Parameters

Below is my attempt at describing all of the parameters in ```vpatient_params.csv```. They were infered from a combination of 2 papers and inspecting the codebase itself it the parameter names were not perfectly clear. I tried to keep them in order of appearance in the ```vpatient_params.csv```. The original Meal Simulation Model of the Glucose-Insulin System (for T2D and Normal Adults) provides a lot more detail of the parameters, there are some parameters not discussed in that paper, and I needed to get the information from the UVA/PADOVA Type 1 Diabetes Simulator: New Features paper.

1. [Meal Simulation Model of the Glucose-Insulin System](https://ieeexplore.ieee.org/abstract/document/4303268/references#references)
2. [The UVA/PADOVA Type 1 Diabetes Simulator: New Features](https://journals.sagepub.com/doi/10.1177/1932296813514502)

I would be happy if others improve, clean, or correct some of the definitions below.

### Definitions

* BW (kg) - Body Weight 
* EGPb (mg/kg/min) - Edogenous Glucose Production, basal steady-state edogenous production, b suffix denotes basal state.
  * $EGP_b = EGP(0)$
  * $EGP(t) = k_{p1} - k_{p2} * G_{p}(t) - k_{p3} * I_{d}(t) - k_{p4} * I_{p0}(t)$
  * EGPb = Ub + Eb <- in normal subjects this is 0. $EGP_b = U_{b} + E_{b}$
  * Ub (mg/kg/min) - Glucose utilization $U(0) = U_{b}$
  * Eb (mg/kg/min) - Renal excretion $E(0) = E_{b}$
* Gb (mg/dl) - plasma Glucose concentration, b suffix denotes basal state: randomly generated from the joint distribution with an average of 120 mg/dl.  In S2008, Gb was randomly generated from the joint distribution with an average of 140 mg/dl (chosen to reflect the knowledge available to the authors at that time). 
  * Gb = Gp/Vg
  * Gp (mg/kg) - glucoses masses in plasma and glucose masses in rapidly equilibrating tissues $G_p(0) = G_{pb}$
  * Gt (mg/kg) - slowly equilibrating tissues $G_{t}(0) = G_{tb}$
  * Vg (dl/kg) - the distribution volume of glucose $V_{G}$
* Ib (pmol/l) - Insulin plasma concentration, b suffix denotes basal state. $I(0) = I_{b}$
  * Ib = Ip/Vi
  * Ip (pmol/l) - inslin masses in plasma $I_{pb} = I_{p}(0)$
  * Vi (l/kg) - distribution volume of insulin $V_{I}$
* kabs ($min^{-1}$) - Rate of appearance process, parameter - $k_{abs}$ is the rate constant of intestinal (gut) absorption 
* kmax ($min^{-1}$) - Rate of appearance process, parameter - $k_{max}$
* kmin ($min^{-1}$)- Rate of appearance process, parameter - $k_{min}$
* b (dimensionless) - Rate of appearance process, parameter
* d (dimensionless) - Rate of appearance process, parameter
* Vg (dl/kg) - Distribution volume of glucose, $V_{G}$
* Vi (l/kg) - Distribution volume of insulin, $V_{I}$
* Ipb (pmol/kg) - insulin masses in plasma, b suffix denotes basal state. $I_{pb} = I_{p}(0)$
* Vmx (mg/kg/min per pmol/l) - Utilization parameter quantifying peripheral insulin action, $V_{mx}$
* Km0 (mg/kg) - Utilization parameter 
* k2 ($min^{-1}$) - Glucose kinetics process, rate parameter (rates $G_p$ variable)
* k1 ($min^{-1}$) - Glucose kinetics process, rate parameter (rates $G_t$ variable)
* p2u ($min^{-1}$) - Utilization parameter  
* m1 ($min^{-1}$) - Insulin kenetics process, rate parameter (rates $I_l$ variables)
* m5 (min*kg/pmol) - Insulin kenetics process, rate parameter (rates )
* CL - I don't see this used in the code and no mention of it in the papers, not sure what I'm missing with this one.
* HEb (dimensionless) - Insulin kenectics process, Hepatic extraction of insulin, i.e., the insulin flux which leaves the liver irreversibly divided by the total insulin flux leaving the liver, b suffix denotes basal state.
  * $HE_b = HE(0) = -m5*S(0) + m6$
  * S (pmol/kg/min) - insulin secretion, I guess this is assumed to be 0 in T1D hence why there is no S parameter, (probably bad assumption)
  * Therefore, in T1D, $m_6$ would be redundant because $HE_b = m_6$ (based on my assumption probably not correct)
* m2 ($min^{-1}$) - Insulin kenetics rate parameter (rates $I_l$ variable)
  * $m_2 = -m_4/HE_{b}$, if assuming S = 0
  * $m_2 = (\frac{S_b}{I_{pb}}-\frac{m_4}{1-HE_{b}})*\frac{1-HE_{b}}{HE_{b}}$ 
* m4 ($min^{-1}$) - Insulin kenetics rate parameter (rates $I_p$ variable), corresponds to peripheral degradation and has been assumed linear
  * $m_4 = \frac{2}{5}*\frac{S_{b}}{I_{pb}}*(1-HE_b)$, generally
* m30 (min^{-1}) - Insulin kenetics process, This really means the initial m3 value because m3 varies over time. 
  * $m_3(0) = \frac{HE_{b}*m_{1}}{1 - HE_{b}}$
  * $m_3(t) = \frac{HE(t)*m_{1}}{1 - HE(t)}$
* Ilb (pmol/kg) - insulin masses in liver, b suffix denotes basal state. $I_{lb} = I_{l}(0)$
* ki ($mg/kg/min$) - Edogenous production process, rate parameter accounting for delay between insulin signal and insulin action
* kp2 ($min^{-1}$) - Edogenous production process, $k_{p2}$ liver (hepatic) glucose effectiveness
* kp3 ($mg/kg/min per pmol/l$) - Edogenous production process, $k_{p3}$ parameter governing amplitude of insulin action on the liver, quantifying hepatic insulin action. 
* f (dimensionless) - Rate of Appearance process, (see Variables not included)
* Gpb(mg/kg) - glucoses masses in plasma and glucose masses in rapidly equilibrating tissues, b indicates basal level.
  * $G_{pb} = G_{p}(0)$
* ke1 ($min^{-1}$) - Renal Excretion processs, used in $E(t) = k_{e1} * [G_{p}(t)-k_{e2}]$ if $G_{p}(t) > k_{e2}$, else $E(t) = 0$
* ke2 (mg/kg) - Renal Excretion processs, used in $E(t) = k_{e1} * [G_{p}(t)-k_{e2}]$ if $G_{p}(t) > k_{e2}$, else $E(t) = 0$
* Fsnc (mg/kg/min) - I believe this is a typo of $F_{cns}$ where I assume cns means central nervous system. This corresponds to glucose utilization by the brain and erythrocytes. 
  * $U_{ii}(t) = F_{cns}$
* Gtb - I do not think this is used in the codebase, $G_{tb} = G_{t}(0)$ is at basal steady state
  * $G_{tb} = \frac{F_{cns}-EGP_{b} + k_{1}*G_{pb}}{k_{2}}$
* Vm0 (mg/kg/min) - Utilization process, $V_{m0}$ maximum utilization by tissue at basal insulin
  * $V_{m0} = \frac{(EGP_{b}-F_{cns})*(K_{m0}+G_{tb})}{G_{tb}}$
* Rdb - can't find reference, not used in code base
* PCRb - can't find reference, not used in code base
#### Subcutaneous Insulin Kinetics
There are four equations that define insulin kinetics, the below variables define how these are calculated
* Equations from papers:
  * $\dot{I}_{sc1} = -(k_{d} + k_{a1})*I_{sc1}(t) + IIR(t)$ 
  * $I_{sc1}(0) = I_{sc1ss}$, see isclss below
  * $\dot{I}_{sc2} = k_{d}*I_{sc1}(t) - k_{a2}*I_{sc2}(t)$ 
  * $I_{sc2}(0) = I_{sc2ss}$, see isc2ss below
* kd - used in calculation of subcutaneous insulin kinetics $k_{d}$ (see above)
* ksc - used in calculation of subcutaneous glucose $k_{sc}$ (see above)
* ka1 - used in calculation of subcutaneous insulin kinetics $k_{a1}$ (see above)
* ka2 - used in calculation of subcutaneous insulin kinetics $k_{a2}$ (see above)
* doskempt - I assume this refers to $k_{empt}$ which is the rate constant of gastric emptying, which is a nonlinear function of $Q_{sto}$
  * $Q_{sto}$ (mg) - is the amount of glucose in the stomach
* u2ss - used for calculating basal insulin in ```t1dpatient.py``` see ```basal = p._params.u2ss * p._params.BW / 6000  # U/min```
* isclss - $I_{sc1}(0) = I_{sc1ss}$, the first steady state of subcutaneous insulin parameter
* isc2ss - $I_{sc2}(0) = I_{sc2ss}$, the second steady state of subcutaneous insulin parameter
* sp1 - could not find in code base
* patient_history - could not find in code base

#### Variables/Equations not included:
These variable are worth noting because the above variables directly impact below, and do not make sense without the context.
* $Ra(t)$ (mg/kg/min) - the glucose rate of apperance in plasma 
  * $Ra(t) = \frac{f*k_{abs}*Q_{gut}(t)}{BW}$
* $k_{p1}$ (mg/kg/min) - is the extrapolated EGP at zero glucose and insulin
