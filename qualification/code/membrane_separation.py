import random
from matplotlib.axis import XAxis
import numpy as np
import pandas as pd
import plotly.express as px
import beccs as phd

###PSO VARIABLES###
w = 1 # inertia term
c1 = 0.8 # cognitive term
c2 = 2 # social term
target_error = 1e-6
n_iterations = 10 # number of interations
n_particles = 30 # number of particles

###VARIABLES###
y_o2in = 0.21 # initial oxygen concentration in the atmospheric air
y_n2in = 1 - y_o2in # initial nitrogen concentration in the atmospheric air
limCF = [1, 1 / y_o2in] # range of concentration factor
limRF = [0, 1] # range of recovery factor


list_RF = list(np.divide(list(range(0, 11)), 10))
plots = len(list_RF)
list_RF = np.array(list_RF * plots)
list_works = np.concatenate(np.array(list(map(lambda t: [t/10] * plots, range(1, 10)))), axis=0)

parameters = [
    phd.temperature(750, 'k'),   #[0] Tcomb
    phd.temperature(170, 'k'),   #[1] Tflue
    phd.pressure(1.01315, 'pa'), #[2] Pcomb
    0.8,                         #[3] yO2
    30,                          #[4] m_rankine
    phd.pressure(1.01315, 'pa'), #[5] PlowRankyne
    phd.pressure(2.5, 'pa'),     #[6] PexaustRankyne  
    phd.pressure(60, 'pa'),      #[7] PhighRankine
    phd.pressure(150, 'pa'),     #[8] PhighBrayton
    phd.pressure(1.01315, 'pa'), #[9] PlowBrayton
    phd.temperature(25, 'k'),    #[10] TlowRankine
    phd.temperature(25, 'k'),    #[11] TlowBrayton
    38.89,                       #[12] mBagasse
    9.73                         #[13] mStraw
]

bagasse = phd.LignoCel(0.075, 0.5, 0.405, 0.39, 0.37, 0.24, 0.02, parameters[12])
straw = phd.LignoCel(0.05, 0.25, 0.68, 0.48, 0.35, 0.17, 0.02, parameters[13])
biogas = phd.BioGas(0.6, 0.4, m=0.2651747054131219)

combustion = phd.CombustionChamber(parameters[0], parameters[2])
combustion.make_flue_gases(bagasse, straw, biogas, 0.8964753)

optmized_spaces = np.array(list(map(lambda w, rf: [rf, phd.pso_simulation(phd.initiate_spaces(target_error, n_particles, w, rf, limCF[1]), n_iterations, 0), w], list_works, list_RF)))
df = pd.DataFrame(optmized_spaces)

###DATA FRAME GENERATION###
df['ne'] = combustion.nO2 / (y_o2in*df[0])
df['N2_1'] = combustion.nO2 / (df[1]*y_o2in) * (1 - df[1]*y_o2in)
df['N2_2'] = df['ne']*(1-y_o2in) - df['N2_1']
df['O2_2'] = combustion.n_o2 * (1/df[0] - 1)
df['Power in MW'] = df['ne'] * phd.r_gases * phd.t0 * df[2] * (y_o2in*np.log(y_o2in) + y_n2in*np.log(y_n2in)) / 1e3
df.rename(columns = {0: 'Recovery factor (nO2out/nO2in)', 1:'Concentration factor (yO2out/yO2in)', 2:'fraction of the ideal separation power'}, inplace = True)
df.to_json('membrane_sep_op.json')

###PLOT###
fig = px.scatter_3d(df, x='Concentration factor (yO2out/yO2in)', y='Recovery factor (nO2out/nO2in)', z='Power in MW', color='N2 (mol/s)', symbol='fraction of the ideal separation power', color_continuous_scale='bluered')
fig.update_layout(legend=dict(yanchor="top", y=0.9, xanchor="right", x=0))
fig.update_layout(xaxis_title="N2 molar fraction", yaxis_title="Power in MW")
fig.write_html('membrane_sep_op')
fig.show()
