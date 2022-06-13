import random
import json
import numpy as np
import pandas as pd
import plotly.express as px
from CoolProp.CoolProp import PropsSI as cp
import beccs as phd

###PSO VARIABLES###
w = 1 # inertia term
c1 = 0.8 # cognitive term
c2 = 2 # social term
target_error = 1e-6
n_iterations = 10 # number of interations
n_particles = 10 # number of particles

###VARIABLES###
y_o2in = 0.21 # initial oxygen concentration in the atmospheric air
y_n2in = 1 - y_o2in # initial nitrogen concentration in the atmospheric air
limCF = [1, 1 / y_o2in] # range of concentration factor
limRF = [0, 1] # range of recovery factor

class CompressorParticle:
    def __init__(self, Pin: float, Pout:float, n_compressor: int) -> None:
        '''
        Class that describes the particle of the simulation method

        :param float RF: recovery factor (oxygen molar flow that enters the combustion chamber / oxygen molar flow that enters the membrane separation process)
        '''
        if n_compressor == 2:
            pressure_pos = np.array([(Pin+Pout) / 2])
        else:
            pressure_pos = np.linspace(Pin + 1e3, Pout, n_compressor - 1, False)
        self.pressure_position = pressure_pos
        self.pbest_position = self.pressure_position
        self.pbest_value = float('-inf') #initiate the best individual value
        self.velocity = np.array([0]*(len(self.pressure_position))) #initiate the velocity element of PSO

    def __str__(self):
        print("Intermediary positions ", self.pressure_position, " Tandem Work ", self.pbest_value)
    
    def move(self) -> None:
        '''
        Function that moves the particle in the defined space
        '''
        self.pressure_position = self.pressure_position + self.velocity
 
class Tandem:
    def __init__(self, n_compressor: float, n_particles: int, Pout: float, flow: phd.FlueGases, Tin: float) -> None:
        '''
        Class that describes the Space that the Particle will move in PSO simulation

        :param float target_error: permissible error between the target and the fitfunction value.
        :param int n_particles: number of particles in the PSO simulation
        :param float n: fraction of the minimum separation work
        '''
        self.n_particles = n_particles
        self.Pin = flow.Pflue 
        self.Pout = Pout
        self.flow = flow
        self.Tin = Tin
        self.n_compressor = n_compressor
                
        self.particles = None #list that saves the simulation particles
        self.heatex_comp = None # exergy content of the tandem compressor
        self.gbest_value = float('-inf') #initiate the global best value
        self.gbest_position = None #initiate the global best position
    

    def fitness(self, particle:CompressorParticle) -> float:
        '''
        Function that describes the goal of the PSO simulation

        :param Particle particle: a particle of the simulation
        '''
        def total_work(i: int):
            '''
            try:
                if particle.pressure_position[i-1] > particle.pressure_position[i]:
                    aux = particle.pressure_position[i]
                    particle.pressure_position[i] = particle.pressure_position[i-1]
                    particle.pressure_position[i-1] = aux
            except:
                pass
            '''
            if i == 0:
                if particle.pressure_position[i] < self.Pin:
                    particle.pressure_position[i] = self.Pin + 1e4
                Pin = self.Pin
                Pout = particle.pressure_position[i]
            elif i == self.n_compressor-1:
                if particle.pressure_position[-1] > self.Pout:
                    particle.pressure_position[-1] = self.Pout - 1e4
                Pin = particle.pressure_position[-1]
                Pout = self.Pout
            else:
                if particle.pressure_position[i-1] > particle.pressure_position[i]:
                    aux = particle.pressure_position[i]
                    particle.pressure_position[i] = particle.pressure_position[i-1]
                    particle.pressure_position[i-1] = aux
                Pin = particle.pressure_position[i-1]
                Pout = particle.pressure_position[i]
                
            p = phd.FlueGases(Pin, self.Tin, self.flow.nflue, self.flow.yN2, self.flow.yCO2, self.flow.yH2O)
            return p.work_or_heat(p.compressor(Pout, 1))
        return np.sum(list(map(total_work, range(self.n_compressor))), axis=0)

    def set_pbest(self):
        def return_pbest(particle: CompressorParticle):
            fitness_cadidate = self.fitness(particle)[0]
            if abs(particle.pbest_value) > abs(fitness_cadidate):
                particle.pbest_value = fitness_cadidate
                particle.pbest_position = particle.pressure_position
            return particle
        
        self.particles = list(map(lambda p: return_pbest(p), self.particles))
            
    def set_gbest(self):
        def return_gbest(n):
            fitness = self.fitness(self.particles[n])
            best_fitness_cadidate = fitness[0]
            ex_heat = fitness[1]
            if abs(self.gbest_value) > abs(best_fitness_cadidate):
                self.gbest_value = best_fitness_cadidate
                self.gbest_position = self.particles[n].pressure_position
                self.heatex_comp = ex_heat
            if n < len(self.particles)-1:
                n+=1
                return return_gbest(n)
        return_gbest(0)

    def move_particles(self):
        def moves_like_Jagger(n):
            global W
            new_velocity = (W*self.particles[n].velocity) + (c1*random.random()) * (self.particles[n].pbest_position - self.particles[n].pressure_position) + \
                            (random.random()*c2) * (self.gbest_position - self.particles[n].pressure_position)
            self.particles[n].velocity = new_velocity
            
            #if self.particles[n].ConcentrationFactor + new_velocity <= limCF[1]:
            self.particles[n].move()
            if n < len(self.particles) - 1:
                n+=1
                return moves_like_Jagger(n)
        moves_like_Jagger(0)

            
def initiate_tandem(n_compressor: int, n_particles: int, Pout: float, flow: phd.FlueGases):
    search_tandem = Tandem(n_compressor, n_particles, Pout, flow, parameters[14])
    search_tandem.particles = list(map(lambda _: CompressorParticle(search_tandem.Pin, Pout, n_compressor), range(n_particles)))
    return search_tandem

def PSO_simulation(search_space: Tandem, n_iterations: int, n: int=0) -> Tandem:
    search_space.set_pbest()    
    search_space.set_gbest()
    if n < n_iterations:
        n+=1
        search_space.move_particles()
        return PSO_simulation(search_space, n_iterations, n)
    else:
        return [search_space.gbest_value, search_space.heatex_comp]

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
    9.73,                        #[13] mStraw
    phd.temperature(80, 'k'),    #[14] TscO2 final in K      
    phd.pressure(80, 'pa')       #[15] scCO2 pressure       
]

bagasse = phd.LignoCel(0.075, 0.5, 0.405, 0.39, 0.37, 0.24, 0.02, parameters[12])
straw = phd.LignoCel(0.05, 0.25, 0.68, 0.48, 0.35, 0.17, 0.02, parameters[13])
biogas = phd.BioGas(0.6, 0.4, m=0.2651747054131219)

combustion = phd.CombustionChamber(parameters[0], parameters[2])
combustion.make_flue_gases(bagasse, straw, biogas, parameters[3])

###UPDATE DATAFRAME###
df = pd.read_json('/home/luismazini/Documents/phd/oxycombustion/membrane_sep_op.json')
df.loc[len(df.index)] = [1, 1 / y_o2in, 1, combustion.n_o2 / y_o2in, 0, combustion.n_o2 / y_o2in * (1-y_o2in), 0, combustion.n_o2 / y_o2in * phd.r_gases * phd.t0 * (y_o2in*np.log(y_o2in) + y_n2in*np.log(y_n2in)) / 1e3]
list_df = np.array(df)

list_combustion = list(map(lambda yo2: combustion.make_flue_gases(bagasse, straw, biogas, yo2[1] * y_o2in), list_df))
list_dwt = list(map(lambda comb: [comb.dewatering(), comb.nflue * comb.y_co2 * phd.prop_cp['co2']['molar_mass']], list_combustion))

optmized_spaces = np.array(list(map(lambda df_element, dwt: [
    df_element[0],                                                                             # [0] Recovery factor
    df_element[1],                                                                             # [1] Concentration factor
    df_element[2],                                                                             # [2] Fraction of the ideal separation power
    biogas.exergy() / 1e3,                                                                     # [3] Biogas exergy content
    bagasse.exergy() / 1e3,                                                                    # [4] Bagasse exergy content
    straw.exergy() / 1e3,                                                                      # [5] Sraw exergy content
    phd.pso_simulation_tandem(                                                                 # [6] Tandem power consumption in MW and the Heat exergy content in MW
        phd.initiate_tandem(5, n_particles, parameters[15], dwt[0][0]), n_iterations), 
    dwt[0][1] / 1e3,                                                                           # [7] Dewatering heat exergy content in MW
    df_element[7],                                                                             # [8] Separation power in MW
    - dwt[1]*(phd.prop_cp['co2']['chem_x']+phd.ph_exergy(                                      # [9] sCO2 exergy content
        cp('H', 'T', parameters[14], 'P', parameters[15], 'CO2') / 1e3,
        cp('S', 'T', parameters[14], 'P', parameters[15], 'CO2') / 1e3,
        cp('H', 'T', phd.t0, 'P', phd.p0, 'CO2') / 1e3,
        cp('S', 'T', phd.t0, 'P', phd.p0, 'CO2') / 1e3)) / 1e3,
    - df_element[5]*phd.prop_cp['n2']['molar_mass']*phd.prop_cp['n2']['chem_x']/1e3,           # [10] N2 B exit exergy content in MW
    - df_element[6]*phd.prop_cp['oxygen']['molar_mass']*phd.prop_cp['oxygen']['chem_x']/1e3,   # [11] O2 B exit exergy content in MW
    - df_element[4]*(phd.prop_cp['n2']['chem_x']+phd.ph_exergy(                                # [12] N2 non-condensable exergy content in MW
        cp('H', 'T', parameters[14], 'P', parameters[15], 'N2') / 1e3,
        cp('S', 'T', parameters[14], 'P', parameters[15], 'N2') / 1e3,
        cp('H', 'T', phd.t0, 'P', phd.p0, 'N2') / 1e3,
        cp('S', 'T', phd.t0, 'P', phd.p0, 'N2') / 1e3)) / 1e3
    ], list_df, list_dwt)))

df = pd.DataFrame(optmized_spaces)
df = df.dropna()
df.rename(columns = {
    0: 'Recovery factor (nO2out/nO2in)', 
    1: 'Concentration factor (yO2out/yO2in)', 
    2: 'Ideal separation power %',
    3: 'Biogas exergy content (MW)',
    4: 'Bagasse exergy content (MW)',
    5: 'Straw exergy content (MW)',
    7: 'Dewatering heat exergy content (MW)',
    8: 'Separation power in MW',
    9: 'Supercritical carbon dioxide exergy content (MW)',
    10: 'N2 outflow (separation) exergy content (MW)',
    11: 'O2 outflow (separation) exergy content (MW)',
    12: 'N2 outflow (non-condensables) exergy content (MW)'}, inplace = True)
df['Compression power in MW', 'Compression heat exergy in MW'] = df[6].to_list()
df.drop(6, axis=1, inplace=True)
df.to_json('total_power_consumption.json')
