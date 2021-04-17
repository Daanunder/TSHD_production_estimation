Documentation
============
This is the documentation for an object based module that estimates production of a TSHD dredger based on a simple model. As explained in [the background file](https://github.com/Daanunder/TSHD\_production\_estimation/blob/f6204a9109c82bab55e72d9469810729a406d4d1/Background.ipynb 'Background explanation of the model'). The model calculates the produced mixture density of a jet- and cutting dredger based on the moment balance given a certain velocity. It therefore consists of two parts, the jet production estimation based on Miedema (2019) and the cutting production estimation based on the moment balance of forces. Below is a short description of the setup and methods of the TSHD object and finally some examples of the output. 

<!-- make sure to run the following: ./gh-md-toc \-\-insert README.test.md-->
<!--ts-->
   * [Documentation](#documentation)
      * [Requirements](#requirements)
      * [Initial Setup](#initial-setup)
         * [General parameters](#general-parameters)
         * [TSHD Parameters](#tshd-parameters)
         * [Model parameters](#model-parameters)
         * [Changing parameters](#changing-parameters)
      * [Jet Production](#jet-production)
         * [Usage](#usage)
         * [Functions](#functions)
      * [Cutting Production](#cutting-production)
         * [Usage](#usage-1)
            * [Calculate initial moment and forces for all velocities](#calculate-initial-moment-and-forces-for-all-velocities)
            * [Run numerical iteration to obtain actual forces and moments](#run-numerical-iteration-to-obtain-actual-forces-and-moments)
            * [Plot iterations, forces and moments](#plot-iterations-forces-and-moments)
         * [Function](#function)
      * [Total Production](#total-production)
         * [Usage](#usage-2)
            * [Create total production data](#create-total-production-data)
            * [Plot data](#plot-data)
            * [Compare influence of different parameters](#compare-influence-of-different-parameters)
         * [Functions](#functions-1)
      * [Extra function](#extra-function)

<!-- Added by: garry, at: Fri 16 Apr 2021 12:36:18 PM UTC -->

<!--te-->

## Requirements
* Numpy
* Pandas
* Matplotlib.pyplot
* Seaborn

## Initial Setup
One can initiate an object as usual:
```python
tshd = trailing_suction_hopper_dredger()
```

### General parameters
By default the following general parameters are defined, related to the environmental conditions;

```python
self.water_density = 1025 # kg/m3
self.sand_density = 2650 # kg/m3
self.steel_density = 7800 # kg/m3
self.permeability = 1*10**-4 #m/s
self.situ_porosity = 0.4 #m/s
self.internal_friction_angle = 35*np.pi/180 #rad
self.steel_sand_friction_angle = 26*np.pi/180 #rad
self.z_coordinate = 30 #m
```

### TSHD Parameters
For the TSHD vessel and dredging arm general parameters and values related to the blade, the visor and the nozzle/jet are defined by default. Where the 'trailing_veloc' is the upper limit of the velocity range for which the production is estimated. The other parameters are self explanatory, as follows; 
```python
# Ship
self.trailing_veloc = 2 #m/s
self.pipe_diam  = 0.762 #m
self.line_speed = 6 #m/s

```
```python
# Blade parameters
self.blade_lenght = 0.1 #m
self.blade_width = 2
self.blade_angle = 45 * np.pi/180 #rad

```
```python
# Visor parameters
self.visor_width = 4 #m
self.visor_height = 2 #m
self.visor_weight = 5500 #kg
self.visor_hor_radius = 2 #m
self.initial_visor_angle = 20 *np.pi/180 #rad
self.lower_suction_angle = 45 * np.pi/180 #rad

```
```python
# Nozzle Parameters
self.pressure = 1*10**3 #kpa
self.nozzle_diam = 0.0254 #m
self.jet_exit_veloc = 5 #m/s
self.nozzles_drag_head = 16 # - 
```
```python
# Jet parameters
self.vel_width_ratio = 1
self.effective_width_power = 0.8
```
### Model parameters
A few other model specific parameters are defined as follows; 
```python
# Model configuration
self.lower_limit_beta = 5*np.pi/180
self.upper_limit_beta = 90*np.pi/180
```
Where the above limits are required to avoid zero-division errors and enable smooth computation of the non-dimensional cutting forces. The paramters below are, respectively:
* hi_method: The method to calculate the jet depth (both 'CSB' and 'Miedema' are accepted)
* cavitating: A boolean to enable a check for cavitational cutting [NOT IMPLEMENTED]
* nodes: The number of nodes used for the velocity range for which the production is estimated. 
* accuracy: The accuracy limit for the moment force balance iteration

```python
# Misc
self.hi_method = 'Miedema'
self.cavitating = True
self.nodes = 5000
self.accuracy = 10 #Nm
    
```

### Changing parameters 
Everytime a parameter is changed manually the 'reinitiate' function has to be called since implicitly some arguments are defined after initiation of the model instance. As deomonstrated below:
```python
tshd = trailing_suction_hopper_dredger()
tshd.trailing_velocity = 5.0
tshd._reinitiate()
```
### Plot initial model
To plot the configured model, at any time, one can call the function below to plot the configured model for a certain trailing velocity (defined by it's index called **cur_index**).
```python
tshd = trailing_suction_hopper_dredger()
tshd.plot_model()
```

Where the function works as follows:
**tshd.plot\_model(self, cur\_index=None):**
- Creates (2x1) plot of the dredge arm and draghed
- Requires cur_index [Defaults to self.initial_visor_angle]
- Returns **plt.show()**

## Jet Production
For the calculation of the jet production a few functions are called; first the jet depth is calculated based on the defined method, then the jet width, the draghead pressure and the volumentric flow in the pipe are calculated, this gives the mixture density based on the constant volume system of the dredger. All these calculations can be done by calling a single function: 

### Usage
```python
tshd.calculate_jet_production()
```

And subsequently the data can be stored and returned in a DataFrame format by calling the following function: 

```python
tshd.create_jet_production_data()
```

### Functions
The first function **tshd.calculate\_jet\_production** uses the functions below; 

**self.\_calculate_hi\_jet(self, vc=None)**
- Calculates cutting depth
- Requires trailing velocity [Defaults to self.vc\_data]
- Checks self.hi\_method
- Limits data to self.max\_hi
- Returns self.hi\_jet

self.**\_calculate\_jet\_width(self)**
- Calculates cavity width
- limits cavity width to the nozzle distance
- returns jet width

self.**\_calculate\_draghead\_pressure (self, hi=None)**
- Calculates draghead pressure
- Requires limited cutting depth [Defaults to self.hi\_jet]
- Returns self.draghead\_pressure

self.**\_calculate\_Q\_pipe(self)**
- Calculates discharge based on pipe diameter and line speed
- Returns self.q\_pipe

self.**\_calculate\_constant\_volume\_system(self, pressure=None)**
- Calculates the Constant Volume System
- Requires pressure in the draghead [Defaults to self.draghead\_pressure]
- Returns self.cvs

self.**\_calculate\_mixture\_density(self, cvs=None)**
- Calculates the density of the mixture
- Requires constant volume system self.cvs
- Returns self.mixture\_density


## Cutting Production
In order to calculate the cutting production we first set up the moment balance to obtain the actual cutting depth. We do this based on cutting forces and the (relative) gravity force on the visor. Please check [the background file](https://github.com/Daanunder/TSHD\_production\_estimation/blob/f6204a9109c82bab55e72d9469810729a406d4d1/Background.ipynb 'Background explanation of the model') for the theory behind this.

### Usage
#### Calculate initial moment and forces for all velocities
To get a starting point for the iterations, with the aim to obtain the balance angle for every given trailing velocity, we first calculate the forces and moments that would be generated for every velocity given the initially defined visor angle; 

```python
tshd.calculate_forces_and_moments()
```

If we then define the 'switchpoint' as the velocity where the moment as a result of the cutting force exceeds the moment induced by the gravity force - i.e. the velocity at which the visor will come loose from its support, we can inspect the different parameters around this point.

```python
tshd.switch_point = np.argmax(np.where(df['M_res [kNm]']>0, -np.inf, df['M_res [kNm]']))
tshd.df[tshd.switch_point-20:tshd.switch_point+20]
```

#### Run numerical iteration to obtain actual forces and moments and plot iterations
Finally we can 
```python
df = tshd.run_main_iteration(log=False, plot=True)
```

#### Plot model for three velocities
```python
tshd.plot_model(tshd.breakpoint-100)
tshd.plot_model(tshd.breakpoint)
tshd.plot_model(tshd.breakpoint+100)
```

### Functions

#### Functions used by **tshd.calculate\_forces\_and\_momens()**; 
**self.\_calculate\_gravity\_force(self):**
- Calculates self weight forcing based on self.visor\_weight
- Returns self.gravity\_force

**self.\_calculate\_hi\_cutting(self, cur\_index=None):**
- Calculates cutting depth based on the visor angle, the visor length and the hi\_jet
- Requires hi\_jet 
- Return self.hi\_cutting
    
**self.\_calculate\_shear\_angle(self, cur\_index=None):**
- Calculates the shear angle of the soil based on the hi cutting, blade angle and friction angles based on empirical relationship
- Requires hi\_cutting
- Returns self.shear\_angle
        
**self.\_calculate\_nondimensional\_forces(self, cur\_index=None):**
- Calculates horizontal and vertical dimensionless cutting forces d1 and d2
- Requires self.shear\_angle
- Returns list [self.d1, self.d2]
    
   
**self.\_calculate\_cutting\_force\_on\_blade(self, cur\_index=None):**
- Calculates hor. and vert. cutting forces on blade based on the non dimensional cutting force, cutting depth and contextual parameters
- Requires self.hi\_cutting, self.d1, self.d2
- Returns list [self.horizontal\_cutting\_force, self.vertical\_cutting\_force]

**self.\_calculate\_arms(self, cur\_index=None):**
- Calculates the arm of the moment-force for the horizontal cutting force and the gravity force based on the visor angle
- Returns list [self.hor\_cutting\_force\_arm, self.gravity\_force\_arm]
    
**self.\_calculate\_moments(self):**
- Calculates the balance of moments. Based on the moments induced by forces, as of yet including; gravity force, horizontal cutting force.
- Requires self.gravity\_force\_arm self.cutting\_force\_arm self.horizontal\_cutting\_force self.gravity\_force
- Returns self.moment\_balance
    

#### Functions used by **tshd.run\_main\_iteration(log=[True,False], plot=[True,False])**; 

**self.\_calculate\_angle\_from\_momentbalance(self,cur\_index):**
- Calculates the balance angle based on transcendental equation by means of a quick iteration. This equation is a result of the moment balance for a situation in which forces and thus moments are assumed to be known.
- Returns balance_angle
                
**self.iterate\_angles(self, log=False):**
- Calculates the angle the visor will make with the horizontal based on the iterative moment balance. In short it works as follows;
    - It gets the first row index (trailing velocity) where self.moment\_balance is not positive and loops over all subsequent indices.
    - It sets some bounds for the max and min visor angle based on realistic constraints of the visor
    - Uses bisection method untill self.moment\_balance < self.accuracy
- Requires 
- All iterations are saved in a dataframe called self.iter\_df
- Using log=True all iterations are printed to the screen
- Returns self.iter\_df
    
**self.plot\_iterations(self, N=15):**
- Plots iterations as computed by self.iterate\_angles()
- Returns plt.show()
    
**self.create\_cutting\_production_data(self):**
- Creates dataframe including cutting forces, moments and depth 
- Returns self.df

## Total Production
### Usage

#### Create total production data

tshd.create_total_production_data()

#### Plot data

tshd.plot_production_data()

#### Compare influence of different parameters

tshd = trailing_suction_hopper_dredger()
tshd.model_comparison('initial_visor_angle', [15\*np.pi/180,60\*np.pi/180], N=5)
tshd.model_comparison('effective_width_power', [0.5, 1.0], N=5)
tshd.model_comparison('internal_friction_angle', [26/180*np.pi, 45/180*np.pi], N=5)
tshd.model_comparison('hi_method', ['Miedema', 'CSB'], explicit_range=True)


### Functions
    **self.create\_total\_production\_data(self):
        self.production\_df['hi\_cut'] = self.df['hi\_cut [m]']
        self.production\_df['hi\_total [m]'] = self.df['hi\_jet [m]'] + self.df['hi\_cut [m]']
        self.production\_df['total p\_draghead'] = self.\_calculate\_draghead\_pressure(hi = self.production\_df['hi\_total [m]'])
        self.\_calculate\_Q\_pipe()
        self.production\_df['total cvs'] = self.\_calculate\_constant\_volume\_system(pressure=self.production\_df['total p\_draghead'])
        self.production\_df['total mixture density'] = self.\_calculate\_mixture\_density(cvs = self.production\_df['total cvs'])
        
        return self.production\_df
    
    
    **self.plot\_production\_data(self, simple=False, extra\_label=None, ax=None):
        if simple:
            if ax == None:
                fig, ax = plt.subplots(figsize=(10,10))
            line = self.production\_df.iloc[:,[0, -1]].plot(x='vc [m/s]', ax=ax)
            label = self.production\_df.columns[-1] + ' - ' + extra\_label
            return line, label

        else:
            fig,[[ax1, ax2],[ax3, ax4]] = plt.subplots(2,2, figsize=(20,20))

            self.production\_df.iloc[:,[0, -1, 5]].plot(x='vc [m/s]', ax=ax1)
            ax1.axvline(self.production\_df['vc [m/s]'][self.breakpoint], color='red', ls='dashdot', lw=0.5, label='Velocity where hi\_cut > hb')
            ax1.legend()

            self.production\_df.iloc[:,[0, -4, -5, 1]].plot(x='vc [m/s]', ax=ax2)
            ax2.axvline(self.production\_df['vc [m/s]'][self.breakpoint], color='red', ls='dashdot', lw=0.5, label='Velocity where hi\_cut > hb')
            ax2.legend()

            self.production\_df.iloc[:,[0, -3, 2]].plot(x='vc [m/s]', ax=ax3)
            ax3.axvline(self.production\_df['vc [m/s]'][self.breakpoint], color='red', ls='dashdot', lw=0.5, label='Velocity where hi\_cut > hb')
            ax3.legend()

            self.production\_df.iloc[:,[0, -2, 4]].plot(x='vc [m/s]', ax=ax4)
            ax4.axvline(self.production\_df['vc [m/s]'][self.breakpoint], color='red', ls='dashdot', lw=0.5, label='Velocity where hi\_cut > hb')
            ax4.legend()
        
            return plt.show()
    
    
    **self.run\_main\_iteration(self, log=False, plot=False):
        tshd.\_reinitiate()
        tshd.create\_jet\_production\_data()
        tshd.calculate\_forces\_and\_moments()
        tshd.iterate\_angles(log=log)
        if plot:
            tshd.plot\_iterations()
        df = tshd.create\_dataframe()
        return df
```python

```

    
    
    

## Extra function

```python
            
    **self.model\_comparison(self, parameter, p\_range, N=5, log=False, explicit\_range=False):

        if not hasattr(self, parameter):
            raise(AttributeError,'This parameter does not exist. Be careful to select a parameter and a range that may be realistically compared.')
        if parameter in ['blade\_angle', 'initial\_visor\_angle', 'lower\_suction\_angle', 'internal\_friction\_angle', 'steel\_sand\_friction\_angle']:
            if max(abs(np.array(p\_range))) > np.pi:
                print('Warning! Angles have to be given in radians. It seems this angle is > 180 degrees. Check ya self before ya reck ya self.')
                
                
        fig, ax = plt.subplots(figsize=(10,10))
        labels = []
        if not explicit\_range:
            p\_range = np.linspace(p\_range[0], p\_range[1], N)
            
        for p in p\_range:
            if parameter in ['blade\_angle', 'initial\_visor\_angle', 'lower\_suction\_angle', 'internal\_friction\_angle', 'steel\_sand\_friction\_angle']:
                p\_string = p*180/np.pi
            else:
                p\_string = p
            print(f'Running new iteration for {parameter} = {p\_string}')
            
            setattr(self, parameter, p)
            self.run\_main\_iteration(log=log)
            self.create\_total\_production\_data()
            
            extra\_label = f'{parameter} = {round(p\_string,2)}'
            line, label = self.plot\_production\_data(simple=True, extra\_label=extra\_label, ax=ax)
            labels.append(label)
        
        handles, wrong\_labels = ax.get\_legend\_handles\_labels()
        ax.legend(handles, labels)
        return plt.show()

```
