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
      * [Total Production](#total-production)
      * [Extra function](#extra-function)

<!-- Added by: garry, at: Fri 19 Mar 2021 10:30:30 PM UTC -->

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
tshd.trailing_velocity = 5.0
tshd._reinitiate()
```

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
**self.\_calculate_hi\_jet(self, vc=None)**
- Calculates cutting depth
- Requires trailing velocity [Defaults to self.vc\_data]
- Checks self.hi\_method
- Limits data to self.max\_hi
- Returns self.hi\_jet

**\_calculate\_jet\_width(self)**
- Calculates cavity width
- limits cavity width to the nozzle distance
- returns jet width

**\_calculate\_draghead\_pressure (self, hi=None)**
- Calculates draghead pressure
- Requires limited cutting depth [Defaults to self.hi\_jet]
- Returns self.draghead\_pressure

**\_calculate\_Q\_pipe(self)**
- Calculates discharge based on pipe diameter and line speed
- Returns self.q\_pipe

**\_calculate\_constant\_volume\_system(self, pressure=None)**
- Calculates the Constant Volume System
- Requires pressure in the draghead [Defaults to self.draghead\_pressure]
- Returns self.cvs

**\_calculate\_mixture\_density(self, cvs=None)**
- Calculates the density of the mixture
- Requires constant volume system self.cvs
- Returns self.mixture\_density


## Cutting Production
For the cuttin 

```python
    ##############
    ### FORCES ###
    ##############
    # - F impuls mixture V
    # - F vacuum V
    # - F sled visor V
    # - F gravity V
    # - F cutting V


    #Define a function 'Impulseforce1()' due to redirection of the sand mixture
    def \_calculate\_impulseforce\_sand\_redirection(self):
        '''
        NOT IMPLEMENTED IN MOMENT BALANCE
        
        Calculates impulseforce as a result of the sand water mixture redirection in the visor
        Requires self.mixture\_density
        Returns self.impulsforce\_sand\_redirection, impulsvertical, impulshorizontal
        '''
        #self.sand\_water\_situ\_density = self.sand\_density*(1-n)+self.water\_density*self.situ\_porosity
        
        beta\_v = np.pi-self.lower\_suction\_angle
        self.pipe\_area = np.pi/4*self.pipe\_diam**2
        impulsehorizontal1 = (self.mixture\_density*self.pipe\_diam*self.line\_speed**2*(1-np.cos(beta\_v)))**2                 #squared
        impulsevertical1 = (self.mixture\_density*self.pipe\_diam*self.line\_speed**2*(np.sin(beta\_v)))**2                     #squared
        impulsetotal1 = np.sqrt(impulsehorizontal1+impulsevertical1)
        self.impulsforce\_sand\_redirection = [impulsehorizontal1, impulsevertical1]
        return impulsetotal1, impulsevertical1, impulsehorizontal1

    #Define a function 'Vacuumforce'
    def \_calculate\_vacuumforce(self):
        '''
        NOT IMPLEMENTED IN MOMENT BALANCE
        
        Calculates vacuumforce as result of negative pressure at the draghead mouth
        Requires self.mixture\_density
        Returns self.vacuumforce 

        Vacuumforce ontbinding?
        '''
        vacuumforce = C*0.5*self.mixture\_density*self.line\_speed**2*self.visor\_area
        self.vacuumforce = vacuumforce
        return vacuumforce

    def \_calculate\_gravity\_force(self):
        '''
        Calculates self weight forcing based on self.visor\_weight
        Returns self.gravity\_force
        '''
        relative\_density = (self.steel\_density - self.water\_density)/self.steel\_density
        self.gravity\_force = self.visor\_weight * relative\_density * 9.81
        return self.gravity\_force


    '''
    Calculating cutting forces:
    - First we determine the horizontal non-dimensional forces d1 and c1
    - Then we calculate the cutting forces on the blade
    - We find the angle for which a balance of moments is found (So far only for Fgravity & Fcutting)
    - We calculate the cutting forces again and iterate
    '''
    
    def \_hi\_cutting\_formulae(self, visor\_angle, hi\_jet):
        total\_height = np.sin(visor\_angle) * self.visor\_hor\_radius + self.blade\_height
        hi\_cutting = np.clip(total\_height - hi\_jet, 0, self.blade\_height)
        return hi\_cutting
    
    def \_calculate\_hi\_cutting(self, cur\_index=None):
        '''
        Calculates cutting depth based on the visor angle, the visor length and the hi\_jet
        Requires hi\_jet
        '''
        if cur\_index == None:
            visor\_angle = self.initial\_visor\_angle
            hi\_jet = self.hi\_jet
            self.hi\_cutting = self.\_hi\_cutting\_formulae(visor\_angle, hi\_jet)

        else:
            visor\_angle = self.visor\_angle\_array[cur\_index]
            hi\_jet = self.hi\_jet[cur\_index]
            self.hi\_cutting[cur\_index] = self.\_hi\_cutting\_formulae(visor\_angle, hi\_jet)
            
        return self.hi\_cutting
    
    def \_shear\_angle\_formulae(self, hi\_cutting):
        beta = 1 - 1/6*self.blade\_angle -2/7*(self.internal\_friction\_angle+self.steel\_sand\_friction\_angle)-0.057 *self.blade\_height/hi\_cutting
        beta = np.clip(beta, self.lower\_limit\_beta, self.upper\_limit\_beta)
        return beta
        
    def \_calculate\_shear\_angle(self, cur\_index=None):
        '''
        Calculates the shear angle of the soil based on the hi cutting, blade angle and friction angles
        Requires hi\_cutting
        Returns shear angle based on empirical relationship
        '''
        
        if cur\_index==None:
            hi\_cutting = self.hi\_cutting
            self.shear\_angle = self.\_shear\_angle\_formulae(hi\_cutting)
        else:
            hi\_cutting = self.hi\_cutting[cur\_index]
            self.shear\_angle[cur\_index] = self.\_shear\_angle\_formulae(hi\_cutting)
            
        #self.shear\_angle1 = 61.29*np.pi/180 + 0.345 * self.blade\_height / hi\_cutting - 0.3068 * self.blade\_angle- 0.4736*self.steel\_sand\_friction\_angle - 0.248 * self.internal\_friction\_angle
        #self.shear\_angle1 = np.clip(self.shear\_angle1, self.lower\_limit\_beta, self.upper\_limit\_beta)
        
        return self.shear\_angle
    
    def \_nondimensional\_force\_formulae(self, hi, alpha, beta, phi, delta):
        if hi > 0:

               # Horizontal 
            part\_1 = np.sin(phi)/np.sin(beta) + self.blade\_height/hi*np.sin(min(alpha + beta + phi, np.pi))/np.sin(alpha)

            part\_2 = np.sin(alpha + delta)/np.sin(min(phi + beta + alpha + delta, np.pi))
            part\_3 = self.blade\_height / hi *np.sin(alpha)/np.sin(alpha)
            d1 = np.clip(part\_1*part\_2 - part\_3, 0, None)

            # vertical 
            part\_2 = np.cos(alpha + delta)/np.sin(min(phi + beta + alpha + delta, np.pi))
            part\_3 = self.blade\_height / hi *np.cos(alpha)/np.sin(alpha)
            d2 = np.clip(part\_1*part\_2 - part\_3, 0, None)

        else: 
            d1 = 0
            d2 = 0
            
        return d1, d2
                    
            

    def \_calculate\_nondimensional\_forces(self, cur\_index=None):

        '''
        Calculates horizontal and vertical dimensionless cutting forces d1 and d2
        Requires self.shear\_angle
        Returns [self.d1, self.d2]
        '''
        alpha = self.blade\_angle
        delta = self.steel\_sand\_friction\_angle
        phi = self.internal\_friction\_angle
        beta\_array = self.shear\_angle 

        if cur\_index==None:
            hi\_cutting = self.hi\_cutting

            self.d1 = np.zeros(self.nodes)
            self.d2 = np.zeros(self.nodes)

            for k, hi in enumerate(hi\_cutting):
                beta = beta\_array[k]
                d1, d2 = self.\_nondimensional\_force\_formulae(hi, alpha, beta, phi, delta)
                self.d1[k] = d1 
                self.d2[k] = d2
                
        else:
            hi\_cutting = self.hi\_cutting[cur\_index]
            beta = beta\_array[cur\_index]
            d1, d2 = self.\_nondimensional\_force\_formulae(hi\_cutting, alpha, beta, phi, delta)
            self.d1[cur\_index] = d1
            self.d2[cur\_index] = d2
            

        return self.d1, self.d2
    
    
    def \_check\_cavitation(self, hi=None):
        '''
        NOT IMPLEMENTED 
        
        What is km? And how can it be determined?
        Is cutting velocity the same as trailing velocity?
        Why z+10? Because of the cavitation limit of 10 m.w.c.?
        For the first cavitating assumptions uses a single boolean value for self.cavitating, 
        still need to change for a full check per velocity.
        '''
        
        # FOR INITIAL ASSUMPTION OF CAVITATING CUTTING
        self.cavitating = True
        return True
        
        # Actual check NOT IN USE
        if hi == None:
            hi = self.hi\_cutting
        if trailing\_velocity == None:
            trailing\_velocity = self.trailing\_veloc
            
        cavitation\_velocity = self.d1 / self.d2 * (self.z\_coordinate+10)/hi * km/self.dilatation\_ratio
        self.cavitating = trailing\_velocity > cavitation\_velocity

    def \_cutting\_force\_formulae(self, d1, d2, hi\_cutting):
        fh = d1 * self.water\_density * 9.81 * (self.z\_coordinate + 10) * hi\_cutting * self.blade\_width
        fv = d2 * self.water\_density * 9.81 * (self.z\_coordinate + 10) * hi\_cutting * self.blade\_width
        return fh, fv
    
    def \_calculate\_cutting\_force\_on\_blade(self, cur\_index=None):
        '''
        Calculates hor. and vert. cutting forces on blade based on the non dimensional cutting force, cutting depth and contextual parameters
        Requires self.hi\_cutting self.d1 self.d2
        Returns self.horizontal\_cutting\_force, self.vertical\_cutting\_force
        '''
        if cur\_index == None:
            hi\_cutting = self.hi\_cutting
            d1 = self.d1
            d2 = self.d2
            if self.cavitating:
                self.horizontal\_cutting\_force, self.vertical\_cutting\_force = self.\_cutting\_force\_formulae(d1, d2, hi\_cutting)
            
            else:
                print('Non-cavitating has not been implemented yet!')
                pass
            
        else:
            hi\_cutting = self.hi\_cutting[cur\_index]
            d1 = self.d1[cur\_index]
            d2 = self.d2[cur\_index]
            self.horizontal\_cutting\_force[cur\_index], self.vertical\_cutting\_force[cur\_index] = self.\_cutting\_force\_formulae(d1, d2, hi\_cutting)
        return self.horizontal\_cutting\_force, self.vertical\_cutting\_force

    
    ###############
    ### MOMENTS ###
    ###############

    def \_calculate\_arms(self, cur\_index=None):
        if cur\_index == None:
            visor\_angle = self.visor\_angle\_array
            self.hor\_cutting\_force\_arm = self.visor\_hor\_radius*np.sin(visor\_angle)+self.blade\_height       
            self.gravity\_force\_arm = self.visor\_center\_of\_gravity\_hor*np.cos(visor\_angle)+self.visor\_center\_of\_gravity\_ver*np.sin(visor\_angle)

        else:
            visor\_angle = self.visor\_angle\_array[cur\_index]
            self.hor\_cutting\_force\_arm[cur\_index] = self.visor\_hor\_radius*np.sin(visor\_angle)+self.blade\_height        
            self.gravity\_force\_arm[cur\_index] = self.visor\_center\_of\_gravity\_hor*np.cos(visor\_angle)+self.visor\_center\_of\_gravity\_ver*np.sin(visor\_angle)

        return self.hor\_cutting\_force\_arm, self.gravity\_force\_arm 
    
    
    def \_calculate\_moments(self):
        '''
        Calculates the moments induced by forces, as of yet including; gravity force, horizontal cutting force
        Requires self.gravity\_force\_arm self.cutting\_force\_arm self.horizontal\_cutting\_force self.gravity\_force
        Returns balance of moments
        '''
        self.hor\_cutting\_moment = - self.hor\_cutting\_force\_arm * self.horizontal\_cutting\_force #clockwise
        self.gravity\_moment = self.gravity\_force\_arm * self.gravity\_force # counter-clockwise
        
        self.moment\_balance = self.hor\_cutting\_moment + self.gravity\_moment
        return self.moment\_balance
    
    
        
    def calculate\_forces\_and\_moments(self, cur\_index=None):
        ## Forces
        # Gravity
        self.\_calculate\_gravity\_force()
        
        #Cutting
        self.\_calculate\_hi\_cutting(cur\_index)
        self.\_calculate\_shear\_angle(cur\_index)
        self.\_calculate\_nondimensional\_forces(cur\_index)
        self.\_check\_cavitation()
        self.\_calculate\_cutting\_force\_on\_blade(cur\_index)
        
        ## Moments
        # Arms
        self.\_calculate\_arms(cur\_index)
        
        # Balance
        self.\_calculate\_moments()
        
    
    def \_calculate\_angle\_from\_momentbalance(self,cur\_index):
        Fc = self.horizontal\_cutting\_force[cur\_index]
        Fg = self.gravity\_force
        Gy = self.visor\_center\_of\_gravity\_ver
        Gx = self.visor\_center\_of\_gravity\_hor
        Rv = self.visor\_hor\_radius
        
        err = 1
        prev\_angle = self.visor\_angle\_array[cur\_index]
        balance\_angle = None
        
        while err > 0.001:
            balance\_angle = np.arctan2(-self.blade\_height/np.cos(prev\_angle) * Fc + Gx*Fg, -Fg*Gy + Fc*Rv)
            prev\_angle = balance\_angle
            err = abs(prev\_angle - balance\_angle)
        
        return balance\_angle
                
    def iterate\_angles(self, log=False):
        
        # get first value where momentbalance is not positive
        kmin = np.where(self.moment\_balance < -self.accuracy, self.moment\_balance, -np.inf).argmax()
        self.kmin = kmin
        # set up dataframe to save iterations
        self.iter\_df = pd.DataFrame(index=np.arange(self.nodes - kmin), columns=['iters'])
        
        # Loop over all rows where moment balance is not positive
        for cur\_index in range(kmin, self.nodes):
            iters = []
            iters\_moment = []
            
            if log:
                print('\n')
                print(cur\_index)
                
            
            positive\_moment\_allowed = False
            #upbound = min(self.initial\_visor\_angle, np.arcsin(self.hi\_jet[cur\_index]/self.visor\_hor\_radius))
            upbound = self.initial\_visor\_angle
            lowbound = -np.arcsin(self.blade\_height/self.visor\_hor\_radius)  
        
            blade\_bound = np.arcsin(self.hi\_jet[cur\_index]/self.visor\_hor\_radius)
            # bisection method untill momentbalance is approx. zero
            if log:
                print('| Initial angle | ', 'Balance angle | ', 'New angle | ', 'Resulting moment | ')
            while True:

                # calculating new balance with bisect angle based on old angle and "balance angle"
                old\_angle = self.visor\_angle\_array[cur\_index]
                balance\_angle = self.\_calculate\_angle\_from\_momentbalance(cur\_index)
                
                if balance\_angle > blade\_bound and self.hi\_cutting[cur\_index] >= self.blade\_height:
                    positive\_moment\_allowed = True
                    if log:
                        print('Positive moment allowed')
                balance\_angle = np.clip(balance\_angle, lowbound, min(upbound, blade\_bound))
                
                # saving iteration for visualisation
                iters.append(self.visor\_angle\_array[cur\_index])
                
                # Bisection method
                new\_angle = (old\_angle+balance\_angle)/2
                self.visor\_angle\_array[cur\_index] = new\_angle

                self.calculate\_forces\_and\_moments(cur\_index)
                e = self.moment\_balance[cur\_index]

                if self.moment\_balance[cur\_index] > 0:
                    lowbound = new\_angle
                    
                # Save moments iteration for visualisation
                iters\_moment.append(self.moment\_balance[cur\_index])
                              
                if log:
                    print(old\_angle*180/np.pi, balance\_angle*180/np.pi, self.visor\_angle\_array[cur\_index]*180/np.pi, self.moment\_balance[cur\_index])
                
                
                # if accuracy is reached break and continue with next row (velocity)
                if abs(e) < self.accuracy or (positive\_moment\_allowed == True and e > 0):
                    self.visor\_angle\_array[cur\_index:] = new\_angle
                    self.iter\_df.iloc[cur\_index-kmin] = [np.array([[iters], [iters\_moment]])]
                    break
        
        return self.iter\_df
    
    def plot\_iterations(self, N=15):
        fig1,[ax1, ax2] = plt.subplots(1,2, figsize=(20,10))
        for q in np.arange(0,self.iter\_df.size, round(self.iter\_df.size/N)):
            angle = self.iter\_df['iters'][q][0][0]*180/np.pi
            moment = self.iter\_df['iters'][q][1][0]/1000
            n = np.arange(len(angle))
            ax1.plot(n, angle, label=f'vc = {round(self.vc\_data[q],2)}')
            ax2.plot(n, moment, label=f'vc = {round(self.vc\_data[q],2)}')
            ax1.legend()
            ax2.legend()
            
        return plt.show()
    
    
    def create\_dataframe(self):
        df = pd.DataFrame(index=np.arange(self.nodes), columns=['vc [m/s]', 'hi\_jet [m]', 'hi\_cut [m]', 'beta [rad]', 'beta [deg]', 'R\_g [m]', 'R\_ch [m]','d1 [-]', 'd2 [-]', 'Fh [kN]','Fv [kN]', 'Fg [kN]', 'M\_res [kNm]'])
        df['vc [m/s]'] = self.vc\_data
        df['hi\_jet [m]'] = self.hi\_jet
        df['hi\_cut [m]'] = self.hi\_cutting
        df['beta [rad]'] = self.shear\_angle
        df['beta [deg]'] = df['beta [rad]']*180/np.pi
        df['R\_g [m]'] = self.gravity\_force\_arm
        df['R\_ch [m]'] = self.hor\_cutting\_force\_arm
        df['d1 [-]'] = self.d1
        df['d2 [-]'] = self.d2
        df['Fh [kN]'] = self.horizontal\_cutting\_force / 1000
        df['Fv [kN]'] = self.vertical\_cutting\_force / 1000
        df['Fg [kN]'] = self.gravity\_force*np.ones(self.nodes) / 1000
        df['M\_res [kNm]'] = self.moment\_balance / 1000
        df = df.round(3)
        self.df = df
        
        self.breakpoint = np.argmin(np.where(self.df['hi\_cut [m]'] > self.blade\_height, self.df['hi\_cut [m]'], np.inf))

        return self.df
    
    
    def plot\_model(self, cur\_index=None):
        if cur\_index == None:
            visor\_angle = self.initial\_visor\_angle
        else:
            visor\_angle = self.visor\_angle\_array[cur\_index]

        # Create data points
        pipe\_arm\_lenght = self.z\_coordinate/self.lower\_suction\_angle
        pipe\_arm = [[0,np.cos(self.lower\_suction\_angle)*pipe\_arm\_lenght],[0,self.z\_coordinate]]

        visor = [[pipe\_arm[0][1], pipe\_arm[0][1]+np.cos(visor\_angle)*self.visor\_hor\_radius],[pipe\_arm[1][1], pipe\_arm[1][1]+np.sin(visor\_angle)*self.visor\_hor\_radius]]

        blade = [[visor[0][1], visor[0][1]-np.cos(self.blade\_angle)*self.blade\_lenght], [visor[1][1], visor[1][1]+self.blade\_height]]

        # Define plot and zoom
        fig,axes = plt.subplots(1,2)
        fig.set\_size\_inches(20, 10)

        ax = axes[0]
        axz = axes[1]

        ax.set\_ylim(self.z\_coordinate+5, -5)
        ax.set\_xlim(pipe\_arm[0][1]+5, -5) 
        axz.set\_ylim(self.z\_coordinate+1.5, self.z\_coordinate-1.5)
        axz.set\_xlim(pipe\_arm[0][1]+self.visor\_hor\_radius*1.1, pipe\_arm[0][1]-1) 
        ax.set\_aspect('equal')
        axz.set\_aspect('equal')

            
        origin = [np.cos(self.lower\_suction\_angle)*pipe\_arm\_lenght, self.z_coordinate]
        
        arc = Arc(origin, 0.7*self.visor\_hor\_radius, 0.4*self.visor\_hor\_radius, 0, 0, visor\_angle*180/np.pi, color='black', lw=1)
        axz.add\_patch(arc)        

        #plot blade
        ax.plot(blade[0], blade[1], label='Blade')
        axz.plot(blade[0], blade[1], label='Blade')

        #plot visor
        ax.plot(visor[0], visor[1], label='Visor')
        axz.plot(visor[0], visor[1], label='Visor')

        # plot arm
        ax.plot(pipe\_arm[0],pipe\_arm[1], label='Lower suction pipe}')
        axz.plot(pipe\_arm[0], pipe\_arm[1], label='Lower suction pipe')

        # Plot waterline and bottom
        ax.axhline(self.z\_coordinate, color='brown', ls='dashed', label='bottom')
        axz.axhline(self.z\_coordinate, color='brown', ls='dashed', label='bottom')    
        ax.axhline(0, color='blue', ls='dashed', label='sealevel')
        axz.axhline(0, color='blue', ls='dashed', label='sealevel')

        # Plot jet and cutting depth

        if cur\_index:            
            ax.axhline(self.hi\_jet[cur\_index]+self.z\_coordinate+self.hi\_cutting[cur\_index], color='green', ls='dashed', label='cutting depth')
            ax.axhline(self.hi\_jet[cur\_index]+self.z\_coordinate, color='orange',ls='dashed', label='jet depth')
            
            axz.axhline(self.hi\_jet[cur\_index]+self.z\_coordinate+self.hi\_cutting[cur\_index], color='green',ls='dashed', label='cutting depth')    
            axz.axhline(self.hi\_jet[cur\_index]+self.z\_coordinate, color='orange', ls='dashed', label='jet depth')
            
            ax.set\_title(f'Velocity = {self.vc\_data[cur\_index]}')
            axz.set\_title(f'Velocity = {self.vc\_data[cur\_index]}')
        
        axz.text(np.cos(self.lower\_suction\_angle)*pipe\_arm\_lenght+0.5*self.visor\_hor\_radius, self.z\_coordinate+0.05*self.visor\_hor\_radius, str(round(visor\_angle*180/np.pi,2))+u"\u00b0")
            
        ax.legend()
        axz.legend()

                
        return plt.show()

```

## Total Production

```
    def create\_total\_production\_data(self):
        self.production\_df['hi\_cut'] = self.df['hi\_cut [m]']
        self.production\_df['hi\_total [m]'] = self.df['hi\_jet [m]'] + self.df['hi\_cut [m]']
        self.production\_df['total p\_draghead'] = self.\_calculate\_draghead\_pressure(hi = self.production\_df['hi\_total [m]'])
        self.\_calculate\_Q\_pipe()
        self.production\_df['total cvs'] = self.\_calculate\_constant\_volume\_system(pressure=self.production\_df['total p\_draghead'])
        self.production\_df['total mixture density'] = self.\_calculate\_mixture\_density(cvs = self.production\_df['total cvs'])
        
        return self.production\_df
    
    
    def plot\_production\_data(self, simple=False, extra\_label=None, ax=None):
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
    
    
    def run\_main\_iteration(self, log=False, plot=False):
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
            
    def model\_comparison(self, parameter, p\_range, N=5, log=False, explicit\_range=False):

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
