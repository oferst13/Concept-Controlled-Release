# Resources
*.csv - rain files \n
*_releases#.csv - optimized release policies - check prefix \n
benchmark.py - run benchmark
optimize.py - run GA for selected rain file
plot_with_rain.py - run the selected release policy and plot
swmm_compare.py - compare flow patterns of kinematic and dynamic wave
plot_tank_storage.py - plot tank storage throughout the simulated event
plot_release_policy.py - plot release policy throughout the simulated event


Code for GA seeding:
pop=np.array([np.zeros(self.dim+1)]*self.pop_s)
        solo=np.zeros(self.dim+1)
        solo0=solo.copy()
        solo1=np.ones(self.dim+1)
        var=np.zeros(self.dim)
        obj0 = self.sim(var)
        solo0[self.dim] = obj0
        pop[0] = solo0.copy()
        var1=np.ones(self.dim)
        obj1 = self.sim(var1)
        solo1[self.dim] = obj1
        pop[1] = solo1.copy()
        for p in range(2,self.pop_s):
