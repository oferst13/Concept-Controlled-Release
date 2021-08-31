# Resources
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
