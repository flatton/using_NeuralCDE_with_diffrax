# Getting started
## Controlled differential equations (CDEs)

```python
from diffrax import AbstractPath, ControlTerm, diffeqsolve, Dopri5


class QuadraticPath(AbstractPath):
    @property
    def t0(self):
        return 0

    @property
    def t1(self):
        return 3

    def evaluate(self, t0, t1=None, left=True):
        del left
        if t1 is not None:
            return self.evaluate(t1) - self.evaluate(t0)
        return t0 ** 2


vector_field = lambda t, y, args: -y
control = QuadraticPath()
term = ControlTerm(vector_field, control).to_ode()
solver = Dopri5()
sol = diffeqsolve(term, solver, t0=0, t1=3, dt0=0.05, y0=1)

print(sol.ts)  # DeviceArray([3.])
print(sol.ys)  # DeviceArray([0.00012341])
```