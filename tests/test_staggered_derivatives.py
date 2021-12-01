from jaxdf.discretization import StaggeredRealFourier
from jaxdf.geometry import Domain, Staggered
from jaxdf import operators as jops
from jaxdf import operator

def test_3d_derivative():
    domain = Domain((32,32,32),(1.,1., 1.))
    u_params, u = StaggeredRealFourier(domain).empty_field(name="u")
    u_params = u_params.at[15:20].set(1.)
    fwd_grad = jops.staggered_grad(1., 1., Staggered.FORWARD)

    @operator()
    def test_op(u):
        return fwd_grad(u)

    T = test_op(u=u)
    gp = T.get_global_params()
    f = T.get_field_on_grid(0)

    q = f(gp, {"u": u_params})

if __name__ == "__main__":
    test_3d_derivative()