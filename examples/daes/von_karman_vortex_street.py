"""Von Karman vortex street: incompressible Navier-Stokes as an index-1 DAE.

This example couples `FEniCSx <https://fenicsproject.org/>`_ (finite element
discretization in space) with `solve_dae`'s Radau IIA method (time
integration) to simulate the unsteady, incompressible flow of a viscous
fluid around a cylinder, i.e. the classical Karman vortex street.

Problem setup
-------------
The domain is a channel with a circular obstacle, following the well known
DFG flow-around-a-cylinder benchmark (case 2D-2, Reynolds number 100):

    Schaefer, M., Turek, S. (1996). Benchmark computations of laminar flow
    around a cylinder. In: Flow Simulation with High-Performance Computers
    II. https://doi.org/10.1007/978-3-322-89849-4_39

A parabolic, ramped-up inflow profile enters on the left, no-slip
conditions are imposed on the channel walls and the cylinder, and a
natural (do-nothing) condition is used on the outflow boundary.

DAE formulation
----------------
Using a Taylor-Hood (P2/P1) element pair for velocity/pressure and the
method of lines, the semi-discrete weak form of the incompressible
Navier-Stokes equations

    du/dt = -(u . grad) u + nu * div(grad(u)) - grad(p)
    0     = div(u)

is a linear-implicit, index-1 DAE M @ y' = f(t, y) with y = (u, p): the
mass matrix M is singular (it has no du/dt-row for the pressure block), so
the incompressibility constraint div(u) = 0 is enforced *algebraically*
rather than differentiated, exactly as in Hairer & Wanner's index-1
formulation of Navier-Stokes/Stokes-type problems. `solve_dae` solves this
DAE directly via its implicit Radau IIA collocation method, using the exact
Jacobians of the finite element residual assembled by FEniCSx.

Dependencies
------------
This script needs FEniCSx (dolfinx) in addition to `solve_dae`, which is
not part of `solve_dae`'s regular dependencies since it cannot be installed
with plain ``pip`` on every platform. Recommended installation routes:

* conda/mamba (any OS)::

    conda install -c conda-forge fenics-dolfinx mpich pyvista gmsh python-gmsh matplotlib

* Debian/Ubuntu (has packaged dolfinx 0.10+)::

    sudo apt install fenicsx python3-gmsh

* Google Colab, without any local installation, using the `FEM on Colab
  <https://fem-on-colab.github.io/>`_ project (run as the first notebook
  cell)::

    import os, sys
    if "google.colab" in sys.modules:
        !wget -nc https://fem-on-colab.github.io/releases/fenicsx-install-release-real.sh -O /tmp/fenicsx-install.sh
        !bash /tmp/fenicsx-install.sh
        !pip install solve_dae gmsh

  or simply open the ready-made notebook
  `von_karman_vortex_street.ipynb <von_karman_vortex_street.ipynb>`_, which
  does all of the above automatically and packages the resulting GIF/VTK
  output for download.

Running this script produces ``von_karman_vortex_street.gif`` (velocity
magnitude over time) in the current working directory, as well as a
``von_karman_vortex_street_vtk/`` folder holding ``u.pvd``/``p.pvd`` time
series that can be opened in ParaView for a full velocity/pressure field
inspection.
"""
import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import basix
import basix.ufl
import dolfinx.fem as dfem
import dolfinx.fem.petsc as dfem_petsc
import dolfinx.io.gmsh as dgmsh
from dolfinx.io import VTKFile
import ufl

from scipy.sparse import csr_matrix
from solve_dae.integrate import solve_dae

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.animation as animation


def dolfinx_to_scipy(A):
    """Convert an assembled PETSc matrix to a SciPy CSR matrix."""
    A_csr = A.getValuesCSR()
    return csr_matrix(A_csr[::-1], shape=A.getSize())


def create_mesh(characteristic_length, extends=(2.2, 0.41), center=(0.2, 0.2), radius=0.05):
    """Build a "channel with circular obstacle" mesh using gmsh.

    Facet tags follow the DFG benchmark convention: 1 = inflow (left),
    2 = bottom wall, 3 = outflow (right), 4 = top wall, 5 = cylinder.
    """
    import gmsh

    gmsh.initialize()
    model = gmsh.model
    model.add("channel_with_cylinder")
    model.setCurrent("channel_with_cylinder")

    dim = 2
    channel = model.occ.addRectangle(0, 0, 0, *extends)
    disk = model.occ.addDisk(*center, 0, radius, radius)
    fluid = model.occ.cut([(dim, channel)], [(dim, disk)])
    model.occ.synchronize()

    volumes = model.getEntities(dim=dim)
    assert volumes == fluid[0]
    fluid_marker = 11
    model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], fluid_marker)
    model.setPhysicalName(volumes[0][0], fluid_marker, "fluid")

    left_marker, bottom_marker, right_marker, top_marker, disk_marker = 1, 2, 3, 4, 5
    boundaries = [
        (left_marker, lambda x: np.isclose(x[0], 0)),
        (bottom_marker, lambda x: np.isclose(x[1], 0)),
        (right_marker, lambda x: np.isclose(x[0], extends[0])),
        (top_marker, lambda x: np.isclose(x[1], extends[1])),
    ]

    left, bottom, right, top = None, None, None, None
    disk_lines = []
    for line in model.getEntities(dim=dim - 1):
        com = model.occ.getCenterOfMass(line[0], line[1])
        for marker, locator in boundaries:
            if locator(com):
                if marker == left_marker:
                    left = line[1]
                elif marker == bottom_marker:
                    bottom = line[1]
                elif marker == right_marker:
                    right = line[1]
                elif marker == top_marker:
                    top = line[1]
                break
        else:
            # none of the straight boundaries matched -> part of the disk
            disk_lines.append(line[1])

    model.addPhysicalGroup(dim - 1, [left], left_marker, "left")
    model.addPhysicalGroup(dim - 1, [bottom], bottom_marker, "bottom")
    model.addPhysicalGroup(dim - 1, [right], right_marker, "right")
    model.addPhysicalGroup(dim - 1, [top], top_marker, "top")
    model.addPhysicalGroup(dim - 1, disk_lines, disk_marker, "disk")

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", characteristic_length)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", characteristic_length)
    model.mesh.generate(dim=dim)

    mesh_data = dgmsh.model_to_mesh(model, MPI.COMM_WORLD, rank=0, gdim=dim)
    gmsh.finalize()

    markers = dict(left=left_marker, bottom=bottom_marker, right=right_marker,
                   top=top_marker, disk=disk_marker)
    return mesh_data, markers, extends


if __name__ == "__main__":
    # ----------------------------------------------------------------
    # physical parameters (DFG 2D-2 benchmark: mean inflow velocity 1,
    # cylinder diameter 0.1, kinematic viscosity 1e-3 -> Reynolds number
    # Re = 1 * 0.1 / 1e-3 = 100)
    # ----------------------------------------------------------------
    rho = 1.0
    nu = 1.0e-3
    Ubar = 1.5  # peak (centerline) inflow velocity

    mesh_data, markers, extends = create_mesh(characteristic_length=2.5e-2)
    mesh = mesh_data.mesh
    facet_tags = mesh_data.facet_tags
    gdim = mesh.geometry.dim

    # Taylor-Hood element pair (P2 velocity, P1 pressure)
    k = 1
    Pu = basix.ufl.element("Lagrange", mesh.basix_cell(), k + 1, shape=(gdim,))
    Pp = basix.ufl.element("Lagrange", mesh.basix_cell(), k)
    V = dfem.functionspace(mesh, basix.ufl.mixed_element([Pu, Pp]))

    u, p = ufl.TrialFunctions(V)
    v_u, v_p = ufl.TestFunctions(V)

    w = dfem.Function(V)   # DAE state y = (u, p)
    wp = dfem.Function(V)  # DAE state derivative y' = (u', p')
    u_h, p_h = ufl.split(w)

    # index-1 DAE: incompressibility div(u) = 0 replaces the (non-existent)
    # time derivative of the pressure -> mass matrix `am` has an empty
    # pressure-row block, `F` carries the full nonlinear spatial operator.
    F = (
        nu * ufl.inner(ufl.grad(v_u), ufl.grad(u_h)) * ufl.dx
        + rho * ufl.inner(v_u, ufl.grad(u_h) * u_h) * ufl.dx
        + ufl.inner(v_p, ufl.div(u_h)) * ufl.dx
    )
    am = (
        ufl.inner(v_u, u) * ufl.dx
        - ufl.inner(ufl.div(v_u), p) * ufl.dx
    )
    J = ufl.derivative(F, w)

    # ----------------------------------------------------------------
    # boundary conditions
    # ----------------------------------------------------------------
    V_u, _ = V.sub(0).collapse()
    mesh.topology.create_connectivity(gdim - 1, gdim)

    bcs = []

    # no-slip: bottom wall, top wall, cylinder
    wall_velocity = dfem.Function(V_u)
    wall_velocity.x.array[:] = 0.0
    for tag in (markers["bottom"], markers["top"], markers["disk"]):
        fcts = facet_tags.find(tag)
        dofs = dfem.locate_dofs_topological((V.sub(0), V_u), gdim - 1, fcts)
        bcs.append(dfem.dirichletbc(wall_velocity, dofs, V.sub(0)))

    def smoothstep(x, x_min=0.0, x_max=1.0):
        """C2 smoothstep, used to ramp up the inflow velocity from rest."""
        x = np.clip((x - x_min) / (x_max - x_min), 0.0, 1.0)
        return 6 * x**5 - 15 * x**4 + 10 * x**3

    def ramp(t, t_ramp=2.0):
        return smoothstep(t, 0.0, t_ramp)

    def inflow_profile(x, t):
        fy = 4.0 * x[1] * (extends[1] - x[1]) / extends[1] ** 2
        return np.stack((Ubar * ramp(t) * fy, np.zeros(x.shape[1])))

    inflow_velocity = dfem.Function(V_u)
    inflow_velocity.interpolate(lambda x: inflow_profile(x, 0.0))
    fcts = facet_tags.find(markers["left"])
    dofs = dfem.locate_dofs_topological((V.sub(0), V_u), gdim - 1, fcts)
    bcs.append(dfem.dirichletbc(inflow_velocity, dofs, V.sub(0)))
    # outflow (right boundary): natural (do-nothing) condition, no BC needed.

    # ----------------------------------------------------------------
    # assembly
    # ----------------------------------------------------------------
    form_am = dfem.form(am)
    residual_form = dfem.form(F)
    jacobian_form = dfem.form(J)

    M = dfem_petsc.create_matrix(form_am)
    A = dfem_petsc.create_matrix(jacobian_form)

    dfem_petsc.assemble_matrix(M, form_am, bcs=bcs)
    M.assemble()
    M_scipy = dolfinx_to_scipy(M)

    def fun_dae(t, y, yp):
        inflow_velocity.interpolate(lambda x: inflow_profile(x, t))

        w.x.array[:] = y
        wp.x.array[:] = yp

        L = dfem_petsc.assemble_vector(residual_form)
        L.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        L.scale(-1)

        A.zeroEntries()
        dfem_petsc.assemble_matrix(A, jacobian_form, bcs=bcs)
        A.assemble()

        # lift the Dirichlet rows: b - J (u_D - u_{k-1}), then set du|_bc = u_D - u_{k-1}
        dfem_petsc.apply_lifting(L, [jacobian_form], [bcs], x0=[w.x.petsc_vec], alpha=1)
        dfem_petsc.set_bc(L, bcs, w.x.petsc_vec, 1.0)
        L.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)

        return M_scipy @ yp - L.array

    def jac_dae(t, y, yp):
        w.x.array[:] = y
        wp.x.array[:] = yp

        A.zeroEntries()
        dfem_petsc.assemble_matrix(A, jacobian_form, bcs=bcs)
        A.assemble()

        return dolfinx_to_scipy(A), M_scipy

    # ----------------------------------------------------------------
    # time integration
    # ----------------------------------------------------------------
    # the fluid starts at rest, which is already a consistent initial
    # state since the ramped-up inflow is zero at t = 0
    t0, t1 = 0.0, 10.0
    w.x.array[:] = 0.0
    wp.x.array[:] = 0.0
    y0 = w.x.array.copy()
    yp0 = wp.x.array.copy()

    t_eval = np.linspace(t0, t1, num=300)

    start = time.time()
    sol = solve_dae(
        fun_dae, (t0, t1), y0, yp0,
        t_eval=t_eval,
        jac=jac_dae,
        method="Radau",
        rtol=1e-4,
        atol=1e-4,
        first_step=1e-3,
    )
    print(f"elapsed time:  {time.time() - start:.2f} s")
    print(f"success:       {sol.success}")
    print(f"status:        {sol.status} ({sol.message})")
    print(f"nfev / njev / nlu: {sol.nfev} / {sol.njev} / {sol.nlu}")

    # ----------------------------------------------------------------
    # VTK output: velocity and pressure time series, for ParaView
    # ----------------------------------------------------------------
    u_out, p_out = w.sub(0).collapse(), w.sub(1).collapse()
    u_out.name, p_out.name = "u", "p"
    with (
        VTKFile(mesh.comm, "von_karman_vortex_street_vtk/u.pvd", "w") as vtk_u,
        VTKFile(mesh.comm, "von_karman_vortex_street_vtk/p.pvd", "w") as vtk_p,
    ):
        for ti, yi in zip(sol.t, sol.y.T):
            w.x.array[:] = yi
            u_out.x.array[:] = w.sub(0).collapse().x.array
            p_out.x.array[:] = w.sub(1).collapse().x.array
            vtk_u.write_function(u_out, ti)
            vtk_p.write_function(p_out, ti)
    print("Saved velocity/pressure time series to von_karman_vortex_street_vtk/")

    # ----------------------------------------------------------------
    # visualization: velocity magnitude, animated over time
    # ----------------------------------------------------------------
    P1 = dfem.functionspace(mesh, ("Lagrange", 1))
    speed_h = dfem.Function(P1)
    speed_expr = dfem.Expression(ufl.sqrt(ufl.inner(u_h, u_h)), P1.element.interpolation_points)

    coords = P1.tabulate_dof_coordinates()[:, :2]
    triangles = mesh.topology.connectivity(gdim, 0).array.reshape(-1, 3)
    triangulation = mtri.Triangulation(coords[:, 0], coords[:, 1], triangles)

    fig, ax = plt.subplots(figsize=(8, 2))
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    title = ax.set_title("")
    levels = np.linspace(0, 2.2 * Ubar, 31)

    def update(frame):
        w.x.array[:] = sol.y[:, frame]
        speed_h.interpolate(speed_expr)
        for coll in ax.collections:
            coll.remove()
        ax.tricontourf(triangulation, speed_h.x.array, levels=levels, cmap="viridis", extend="max")
        title.set_text(f"|u|,  t = {sol.t[frame]:.2f} s")

    ani = animation.FuncAnimation(fig, update, frames=len(sol.t))
    ani.save("von_karman_vortex_street.gif", writer=animation.PillowWriter(fps=20))
    print("Saved animation to von_karman_vortex_street.gif")
